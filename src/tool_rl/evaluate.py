"""Math evaluation with one independent model replica per GPU."""
import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from .environment import ToolEnvironment
from .experiment_data import evaluator_fingerprint, load_split, math_only, select_rows
from .io import metadata, sha256, write_json
from .records import stable_seed
from .rewards import grade
from .rollout import run_agent_episode, run_native_tool_episode
from .prompt_variants import OPTIONAL_TOOL, PROMPT_VARIANTS, COMPACT_MATH, make_math_training_prompt
from .runtime import HuggingFaceTurns, NativeToolHuggingFaceTurns, load_runtime
from .sandbox import DockerSandbox


# Public plans are frozen against this snapshot, including its prompt/controller.
COMPATIBLE_FROZEN_EVALUATORS = set()


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)
    result = {}
    for benchmark, group in sorted(grouped.items()):
        requested = sum(row["tool_metrics"]["requested_calls"] for row in group)
        executed = sum(row["tool_metrics"]["executed_calls"] for row in group)
        result[benchmark] = {
            "n": len(group),
            "accuracy": sum(row["score"]["correct"] for row in group) / len(group),
            "tool_call_rate": sum(row["tool_metrics"]["requested_calls"] > 0 for row in group) / len(group),
            "valid_tool_call_rate": sum(row["tool_metrics"]["valid_calls"] for row in group) / requested if requested else None,
            "mean_calls_per_problem": requested / len(group),
            "mean_executed_calls": executed / len(group),
            "tool_errors_per_problem": sum(row["tool_metrics"]["tool_errors"] for row in group) / len(group),
            "invalid_calls_per_problem": sum(row["tool_metrics"].get("invalid_calls", 0) for row in group) / len(group),
            "timeouts_per_problem": sum(row["tool_metrics"]["timeouts"] for row in group) / len(group),
            "mean_response_tokens": sum(row["generated_tokens"] for row in group) / len(group),
            "truncation_rate": sum(row["truncated"] for row in group) / len(group),
            "format_rate": sum(row["score"]["format_ok"] for row in group) / len(group),
        }
    return result


def distributed_context(environ=None):
    environ = os.environ if environ is None else environ
    try:
        world_size = int(environ.get("WORLD_SIZE", "1"))
        rank = int(environ.get("RANK", "0"))
        local_rank = int(environ.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE, RANK, and LOCAL_RANK must be integers") from exc
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid distributed inference rank configuration")
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def shard_rows(rows, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid evaluation shard")
    return list(rows[rank::world_size])


def component_fingerprints():
    root = Path(__file__).parent
    groups = {
        "prompt": ("protocol.py", "prompt_variants.py"),
        "tool_schema": ("environment.py", "qwen25.py"),
        "parser": ("parser.py", "runtime.py"),
        "verifier": ("math_answers.py", "rewards.py", "verifier_checks.py", "worker.py"),
    }
    result = {}
    for label, names in groups.items():
        digest = hashlib.sha256()
        for name in names:
            digest.update(name.encode())
            digest.update((root / name).read_bytes())
        result[label + "_sha256"] = digest.hexdigest()
    return result


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _wait_for_json(path, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for distributed evaluation file: {path}")


def _evaluate_problem(model, tokenizer, problem, reference, policy, sandbox, tools_enabled, prompt_variant=OPTIONAL_TOOL):
    from transformers import set_seed

    set_seed(stable_seed(policy["seed"], problem["id"]))
    started = time.monotonic()
    backend_type = (
        NativeToolHuggingFaceTurns
        if prompt_variant == COMPACT_MATH
        else HuggingFaceTurns
    )
    backend = backend_type(model, tokenizer, policy)
    environment = ToolEnvironment(sandbox, policy["max_execute_calls"])
    prompt = make_math_training_prompt(
        problem, policy["max_execute_calls"], tools_enabled, prompt_variant
    )
    rollout = (
        run_native_tool_episode
        if prompt_variant == COMPACT_MATH
        else run_agent_episode
    )
    rollout_kwargs = {
        "max_execute_calls": policy["max_execute_calls"],
        "max_invalid_calls": policy["max_invalid_calls"],
        "max_agent_turns": policy["max_agent_turns"],
        "tools_enabled": tools_enabled,
    }
    if prompt_variant != COMPACT_MATH:
        rollout_kwargs["require_execute"] = False
    trajectory = rollout(prompt, backend, environment, **rollout_kwargs)
    trajectory["score"] = grade(dict(reference, id=problem["id"]), trajectory["messages"], sandbox, truncated=trajectory["truncated"] or trajectory["error"] is not None)
    trajectory.update(id=problem["id"], benchmark=problem["benchmark"], domain=problem["domain"], seconds=time.monotonic() - started)
    return trajectory


def evaluate_rows(model, tokenizer, problems, references, policy, sandbox, output, run_info, *, tools_enabled=True, prompt_variant=OPTIONAL_TOOL):
    """Single-process evaluator used by the development smoke gate."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output is nonempty; choose a fresh run directory")
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    manifest = {**metadata(), **run_info, **component_fingerprints(), "policy": policy, "tools_enabled": tools_enabled, "prompt_variant": prompt_variant, "selected_ids": [row["id"] for row in problems], "evaluator_sha256": evaluator_fingerprint(), "status": "running"}
    write_json(output / "manifest.json", manifest)
    results = []
    try:
        with (output / "trajectories.jsonl").open("w", encoding="utf-8", newline="\n") as trajectories, (output / "metrics.jsonl").open("w", encoding="utf-8", newline="\n") as metrics:
            for index, problem in enumerate(problems):
                trajectory = _evaluate_problem(model, tokenizer, problem, references[problem["id"]], policy, sandbox, tools_enabled, prompt_variant)
                trajectories.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                trajectories.flush()
                metric = {key: trajectory[key] for key in ("id", "benchmark", "score", "tool_metrics", "generated_tokens", "truncated", "error", "seconds")}
                metrics.write(json.dumps(metric) + "\n")
                metrics.flush()
                results.append(trajectory)
                print(f"{index + 1}/{len(problems)} {problem['id']} correct={trajectory['score']['correct']}", flush=True)
        write_json(output / "summary.json", summarize(results))
        manifest.update(status="complete", count=len(results))
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return summarize(results)


def _prepare_distributed_output(output, context, timeout_s):
    output = Path(output)
    marker = output / "coordinator.json"
    if context["rank"] == 0:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Evaluation output is nonempty; choose a fresh run directory")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(marker, {"status": "initialized", **context, "strategy": "independent_model_replicas"})
    else:
        value = _wait_for_json(marker, timeout_s)
        if value.get("world_size") != context["world_size"]:
            raise ValueError("Distributed workers disagree on WORLD_SIZE")
    return output


def evaluate_rank(model, tokenizer, problems, references, policy, sandbox, output, run_info, context, *, tools_enabled=True, prompt_variant=OPTIONAL_TOOL):
    """Evaluate one deterministic rank shard and write only rank-owned files."""
    rank, world_size = context["rank"], context["world_size"]
    local = shard_rows(problems, rank, world_size)
    manifest_path = output / f"manifest.rank{rank}.json"
    manifest = {**metadata(), **run_info, **component_fingerprints(), "policy": policy, "tools_enabled": tools_enabled, "prompt_variant": prompt_variant, "selected_ids": [row["id"] for row in local], "evaluator_sha256": evaluator_fingerprint(), "distributed": context, "status": "running"}
    _atomic_json(manifest_path, manifest)
    results = []
    try:
        with (output / f"trajectories.rank{rank}.jsonl").open("x", encoding="utf-8", newline="\n") as trajectories, (output / f"metrics.rank{rank}.jsonl").open("x", encoding="utf-8", newline="\n") as metrics:
            for index, problem in enumerate(local):
                trajectory = _evaluate_problem(model, tokenizer, problem, references[problem["id"]], policy, sandbox, tools_enabled, prompt_variant)
                trajectories.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                trajectories.flush()
                metric = {key: trajectory[key] for key in ("id", "benchmark", "score", "tool_metrics", "generated_tokens", "truncated", "error", "seconds")}
                metrics.write(json.dumps(metric) + "\n")
                metrics.flush()
                results.append(trajectory)
                print(f"rank={rank} {index + 1}/{len(local)} {problem['id']} correct={trajectory['score']['correct']}", flush=True)
        manifest.update(status="complete", count=len(results))
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _atomic_json(manifest_path, manifest)
        _atomic_json(output / f"done.rank{rank}.json", {"rank": rank, "status": manifest["status"], "count": len(results)})
    return results


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_rank_outputs(output, problems, world_size):
    """Validate, merge, and summarize rank-owned outputs on rank 0."""
    output = Path(output)
    expected_ids = [row["id"] for row in problems]
    order = {identity: index for index, identity in enumerate(expected_ids)}
    trajectories, metrics, manifests = [], [], []
    for rank in range(world_size):
        manifest = json.loads((output / f"manifest.rank{rank}.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Rank {rank} did not complete")
        local_trajectories = _read_jsonl(output / f"trajectories.rank{rank}.jsonl")
        local_metrics = _read_jsonl(output / f"metrics.rank{rank}.jsonl")
        expected_local = expected_ids[rank::world_size]
        if manifest.get("selected_ids") != expected_local:
            raise ValueError(f"Rank {rank} evaluated the wrong frozen IDs")
        if [row["id"] for row in local_trajectories] != expected_local or [row["id"] for row in local_metrics] != expected_local:
            raise ValueError(f"Rank {rank} output IDs do not match its shard")
        trajectories.extend(local_trajectories)
        metrics.extend(local_metrics)
        manifests.append(manifest)
    if len(trajectories) != len(expected_ids) or {row["id"] for row in trajectories} != set(expected_ids):
        raise ValueError("Merged trajectories do not cover the frozen evaluation set exactly once")
    runtime_keys = ("model", "revision", "origin", "template_sha256", "cuda", "parameter_dtype")
    reference_runtime = {key: manifests[0]["runtime"][key] for key in runtime_keys}
    invariant_keys = ("files", "dataset_revisions", "plan_sha256", "policy", "tools_enabled", "prompt_variant", "evaluator_sha256", "prompt_sha256", "tool_schema_sha256", "parser_sha256", "verifier_sha256", "sandbox")
    reference_invariants = {key: manifests[0].get(key) for key in invariant_keys}
    for manifest in manifests[1:]:
        if {key: manifest["runtime"][key] for key in runtime_keys} != reference_runtime:
            raise ValueError("Distributed workers loaded different model/runtime revisions")
        if {key: manifest.get(key) for key in invariant_keys} != reference_invariants:
            raise ValueError("Distributed workers used different evaluation inputs or components")
    trajectories.sort(key=lambda row: order[row["id"]])
    metrics.sort(key=lambda row: order[row["id"]])
    with (output / "trajectories.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in trajectories)
    with (output / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in metrics)
    summary = summarize(trajectories)
    write_json(output / "summary.json", summary)
    final = dict(manifests[0])
    workers = [{key: manifest["runtime"][key] for key in ("rank", "local_rank", "device", "gpu")} for manifest in manifests]
    final.update(status="complete", count=len(trajectories), selected_ids=expected_ids, distributed={"strategy": "independent_model_replicas", "world_size": world_size, "ddp": False, "workers": workers})
    final["runtime"] = {**reference_runtime, "workers": workers}
    write_json(output / "manifest.json", final)
    _atomic_json(output / "coordinator.json", {"status": "complete", "world_size": world_size, "count": len(trajectories), "strategy": "independent_model_replicas"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--data-dir", default="data/frozen")
    parser.add_argument("--plan", help="Frozen final-evaluation plan; required for test")
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--config", default="configs/eval-compact-math.json", help="Development policy; test uses its frozen plan")
    parser.add_argument("--limit-per-benchmark", type=int, default=4, help="Development smoke size only")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-tools", action="store_true", help="Optional ablation, not the primary RL comparison")
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default=OPTIONAL_TOOL)
    parser.add_argument("--merge-timeout-seconds", type=float, default=86400)
    parser.add_argument("--local-rank", "--local_rank", dest="launcher_local_rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    context = distributed_context()
    if args.launcher_local_rank is not None and args.launcher_local_rank != context["local_rank"]:
        parser.error("torchrun local-rank argument disagrees with LOCAL_RANK")
    if args.merge_timeout_seconds <= 0:
        parser.error("Merge timeout must be positive")
    public, references, data_manifest = load_split(args.data_dir, args.split)
    info = {"split": args.split, "files": data_manifest["files"][args.split], "dataset_revisions": data_manifest["dataset_revisions"]}
    if args.split == "test":
        if not args.plan:
            parser.error("Freeze final evaluation IDs before running test: python -m tool_rl.experiment_data")
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        if plan.get("schema_version") != 2 or plan.get("domain") != "math" or plan.get("split") != "test":
            parser.error("Expected a schema-v2 frozen math evaluation plan")
        compatible_evaluators = {evaluator_fingerprint(), *COMPATIBLE_FROZEN_EVALUATORS}
        if plan["files"] != info["files"] or plan["evaluator_sha256"] not in compatible_evaluators:
            parser.error("Frozen plan data/evaluator changed; do not compare incompatible runs")
        if plan.get("prompt_variant") != args.prompt_variant:
            parser.error("Prompt variant differs from the frozen plan")
        lookup = {row["id"]: row for row in public}
        problems = [lookup[identity] for identity in plan["selected_ids"]]
        if len({row["id"] for row in problems}) != len(problems):
            parser.error("Duplicate evaluation IDs")
        policy = plan["policy"]
        info["plan_sha256"] = sha256(args.plan)
    else:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.limit_per_benchmark < 1:
            parser.error("Development sample limit must be positive")
        problems = select_rows(public, {key: args.limit_per_benchmark for key in config["benchmark_limits"]}, config["seed"], "smoke-development")
        policy = {key: config[key] for key in (
            "seed", "max_completion_length", "max_prompt_tokens",
            "max_execute_calls", "max_invalid_calls", "max_agent_turns",
            "temperature", "top_p", "enable_thinking",
        )}
    math_only(problems)
    if context["world_size"] > len(problems):
        parser.error("WORLD_SIZE exceeds the number of evaluation problems")
    if args.dry_run:
        if context["rank"] == 0:
            print(json.dumps({"status": "validated_only", "counts": dict(Counter(row["benchmark"] for row in problems)), "policy": policy, "prompt_variant": args.prompt_variant, "distributed": {"strategy": "independent_model_replicas", "world_size": context["world_size"], "shard_sizes": [len(shard_rows(problems, rank, context["world_size"])) for rank in range(context["world_size"])]}}, indent=2))
        return
    output = _prepare_distributed_output(args.output, context, args.merge_timeout_seconds)
    sandbox = DockerSandbox()
    info["sandbox"] = sandbox.preflight()
    verifier_path = output / "math-verifier-checks.json"
    if any(references[row["id"]].get("verifier") == "symbolic" for row in problems):
        if context["rank"] == 0:
            from .verifier_checks import run_checks
            _atomic_json(verifier_path, run_checks(sandbox, args.data_dir))
        check = _wait_for_json(verifier_path, args.merge_timeout_seconds)
        if not check.get("passed"):
            raise RuntimeError("Symbolic verifier checks failed; do not report math accuracy")
        info["math_verifier_checks"] = check
    model, tokenizer, runtime_info = load_runtime(args.model, revision=args.revision, seed=policy["seed"])
    info["runtime"] = runtime_info
    evaluate_rank(model, tokenizer, problems, references, policy, sandbox, output, info, context, tools_enabled=not args.no_tools, prompt_variant=args.prompt_variant)
    if context["rank"] == 0:
        done = [_wait_for_json(output / f"done.rank{rank}.json", args.merge_timeout_seconds) for rank in range(context["world_size"])]
        if any(row.get("status") != "complete" for row in done):
            raise RuntimeError("At least one evaluation rank failed; rank logs were preserved")
        print(json.dumps(merge_rank_outputs(output, problems, context["world_size"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
