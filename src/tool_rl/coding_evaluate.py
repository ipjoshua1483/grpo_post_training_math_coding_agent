"""Held-out MBPP/HumanEval pass@1 evaluation with coding-only tool use."""
import argparse
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from .coding_data import coding_evaluator_fingerprint, freeze_coding_plan, load_coding_test
from .coding_environment import CodingToolEnvironment
from .coding_rewards import grade_code
from .coding_sandbox import CodingSandbox
from .io import metadata, sha256, write_json
from .records import stable_seed
from .prompt_variants import OPTIONAL_TOOL, PROMPT_VARIANTS, make_coding_training_prompt
from .rollout import run_agent_episode
from .runtime import HuggingFaceTurns, load_runtime


POLICY_KEYS = (
    "seed", "max_completion_length", "max_prompt_tokens",
    "max_execute_calls", "max_invalid_calls", "max_agent_turns",
    "temperature", "top_p", "enable_thinking",
)

COMPATIBLE_FROZEN_EVALUATORS = set()


def validate_plan(plan, rows, data_manifest, config=None, prompt_variant=None):
    if plan.get("schema_version") != 2 or plan.get("domain") != "code" or plan.get("split") != "test":
        raise ValueError("Expected a schema-v2 held-out coding evaluation plan")
    if plan.get("files") != data_manifest["files"]["test"]:
        raise ValueError("Coding plan does not match the frozen held-out data")
    revisions = {
        key: data_manifest["dataset_revisions"][key]
        for key in ("google-research-datasets/mbpp", "openai/openai_humaneval")
    }
    if plan.get("dataset_revisions") != revisions:
        raise ValueError("Coding plan dataset revisions do not match the frozen data")
    compatible_evaluators = {coding_evaluator_fingerprint(), *COMPATIBLE_FROZEN_EVALUATORS}
    if plan.get("evaluator_sha256") not in compatible_evaluators:
        raise ValueError("Coding evaluator changed after plan freeze; create a fresh plan before comparing runs")
    if prompt_variant is not None and plan.get("prompt_variant") != prompt_variant:
        raise ValueError("Prompt variant differs from the frozen coding plan")
    identities = plan.get("selected_ids")
    if not isinstance(identities, list) or not identities or len(identities) != len(set(identities)):
        raise ValueError("Coding plan IDs must be a nonempty unique list")
    lookup = {row["id"]: row for row in rows}
    try:
        problems = [lookup[identity] for identity in identities]
    except KeyError as exc:
        raise ValueError(f"Coding plan contains an unknown ID: {exc}") from exc
    actual_counts = {
        benchmark: sum(row["benchmark"] == benchmark for row in problems)
        for benchmark in ("humaneval", "mbpp")
    }
    if plan.get("counts") != actual_counts:
        raise ValueError("Coding plan benchmark counts do not match its selected IDs")
    policy = plan.get("policy")
    if not isinstance(policy, dict) or set(policy) != set(POLICY_KEYS):
        raise ValueError("Coding plan has an invalid inference policy")
    if config is not None:
        expected_policy = {key: config.get(key) for key in POLICY_KEYS}
        if config.get("benchmark_limits") != actual_counts or expected_policy != policy:
            raise ValueError("Coding config differs from the frozen plan; use the plan's original config")
    return problems, policy


def summarize_coding(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(row)
    summary = {}
    for benchmark, group in sorted(grouped.items()):
        requested = sum(row["tool_metrics"]["requested_calls"] for row in group)
        summary[benchmark] = {
            "n": len(group),
            "pass_at_1": sum(row["score"]["correct"] for row in group) / len(group),
            "format_rate": sum(row["score"]["format_ok"] for row in group) / len(group),
            "tool_call_rate": sum(row["tool_metrics"]["requested_calls"] > 0 for row in group) / len(group),
            "valid_tool_call_rate": (
                sum(row["tool_metrics"]["valid_calls"] for row in group) / requested
                if requested else None
            ),
            "mean_calls_per_problem": requested / len(group),
            "tool_errors_per_problem": sum(row["tool_metrics"]["tool_errors"] for row in group) / len(group),
            "invalid_calls_per_problem": sum(row["tool_metrics"].get("invalid_calls", 0) for row in group) / len(group),
            "tool_timeouts_per_problem": sum(row["tool_metrics"]["timeouts"] for row in group) / len(group),
            "verifier_timeouts": sum(row["score"].get("verification", {}).get("timed_out") is True for row in group),
            "mean_response_tokens": sum(row["generated_tokens"] for row in group) / len(group),
            "truncation_rate": sum(row["truncated"] for row in group) / len(group),
        }
    return summary


def distributed_context(environ=None):
    environ = os.environ if environ is None else environ
    try:
        world_size = int(environ.get("WORLD_SIZE", "1"))
        rank = int(environ.get("RANK", "0"))
        local_rank = int(environ.get("LOCAL_RANK", "0"))
    except ValueError as exc:
        raise ValueError("WORLD_SIZE, RANK, and LOCAL_RANK must be integers") from exc
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid distributed coding-evaluation rank configuration")
    return {"rank": rank, "local_rank": local_rank, "world_size": world_size}


def shard_rows(rows, rank, world_size):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid coding-evaluation shard")
    return list(rows[rank::world_size])


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(temporary, path)


def _wait_for_json(path, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for distributed coding-evaluation file: {path}")


def _prepare_distributed_output(output, context, timeout_s):
    output = Path(output)
    marker = output / "coordinator.json"
    if context["rank"] == 0:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Evaluation output is nonempty; choose a fresh run directory")
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(marker, {
            "status": "initialized", **context,
            "strategy": "independent_model_replicas",
        })
    else:
        value = _wait_for_json(marker, timeout_s)
        if value.get("world_size") != context["world_size"]:
            raise ValueError("Distributed coding workers disagree on WORLD_SIZE")
    return output


def _run_coding_variant(problem, backend, sandbox, policy, tools_enabled, prompt_variant):
    prompt = make_coding_training_prompt(
        problem, policy["max_execute_calls"], tools_enabled, prompt_variant
    )
    environment = CodingToolEnvironment(problem, sandbox, policy["max_execute_calls"])
    entry_point = problem.get("entry_point") or (problem.get("required_symbols") or [None])[0]
    return run_agent_episode(
        prompt, backend, environment,
        max_execute_calls=policy["max_execute_calls"],
        max_invalid_calls=policy["max_invalid_calls"],
        max_agent_turns=policy["max_agent_turns"],
        tools_enabled=tools_enabled,
        require_execute=False,
        entry_point=entry_point,
    )


def evaluate_coding_rows(model, tokenizer, problems, references, policy, sandbox, output, run_info, *, tools_enabled=True, prompt_variant=OPTIONAL_TOOL):
    from transformers import set_seed

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output is nonempty; choose a fresh run directory")
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    manifest = {
        **metadata(), **run_info, "policy": policy, "tools_enabled": tools_enabled,
        "prompt_variant": prompt_variant,
        "tool_scope": "coding_only" if tools_enabled else "disabled",
        "hidden_tests_exposed_to_model": False,
        "selected_ids": [row["id"] for row in problems],
        "evaluator_sha256": coding_evaluator_fingerprint(), "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    results = []
    try:
        with (output / "trajectories.jsonl").open("w", encoding="utf-8", newline="\n") as trajectories, (output / "metrics.jsonl").open("w", encoding="utf-8", newline="\n") as metrics:
            for index, problem in enumerate(problems):
                set_seed(stable_seed(policy["seed"], problem["id"]))
                started = time.monotonic()
                backend = HuggingFaceTurns(model, tokenizer, policy)
                trajectory = _run_coding_variant(
                    problem, backend, sandbox, policy, tools_enabled, prompt_variant
                )
                reference = references[problem["id"]]
                trajectory["score"] = grade_code(
                    reference, trajectory["messages"], sandbox,
                    truncated=trajectory["truncated"] or trajectory["error"] is not None,
                )
                trajectory.update(
                    id=problem["id"], benchmark=problem["benchmark"], domain="code",
                    seconds=time.monotonic() - started,
                )
                trajectories.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                trajectories.flush()
                metric = {key: trajectory[key] for key in (
                    "id", "benchmark", "score", "tool_metrics", "generated_tokens",
                    "truncated", "error", "seconds",
                )}
                metrics.write(json.dumps(metric) + "\n")
                metrics.flush()
                results.append(trajectory)
                print(f"{index + 1}/{len(problems)} {problem['id']} pass={trajectory['score']['correct']}", flush=True)
        summary = summarize_coding(results)
        write_json(output / "summary.json", summary)
        manifest.update(status="complete", count=len(results))
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return summary


def _evaluate_coding_problem(model, tokenizer, problem, reference, policy, sandbox, tools_enabled, prompt_variant=OPTIONAL_TOOL):
    from transformers import set_seed

    set_seed(stable_seed(policy["seed"], problem["id"]))
    started = time.monotonic()
    backend = HuggingFaceTurns(model, tokenizer, policy)
    trajectory = _run_coding_variant(
        problem, backend, sandbox, policy, tools_enabled, prompt_variant
    )
    trajectory["score"] = grade_code(
        reference, trajectory["messages"], sandbox,
        truncated=trajectory["truncated"] or trajectory["error"] is not None,
    )
    trajectory.update(
        id=problem["id"], benchmark=problem["benchmark"], domain="code",
        seconds=time.monotonic() - started,
    )
    return trajectory


def evaluate_coding_rank(
    model, tokenizer, problems, references, policy, sandbox, output, run_info,
    context, *, tools_enabled=True, prompt_variant=OPTIONAL_TOOL,
):
    """Evaluate one deterministic coding shard and write only rank-owned files."""
    rank, world_size = context["rank"], context["world_size"]
    local = shard_rows(problems, rank, world_size)
    manifest_path = output / f"manifest.rank{rank}.json"
    manifest = {
        **metadata(), **run_info, "policy": policy, "tools_enabled": tools_enabled,
        "prompt_variant": prompt_variant,
        "tool_scope": "coding_only" if tools_enabled else "disabled",
        "hidden_tests_exposed_to_model": False,
        "selected_ids": [row["id"] for row in local],
        "evaluator_sha256": coding_evaluator_fingerprint(),
        "distributed": context, "status": "running",
    }
    _atomic_json(manifest_path, manifest)
    results = []
    try:
        trajectory_path = output / f"trajectories.rank{rank}.jsonl"
        metric_path = output / f"metrics.rank{rank}.jsonl"
        with trajectory_path.open("x", encoding="utf-8", newline="\n") as trajectories, metric_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as metrics:
            for index, problem in enumerate(local):
                trajectory = _evaluate_coding_problem(
                    model, tokenizer, problem, references[problem["id"]],
                    policy, sandbox, tools_enabled, prompt_variant,
                )
                trajectories.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
                trajectories.flush()
                metric = {key: trajectory[key] for key in (
                    "id", "benchmark", "score", "tool_metrics", "generated_tokens",
                    "truncated", "error", "seconds",
                )}
                metrics.write(json.dumps(metric) + "\n")
                metrics.flush()
                results.append(trajectory)
                print(
                    f"rank={rank} {index + 1}/{len(local)} {problem['id']} "
                    f"pass={trajectory['score']['correct']}",
                    flush=True,
                )
        manifest.update(status="complete", count=len(results))
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _atomic_json(manifest_path, manifest)
        _atomic_json(output / f"done.rank{rank}.json", {
            "rank": rank, "status": manifest["status"], "count": len(results),
        })
    return results


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def merge_coding_rank_outputs(output, problems, world_size):
    """Validate and merge all coding rank outputs in frozen-plan order."""
    output = Path(output)
    expected_ids = [row["id"] for row in problems]
    order = {identity: index for index, identity in enumerate(expected_ids)}
    trajectories, metrics, manifests = [], [], []
    for rank in range(world_size):
        manifest = json.loads(
            (output / f"manifest.rank{rank}.json").read_text(encoding="utf-8")
        )
        if manifest.get("status") != "complete":
            raise RuntimeError(f"Coding rank {rank} did not complete")
        local_trajectories = _read_jsonl(output / f"trajectories.rank{rank}.jsonl")
        local_metrics = _read_jsonl(output / f"metrics.rank{rank}.jsonl")
        expected_local = expected_ids[rank::world_size]
        if manifest.get("selected_ids") != expected_local:
            raise ValueError(f"Coding rank {rank} evaluated the wrong frozen IDs")
        if (
            [row["id"] for row in local_trajectories] != expected_local
            or [row["id"] for row in local_metrics] != expected_local
        ):
            raise ValueError(f"Coding rank {rank} output IDs do not match its shard")
        trajectories.extend(local_trajectories)
        metrics.extend(local_metrics)
        manifests.append(manifest)
    if (
        len(trajectories) != len(expected_ids)
        or {row["id"] for row in trajectories} != set(expected_ids)
    ):
        raise ValueError("Merged coding trajectories do not cover the frozen set exactly once")
    runtime_keys = (
        "model", "revision", "origin", "template_sha256", "cuda", "parameter_dtype",
    )
    reference_runtime = {key: manifests[0]["runtime"][key] for key in runtime_keys}
    invariant_keys = (
        "files", "dataset_revisions", "plan_sha256", "policy", "tools_enabled",
        "tool_scope", "hidden_tests_exposed_to_model", "prompt_variant", "evaluator_sha256", "sandbox",
    )
    reference_invariants = {key: manifests[0].get(key) for key in invariant_keys}
    for manifest in manifests[1:]:
        if {key: manifest["runtime"][key] for key in runtime_keys} != reference_runtime:
            raise ValueError("Distributed coding workers loaded different model/runtime revisions")
        if {key: manifest.get(key) for key in invariant_keys} != reference_invariants:
            raise ValueError("Distributed coding workers used different evaluation inputs")
    trajectories.sort(key=lambda row: order[row["id"]])
    metrics.sort(key=lambda row: order[row["id"]])
    with (output / "trajectories.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        stream.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in trajectories)
    with (output / "metrics.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        stream.writelines(json.dumps(row) + "\n" for row in metrics)
    summary = summarize_coding(trajectories)
    write_json(output / "summary.json", summary)
    final = dict(manifests[0])
    workers = [
        {key: manifest["runtime"][key] for key in ("rank", "local_rank", "device", "gpu")}
        for manifest in manifests
    ]
    final.update(
        status="complete", count=len(trajectories), selected_ids=expected_ids,
        distributed={
            "strategy": "independent_model_replicas", "world_size": world_size,
            "ddp": False, "workers": workers,
        },
    )
    final["runtime"] = {**reference_runtime, "workers": workers}
    write_json(output / "manifest.json", final)
    _atomic_json(output / "coordinator.json", {
        "status": "complete", "world_size": world_size, "count": len(trajectories),
        "strategy": "independent_model_replicas",
    })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--data-dir", default="data/frozen")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config", help="Optional config to verify against the frozen plan")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-tools", action="store_true", help="Disable model tool access for the matched ablation")
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default=OPTIONAL_TOOL)
    parser.add_argument("--merge-timeout-seconds", type=float, default=86400)
    parser.add_argument(
        "--local-rank", "--local_rank", dest="launcher_local_rank",
        type=int, help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    context = distributed_context()
    if args.launcher_local_rank is not None and args.launcher_local_rank != context["local_rank"]:
        parser.error("torchrun local-rank argument disagrees with LOCAL_RANK")
    if args.merge_timeout_seconds <= 0:
        parser.error("Merge timeout must be positive")

    rows, references, data_manifest = load_coding_test(args.data_dir)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    config = None
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    try:
        problems, policy = validate_plan(plan, rows, data_manifest, config, args.prompt_variant)
    except ValueError as exc:
        parser.error(str(exc))
    if context["world_size"] > len(problems):
        parser.error("WORLD_SIZE exceeds the number of coding evaluation problems")
    if args.dry_run:
        if context["rank"] == 0:
            print(json.dumps({
                "status": "validated_only",
                "counts": dict(Counter(row["benchmark"] for row in problems)),
                "policy": policy,
                "tools_enabled": not args.no_tools,
                "prompt_variant": args.prompt_variant,
                "distributed": {
                    "strategy": "independent_model_replicas",
                    "world_size": context["world_size"],
                    "shard_sizes": [
                        len(shard_rows(problems, rank, context["world_size"]))
                        for rank in range(context["world_size"])
                    ],
                },
            }, indent=2))
        return

    output = _prepare_distributed_output(
        args.output, context, args.merge_timeout_seconds
    )
    sandbox = CodingSandbox()
    info = {
        "split": "test", "plan_sha256": sha256(args.plan),
        "files": data_manifest["files"]["test"],
        "dataset_revisions": plan["dataset_revisions"], "sandbox": sandbox.preflight(),
    }
    model, tokenizer, runtime_info = load_runtime(
        args.model, revision=args.revision, seed=policy["seed"]
    )
    info["runtime"] = runtime_info
    evaluate_coding_rank(
        model, tokenizer, problems, references, policy, sandbox, output, info,
        context, tools_enabled=not args.no_tools, prompt_variant=args.prompt_variant,
    )
    if context["rank"] == 0:
        done = [
            _wait_for_json(
                output / f"done.rank{rank}.json", args.merge_timeout_seconds
            )
            for rank in range(context["world_size"])
        ]
        if any(row.get("status") != "complete" for row in done):
            raise RuntimeError("At least one coding-evaluation rank failed")
        print(json.dumps(
            merge_coding_rank_outputs(output, problems, context["world_size"]),
            indent=2,
        ), flush=True)


def plan_main():
    parser = argparse.ArgumentParser(description="Freeze deterministic MBPP/HumanEval evaluation IDs")
    parser.add_argument("--data-dir", default="data/frozen")
    parser.add_argument("--config", default="configs/eval-compact-code.json")
    parser.add_argument("--output", default="data/eval-code-plan.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    plan = freeze_coding_plan(args.data_dir, config, args.output)
    print(json.dumps({"output": args.output, "counts": plan["counts"]}, indent=2))


if __name__ == "__main__":
    main()
