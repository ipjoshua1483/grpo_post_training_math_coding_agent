"""Joint math/coding TRL GRPO with explicit tools and deterministic rewards."""
import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from collections import Counter, defaultdict
from .coding_environment import CodingToolEnvironment
from .coding_rewards import grade_code
from .coding_sandbox import CodingSandbox
from .environment import ToolEnvironment
from .experiment_data import load_split, select_rows
from .io import metadata, write_json
from .prompt_variants import (
    OPTIONAL_TOOL,
    PROMPT_VARIANTS,
    COMPACT_MATH,
    COMPACT_CODE,
    make_coding_training_prompt,
    make_math_training_prompt,
    validate_prompt_variant,
)
from .rewards import grade
from .rollout import run_agent_episode, run_native_tool_episode
from .runtime import HuggingFaceTurns, NativeToolHuggingFaceTurns, load_runtime
from .sandbox import DockerSandbox
from .records import stable_seed


def domain_policy(config, domain):
    policies = config.get("domain_policies", {})
    if domain in policies:
        return policies[domain]
    return {key: config[key] for key in ("max_execute_calls", "max_invalid_calls", "max_agent_turns")}


def balanced_domain_rows(rows, seed):
    """Seed, cap, and interleave domains so every adjacent pair is math/code."""
    pools = {domain: sorted((row for row in rows if row["domain"] == domain), key=lambda row: row["id"]) for domain in ("math", "code")}
    if not pools["math"] or not pools["code"]:
        raise ValueError("Balanced joint training requires both math and coding rows")
    for domain, pool in pools.items():
        rng = random.Random(stable_seed(seed, f"joint-training/{domain}"))
        rng.shuffle(pool)
    count = min(len(pools["math"]), len(pools["code"]))
    return [row for pair in zip(pools["math"][:count], pools["code"][:count], strict=True) for row in pair]


def rollout_seed(seed, step, rank, local_index):
    """Give every sampled completion a stable, rank-distinct RNG stream."""
    return stable_seed(seed, f"rollout/{step}/{rank}/{local_index}")


def training_limits(config, override=None):
    configured = config.get("train_limits", {})
    if configured and set(configured) != set(config["train_benchmarks"]):
        raise ValueError("train_limits must define every configured training benchmark")
    return {
        benchmark: override if override is not None else configured.get(benchmark)
        for benchmark in config["train_benchmarks"]
    }


def validate_config(config, world_size=None):
    validate_prompt_variant(config.get("prompt_variant", OPTIONAL_TOOL))
    domain_prompt_variants = config.get("domain_prompt_variants", {})
    if not isinstance(domain_prompt_variants, dict) or set(domain_prompt_variants) - {"math", "code"}:
        raise ValueError("domain_prompt_variants may contain only math and code")
    for domain, variant in domain_prompt_variants.items():
        validate_prompt_variant(variant)
        if variant == COMPACT_MATH and domain != "math":
            raise ValueError("compact_math is math-only")
        if variant == COMPACT_CODE and domain != "code":
            raise ValueError("compact_code is coding-only")
    if "kodcode" in config["train_benchmarks"] and domain_prompt_variants.get(
        "code", config.get("prompt_variant", OPTIONAL_TOOL)
    ) == COMPACT_MATH:
        raise ValueError("compact_math is math-only; configure compact_code or another coding prompt")
    if set(config["train_benchmarks"]) & {"gsm8k", "math"} and domain_prompt_variants.get(
        "math", config.get("prompt_variant", OPTIONAL_TOOL)
    ) == COMPACT_CODE:
        raise ValueError("compact_code is coding-only; configure a math prompt")
    for key in (
        "max_steps", "num_generations", "per_device_train_batch_size",
        "gradient_accumulation_steps", "max_completion_length", "max_prompt_tokens",
        "logging_steps", "save_steps",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for domain in ("math", "code"):
        if domain == "code" and "kodcode" not in config["train_benchmarks"]:
            continue
        policy = domain_policy(config, domain)
        for key in ("max_execute_calls", "max_invalid_calls", "max_agent_turns"):
            if type(policy.get(key)) is not int or policy[key] < 1:
                raise ValueError(f"{domain}.{key} must be a positive integer")
    for benchmark, limit in training_limits(config).items():
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError(f"train_limits.{benchmark} must be a positive integer or null")
    world_size = world_size or int(os.environ.get("WORLD_SIZE", "1"))
    batch = config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] * world_size
    if config["num_generations"] < 2 or batch % config["num_generations"]:
        raise ValueError("GRPO requires at least two generations and complete prompt groups")
    if config.get("balanced_domains") and (batch // config["num_generations"]) % 2:
        raise ValueError("Balanced joint training requires an even number of prompt groups per optimizer update")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0 or not math.isfinite(config["beta"]) or config["beta"] <= 0:
        raise ValueError("Positive learning rate/beta required; nonzero beta enables KL logging")


def checkpoint_step(path):
    path = Path(path)
    prefix = "checkpoint-"
    if not path.is_dir() or not path.name.startswith(prefix) or not path.name[len(prefix):].isdigit():
        raise ValueError(f"Invalid trainer checkpoint: {path}")
    return int(path.name[len(prefix):])


def resolve_resume_checkpoint(output, requested):
    """Resolve only checkpoints owned by this run directory."""
    if requested is None:
        return None
    output = Path(output).resolve()
    if not output.is_dir():
        raise ValueError("Cannot resume because the training output directory does not exist")
    if requested == "latest":
        candidates = []
        for path in output.glob("checkpoint-*"):
            try:
                candidates.append((checkpoint_step(path), path.resolve()))
            except ValueError:
                continue
        if not candidates:
            raise ValueError("No trainer checkpoints are available to resume")
        return max(candidates)[1]
    path = Path(requested).resolve()
    if path.parent != output:
        raise ValueError("Resume checkpoint must be directly inside the training output directory")
    checkpoint_step(path)
    return path


def trim_logs_for_resume(output, step):
    """Discard audit rows produced after the saved optimizer checkpoint."""
    output = Path(output)
    paths = [
        output / "metrics.jsonl", output / "mask-audit.jsonl",
        output / "gradient-audit.jsonl", output / "update-audit.jsonl",
        *output.glob("rollouts.rank*.jsonl"),
        *output.glob("domain-rewards.rank*.jsonl"),
        *output.glob("metrics.rank*.jsonl"),
        *output.glob("mask-audit.rank*.jsonl"),
        *output.glob("gradient-audit.rank*.jsonl"),
        *output.glob("update-audit.rank*.jsonl"),
    ]
    for path in paths:
        if not path.exists():
            continue
        kept = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("step", -1)) < step:
                kept.append(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary = path.with_name(path.name + ".resume.tmp")
        temporary.write_text("".join(kept), encoding="utf-8", newline="\n")
        os.replace(temporary, path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ParameterProbe:
    """Small deterministic weight samples to confirm actual optimizer updates."""
    def __init__(self, model):
        import torch
        self.rows = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.numel() >= 256:
                stride = max(1, parameter.numel() // 2048)
                indices = torch.arange(0, parameter.numel(), stride, device=parameter.device)[:2048]
                before = parameter.detach().reshape(-1)[indices].float().cpu().clone()
                self.rows.append((name, parameter, indices, before))
                if len(self.rows) == 8:
                    break

    def delta(self):
        total = 0.0
        for _, parameter, indices, before in self.rows:
            after = parameter.detach().reshape(-1)[indices].float().cpu()
            total += float((after - before).square().sum())
        return math.sqrt(total)


def run_training(data_dir, config, output, *, limit_per_benchmark=None, revision=None, runtime=None, resume_from_checkpoint=None):
    validate_config(config)
    public, references, data_manifest = load_split(data_dir, "train", purpose="training")
    limits = training_limits(config, limit_per_benchmark)
    rows = select_rows(public, limits, config["seed"], "training")
    if config.get("balanced_domains"):
        rows = balanced_domain_rows(rows, config["seed"])
    elif any(row["domain"] == "code" for row in rows):
        raise ValueError("Joint training must set balanced_domains=true")
    output = Path(output)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    resume_path = resolve_resume_checkpoint(output, resume_from_checkpoint)
    existing_manifest = None
    if rank == 0 and output.exists() and any(output.iterdir()) and resume_path is None:
        raise ValueError("Training output is nonempty; use a fresh directory")
    if resume_path is not None:
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Resume requires the original run manifest")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("code_sha256") != metadata().get("code_sha256"):
            raise ValueError("Resume code fingerprint changed")
        expected_ids = [row["id"] for row in rows]
        if existing_manifest.get("status") == "complete":
            raise ValueError("Refusing to resume an already complete training run")
        if existing_manifest.get("config") != config or existing_manifest.get("files") != data_manifest["files"]["train"] or existing_manifest.get("selected_ids") != expected_ids:
            raise ValueError("Resume configuration, data fingerprint, or training IDs changed")
        if rank == 0:
            trim_logs_for_resume(output, checkpoint_step(resume_path))
    domains = {row["domain"] for row in rows}
    math_sandbox = DockerSandbox() if "math" in domains else None
    coding_sandbox = CodingSandbox() if "code" in domains else None
    sandbox_info = {}
    if math_sandbox is not None:
        sandbox_info["math"] = math_sandbox.preflight()
    if coding_sandbox is not None:
        sandbox_info["code"] = coding_sandbox.preflight()
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    set_seed(config["seed"])
    model, tokenizer, runtime_info = runtime or load_runtime(config["model"], revision=revision, seed=config["seed"], training=True)
    if existing_manifest is not None:
        old_runtime = existing_manifest.get("runtime", {})
        for key in ("model", "revision", "origin", "template_sha256"):
            if old_runtime.get(key) != runtime_info.get(key):
                raise ValueError(f"Resume runtime changed: {key}")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters() if parameter.requires_grad):
        raise ValueError("Use float32 trainable weights with bf16 mixed precision to preserve small optimizer updates")
    model.train()
    dataset_rows = []
    prompt_records = {}
    for row in rows:
        policy = domain_policy(config, row["domain"])
        prompt_variant = config.get("domain_prompt_variants", {}).get(
            row["domain"], config.get("prompt_variant", OPTIONAL_TOOL)
        )
        prompt = (
            make_math_training_prompt(row, policy["max_execute_calls"], prompt_variant=prompt_variant)
            if row["domain"] == "math"
            else make_coding_training_prompt(row, policy["max_execute_calls"], prompt_variant=prompt_variant)
        )
        template_tools = None
        if prompt_variant == COMPACT_MATH:
            if row["domain"] != "math":
                raise ValueError("compact_math training is math-only")
            template_tools = [
                ToolEnvironment(math_sandbox, policy["max_execute_calls"]).execute_python
            ]
        count = len(tokenizer.apply_chat_template(prompt, tools=template_tools, tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False))
        if count > config["max_prompt_tokens"] or count + config["max_completion_length"] > model.config.max_position_embeddings:
            raise ValueError(f"Prompt {row['id']} exceeds training context budget")
        prompt_key = json.dumps(prompt, sort_keys=True, ensure_ascii=False)
        if prompt_key in prompt_records:
            raise ValueError("Two training rows rendered to the same prompt")
        prompt_records[prompt_key] = row
        dataset_rows.append({"prompt": prompt, "reference_json": json.dumps(references[row["id"]]), "problem_id": row["id"]})
    dataset = Dataset.from_list(dataset_rows)
    output.mkdir(parents=True, exist_ok=True)
    reward_batches = []
    rollout_policy = {
        key: config[key] for key in (
            "max_completion_length", "max_prompt_tokens", "temperature", "top_p"
        )
    }
    rollout_policy["enable_thinking"] = False

    def explicit_rollout(prompts, trainer):
        """Use the same parser/controller as evaluation and return an env mask to TRL."""
        policy_model = trainer.accelerator.unwrap_model(trainer.model)
        was_training = policy_model.training
        policy_model.eval()
        prompt_ids, completion_ids, env_masks = [], [], []
        agent_messages, agent_metrics, executions = [], [], []
        agent_truncated, agent_error, agent_seeds = [], [], []
        try:
            for local_index, prompt in enumerate(prompts):
                problem = prompt_records.get(json.dumps(prompt, sort_keys=True, ensure_ascii=False))
                if problem is None:
                    raise RuntimeError("Rollout prompt is not present in the frozen training selection")
                episode_seed = rollout_seed(config["seed"], trainer.state.global_step, rank, local_index)
                torch.manual_seed(episode_seed)
                torch.cuda.manual_seed_all(episode_seed)
                policy = domain_policy(config, problem["domain"])
                rollout_prompt_variant = config.get("domain_prompt_variants", {}).get(
                    problem["domain"], config.get("prompt_variant", OPTIONAL_TOOL)
                )
                native_tools = rollout_prompt_variant == COMPACT_MATH
                backend_type = NativeToolHuggingFaceTurns if native_tools else HuggingFaceTurns
                backend = backend_type(policy_model, tokenizer, rollout_policy)
                if problem["domain"] == "math":
                    environment = ToolEnvironment(math_sandbox, policy["max_execute_calls"])
                    entry_point = None
                else:
                    environment = CodingToolEnvironment(problem, coding_sandbox, policy["max_execute_calls"])
                    entry_point = problem.get("entry_point") or (problem.get("required_symbols") or [None])[0]
                rollout = run_native_tool_episode if native_tools else run_agent_episode
                rollout_kwargs = {
                    "max_execute_calls": policy["max_execute_calls"],
                    "max_invalid_calls": policy["max_invalid_calls"],
                    "max_agent_turns": policy["max_agent_turns"],
                    "tools_enabled": True,
                }
                if not native_tools:
                    rollout_kwargs.update(
                        require_execute=False,
                        entry_point=entry_point,
                    )
                trajectory = rollout(prompt, backend, environment, **rollout_kwargs)
                if len(backend.completion_ids) != len(backend.completion_mask):
                    raise RuntimeError("Explicit rollout token/mask lengths disagree")
                prompt_ids.append(backend.prompt_ids)
                completion_ids.append(backend.completion_ids)
                env_masks.append(backend.completion_mask)
                agent_messages.append(trajectory["messages"])
                agent_metrics.append(trajectory["tool_metrics"])
                executions.append(trajectory["executions"])
                agent_truncated.append(trajectory["truncated"])
                agent_error.append(trajectory["error"])
                agent_seeds.append(episode_seed)
        finally:
            if was_training:
                policy_model.train()
        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": None,
            "env_mask": env_masks,
            "agent_messages": agent_messages,
            "agent_metrics": agent_metrics,
            "executions": executions,
            "agent_truncated": agent_truncated,
            "agent_error": agent_error,
            "agent_seed": agent_seeds,
        }

    def outcome_reward(
        prompts, completions, completion_ids, reference_json, problem_id,
        agent_messages, agent_metrics, executions, agent_truncated, agent_error,
        agent_seed,
        trainer_state, **kwargs,
    ):
        scores = []
        batch_domains = defaultdict(list)
        with (output / f"rollouts.rank{rank}.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            rows_to_score = zip(
                prompts, completion_ids, reference_json, problem_id,
                agent_messages, agent_metrics, executions, agent_truncated, agent_error, agent_seed,
                strict=True,
            )
            for prompt, tokens, reference, identity, messages, metrics, calls, was_truncated, episode_error, episode_seed in rows_to_score:
                truncated = was_truncated or episode_error is not None
                reference = json.loads(reference)
                if reference["domain"] == "math":
                    result = grade(reference, messages, math_sandbox, truncated=truncated)
                elif reference["domain"] == "code":
                    result = grade_code(reference, messages, coding_sandbox, truncated=truncated)
                else:
                    raise ValueError("Unsupported reward domain")
                scores.append(result["reward"])
                batch_domains[reference["domain"]].append(result["reward"])
                stream.write(json.dumps({
                    "step": trainer_state.global_step, "id": identity,
                    "benchmark": reference["benchmark"], "domain": reference["domain"], "prompt": prompt,
                    "messages": messages, "completion_ids": tokens, "score": result,
                    "truncated": truncated, "error": episode_error,
                    "rollout_seed": episode_seed,
                    "tool_metrics": metrics, "executions": calls,
                }, allow_nan=False) + "\n")
        reward_event = {
            "step": trainer_state.global_step,
            "domains": {
                domain: {
                    "n": len(values), "mean_reward": sum(values) / len(values),
                    "reward_distribution": {str(value): values.count(value) for value in (0.0, 1.0)},
                }
                for domain, values in sorted(batch_domains.items())
            },
        }
        reward_batches.append(reward_event)
        with (output / f"domain-rewards.rank{rank}.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(reward_event, allow_nan=False) + "\n")
        return scores

    class AuditedGRPOTrainer(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            # Pinning TRL is essential: this hook audits the env_mask returned by
            # our explicit controller while leaving GRPO's loss/optimizer intact.
            result = super()._generate_and_score_completions(inputs)
            mask, completion = result.get("tool_mask"), result["completion_mask"]
            if mask is None or not bool(((mask == 0) | (mask == 1)).all()):
                raise RuntimeError("Missing/invalid explicit observation mask")
            trainable = completion * mask
            if not int(trainable.sum()):
                raise RuntimeError("No trainable completion tokens; inspect truncation and increase token budget")
            if not bool(torch.isfinite(result["advantages"]).all()):
                raise RuntimeError("Non-finite GRPO advantages")
            event = {"step": self.state.global_step, "model_tokens_in_loss": int(trainable.sum()), "observation_tokens_excluded": int(((mask == 0) & (completion == 1)).sum()), "nonzero_advantages": int((result["advantages"] != 0).sum())}
            audit_path = output / ("mask-audit.jsonl" if rank == 0 else f"mask-audit.rank{rank}.jsonl")
            with audit_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event) + "\n")
            return result

    probe = ParameterProbe(model)
    gradient_events = []
    class RunLogging(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            total = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    norm = float(parameter.grad.detach().float().norm())
                    if not math.isfinite(norm):
                        raise RuntimeError("Non-finite gradient")
                    total += norm * norm
            event = {"step": state.global_step + 1, "gradient_norm": math.sqrt(total)}
            gradient_events.append(event)
            audit_path = output / ("gradient-audit.jsonl" if rank == 0 else f"gradient-audit.rank{rank}.jsonl")
            with audit_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event) + "\n")

        def on_step_end(self, args, state, control, **kwargs):
            event = {"step": state.global_step, "parameter_sample_delta_l2": probe.delta(), "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            audit_path = output / ("update-audit.jsonl" if rank == 0 else f"update-audit.rank{rank}.jsonl")
            with audit_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")

        def on_log(self, args, state, control, logs=None, **kwargs):
            metrics_path = output / ("metrics.jsonl" if rank == 0 else f"metrics.rank{rank}.jsonl")
            with metrics_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"step": state.global_step, **(logs or {})}, allow_nan=False) + "\n")

    args = GRPOConfig(
        output_dir=str(output), seed=config["seed"], data_seed=config["seed"],
        max_steps=config["max_steps"], learning_rate=config["learning_rate"],
        num_generations=config["num_generations"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_completion_length=config["max_completion_length"],
        temperature=config["temperature"], top_p=config["top_p"], beta=config["beta"],
        loss_type="dapo", scale_rewards="group", num_iterations=1,
        bf16=config["bf16"], gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config["logging_steps"], save_steps=config["save_steps"],
        save_total_limit=2, report_to="none", remove_unused_columns=False,
        shuffle_dataset=config.get("shuffle_dataset", not config.get("balanced_domains", False)),
        chat_template_kwargs={"enable_thinking": False}, use_vllm=False,
        # Truncated episodes already receive verifier reward 0. Keep their generated
        # policy tokens in the loss so one all-truncated local GRPO group cannot
        # produce an empty completion mask and abort every distributed rank.
        mask_truncated_completions=False, optim="adamw_torch",
    )
    trainer = AuditedGRPOTrainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=dataset, reward_funcs=outcome_reward,
        rollout_func=explicit_rollout, callbacks=[RunLogging()],
    )
    resume_history = list((existing_manifest or {}).get("resume_history", []))
    if resume_path is not None:
        resume_history.append({"checkpoint": str(resume_path), "step": checkpoint_step(resume_path), "resumed_at_unix": time.time()})
    manifest = {
        **(existing_manifest or metadata()), "config": config, "runtime": runtime_info,
        "distributed": {"world_size": world_size}, "sandbox": sandbox_info,
        "files": data_manifest["files"]["train"], "selected_ids": [row["id"] for row in rows],
        "resume_history": resume_history, "status": "running",
    }
    # JSON object keys must be strings; retain domain and benchmark without losing either.
    manifest["selected_counts"] = {
        f"{domain}/{benchmark}": count
        for (domain, benchmark), count in Counter((row["domain"], row["benchmark"]) for row in rows).items()
    }
    if rank == 0:
        write_json(output / "manifest.json", manifest)
        frozen = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (output / "environment.freeze.txt").write_text(frozen.stdout, encoding="utf-8", newline="\n")
    trainer.accelerator.wait_for_everyone()
    start = time.monotonic()
    try:
        trainer.train(resume_from_checkpoint=str(resume_path) if resume_path is not None else None)
        trainer.save_model(str(output / "final"))
        if rank == 0:
            tokenizer.save_pretrained(output / "final")
        trainer.save_state()
        trainer.accelerator.wait_for_everyone()
        summary = None
        if rank == 0:
            write_json(output / "final" / "tool_rl_training.json", {
                "origin": runtime_info["origin"],
                "training_ids": manifest["selected_ids"],
                "seed": config["seed"],
                "domains": sorted(domains),
                "prompt_variants": {
                    domain: config.get("domain_prompt_variants", {}).get(
                        domain, config.get("prompt_variant", OPTIONAL_TOOL)
                    )
                    for domain in sorted(domains)
                },
            })
            all_gradients = [row for path in output.glob("gradient-audit*.jsonl") for row in read_jsonl(path)]
            all_rollouts = [row for path in output.glob("rollouts.rank*.jsonl") for row in read_jsonl(path)]
            rewards = [float(row["score"]["reward"]) for row in all_rollouts]
            by_domain = {}
            for domain in sorted(domains):
                domain_rows = [row for row in all_rollouts if row["domain"] == domain]
                domain_rewards = [float(row["score"]["reward"]) for row in domain_rows]
                requested = sum(row["tool_metrics"].get("requested_calls", 0) for row in domain_rows)
                valid = sum(row["tool_metrics"].get("valid_calls", 0) for row in domain_rows)
                by_domain[domain] = {
                    "n": len(domain_rows),
                    "mean_reward": sum(domain_rewards) / len(domain_rewards) if domain_rewards else None,
                    "reward_distribution": {str(value): domain_rewards.count(value) for value in (0.0, 1.0)},
                    "tool_call_rate": sum(row["tool_metrics"].get("requested_calls", 0) > 0 for row in domain_rows) / len(domain_rows) if domain_rows else None,
                    "valid_tool_call_rate": valid / requested if requested else None,
                    "mean_calls_per_problem": valid / len(domain_rows) if domain_rows else None,
                }
            summary = {"optimizer_steps": trainer.state.global_step, "world_size": world_size, "gradient_events": len(all_gradients), "nonzero_gradient_events": sum(row["gradient_norm"] > 0 for row in all_gradients), "parameter_sample_delta_l2": probe.delta(), "rollout_count": len(all_rollouts), "reward_distribution": {str(value): rewards.count(value) for value in (0.0, 1.0)}, "by_domain": by_domain, "seconds_this_session": time.monotonic() - start, "checkpoint": str(output / "final"), "resumed": resume_path is not None}
            write_json(output / "summary.json", summary)
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        raise
    finally:
        if rank == 0:
            write_json(output / "manifest.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grpo.json")
    parser.add_argument("--data-dir", default="data/joint-frozen")
    parser.add_argument("--output", default="runs/grpo-main")
    parser.add_argument("--revision")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--prompt-variant", choices=PROMPT_VARIANTS)
    parser.add_argument("--limit-per-benchmark", type=int)
    parser.add_argument("--resume-from-checkpoint", nargs="?", const="latest", help="Resume from the latest checkpoint, or from a named checkpoint inside --output")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.steps is not None:
        config["max_steps"] = args.steps
    if args.prompt_variant is not None:
        config["prompt_variant"] = args.prompt_variant
    validate_config(config)
    if args.dry_run:
        rows, _, _ = load_split(args.data_dir, "train", purpose="training")
        rows = select_rows(rows, training_limits(config, args.limit_per_benchmark), config["seed"], "training")
        if config.get("balanced_domains"):
            rows = balanced_domain_rows(rows, config["seed"])
        print(json.dumps({"status": "validated_only", "n": len(rows), "counts": dict(Counter(row["domain"] for row in rows)), "config": config}, indent=2))
        return
    run_training(args.data_dir, config, args.output, limit_per_benchmark=args.limit_per_benchmark, revision=args.revision, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
