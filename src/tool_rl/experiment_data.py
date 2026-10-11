"""Load hashed public/reference splits and freeze domain-specific evaluation."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from .io import sha256, write_json
from .records import read_records, reject_private_fields, stable_seed


def load_split(directory, split, *, purpose="evaluation"):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    public_path, private_path = root / f"{split}.jsonl", root / "references" / f"{split}.jsonl"
    fingerprints = manifest["files"][split]
    if sha256(public_path) != fingerprints["public_sha256"] or sha256(private_path) != fingerprints["references_sha256"]:
        raise ValueError("Frozen dataset fingerprint changed")
    public = read_records(public_path)
    private = {row["id"]: row for row in read_records(private_path)}
    if private.keys() != {row["id"] for row in public}:
        raise ValueError("Reference/public IDs do not match")
    for row in public:
        reject_private_fields(row)
        ref = private[row["id"]]
        if row.get("split") != split or ref.get("split") != split:
            raise ValueError("Incorrect split label")
        if (ref.get("domain"), ref.get("benchmark")) != (row["domain"], row["benchmark"]):
            raise ValueError("Reference metadata mismatch")
        if purpose == "training" and (split != "train" or row.get("source_split") != "train" or row["benchmark"] not in {"gsm8k", "math", "kodcode"}):
            raise ValueError("Training cannot use held-out benchmarks")
    return public, private, manifest


def select_rows(rows, limits, seed, label):
    selected = []
    for benchmark, limit in sorted(limits.items()):
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("Sample limit must be positive or null")
        pool = sorted((row for row in rows if row["benchmark"] == benchmark), key=lambda row: row["id"])
        if not pool:
            raise ValueError(f"No problems for {benchmark}")
        random.Random(stable_seed(seed, label + "/" + benchmark)).shuffle(pool)
        selected.extend(pool if limit is None else pool[:limit])
    return sorted(selected, key=lambda row: (row["benchmark"], row["id"]))


def math_only(rows):
    if any(row["domain"] != "math" for row in rows):
        raise ValueError("A math evaluation plan cannot contain coding rows")


def evaluator_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in ("evaluate.py", "runtime.py", "qwen25.py", "rollout.py", "protocol.py", "prompt_variants.py", "parser.py", "math_answers.py", "environment.py", "rewards.py", "sandbox.py", "worker.py", "experiment_data.py", "verifier_checks.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def freeze_plan(directory, config, output):
    benchmarks = set(config.get("benchmark_limits", {}))
    # Keep separate domain-specific plan builders and evaluator fingerprints while
    # exposing one public freeze command.  A plan can never mix math and code.
    if benchmarks & {"mbpp", "humaneval"}:
        if benchmarks != {"mbpp", "humaneval"}:
            raise ValueError("Coding evaluation plans must contain exactly MBPP and HumanEval")
        from .coding_data import freeze_coding_plan
        return freeze_coding_plan(directory, config, output)
    if Path(output).exists():
        raise ValueError("Evaluation plan already exists; never overwrite frozen IDs")
    rows, _, manifest = load_split(directory, "test")
    selected = select_rows(rows, config["benchmark_limits"], config["seed"], "final-evaluation")
    math_only(selected)
    if any(row.get("source_split") != "test" for row in selected):
        raise ValueError("Final evaluation must use official held-out test problems")
    policy = {key: config[key] for key in (
        "seed", "max_completion_length", "max_prompt_tokens",
        "max_execute_calls", "max_invalid_calls", "max_agent_turns",
        "temperature", "top_p", "enable_thinking",
    )}
    value = {"schema_version": 2, "domain": "math", "split": "test", "selected_ids": [row["id"] for row in selected], "counts": dict(Counter(row["benchmark"] for row in selected)), "files": manifest["files"]["test"], "dataset_revisions": manifest["dataset_revisions"], "policy": policy, "prompt_variant": config.get("prompt_variant", "optional_tool"), "evaluator_sha256": evaluator_fingerprint()}
    write_json(output, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/frozen")
    parser.add_argument("--config", default="configs/eval-compact-math.json")
    parser.add_argument("--output", default="data/eval-plan.json")
    args = parser.parse_args()
    result = freeze_plan(args.data_dir, json.loads(Path(args.config).read_text()), args.output)
    print(f"Frozen evaluation IDs: {result['counts']}")


if __name__ == "__main__":
    main()
