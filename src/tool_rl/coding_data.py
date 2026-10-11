"""Frozen held-out coding records and deterministic evaluation plans."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .io import sha256, write_json
from .records import read_records, reject_private_fields, stable_seed

BENCHMARKS = {"mbpp", "humaneval"}


def load_coding_test(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    public_path = root / "test.jsonl"
    reference_path = root / "references" / "test.jsonl"
    expected = manifest["files"]["test"]
    if sha256(public_path) != expected["public_sha256"] or sha256(reference_path) != expected["references_sha256"]:
        raise ValueError("Frozen test data fingerprint changed")
    public = []
    for row in read_records(public_path):
        if row.get("domain") == "code":
            reject_private_fields(row)
            if row.get("benchmark") not in BENCHMARKS or row.get("split") != "test" or row.get("source_split") != "test":
                raise ValueError("Coding evaluation accepts only official held-out MBPP/HumanEval test records")
            public.append(row)
    references = {
        row["id"]: row for row in read_records(reference_path)
        if row.get("domain") == "code"
    }
    if not public or set(references) != {row["id"] for row in public}:
        raise ValueError("Coding public/reference IDs do not match")
    for row in references.values():
        if row.get("benchmark") not in BENCHMARKS or row.get("split") != "test" or not isinstance(row.get("tests"), dict):
            raise ValueError("Invalid coding reference")
    return public, references, manifest


def coding_evaluator_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for name in (
        "coding_data.py", "coding_environment.py", "coding_evaluate.py",
        "coding_protocol.py", "coding_rewards.py", "coding_sandbox.py",
        "prompt_variants.py",
        "coding_worker.py", "rollout.py", "protocol.py", "runtime.py",
        "qwen25.py", "sandbox.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _select(rows, limits, seed):
    selected = []
    for benchmark in sorted(BENCHMARKS):
        group = sorted((row for row in rows if row["benchmark"] == benchmark), key=lambda row: row["id"])
        random.Random(stable_seed(seed, "coding-eval/" + benchmark)).shuffle(group)
        limit = limits[benchmark]
        if type(limit) is not int or limit < 1 or limit > len(group):
            raise ValueError(f"Invalid {benchmark} limit {limit}; available={len(group)}")
        selected.extend(group[:limit])
    return sorted(selected, key=lambda row: (row["benchmark"], row["id"]))


def freeze_coding_plan(data_dir, config, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a frozen coding evaluation plan")
    if set(config.get("benchmark_limits", {})) != BENCHMARKS:
        raise ValueError("Coding plan must configure both MBPP and HumanEval")
    rows, _, manifest = load_coding_test(data_dir)
    selected = _select(rows, config["benchmark_limits"], config["seed"])
    policy_keys = (
        "seed", "max_completion_length", "max_prompt_tokens",
        "max_execute_calls", "max_invalid_calls", "max_agent_turns",
        "temperature", "top_p", "enable_thinking",
    )
    policy = {key: config[key] for key in policy_keys}
    plan = {
        "schema_version": 2,
        "domain": "code",
        "split": "test",
        "selected_ids": [row["id"] for row in selected],
        "counts": dict(Counter(row["benchmark"] for row in selected)),
        "files": manifest["files"]["test"],
        "dataset_revisions": {
            key: manifest["dataset_revisions"][key]
            for key in ("google-research-datasets/mbpp", "openai/openai_humaneval")
        },
        "policy": policy,
        "prompt_variant": config.get("prompt_variant", "optional_tool"),
        "evaluator_sha256": coding_evaluator_fingerprint(),
    }
    write_json(output, plan)
    return plan
