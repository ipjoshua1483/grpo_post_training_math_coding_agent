"""Freeze official train/dev/test boundaries before any model training."""
import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from .io import sha256, write_json, write_jsonl
from .records import public_record, reference_record, stable_seed
from .rewards import number

MATH_CONFIGS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus")


def last_boxed(text):
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        raise ValueError("MATH solution has no boxed answer")
    start, depth = matches[-1].end(), 1
    for i in range(start, len(text)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    raise ValueError("Unbalanced boxed answer")


def gsm_row(row, split, index):
    if "####" not in row["answer"]:
        raise ValueError("Missing GSM8K answer delimiter")
    answer = row["answer"].rsplit("####", 1)[1].strip()
    if number(answer) is None:
        raise ValueError("Unsupported GSM8K numeric answer")
    return {"id": f"gsm8k/{split}/{index}", "benchmark": "gsm8k", "domain": "math", "source_split": split, "prompt": row["question"], "answer": answer, "verifier": "numeric"}


def math_row(row, split, config, index):
    try:
        answer = last_boxed(row["solution"])
    except ValueError:
        answer = None
    return {"id": f"math/{config}/{split}/{index}", "benchmark": "math", "domain": "math", "source_split": split, "prompt": row["problem"], "answer": answer, "verifier": "symbolic"}


def mbpp_row(row, split):
    definitions = {node.name for node in ast.parse(row["code"]).body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))}
    used = set()
    for test in row["test_list"]:
        used.update(node.id for node in ast.walk(ast.parse(test)) if isinstance(node, ast.Name))
    return {"id": f"mbpp/{row['task_id']}", "benchmark": "mbpp", "domain": "code", "source_split": split, "prompt": row["text"], "required_symbols": sorted(definitions & used), "tests": {"kind": "mbpp", "setup": row.get("test_setup_code", ""), "tests": list(row["test_list"]), "challenge_tests": list(row.get("challenge_test_list", []))}}


def humaneval_row(row):
    return {"id": row["task_id"], "benchmark": "humaneval", "domain": "code", "source_split": "test", "prompt": row["prompt"], "entry_point": row["entry_point"], "tests": {"kind": "humaneval", "test": row["test"], "entry_point": row["entry_point"]}}


def canonical(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def shingles(text):
    words = canonical(text).split()
    return {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


def decontaminate(rows, heldout, threshold=0.85):
    exact = {canonical(row["prompt"]) for row in heldout}
    held_shingles = [shingles(row["prompt"]) for row in heldout]
    index = defaultdict(set)
    for i, tokens in enumerate(held_shingles):
        for token in tokens:
            index[token].add(i)
    kept, removed, seen = [], [], set()
    for row in rows:
        key, tokens = canonical(row["prompt"]), shingles(row["prompt"])
        counts = Counter(i for token in tokens for i in index.get(token, ()))
        near = len(tokens) >= 8 and any(count / (len(tokens) + len(held_shingles[i]) - count) >= threshold for i, count in counts.items())
        if key in exact or key in seen or near:
            removed.append(row["id"])
        else:
            seen.add(key)
            kept.append(row)
    return kept, removed


def sample(rows, limit, seed, label):
    rows = sorted(rows, key=lambda row: row["id"])
    random.Random(stable_seed(seed, label)).shuffle(rows)
    return rows if limit is None else rows[:limit]


def prepare_splits(sources, config):
    heldout = sources["gsm_test"] + sources["math_test"] + sources["math500"] + sources["mbpp_test"] + sources["mbpp_dev"] + sources["mbpp_prompt"] + sources["humaneval"]
    seed, removed, pools, seen_training = config["seed"], [], {}, []
    for benchmark, key in (("gsm8k", "gsm_train"), ("math", "math_train")):
        eligible = [row for row in sources[key] if row["domain"] != "math" or row.get("answer")]
        removed.extend(row["id"] for row in sources[key] if row["domain"] == "math" and not row.get("answer"))
        pools[benchmark], exclusions = decontaminate(eligible, heldout + seen_training)
        removed.extend(exclusions)
        seen_training.extend(pools[benchmark])
    dev = []
    for benchmark in ("gsm8k", "math"):
        ordered = sample(pools[benchmark], None, seed, f"{benchmark}/dev-reserve")
        n = min(config["dev_per_math_dataset"], max(0, len(ordered) - 1))
        dev.extend(ordered[:n])
        pools[benchmark] = ordered[n:]
    train = []
    for benchmark in ("gsm8k", "math"):
        eligible, exclusions = decontaminate(pools[benchmark], dev)
        removed.extend(exclusions)
        train.extend(sample(eligible, config["train_limits"][benchmark], seed, f"{benchmark}/train"))
    test = sources["gsm_test"] + sources["math500"] + sources["mbpp_test"] + sources["humaneval"]
    splits = {}
    for split, rows in (("train", train), ("dev", dev), ("test", test)):
        selected = [dict(row, split=split) for row in rows]
        splits[split] = sorted(selected, key=lambda row: (row["benchmark"], row["id"]))
        if not selected:
            raise ValueError(f"Empty {split} split")
    sets = [set(row["id"] for row in splits[name]) for name in ("train", "dev", "test")]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Overlapping split IDs")
    if any(row["benchmark"] in {"mbpp", "humaneval", "math500"} for row in splits["train"] + splits["dev"]):
        raise ValueError("Held-out benchmarks cannot enter training/development")
    return splits, removed


def load_sources(revisions=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api, snapshots = HfApi(), {}
    def load(repo, config, allowed_splits=None):
        # Versioned Parquet only: never run remote dataset scripts.
        sha = (revisions or {}).get(repo) or snapshots.get(repo) or api.dataset_info(repo, revision="refs/convert/parquet").sha
        snapshots[repo] = sha
        files = api.list_repo_files(repo, repo_type="dataset", revision=sha)
        grouped = defaultdict(list)
        for file in sorted(files):
            parts = file.split("/")
            if len(parts) >= 3 and parts[0] == config and file.endswith(".parquet") and (allowed_splits is None or parts[1] in allowed_splits):
                grouped[parts[1]].append(f"hf://datasets/{repo}@{sha}/{file}")
        if not grouped:
            raise ValueError(f"No Parquet shards for {repo}/{config} at {sha}")
        return load_dataset("parquet", data_files=dict(grouped))
    gsm = load("openai/gsm8k", "main")
    mbpp = load("google-research-datasets/mbpp", "full", allowed_splits={"test"})
    human = load("openai/openai_humaneval", "openai_humaneval")
    math500 = load("HuggingFaceH4/MATH-500", "default")
    sources = {"gsm_train": [gsm_row(r, "train", i) for i, r in enumerate(gsm["train"])], "gsm_test": [gsm_row(r, "test", i) for i, r in enumerate(gsm["test"])], "math_train": [], "math_test": [], "math500": [], "humaneval": [humaneval_row(r) for r in human["test"]]}
    # Strict interpretation: all MBPP/HumanEval material is evaluation-only.
    sources.update(mbpp_train=[], mbpp_dev=[], mbpp_prompt=[])
    sources["mbpp_test"] = [mbpp_row(r, "test") for r in mbpp["test"]]
    for i, row in enumerate(math500["test"]):
        sources["math500"].append({"id": f"math500/test/{i}", "benchmark": "math500", "domain": "math", "source_split": "test", "prompt": row["problem"], "answer": row["answer"], "verifier": "symbolic"})
    for config in MATH_CONFIGS:
        dataset = load("EleutherAI/hendrycks_math", config)
        for split in ("train", "test"):
            sources[f"math_{split}"].extend(math_row(r, split, config, i) for i, r in enumerate(dataset[split]))
    return sources, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.json")
    parser.add_argument("--output", default="data/frozen")
    parser.add_argument("--revisions-from", help="Reuse all dataset commit SHAs from a prior manifest")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if set(config["train_limits"]) != {"gsm8k", "math"}:
        parser.error("Base-data preparation expects GSM8K/MATH train sources; add KodCode with joint_data")
    if config["dev_per_math_dataset"] < 1 or any(n is not None and n < 1 for n in config["train_limits"].values()):
        parser.error("Train limits/dev size must be positive; null means all training rows")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Refusing to overwrite frozen data; choose a fresh directory")
    revisions = None
    if args.revisions_from:
        revisions = json.loads(Path(args.revisions_from).read_text())["dataset_revisions"]
        expected = {"openai/gsm8k", "google-research-datasets/mbpp", "openai/openai_humaneval", "HuggingFaceH4/MATH-500", "EleutherAI/hendrycks_math"}
        if revisions.keys() != expected:
            parser.error("Prior manifest must pin every dataset")
    sources, snapshots = load_sources(revisions)
    splits, exclusions = prepare_splits(sources, config)
    fingerprints = {}
    for split, rows in splits.items():
        public_path, reference_path = output / f"{split}.jsonl", output / "references" / f"{split}.jsonl"
        write_jsonl(public_path, [public_record(row) for row in rows])
        write_jsonl(reference_path, [reference_record(row) for row in rows])
        fingerprints[split] = {"public_sha256": sha256(public_path), "references_sha256": sha256(reference_path)}
    write_json(output / "manifest.json", {"schema_version": 2, "config": config, "dataset_revisions": snapshots, "files": fingerprints, "counts": {split: dict(Counter(row["benchmark"] for row in rows)) for split, rows in splits.items()}, "source_counts": {key: len(value) for key, value in sources.items()}, "excluded_training_ids": exclusions, "unverifiable_training_ids": [row["id"] for row in sources["math_train"] if not row.get("answer")], "notes": ["Official test splits remain evaluation-only", "GSM8K/MATH train splits supply math training", "Math dev reserved from official train", "MBPP, HumanEval and MATH-500 never enter training or development", "KodCode training data is added separately by joint_data", "Full evaluation sets preserved, including non-JSON coding tests", "MATH training examples without extractable boxed answers excluded and recorded", "All MATH test prompts retained for overlap checks even without extractable answers", "No benchmark reference implementations saved", "Near-duplicate filter: 5-word shingle Jaccard >= 0.85", "Cannot remove base-model pretraining contamination"]})
    print(f"Frozen splits in {output}; see manifest.json for counts and revisions")


if __name__ == "__main__":
    main()
