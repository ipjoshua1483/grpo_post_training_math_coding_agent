"""Add a pinned, filtered KodCode subset to an existing frozen data bundle."""

import argparse
import ast
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .data import decontaminate
from .experiment_data import load_split
from .io import sha256, write_json, write_jsonl
from .records import public_record, reference_record, stable_seed


KODCODE_REPO = "KodCode/KodCode-Light-RL-10K"
DIFFICULTIES = ("easy", "medium", "hard")
FORBIDDEN_TEST_NAMES = {
    "pytest", "capsys", "capfd", "monkeypatch", "tmp_path", "tmpdir",
    "recwarn", "request", "mocker",
}


def _external_imports(source):
    if not isinstance(source, str):
        raise ValueError("KodCode solution must be Python source")
    tree = ast.parse(source)
    modules = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    modules.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    return modules - set(sys.stdlib_module_names)


def _entry_point(row):
    info = row.get("test_info")
    if not isinstance(info, list) or len(info) != 1 or not isinstance(info[0], dict):
        raise ValueError("KodCode row must have exactly one test_info record")
    name = info[0].get("function_name")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("KodCode function_name is missing or invalid")
    return name


def _sanitize_tests(source, minimum):
    if not isinstance(source, str):
        raise ValueError("KodCode tests must be Python source")
    tree = ast.parse(source)
    imported_solution_names = set()
    body = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "solution":
            imported_solution_names.update(alias.asname or alias.name for alias in node.names)
            continue
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            # Pytest-style source sometimes includes a direct-run footer. Hidden
            # tests are invoked individually by the worker, so remove the footer.
            rendered = ast.unparse(node.test)
            if "__name__" in rendered and "__main__" in rendered:
                continue
        body.append(node)
    tree.body = body
    used_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    imported_modules = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    if FORBIDDEN_TEST_NAMES & (used_names | imported_modules):
        raise ValueError("KodCode tests require unsupported pytest fixtures or packages")
    if imported_modules - set(sys.stdlib_module_names):
        raise ValueError("KodCode tests require packages absent from the restricted sandbox")
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            args = node.args
            if isinstance(node, ast.AsyncFunctionDef) or args.posonlyargs or args.args or args.kwonlyargs or args.vararg or args.kwarg or node.decorator_list:
                raise ValueError("KodCode tests must be undecorated zero-argument functions")
            functions.append(node.name)
    if len(functions) < minimum:
        raise ValueError(f"KodCode row has fewer than {minimum} runnable tests")
    if len(functions) != len(set(functions)):
        raise ValueError("KodCode test function names must be unique")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).strip() + "\n", functions, imported_solution_names


def kodcode_row(row, index, *, minimum_tests=5):
    """Convert one source row, rejecting suites the restricted worker cannot run."""
    prompt = row.get("question")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("KodCode question is missing")
    entry_point = _entry_point(row)
    if _external_imports(row.get("solution")):
        raise ValueError("KodCode solution requires packages absent from the restricted sandbox")
    source, functions, imported_names = _sanitize_tests(row.get("test"), minimum_tests)
    if imported_names and entry_point not in imported_names and "*" not in imported_names:
        raise ValueError("KodCode tests do not import the declared function")
    difficulty = str(row.get("gpt_difficulty", "")).lower()
    if difficulty not in DIFFICULTIES:
        raise ValueError("KodCode difficulty must be easy, medium, or hard")
    identity = row.get("question_id", index)
    return {
        "id": f"kodcode/train/{identity}",
        "benchmark": "kodcode",
        "domain": "code",
        "source_split": "train",
        "prompt": prompt.strip(),
        "entry_point": entry_point,
        "required_symbols": [entry_point],
        "difficulty": difficulty,
        "tests": {"kind": "kodcode", "source": source, "functions": functions},
    }


def _allocation(total, weights):
    raw = {name: total * float(weights[name]) for name in DIFFICULTIES}
    allocated = {name: int(raw[name]) for name in DIFFICULTIES}
    for name in sorted(DIFFICULTIES, key=lambda value: (-(raw[value] - allocated[value]), value)):
        if sum(allocated.values()) == total:
            break
        allocated[name] += 1
    return allocated


def stratified_sample(rows, total, seed, label, weights):
    if type(total) is not int or total < 1:
        raise ValueError("KodCode sample size must be positive")
    if set(weights) != set(DIFFICULTIES) or any(float(value) < 0 for value in weights.values()):
        raise ValueError("difficulty_weights must define nonnegative easy/medium/hard weights")
    scale = sum(float(value) for value in weights.values())
    if scale <= 0:
        raise ValueError("difficulty_weights must have positive mass")
    normalized = {key: float(value) / scale for key, value in weights.items()}
    targets = _allocation(total, normalized)
    pools = defaultdict(list)
    for row in rows:
        pools[row["difficulty"]].append(row)
    selected = []
    remainder = []
    for difficulty in DIFFICULTIES:
        pool = sorted(pools[difficulty], key=lambda row: row["id"])
        random.Random(stable_seed(seed, f"{label}/{difficulty}")).shuffle(pool)
        take = min(targets[difficulty], len(pool))
        selected.extend(pool[:take])
        remainder.extend(pool[take:])
    if len(selected) < total:
        random.Random(stable_seed(seed, f"{label}/remainder")).shuffle(remainder)
        selected.extend(remainder[: total - len(selected)])
    if len(selected) != total:
        raise ValueError(f"Only {len(selected)} eligible KodCode rows for requested sample of {total}")
    return sorted(selected, key=lambda row: row["id"])


def prepare_kodcode_splits(source_rows, heldout_rows, config):
    eligible, rejected = [], []
    for index, source in enumerate(source_rows):
        try:
            eligible.append(kodcode_row(source, index, minimum_tests=config["min_tests"]))
        except (SyntaxError, ValueError, TypeError):
            rejected.append(str(source.get("question_id", index)))
    eligible, contaminated = decontaminate(eligible, heldout_rows)
    rejected.extend(contaminated)
    dev = stratified_sample(
        eligible, config["dev_limit"], config["seed"], "kodcode/dev",
        config["difficulty_weights"],
    )
    dev_ids = {row["id"] for row in dev}
    train_pool = [row for row in eligible if row["id"] not in dev_ids]
    train_pool, overlap = decontaminate(train_pool, dev)
    rejected.extend(overlap)
    train = stratified_sample(
        train_pool, config["train_limit"], config["seed"], "kodcode/train",
        config["difficulty_weights"],
    )
    return {
        "train": [dict(row, split="train") for row in train],
        "dev": [dict(row, split="dev") for row in dev],
    }, sorted(set(rejected))


def load_kodcode(revision=None):
    """Load only revision-pinned Parquet; never execute a remote dataset script."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    resolved = revision or api.dataset_info(KODCODE_REPO, revision="refs/convert/parquet").sha
    files = api.list_repo_files(KODCODE_REPO, repo_type="dataset", revision=resolved)
    parquet = [f"hf://datasets/{KODCODE_REPO}@{resolved}/{name}" for name in sorted(files) if name.endswith(".parquet")]
    if not parquet:
        raise ValueError(f"No Parquet files found for {KODCODE_REPO} at {resolved}")
    dataset = load_dataset("parquet", data_files={"train": parquet})["train"]
    return list(dataset), resolved


def freeze_joint_data(base_directory, output, config, source_rows, revision):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to overwrite frozen data; choose a fresh directory")
    base = {}
    references = {}
    manifests = None
    for split in ("train", "dev", "test"):
        base[split], references[split], manifest = load_split(
            base_directory, split, purpose="training" if split == "train" else "evaluation"
        )
        manifests = manifest
    heldout = base["test"]
    coding, rejected = prepare_kodcode_splits(source_rows, heldout, config)
    files, counts = {}, {}
    for split in ("train", "dev", "test"):
        code_rows = coding.get(split, [])
        public = list(base[split]) + [public_record(row) for row in code_rows]
        private = list(references[split].values()) + [reference_record(row) for row in code_rows]
        public.sort(key=lambda row: (row["benchmark"], row["id"]))
        private.sort(key=lambda row: (row["benchmark"], row["id"]))
        public_path = output / f"{split}.jsonl"
        private_path = output / "references" / f"{split}.jsonl"
        write_jsonl(public_path, public)
        write_jsonl(private_path, private)
        files[split] = {
            "public_sha256": sha256(public_path),
            "references_sha256": sha256(private_path),
        }
        counts[split] = dict(Counter(row["benchmark"] for row in public))
    revisions = dict(manifests.get("dataset_revisions", {}))
    revisions[KODCODE_REPO] = revision
    result = {
        "schema_version": 3,
        "config": config,
        "base_manifest_sha256": sha256(Path(base_directory) / "manifest.json"),
        "dataset_revisions": revisions,
        "files": files,
        "counts": counts,
        "source_counts": {
            "kodcode_total": len(source_rows),
            "kodcode_selected_train": len(coding["train"]),
            "kodcode_selected_dev": len(coding["dev"]),
            "kodcode_excluded": len(rejected),
        },
        "excluded_kodcode_ids": rejected,
        "notes": [
            "Existing frozen math and held-out evaluation rows were preserved",
            "KodCode is training/development only and was decontaminated against held-out prompts",
            "KodCode suites require one function and at least five runnable zero-argument tests",
            "Sampling is deterministic and stratified by gpt_difficulty",
            "Hidden tests remain only in reference files",
        ],
    }
    write_json(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", default="data/frozen")
    parser.add_argument("--config", default="configs/joint-data.json")
    parser.add_argument("--output", default="data/joint-frozen")
    parser.add_argument("--revision")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    rows, revision = load_kodcode(args.revision)
    result = freeze_joint_data(args.base_data, args.output, config, rows, revision)
    print(f"Frozen joint splits in {args.output}: {result['counts']}")


if __name__ == "__main__":
    main()
