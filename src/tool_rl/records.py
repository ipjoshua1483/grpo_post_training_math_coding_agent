"""Public problem records and private verification references."""
import hashlib
import json
from pathlib import Path


def read_records(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Records must be nonempty and have unique IDs")
    return rows


def stable_seed(seed, label):
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:4], "big")


def public_record(row):
    keys = {"id", "benchmark", "domain", "split", "prompt", "entry_point", "required_symbols", "source_split"}
    result = {key: value for key, value in row.items() if key in keys}
    if not result.get("prompt") or result.get("domain") not in ("math", "code"):
        raise ValueError("Invalid public problem")
    return result


def reference_record(row):
    keys = {"id", "benchmark", "domain", "split", "answer", "verifier", "tests"}
    return {key: value for key, value in row.items() if key in keys}


def reject_private_fields(row):
    if {"answer", "tests", "solution", "code", "canonical_solution", "test_list"} & row.keys():
        raise ValueError("Baseline inputs must contain only public problems, not answer keys/tests")

