"""Artifact IO and reproducibility metadata, without recording credentials."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Dataset must be nonempty with unique IDs")
    for row in rows:
        if row.get("domain") not in ("math", "code") or not row.get("prompt"):
            raise ValueError("Each row requires a domain and a prompt")
        if row["domain"] == "math" and "answer" not in row:
            raise ValueError("Math row requires an answer")
        if row["domain"] == "code" and (not row.get("entry_point", "").isidentifier() or not row.get("tests")):
            raise ValueError("Code row requires an entry point and test cases")
    return rows


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")


def metadata() -> dict:
    versions = {}
    for name in ("torch", "transformers", "trl", "accelerate", "datasets", "jmespath"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    root = Path(__file__).resolve().parents[2]
    code_files = sorted((root / "src" / "tool_rl").glob("*.py"))
    digest = hashlib.sha256()
    for path in code_files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
        commit = git.stdout.strip() if git.returncode == 0 else None
        is_dirty = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    except OSError:
        commit, is_dirty = None, None
    return {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": commit,
        "git_dirty": is_dirty,
        "code_sha256": digest.hexdigest(),
    }
