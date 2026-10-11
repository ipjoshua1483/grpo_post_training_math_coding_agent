"""Dispatch the shared evaluation command from a frozen plan's domain."""
import argparse
import json
import sys
from pathlib import Path


def selected_domain(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plan")
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not args.plan:
        return "math"
    try:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read frozen evaluation plan: {exc}") from exc
    domain = plan.get("domain", "math")  # Math plan schema v1 predates this field.
    if domain not in {"math", "code"}:
        raise SystemExit(f"Unsupported evaluation plan domain: {domain!r}")
    return domain


def main():
    if selected_domain() == "code":
        from .coding_evaluate import main as evaluate
    else:
        from .evaluate import main as evaluate
    evaluate()
