import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tool_rl.coding_worker import handle
from tool_rl.experiment_data import load_split
from tool_rl.io import sha256, write_json, write_jsonl
from tool_rl.joint_data import kodcode_row, prepare_kodcode_splits
from tool_rl.records import public_record, reference_record
from tool_rl.train import balanced_domain_rows, rollout_seed, training_limits, validate_config


def source_row(index, difficulty="easy", *, fixture=False):
    argument = "capsys" if fixture else ""
    tests = [
        f"def test_{i}({argument}):\n    assert target({i}) == {i + 1}"
        for i in range(6)
    ]
    return {
        "question_id": str(index),
        "question": f"Implement target for unique task {index}.",
        "solution": "def target(x):\n    return x + 1",
        "test": "from solution import target\n\n" + "\n\n".join(tests),
        "test_info": [{"function_name": "target"}],
        "gpt_difficulty": difficulty,
    }


class JointTrainingTests(unittest.TestCase):
    def test_kodcode_conversion_strips_solution_import_and_keeps_hidden_tests(self):
        row = kodcode_row(source_row(1), 1, minimum_tests=5)
        self.assertEqual(row["required_symbols"], ["target"])
        self.assertNotIn("from solution", row["tests"]["source"])
        self.assertEqual(len(row["tests"]["functions"]), 6)
        self.assertNotIn("tests", public_record(row))
        self.assertIn("tests", reference_record(row))

    def test_kodcode_rejects_fixture_dependent_tests(self):
        with self.assertRaisesRegex(ValueError, "unsupported|zero-argument"):
            kodcode_row(source_row(2, fixture=True), 2, minimum_tests=5)

    def test_kodcode_rejects_nonstdlib_solution_dependencies(self):
        row = source_row(3)
        row["solution"] = "import pandas\ndef target(x):\n    return x + 1"
        with self.assertRaisesRegex(ValueError, "packages absent"):
            kodcode_row(row, 3, minimum_tests=5)

    def test_seeded_stratified_split_is_deterministic_and_disjoint(self):
        rows = []
        for index in range(60):
            difficulty = ("easy", "medium", "hard")[index % 3]
            rows.append(source_row(index, difficulty))
        config = {
            "seed": 42, "train_limit": 20, "dev_limit": 10, "min_tests": 5,
            "difficulty_weights": {"easy": 0.6, "medium": 0.35, "hard": 0.05},
        }
        first, _ = prepare_kodcode_splits(rows, [], config)
        second, _ = prepare_kodcode_splits(rows, [], config)
        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 20)
        self.assertEqual(len(first["dev"]), 10)
        self.assertFalse({r["id"] for r in first["train"]} & {r["id"] for r in first["dev"]})

    def test_training_order_is_seeded_balanced_and_capped(self):
        rows = [
            *({"id": f"m/{i}", "domain": "math"} for i in range(5)),
            *({"id": f"c/{i}", "domain": "code"} for i in range(3)),
        ]
        first = balanced_domain_rows(rows, 42)
        self.assertEqual(first, balanced_domain_rows(rows, 42))
        self.assertEqual(len(first), 6)
        self.assertEqual([row["domain"] for row in first], ["math", "code"] * 3)

    def test_rollout_seeds_are_reproducible_and_distinct_across_ranks(self):
        values = {
            rollout_seed(42, step, rank, index)
            for step in range(2) for rank in range(4) for index in range(8)
        }
        self.assertEqual(len(values), 64)
        self.assertEqual(rollout_seed(42, 3, 1, 7), rollout_seed(42, 3, 1, 7))



    def test_kodcode_worker_counts_each_hidden_test(self):
        spec = {
            "kind": "kodcode",
            "source": "def test_a():\n    assert target(1) == 2\n\ndef test_b():\n    assert target(2) == 3\n",
            "functions": ["test_a", "test_b"],
        }
        with patch.dict(os.environ, {"TOOL_RL_SANDBOXED": "1"}):
            passed = handle({"mode": "unit_tests", "code": "def target(x): return x + 1", "tests": spec})
            failed = handle({"mode": "unit_tests", "code": "def target(x): return x", "tests": spec})
        self.assertEqual(passed["value"]["tests_passed"], 2)
        self.assertEqual(failed["value"]["tests_passed"], 0)

    def test_signed_training_loader_accepts_kodcode_but_not_heldout_code(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            public = [{
                "id": "kodcode/train/1", "benchmark": "kodcode", "domain": "code",
                "split": "train", "source_split": "train", "prompt": "Implement target.",
                "entry_point": "target", "required_symbols": ["target"],
            }]
            private = [{
                "id": "kodcode/train/1", "benchmark": "kodcode", "domain": "code",
                "split": "train", "tests": {"kind": "kodcode", "source": "def test_a(): pass", "functions": ["test_a"]},
            }]
            write_jsonl(root / "train.jsonl", public)
            write_jsonl(root / "references" / "train.jsonl", private)
            files = {"train": {"public_sha256": sha256(root / "train.jsonl"), "references_sha256": sha256(root / "references" / "train.jsonl")}}
            write_json(root / "manifest.json", {"files": files})
            loaded, _, _ = load_split(root, "train", purpose="training")
            self.assertEqual(loaded, public)


if __name__ == "__main__":
    unittest.main()
