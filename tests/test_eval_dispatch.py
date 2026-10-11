import copy
import unittest

from tool_rl.coding_data import coding_evaluator_fingerprint
from tool_rl.coding_evaluate import POLICY_KEYS, validate_plan


class EvaluationDispatchTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "HumanEval/0", "benchmark": "humaneval", "domain": "code"},
            {"id": "mbpp/0", "benchmark": "mbpp", "domain": "code"},
        ]
        self.policy = {
            "seed": 42,
            "max_completion_length": 256,
            "max_prompt_tokens": 1024,
            "max_execute_calls": 4,
            "max_invalid_calls": 2,
            "max_agent_turns": 7,
            "temperature": 0.0,
            "top_p": 0.95,
            "enable_thinking": False,
        }
        self.manifest = {
            "files": {"test": {"public_sha256": "public", "references_sha256": "private"}},
            "dataset_revisions": {
                "google-research-datasets/mbpp": "mbpp-revision",
                "openai/openai_humaneval": "humaneval-revision",
            },
        }
        self.plan = {
            "schema_version": 2,
            "domain": "code",
            "split": "test",
            "selected_ids": ["HumanEval/0", "mbpp/0"],
            "counts": {"humaneval": 1, "mbpp": 1},
            "files": self.manifest["files"]["test"],
            "dataset_revisions": self.manifest["dataset_revisions"],
            "policy": self.policy,
            "evaluator_sha256": coding_evaluator_fingerprint(),
        }

    def test_valid_coding_plan_and_matching_config(self):
        config = {"benchmark_limits": self.plan["counts"], **self.policy}
        problems, policy = validate_plan(self.plan, self.rows, self.manifest, config)
        self.assertEqual([row["id"] for row in problems], self.plan["selected_ids"])
        self.assertEqual(policy, self.policy)
        self.assertEqual(set(policy), set(POLICY_KEYS))

    def test_changed_config_is_rejected(self):
        config = {"benchmark_limits": self.plan["counts"], **self.policy, "max_execute_calls": 3}
        with self.assertRaisesRegex(ValueError, "differs from the frozen plan"):
            validate_plan(self.plan, self.rows, self.manifest, config)

    def test_tampered_ids_or_counts_are_rejected(self):
        duplicate = copy.deepcopy(self.plan)
        duplicate["selected_ids"] = ["mbpp/0", "mbpp/0"]
        with self.assertRaisesRegex(ValueError, "nonempty unique"):
            validate_plan(duplicate, self.rows, self.manifest)
        wrong_counts = copy.deepcopy(self.plan)
        wrong_counts["counts"]["mbpp"] = 2
        with self.assertRaisesRegex(ValueError, "counts"):
            validate_plan(wrong_counts, self.rows, self.manifest)


if __name__ == "__main__":
    unittest.main()
