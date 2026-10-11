import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tool_rl.coding_data import freeze_coding_plan, load_coding_test
from tool_rl.coding_environment import CodingToolEnvironment
from tool_rl.coding_evaluate import (
    distributed_context, merge_coding_rank_outputs, shard_rows,
)
from tool_rl.coding_protocol import make_coding_prompt, run_coding_episode
from tool_rl.coding_rewards import grade_code
from tool_rl.coding_worker import handle
from tool_rl.protocol import parse_assistant


class FakeSandbox:
    def __init__(self, test_result=None):
        self.scratch = []
        self.verifier_payloads = []
        self.test_result = test_result or {
            "ok": True,
            "value": {"tests_passed": 3, "tests_total": 3, "failures": []},
            "exit_code": 0,
            "timed_out": False,
            "output_limited": False,
        }

    def execute_python(self, code):
        self.scratch.append(code)
        return {"ok": True, "stdout": "6\n", "stderr": "", "exit_code": 0, "timed_out": False, "output_limited": False}

    def run_unit_tests(self, code, tests):
        self.verifier_payloads.append({"code": code, "tests": tests})
        return self.test_result


class ScriptedModel:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.observations = []

    def start(self, prompt, tools):
        self.prompt, self.tools = prompt, tools

    def next_turn(self):
        return {"message": {"role": "assistant", "content": next(self.turns)}, "terminated": True, "generated_tokens": 12}

    def observe(self, messages):
        self.observations.extend(messages)


class CodingEvalTests(unittest.TestCase):
    def setUp(self):
        self.problem = {
            "id": "fixture/code/0", "benchmark": "mbpp", "domain": "code",
            "split": "test", "source_split": "test", "prompt": "Write double(x).",
            "required_symbols": ["double"],
        }
        self.reference = {
            "id": self.problem["id"], "benchmark": "mbpp", "domain": "code", "split": "test",
            "tests": {"kind": "mbpp", "setup": "", "tests": ["assert double(3) == 6"], "challenge_tests": []},
        }

    def test_tool_is_restricted_to_code_domain(self):
        with self.assertRaises(ValueError):
            CodingToolEnvironment({"domain": "math"}, FakeSandbox())
        with self.assertRaises(ValueError):
            make_coding_prompt({"domain": "math", "prompt": "2+2"})

    def test_distributed_coding_shards_are_disjoint_and_complete(self):
        rows = [{"id": str(index)} for index in range(7)]
        shards = [shard_rows(rows, rank, 2) for rank in range(2)]
        self.assertEqual([row["id"] for row in shards[0]], ["0", "2", "4", "6"])
        self.assertEqual([row["id"] for row in shards[1]], ["1", "3", "5"])
        self.assertEqual(
            sorted(row["id"] for shard in shards for row in shard),
            [row["id"] for row in rows],
        )
        self.assertEqual(
            distributed_context({"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"}),
            {"world_size": 2, "rank": 1, "local_rank": 1},
        )

    def test_distributed_coding_merge_restores_plan_order(self):
        problems = [
            {"id": "code/0", "benchmark": "humaneval"},
            {"id": "code/1", "benchmark": "mbpp"},
            {"id": "code/2", "benchmark": "mbpp"},
        ]
        runtime = {
            "model": "fixture", "revision": "abc", "origin": {},
            "template_sha256": "template", "cuda": "fixture",
            "parameter_dtype": "torch.bfloat16",
        }
        invariants = {
            "files": {}, "dataset_revisions": {}, "plan_sha256": "plan",
            "policy": {}, "tools_enabled": True, "tool_scope": "coding_only",
            "hidden_tests_exposed_to_model": False,
            "evaluator_sha256": "evaluator", "sandbox": {},
        }
        def trajectory(problem):
            return {
                "id": problem["id"], "benchmark": problem["benchmark"],
                "score": {"correct": True, "format_ok": True, "verification": {}},
                "tool_metrics": {
                    "requested_calls": 0, "valid_calls": 0, "tool_errors": 0,
                    "invalid_calls": 0, "timeouts": 0,
                },
                "generated_tokens": 1, "truncated": False, "error": None,
                "seconds": 0.1,
            }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for rank in range(2):
                local = problems[rank::2]
                manifest = {
                    **invariants, "status": "complete",
                    "selected_ids": [row["id"] for row in local],
                    "runtime": {
                        **runtime, "rank": rank, "local_rank": rank,
                        "device": f"cuda:{rank}", "gpu": "fixture",
                    },
                }
                (output / f"manifest.rank{rank}.json").write_text(json.dumps(manifest))
                rows = [trajectory(row) for row in local]
                payload = "".join(json.dumps(row) + "\n" for row in rows)
                (output / f"trajectories.rank{rank}.jsonl").write_text(payload)
                (output / f"metrics.rank{rank}.jsonl").write_text(payload)
            summary = merge_coding_rank_outputs(output, problems, 2)
            merged = [
                json.loads(line)
                for line in (output / "trajectories.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["id"] for row in merged], [row["id"] for row in problems])
            self.assertEqual(summary["humaneval"]["pass_at_1"], 1.0)
            self.assertEqual(summary["mbpp"]["n"], 2)

    def test_hidden_tests_never_enter_prompt_or_scratch_tool(self):
        sandbox = FakeSandbox()
        call = "<tool_call>" + json.dumps({
            "name": "execute_python",
            "arguments": {"code": "def double(x): return x*2\nprint(double(3))"},
        }) + "</tool_call>"
        model = ScriptedModel([call, "<final>def double(x):\n    return x * 2</final>"])
        trajectory = run_coding_episode(self.problem, model, sandbox, max_execute_calls=4)
        serialized = json.dumps({"prompt": model.prompt, "scratch": sandbox.scratch, "trajectory": trajectory})
        self.assertNotIn("assert double(3) == 6", serialized)
        system = model.prompt[0]["content"]
        self.assertIn("exactly one available tool: execute_python", system)
        self.assertIn("is NOT a tool", system)
        self.assertIsNone(model.tools)
        observed = json.loads(model.observations[0]["content"].split("\n", 1)[1].rsplit("\n", 1)[0])
        self.assertEqual(observed["stdout"].strip(), "6")
        self.assertEqual(trajectory["tool_metrics"]["successful_calls"], 1)
        rendered_prompt = json.dumps(model.prompt)
        self.assertIn("AssertionError", rendered_prompt)
        self.assertIn("revise", rendered_prompt.lower())

    def test_no_tools_ablation_has_no_schema_or_execution(self):
        sandbox = FakeSandbox()
        model = ScriptedModel(["<final>def double(x):\n    return x * 2</final>"])
        trajectory = run_coding_episode(
            self.problem, model, sandbox, max_execute_calls=4, tools_enabled=False
        )
        self.assertIsNone(model.tools)
        self.assertIn("No execution tools are available", model.prompt[0]["content"])
        self.assertNotIn("execute_python", model.prompt[0]["content"])
        self.assertEqual(sandbox.scratch, [])
        self.assertEqual(trajectory["tool_metrics"]["requested_calls"], 0)

    def test_no_tools_ablation_rejects_an_unexpected_tool_call(self):
        sandbox = FakeSandbox()
        call = "<tool_call>" + json.dumps({
            "name": "execute_python", "arguments": {"code": "print(6)"}
        }) + "</tool_call>"
        trajectory = run_coding_episode(
            self.problem, ScriptedModel([call]), sandbox,
            max_execute_calls=4, tools_enabled=False,
        )
        self.assertEqual(trajectory["error"], "tool_disabled")
        self.assertEqual(sandbox.scratch, [])
        self.assertEqual(trajectory["tool_metrics"]["requested_calls"], 1)

    def test_binary_reward_comes_from_unit_test_suite(self):
        sandbox = FakeSandbox()
        result = grade_code(self.reference, "<final>def double(x):\n    return x * 2</final>", sandbox)
        self.assertEqual(result["reward"], 1.0)
        self.assertTrue(result["correct"])
        self.assertEqual((result["tests_passed"], result["tests_total"]), (3, 3))
        self.assertEqual(sandbox.verifier_payloads[0]["tests"], self.reference["tests"])
        wrong = FakeSandbox({"ok": True, "value": {"tests_passed": 2, "tests_total": 3, "failures": [{"index": 2, "type": "AssertionError"}]}, "exit_code": 0})
        result = grade_code(self.reference, "<final>def double(x): return 0</final>", wrong)
        self.assertEqual(result["reward"], 0.0)
        self.assertFalse(result["correct"])

    def test_code_fallbacks_preserve_format_as_a_separate_metric(self):
        for completion, method in (
            ("```python\ndef double(x):\n    return x * 2\n```", "code_fence"),
            ("def double(x):\n    return x * 2", "raw_code"),
        ):
            sandbox = FakeSandbox()
            result = grade_code(self.reference, completion, sandbox)
            self.assertEqual(result["reward"], 1.0)
            self.assertFalse(result["format_ok"])
            self.assertEqual(result["extraction_method"], method)
            self.assertTrue(sandbox.verifier_payloads)

    def test_ambiguous_malformed_or_truncated_submission_is_zero_without_execution(self):
        for completion, truncated in (
            ("Explanation with no code.", False),
            ("```python\ndef double(x): return x*2\n```\n```python\ndef other(x): return x\n```", False),
            ("<final>def double(x): return x*2", False),
            ("<final>def double(x): return x*2</final>", True),
        ):
            sandbox = FakeSandbox()
            self.assertEqual(grade_code(self.reference, completion, sandbox, truncated=truncated)["reward"], 0.0)
            self.assertFalse(sandbox.verifier_payloads)

    def test_mbpp_worker_runs_every_test(self):
        payload = {
            "mode": "unit_tests", "code": "def double(x): return x * 2",
            "tests": {"kind": "mbpp", "setup": "", "tests": ["assert double(3) == 6", "assert double(0) == 0"], "challenge_tests": ["assert double(-2) == -4"]},
        }
        with patch.dict(os.environ, {"TOOL_RL_SANDBOXED": "1"}):
            result = handle(payload)
        self.assertTrue(result["ok"])
        self.assertEqual(result["value"]["tests_passed"], 3)
        self.assertEqual(result["value"]["tests_total"], 3)

    def test_humaneval_worker_calls_official_check(self):
        spec = {"kind": "humaneval", "entry_point": "double", "test": "def check(candidate):\n    assert candidate(3) == 6\n    assert candidate(-2) == -4"}
        with patch.dict(os.environ, {"TOOL_RL_SANDBOXED": "1"}):
            passed = handle({"mode": "unit_tests", "code": "def double(x): return x*2", "tests": spec})
            failed = handle({"mode": "unit_tests", "code": "def double(x): return 0", "tests": spec})
        self.assertEqual(passed["value"]["tests_passed"], 1)
        self.assertEqual(failed["value"]["tests_passed"], 0)

    def test_frozen_real_coding_plan_is_deterministic(self):
        root = Path(__file__).resolve().parents[1]
        data_dir = root / "data" / "frozen"
        if not (data_dir / "manifest.json").exists():
            self.skipTest("Optional full-dataset integration test: build data/frozen first")
        rows, references, _ = load_coding_test(data_dir)
        self.assertEqual(len(rows), 664)
        self.assertEqual(len(references), 664)
        config = {
            "seed": 42, "benchmark_limits": {"humaneval": 2, "mbpp": 2},
            "max_completion_length": 2048, "max_prompt_tokens": 4096,
            "max_execute_calls": 4, "max_invalid_calls": 2,
            "max_agent_turns": 7, "temperature": 0.0, "top_p": 0.95,
            "enable_thinking": False,
        }
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            first = freeze_coding_plan(data_dir, config, Path(temporary) / "a.json")
            second = freeze_coding_plan(data_dir, config, Path(temporary) / "b.json")
        self.assertEqual(first, second)
        self.assertEqual(first["counts"], {"humaneval": 2, "mbpp": 2})

    def test_problem_function_called_as_tool_gets_corrective_feedback_without_execution(self):
        problem = dict(self.problem, required_symbols=["double"])
        bad = '<tool_call>{"name":"double","arguments":{"code":"print(1)"}}</tool_call>'
        valid = "<tool_call>" + json.dumps({
            "name": "execute_python",
            "arguments": {"code": "def double(x): return x * 2\nprint(double(3))"},
        }) + "</tool_call>"
        final = "<final>def double(x):\n    return x * 2</final>"
        model = ScriptedModel([bad, valid, final])
        sandbox = FakeSandbox()
        trajectory = run_coding_episode(problem, model, sandbox)
        self.assertEqual(len(sandbox.scratch), 1)
        self.assertEqual(trajectory["tool_metrics"]["invalid_calls"], 1)
        feedback = model.observations[0]["content"]
        self.assertIn("`double` is the Python function you must implement", feedback)
        self.assertIn("only available tool is execute_python", feedback)


if __name__ == "__main__":
    unittest.main()
