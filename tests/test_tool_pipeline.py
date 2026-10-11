import json
import unittest
from tool_rl.environment import ToolEnvironment
from tool_rl.parser import parse_final, parse_tool_calls
from tool_rl.protocol import parse_assistant
from tool_rl.rewards import grade, math_reward
from tool_rl.rollout import run_episode, run_native_tool_episode
from tool_rl.runtime import parse_qwen_native_response
from tool_rl.sandbox import DockerSandbox, SandboxConfig


class ScriptedSandbox:
    """Deterministic tool fixture, never executes a Python source string."""
    def __init__(self):
        self.sources = []
    def execute_python(self, code):
        self.sources.append(code)
        return {"ok": True, "stdout": "391\n", "stderr": "", "exit_code": 0, "timed_out": False}


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.observations = []
    def start(self, prompt, tools):
        self.prompt, self.tools = prompt, tools
    def next_turn(self):
        return {"message": {"role": "assistant", "content": next(self.responses)}, "terminated": True, "generated_tokens": 20}
    def observe(self, messages):
        self.observations.extend(messages)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.problem = {"id": "fixture/0", "domain": "math", "prompt": "Compute 23 times 17."}
        self.call = '<tool_call>{"name":"execute_python","arguments":{"code":"print(23*17)"}}</tool_call>'

    def test_tool_then_final_feedback(self):
        sandbox = ScriptedSandbox()
        model = ScriptedModel([self.call, "<final>391</final>"])
        trajectory = run_episode(
            self.problem, model, ToolEnvironment(sandbox), require_execute=True
        )
        self.assertEqual(sandbox.sources, ["print(23*17)"])
        observed = json.loads(model.observations[0]["content"].split("\n", 1)[1].rsplit("\n", 1)[0])
        self.assertEqual(observed["stdout"].strip(), "391")
        self.assertEqual(trajectory["tool_metrics"]["successful_calls"], 1)
        self.assertIsNone(model.tools)
        self.assertEqual(grade({"id": "fixture/0", "domain": "math", "answer": "391"}, trajectory["messages"])["reward"], 1)
        self.assertNotIn("answer", model.prompt[-1])

    def test_native_tool_then_final_uses_current_environment(self):
        class NativeModel(ScriptedModel):
            def next_turn(self):
                return {
                    "message": next(self.responses),
                    "terminated": True,
                    "generated_tokens": 20,
                }

        native_call = {
            "role": "assistant",
            "content": "I will calculate it.",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "arguments": {"code": "print(23*17)"},
                },
            }],
        }
        model = NativeModel([native_call, {"role": "assistant", "content": "<final>391</final>"}])
        sandbox = ScriptedSandbox()
        trajectory = run_native_tool_episode(
            [{"role": "system", "content": "fixture"}, self.problem],
            model,
            ToolEnvironment(sandbox),
            max_execute_calls=2,
            max_invalid_calls=2,
            max_agent_turns=6,
        )
        self.assertEqual(sandbox.sources, ["print(23*17)"])
        self.assertEqual(model.tools[0].__name__, "execute_python")
        self.assertEqual(model.observations[0]["role"], "tool")
        self.assertEqual(trajectory["tool_metrics"]["executed_calls"], 1)
        self.assertIsNone(trajectory["error"])

    def test_qwen_native_response_is_parsed_without_response_schema(self):
        message = parse_qwen_native_response(
            'I will calculate.\n<tool_call>{"name":"execute_python",'
            '"arguments":{"code":"print(23*17)"}}</tool_call>'
        )
        self.assertEqual(message["tool_calls"][0]["type"], "function")
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"]["code"],
            "print(23*17)",
        )

    def test_malformed_qwen_native_response_reaches_recovery_path(self):
        message = parse_qwen_native_response("<tool_call>{bad json}</tool_call>")
        self.assertEqual(message["tool_calls"][0]["type"], "invalid")
        self.assertEqual(
            message["tool_calls"][0]["parse_error"],
            "tool_call_must_contain_valid_json",
        )

    def test_direct_qwen_native_response_remains_final_text(self):
        message = parse_qwen_native_response("Therefore, <final>391</final>")
        self.assertNotIn("tool_calls", message)

    def test_early_final_gets_required_tool_feedback(self):
        sandbox = ScriptedSandbox()
        model = ScriptedModel(["<final>391</final>", self.call, "<final>391</final>"])
        trajectory = run_episode(
            self.problem, model, ToolEnvironment(sandbox), require_execute=True
        )
        self.assertEqual(sandbox.sources, ["print(23*17)"])
        self.assertIn("requires at least one", model.observations[0]["content"])
        self.assertEqual(trajectory["tool_metrics"]["executed_calls"], 1)
        self.assertIsNone(trajectory["error"])

    def test_execute_budget_is_independent_from_agent_turns(self):
        sandbox = ScriptedSandbox()
        model = ScriptedModel([self.call, self.call, self.call, "<final>391</final>"])
        trajectory = run_episode(
            self.problem, model, ToolEnvironment(sandbox, max_calls=2),
            max_execute_calls=2, max_agent_turns=4,
        )
        self.assertEqual(len(sandbox.sources), 2)
        self.assertEqual(trajectory["tool_metrics"]["budget_denials"], 1)

    def test_turn_budget_prevents_infinite_agent_loop(self):
        sandbox = ScriptedSandbox()
        trajectory = run_episode(
            self.problem, ScriptedModel([self.call] * 3), ToolEnvironment(sandbox),
            max_agent_turns=2,
        )
        self.assertEqual(trajectory["error"], "agent_turn_budget")
        self.assertIsNone(parse_final(trajectory["messages"]))
        self.assertEqual(len(sandbox.sources), 2)

    def test_no_tools_does_not_execute_requests(self):
        sandbox = ScriptedSandbox()
        trajectory = run_episode(self.problem, ScriptedModel([self.call]), ToolEnvironment(sandbox), tools_enabled=False)
        self.assertFalse(sandbox.sources)
        self.assertIsNone(parse_final(trajectory["messages"]))

    def test_parser_rejects_invalid_calls(self):
        for text in ('<tool_call>{"code":"print(1)"}</tool_call>', '<tool_call>{"name":"execute_python","arguments":{"code":1}}</tool_call>', '<tool_call>{bad}</tool_call>', '<tool_call>{"name":"execute_python","arguments":{"code":"x"}}'):
            with self.assertRaises(ValueError):
                parse_tool_calls(text)

    def test_malformed_call_is_corrected_and_does_not_consume_execution_budget(self):
        sandbox = ScriptedSandbox()
        model = ScriptedModel([
            '<tool_call>{bad json}</tool_call>',
            self.call,
            '<final>391</final>',
        ])
        trajectory = run_episode(self.problem, model, ToolEnvironment(sandbox))
        self.assertEqual(sandbox.sources, ["print(23*17)"])
        self.assertEqual(trajectory["tool_metrics"]["invalid_calls"], 1)
        self.assertEqual(trajectory["tool_metrics"]["executed_calls"], 1)
        self.assertIn("tool call was malformed", model.observations[0]["content"])

    def test_invalid_final_is_not_rewarded(self):
        for text in ("391", "<final>391</final> extra", "<final>391</final><final>391</final>", '<tool_call>{bad}</tool_call><final>391</final>'):
            self.assertIsNone(parse_final(text))

    def test_numeric_equivalence(self):
        for predicted, gold in (("1,000", "1000"), (" 0.5 ", "1/2"), ("1e3", "1000"), ("-0", "0")):
            self.assertEqual(math_reward(predicted, gold), 1)
        self.assertEqual(math_reward("0.5001", "1/2"), 0)

    def test_truncated_final_is_zero(self):
        self.assertEqual(grade({"id": "x", "domain": "math", "answer": "391"}, "<final>391</final>", truncated=True)["reward"], 0)

    def test_reward_does_not_depend_on_tool_use(self):
        self.assertEqual(grade({"id": "x", "domain": "math", "answer": "391"}, "<final>391</final>")["reward"], 1)

    def test_docker_security_command(self):
        command = DockerSandbox().command("tool-rl-test")
        for option in ("--network=none", "--read-only", "--user=65534:65534", "--cap-drop=ALL", "--security-opt=no-new-privileges:true", "--memory=256m", "--memory-swap=256m", "--pids-limit=32", "--cpus=1"):
            self.assertIn(option, command)
        for forbidden in ("--volume", "-v", "--mount", "--gpus", "--privileged", "--pid=host", "--network=host"):
            self.assertNotIn(forbidden, command)

    def test_input_limit_before_docker(self):
        result = DockerSandbox().execute_python("x" * 65537)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "input_limit")

    def test_resource_config_rejects_unbounded_values(self):
        for changes in ({"timeout_s": float("inf")}, {"timeout_s": 0}, {"memory_mb": 0}):
            with self.assertRaises(ValueError):
                SandboxConfig(**changes)


if __name__ == "__main__":
    unittest.main()
