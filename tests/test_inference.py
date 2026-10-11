import json
import unittest

from tool_rl.coding_environment import CodingToolEnvironment
from tool_rl.environment import ToolEnvironment
from tool_rl.inference import run_with_backend
from tool_rl.prompt_variants import COMPACT_CODE, COMPACT_MATH, OPTIONAL_TOOL
from tool_rl.runtime import parse_qwen_native_response


class Backend:
    def __init__(self, responses, native=False):
        self.responses = iter(responses)
        self.native = native
        self.observations = []

    def start(self, prompt, tools):
        self.prompt, self.tools = prompt, tools

    def next_turn(self):
        response = next(self.responses)
        message = parse_qwen_native_response(response) if self.native else {'role': 'assistant', 'content': response}
        return {'message': message, 'terminated': True, 'generated_tokens': 10}

    def observe(self, messages):
        self.observations.extend(messages)


class Sandbox:
    def __init__(self):
        self.sources = []

    def execute_python(self, source):
        self.sources.append(source)
        return {'ok': True, 'stdout': '391\n', 'stderr': '', 'exit_code': 0, 'timed_out': False}

    def run_unit_tests(self, *args):
        raise AssertionError('Open-ended inference must never run hidden tests')


class InferenceTests(unittest.TestCase):
    policy = {'max_execute_calls': 4, 'max_invalid_calls': 2, 'max_agent_turns': 7}

    def test_direct_answer_is_accepted_without_tool_or_final_tags(self):
        problem = {'domain': 'math', 'prompt': 'Compute 23*17'}
        sandbox = Sandbox()
        result = run_with_backend(problem, self.policy, Backend(['The answer is 391.']),
                                  ToolEnvironment(sandbox, 4), OPTIONAL_TOOL)
        self.assertEqual(result['candidate'], '391')
        self.assertFalse(result['format_ok'])
        self.assertIsNone(result['correctness'])
        self.assertEqual(sandbox.sources, [])

    def test_compact_math_executes_observes_and_continues(self):
        call = '<tool_call>'+json.dumps({'name': 'execute_python', 'arguments': {'code': 'print(23*17)'}})+'</tool_call>'
        backend, sandbox = Backend([call, '<final>391</final>'], native=True), Sandbox()
        result = run_with_backend({'domain': 'math', 'prompt': 'Compute 23*17'}, self.policy,
                                  backend, ToolEnvironment(sandbox, 4), COMPACT_MATH)
        self.assertEqual(result['candidate'], '391')
        self.assertEqual(result['tool_metrics']['executed_calls'], 1)
        self.assertEqual(backend.observations[0]['role'], 'tool')

    def test_coding_selftest_and_relaxed_final_extraction(self):
        problem = {'domain': 'code', 'prompt': 'Implement scale(x): multiply x by two.', 'entry_point': 'scale'}
        source = 'def scale(x):\n    return x * 2\nprint(scale(195.5))'
        call = '<tool_call>'+json.dumps({'name': 'execute_python', 'arguments': {'code': source}})+'</tool_call>'
        backend, sandbox = Backend([call, '```python\ndef scale(x):\n    return x * 2\n```']), Sandbox()
        result = run_with_backend(problem, self.policy, backend, CodingToolEnvironment(problem, sandbox, 4), COMPACT_CODE)
        self.assertIn('def scale(x)', result['candidate'])
        self.assertEqual(result['tool_metrics']['executed_calls'], 1)
        self.assertIn('<tool_result>', backend.observations[0]['content'])
        self.assertEqual(sandbox.sources, [source])
        self.assertIsNone(result['correctness'])

    def test_private_references_are_rejected_before_generation(self):
        for field in ('answer', 'tests', 'canonical_solution'):
            problem = {'domain': 'math', 'prompt': 'Compute', field: 'private'}
            with self.assertRaisesRegex(ValueError, 'public problem fields'):
                run_with_backend(problem, self.policy, Backend([]), ToolEnvironment(Sandbox()), OPTIONAL_TOOL)


if __name__ == '__main__':
    unittest.main()
