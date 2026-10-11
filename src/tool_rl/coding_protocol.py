"""Fixed coding prefix and the shared explicit textual agent controller."""

from .coding_environment import CodingToolEnvironment
from .rollout import run_agent_episode


CODING_SYSTEM = """Implement the exact Python function name requested by the problem.
Return complete Python source defining that function and any required helpers.

There is exactly one available tool: execute_python(code: str). A function name
that appears in the problem is NOT a tool; it is a function you must implement.
To run a self-test, emit exactly:
<tool_call>
{{"name":"execute_python","arguments":{{"code":"..."}}}}
</tool_call>
The code string must contain the complete draft implementation and any public
self-tests you choose to write. Each call runs in a fresh restricted container,
so state does not persist. You will receive a <tool_result> JSON block. Inspect
it, revise when needed, and continue. You may execute at most
{max_execute_calls} valid calls. Malformed calls get corrective feedback and do
not consume that execution budget. Official hidden tests are never available to
you and run only after your final submission.

Finish with the complete Python solution. Prefer one <final>...</final> block,
optionally containing a Python code fence. A single Python fenced block or a
complete identifiable function definition can also be graded; formatting is
recorded separately from unit-test correctness.
"""

CODING_SYSTEM_NO_TOOLS = """Implement the exact Python function name requested by the problem.
Return complete Python source defining that function and any required helpers.
No execution tools are available. Function names in the problem are functions
you must implement. Official hidden tests run only after your final submission.
Prefer one <final>...</final> block, optionally containing a Python code fence.
A single Python fenced block or complete identifiable function definition can
also be graded; formatting is recorded separately from correctness.
"""

# Synthetic and intentionally unrelated to HumanEval/MBPP records. The first
# self-test exposes a bug, then the assistant revises, retests, and submits.
CODING_WORKFLOW_EXAMPLE = (
    {"role": "user", "content": "Example: Implement `floor_negatives(values)` so it returns a new list where each negative number is replaced by 0 and every other number is unchanged."},
    {"role": "assistant", "content": "I will draft the exact requested function and test boundary cases.\n<tool_call>\n{\"name\":\"execute_python\",\"arguments\":{\"code\":\"def floor_negatives(values):\\n    return [max(x, 1) for x in values]\\n\\nassert floor_negatives([-3, 0, 4]) == [0, 0, 4]\\nprint('tests passed')\"}}\n</tool_call>"},
    {"role": "user", "content": "<tool_result>\n{\"ok\":false,\"stdout\":\"\",\"stderr\":\"Traceback (most recent call last):\\n  ...\\nAssertionError\\n\",\"exit_code\":1,\"timed_out\":false}\n</tool_result>"},
    {"role": "assistant", "content": "The self-test failed because max(x, 1) changes zero to one. I will revise it to max(x, 0) and test again.\n<tool_call>\n{\"name\":\"execute_python\",\"arguments\":{\"code\":\"def floor_negatives(values):\\n    return [max(x, 0) for x in values]\\n\\nassert floor_negatives([-3, 0, 4]) == [0, 0, 4]\\nassert floor_negatives([]) == []\\nprint('tests passed')\"}}\n</tool_call>"},
    {"role": "user", "content": "<tool_result>\n{\"ok\":true,\"stdout\":\"tests passed\\n\",\"stderr\":\"\",\"exit_code\":0,\"timed_out\":false}\n</tool_result>"},
    {"role": "assistant", "content": "The revised implementation passes both self-tests.\n<final>\ndef floor_negatives(values):\n    return [max(x, 0) for x in values]\n</final>"},
)


def make_coding_prompt(problem, max_execute_calls=4, tools_enabled=True):
    if problem.get("domain") != "code":
        raise ValueError("Coding evaluation accepts only domain=code problems")
    forbidden = {"tests", "solution", "code", "canonical_solution", "test_list"}
    if forbidden & problem.keys():
        raise ValueError("Public coding prompt contains private solution/test data")
    required = list(problem.get("required_symbols") or [])
    if problem.get("entry_point") and problem["entry_point"] not in required:
        required.append(problem["entry_point"])
    text = problem["prompt"]
    if required:
        text += "\nImplement these exact required Python names: " + ", ".join(required)
        text += "\nThese names are functions/definitions to implement; they are not tools."
    system = CODING_SYSTEM.format(max_execute_calls=max_execute_calls) if tools_enabled else CODING_SYSTEM_NO_TOOLS
    examples = CODING_WORKFLOW_EXAMPLE if tools_enabled else ()
    return [{"role": "system", "content": system}, *examples, {"role": "user", "content": text}]


def run_coding_episode(
    problem,
    backend,
    sandbox,
    *,
    max_execute_calls=4,
    max_invalid_calls=2,
    max_agent_turns=7,
    tools_enabled=True,
    require_execute=False,
):
    """Run a coding episode; hidden tests are never accepted as model input."""
    environment = CodingToolEnvironment(problem, sandbox, max_execute_calls)
    prompt = make_coding_prompt(problem, max_execute_calls, tools_enabled)
    entry_point = problem.get("entry_point")
    if not entry_point and problem.get("required_symbols"):
        entry_point = problem["required_symbols"][0]
    return run_agent_episode(
        prompt,
        backend,
        environment,
        max_execute_calls=max_execute_calls,
        max_invalid_calls=max_invalid_calls,
        max_agent_turns=max_agent_turns,
        tools_enabled=tools_enabled,
        require_execute=require_execute,
        entry_point=entry_point,
    )
