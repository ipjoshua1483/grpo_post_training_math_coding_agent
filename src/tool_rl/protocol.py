"""Fixed math prompt and explicit textual tool protocol.

The controller, rather than the model chat template, owns parsing, execution,
budgets, and recovery. This module has no Transformers or TRL dependency so the
protocol can be unit tested exactly.
"""

import json
import re
import ast


TOOL_NAME = "execute_python"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
TOOL_RESULT_OPEN = "<tool_result>"
TOOL_RESULT_CLOSE = "</tool_result>"

MATH_SYSTEM = """Solve the mathematics problem accurately. You may explain your reasoning.

Tool use is optional. The only available tool is execute_python(code: str),
which runs complete Python source in a fresh restricted container. Variables do
not persist between calls. To use it, emit exactly:
<tool_call>
{{"name":"execute_python","arguments":{{"code":"..."}}}}
</tool_call>
You will receive a <tool_result> JSON block and must continue reasoning from that
observation. Never invent a tool result. You may make at most
{max_execute_calls} successful execution requests. Malformed requests receive
corrective feedback and do not consume that execution budget.

You may instead solve the problem directly. End with one clear, unambiguous
numeric answer or mathematical expression. Prefer marking it as
<final>answer</final>, but the tag is not required for correctness. Put only the
answer inside a final tag.
"""

MATH_SYSTEM_NO_TOOLS = """No execution tools are available. Solve the mathematics problem accurately.
You may explain your reasoning.
End with one clear, unambiguous numeric answer or mathematical expression.
Prefer marking it as <final>answer</final>, but the tag is not required for
correctness. Put only the answer inside a final tag.
"""

MATH_DIRECT_EXAMPLE = (
    {"role": "user", "content": "Example: A tray has 6 rows of 7 buttons. How many buttons are there?"},
    {"role": "assistant", "content": "There are 6 times 7 buttons, so there are 42. <final>42</final>"},
)

MATH_TOOL_EXAMPLE = (
    {"role": "user", "content": "Example: What is the sum of the squares of the integers from 1 through 20?"},
    {"role": "assistant", "content": "I will compute the finite sum, then interpret it.\n<tool_call>\n{\"name\":\"execute_python\",\"arguments\":{\"code\":\"print(sum(k*k for k in range(1, 21)))\"}}\n</tool_call>"},
    {"role": "user", "content": "<tool_result>\n{\"ok\":true,\"stdout\":\"2870\\n\",\"stderr\":\"\",\"exit_code\":0,\"timed_out\":false}\n</tool_result>"},
    {"role": "assistant", "content": "The execution returned 2870, so the requested sum is 2870. <final>2870</final>"},
)


def make_prompt(problem: dict, max_execute_calls: int = 2, tools_enabled: bool = True) -> list[dict]:
    """Build the fixed math prefix. Coding has a separate prompt module."""
    from .records import reject_private_fields

    reject_private_fields(problem)
    if problem.get("domain") != "math":
        raise ValueError("make_prompt accepts only domain=math; use make_coding_prompt for code")
    system = MATH_SYSTEM.format(max_execute_calls=max_execute_calls) if tools_enabled else MATH_SYSTEM_NO_TOOLS
    examples = (*MATH_DIRECT_EXAMPLE, *MATH_TOOL_EXAMPLE) if tools_enabled else MATH_DIRECT_EXAMPLE
    return [{"role": "system", "content": system}, *examples, {"role": "user", "content": problem["prompt"]}]


def _parse_python_call_tool_syntax(text: str) -> dict | None:
    """Recognize the model's fenced ``execute_python(code='...')`` form.

    Some Qwen completions follow the tool signature as Python syntax instead
    of emitting the documented XML/JSON envelope. Parse with ``ast`` and
    accept only a top-level call whose code argument is a literal string. The
    source is still executed only by the configured sandbox.
    """
    blocks = re.findall(
        r"```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    for block in blocks:
        if not re.search(r"\bexecute_python\s*\(", block):
            continue
        try:
            tree = ast.parse(block)
        except (SyntaxError, ValueError, TypeError):
            return {"kind": "invalid", "error": "malformed_python_tool_call"}
        for statement in tree.body:
            call = None
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
            elif isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
                call = statement.value
            elif isinstance(statement, ast.AnnAssign) and isinstance(statement.value, ast.Call):
                call = statement.value
            if not isinstance(call, ast.Call):
                continue
            if not isinstance(call.func, ast.Name) or call.func.id != TOOL_NAME:
                continue
            code_value = None
            if not call.args and len(call.keywords) == 1 and call.keywords[0].arg == "code":
                value = call.keywords[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    code_value = value.value
            elif len(call.args) == 1 and not call.keywords:
                value = call.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    code_value = value.value
            if code_value is None:
                return {"kind": "invalid", "error": "python_tool_code_must_be_a_string_literal"}
            return {"kind": "tool_call", "name": TOOL_NAME, "code": code_value, "syntax": "python_call"}
        return {"kind": "invalid", "error": "malformed_python_tool_call"}
    return None


def parse_tool_action(text: str) -> dict:
    """Classify one assistant turn as final, valid tool call, or invalid call."""
    if not isinstance(text, str):
        return {"kind": "invalid", "error": "assistant_content_must_be_text"}
    if len(text) > 65536:
        return {"kind": "invalid", "error": "assistant_response_too_large"}
    opens, closes = text.count(TOOL_CALL_OPEN), text.count(TOOL_CALL_CLOSE)
    if opens == closes == 0:
        python_call = _parse_python_call_tool_syntax(text)
        if python_call is not None:
            return python_call
        return {"kind": "final"}
    if opens != 1 or closes != 1:
        return {"kind": "invalid", "error": "expected_exactly_one_complete_tool_call"}
    start = text.find(TOOL_CALL_OPEN) + len(TOOL_CALL_OPEN)
    end = text.find(TOOL_CALL_CLOSE, start)
    if end < start:
        return {"kind": "invalid", "error": "malformed_tool_call_tags"}
    try:
        value = json.loads(text[start:end].strip())
    except (json.JSONDecodeError, TypeError):
        return {"kind": "invalid", "error": "tool_call_must_contain_valid_json"}
    if not isinstance(value, dict):
        return {"kind": "invalid", "error": "tool_call_json_must_be_an_object"}
    name = value.get("name")
    if name != TOOL_NAME:
        return {"kind": "invalid", "error": "unsupported_tool_name", "requested_name": name}
    if set(value) != {"name", "arguments"}:
        return {"kind": "invalid", "error": "tool_call_requires_name_and_arguments_only"}
    arguments = value.get("arguments")
    if not isinstance(arguments, dict) or set(arguments) != {"code"}:
        return {"kind": "invalid", "error": "arguments_must_contain_only_code"}
    if not isinstance(arguments["code"], str):
        return {"kind": "invalid", "error": "code_must_be_a_string"}
    return {"kind": "tool_call", "name": TOOL_NAME, "code": arguments["code"]}


def invalid_tool_feedback(action: dict, *, entry_point: str | None = None, exhausted: bool = False) -> str:
    """Return explicit recovery text without consuming an execution call."""
    requested = action.get("requested_name")
    if requested:
        detail = f"`{requested}` is not an available tool; it is a Python function name, not a tool. "
    else:
        detail = "The tool call was malformed. "
    if entry_point and requested == entry_point:
        detail = f"`{requested}` is the Python function you must implement, not a tool. "
    suffix = " No more malformed calls are allowed; return the completed solution directly." if exhausted else ""
    message = (
        "Invalid tool call. " + detail
        + "The only available tool is execute_python(code: str). Emit valid JSON with name "
        "execute_python and a string arguments.code containing complete Python source. "
        "Either test that source with execute_python or return the completed answer/solution."
        + suffix
    )
    return format_tool_result({"ok": False, "error": action.get("error", "invalid_tool_call"), "message": message})


def format_tool_result(result: dict | str) -> str:
    """Wrap one bounded JSON observation in the frozen textual syntax."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = {"ok": False, "error": "unstructured_tool_result", "stderr": result[:512]}
    payload = json.dumps(result, ensure_ascii=True, separators=(",", ":"))
    return f"{TOOL_RESULT_OPEN}\n{payload}\n{TOOL_RESULT_CLOSE}"


def final_answer(messages: list[dict] | str) -> str | None:
    if isinstance(messages, str):
        content = messages
    else:
        if not messages or messages[-1].get("role") != "assistant":
            return None
        content = messages[-1].get("content") or ""
    if not isinstance(content, str):
        return None
    forbidden = (TOOL_CALL_OPEN, TOOL_CALL_CLOSE, TOOL_RESULT_OPEN, TOOL_RESULT_CLOSE, "<tool_response>")
    if any(marker in content for marker in forbidden):
        return None
    if content.count("<final>") != 1 or content.count("</final>") != 1:
        return None
    match = re.search(r"<final>(.*?)</final>\s*$", content, re.DOTALL)
    return match.group(1).strip() if match else None


def strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:python)?\s*\n(.*?)\n```", text.strip(), re.DOTALL)
    return match.group(1) if match else text.strip()


def parse_assistant(text: str) -> dict:
    """Strict compatibility parser used by fixtures and protocol diagnostics."""
    action = parse_tool_action(text)
    if action["kind"] == "invalid":
        raise ValueError(action["error"])
    result = {"role": "assistant", "content": text}
    if action["kind"] == "tool_call":
        result["tool_calls"] = [{
            "type": "function",
            "function": {"name": TOOL_NAME, "arguments": {"code": action["code"]}},
        }]
    return result
