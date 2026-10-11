"""Binary coding reward from official unit tests in a restricted container."""
import ast
import re

from .protocol import final_answer, strip_code_fence


_FENCED_CODE = re.compile(
    r"```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```",
    re.IGNORECASE | re.DOTALL,
)


def _last_assistant_content(completion):
    if isinstance(completion, str):
        return completion
    if not completion or completion[-1].get("role") != "assistant":
        return None
    last = completion[-1]
    if last.get("tool_calls"):
        return None
    content = last.get("content") or ""
    return content if isinstance(content, str) else None


def extract_code_submission(completion):
    """Return one conservative code candidate without conflating format and correctness."""
    strict = final_answer(completion)
    if strict is not None:
        return strip_code_fence(strict), True, "final"

    content = _last_assistant_content(completion)
    if not content or "<tool_call>" in content or "</tool_call>" in content:
        return None, False, None
    # A malformed or ambiguous attempted final block should not fall through.
    if "<final>" in content or "</final>" in content:
        return None, False, None

    blocks = _FENCED_CODE.findall(content)
    if len(blocks) == 1 and content.count("```") == 2:
        return blocks[0].strip(), False, "code_fence"
    if blocks or "```" in content:
        return None, False, None

    candidate = content.strip()
    try:
        tree = ast.parse(candidate)
    except (SyntaxError, ValueError, TypeError):
        return None, False, None
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in tree.body):
        return None, False, None
    return candidate, False, "raw_code"


def grade_code(reference, completion, sandbox, *, truncated=False):
    if reference.get("domain") != "code":
        raise ValueError("Coding verifier accepts only domain=code references")
    code, format_ok, extraction_method = extract_code_submission(completion)
    result = {
        "reward": 0.0,
        "correct": False,
        "format_ok": format_ok,
        "extraction_method": extraction_method,
        "tests_passed": 0,
        "tests_total": 0,
    }
    if code is None or truncated:
        return result
    if not code or len(code.encode("utf-8")) > 32768:
        return result
    spec = reference.get("tests")
    if not isinstance(spec, dict) or spec.get("kind") not in {"mbpp", "humaneval", "kodcode"}:
        raise ValueError("Unsupported or missing coding unit tests")
    checked = sandbox.run_unit_tests(code, spec)
    value = checked.get("value") if checked.get("ok") else None
    if isinstance(value, dict):
        passed, total = value.get("tests_passed"), value.get("tests_total")
        if type(passed) is int and type(total) is int and 0 <= passed <= total and total > 0:
            result.update(tests_passed=passed, tests_total=total)
            result["correct"] = passed == total
            result["reward"] = float(result["correct"])
    result["verification"] = {
        "ok": checked.get("ok") is True,
        "error": checked.get("error"),
        "timed_out": checked.get("timed_out") is True,
        "output_limited": checked.get("output_limited") is True,
        "exit_code": checked.get("exit_code"),
    }
    return result
