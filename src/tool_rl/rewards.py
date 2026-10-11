"""Outcome-only verifiers. Golden answers never enter the public Python tool."""

import re
from fractions import Fraction

from .protocol import final_answer
from .math_answers import extract_math_answer, numeric_answer_text


def number(text: str) -> Fraction | None:
    text = text.strip()
    # Only accept an entire number, never search prose for a matching substring.
    if re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+)(?:\.\d+)?", text):
        text = text.replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?\d+/[1-9]\d*", text) or len(text) > 128:
        return None
    # Prevent huge integer allocation from adversarial scientific exponents.
    exponent = re.search(r"[eE]([+-]?\d+)$", text)
    if exponent and abs(int(exponent.group(1))) > 100:
        return None
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def math_reward(predicted: str, gold: str) -> float:
    """Exact numeric comparison after safe currency/unit normalization."""
    expected, actual = number(gold), number(numeric_answer_text(predicted))
    if expected is None:
        raise ValueError("Gold answer is not a supported number")
    return float(actual is not None and actual == expected)


def grade(problem: dict, completion, sandbox=None, *, truncated=False) -> dict:
    if problem.get("domain") != "math":
        raise ValueError("Use coding_rewards.grade_code for coding unit-test rewards")
    answer = final_answer(completion)
    result = {"reward": 0.0, "correct": False, "format_ok": answer is not None, "tests_passed": 0, "tests_total": 0}
    if problem["domain"] == "math":
        extraction = extract_math_answer(completion)
        answer = extraction["answer"]
        result.update(extracted_answer=answer, extraction_method=extraction["method"])
    if answer is None or truncated:
        return result
    if problem["domain"] == "math":
        if problem.get("verifier", "numeric") == "numeric":
            correct = math_reward(answer, problem["answer"]) == 1.0
            if not correct and number(numeric_answer_text(answer)) is None and sandbox is not None and len(answer) <= 1024:
                checked = sandbox.run({"mode": "symbolic", "gold": problem["answer"], "answer": answer, "exact_numeric": True})
                result["verification"] = {key: checked.get(key) for key in ("ok", "error", "timed_out", "elapsed_s")}
                correct = checked.get("ok", False) and checked.get("value") is True
        else:
            if problem.get("verifier") != "symbolic":
                raise ValueError("Unsupported verifier")
            if sandbox is None:
                raise ValueError("Symbolic verification requires the bounded sandbox")
            if len(answer) > 1024:
                return result
            checked = sandbox.run({"mode": "symbolic", "gold": problem["answer"], "answer": answer})
            result["verification"] = {key: checked.get(key) for key in ("ok", "error", "timed_out", "elapsed_s")}
            correct = checked.get("ok", False) and checked.get("value") is True
        result.update(correct=correct, reward=float(correct))
        return result
