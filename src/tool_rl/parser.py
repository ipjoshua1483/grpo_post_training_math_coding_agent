"""Parser API backed by local textual validation, not a tokenizer response schema."""
from .protocol import final_answer, parse_assistant


def parse_tool_calls(text):
    return parse_assistant(text).get("tool_calls", [])


def parse_final(text):
    return final_answer(text)

