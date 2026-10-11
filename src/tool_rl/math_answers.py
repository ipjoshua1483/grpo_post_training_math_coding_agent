"""Extract a submitted mathematical answer without consulting the gold answer.

Correctness extraction is independent from strict protocol-format diagnostics.
No generated expression is evaluated or symbolically parsed on the host.
"""
import re


ANSWER_CUE = re.compile(r"(?:\b(?:the\s+)?(?:final\s+answer|answer)\s*(?:is\s*:?|:|=)\s*)", re.IGNORECASE)
CONCLUSION_CUE = re.compile(r"\b(?:result|profit|total|solution)\b(?:\s+[A-Za-z']+){0,4}\s+(?:is|was|equals)\s*:?\s*", re.IGNORECASE)
FINAL_RELATION = re.compile(
    r"\b(?:is|was|equals?|costs?|cost|totals?|totaled|makes?|made|gives?|gave|"
    r"produces?|produced|yields?|yielded)\b|(?<![<>=])=(?!=)",
    re.IGNORECASE,
)
NUMBER_CANDIDATE = re.compile(
    r"(?<![\w.])(?:\\?\$|USD\s*)?[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d*)?(?:[eE][+-]?\d+)?|(?<![\w.])[+-]?\.\d+(?:[eE][+-]?\d+)?",
    re.IGNORECASE,
)


def assistant_text(completion):
    if isinstance(completion, str):
        text = completion
    else:
        if not completion or completion[-1].get("role") != "assistant" or completion[-1].get("tool_calls"):
            return None
        text = completion[-1].get("content")
    if not isinstance(text, str) or len(text) > 32768:
        return None
    if any(marker in text for marker in (
        "<tool_call>", "</tool_call>", "<tool_result>", "</tool_result>",
        "<tool_response>", "</tool_response>",
    )):
        return None
    return text.strip()


def boxed_answers(text):
    answers = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        start, depth = match.end(), 1
        for index in range(start, len(text)):
            if text[index] == "{" and text[index - 1] != "\\":
                depth += 1
            elif text[index] == "}" and text[index - 1] != "\\":
                depth -= 1
                if depth == 0:
                    answers.append(text[start:index].strip())
                    break
    return answers


def clean_math(text):
    text = text.strip().strip("*").strip().rstrip(".").strip()
    for left, right in ((r"\(", r"\)"), (r"\[", r"\]"), ("$$", "$$"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
            break
    return text


def extracted(candidate, method):
    answer = clean_math(candidate)
    if not answer or re.search(r"\b(?:or|either|alternatively|wrong|incorrect|not)\b", answer, re.IGNORECASE):
        return {"answer": None, "method": "ambiguous_answer"}
    return {"answer": answer, "method": method}


def boxed_submission(text, method):
    boxes = boxed_answers(text)
    if len(boxes) != len(re.findall(r"\\(?:boxed|fbox)\s*\{", text)):
        return {"answer": None, "method": "incomplete_boxed"}
    if len({clean_math(box) for box in boxes}) != 1:
        return {"answer": None, "method": "ambiguous_boxed"}
    return extracted(boxes[0], method)


def cue_submission(text, method):
    cues = list(ANSWER_CUE.finditer(text))
    # An explicit final conclusion takes precedence over earlier tentative answers.
    final_cues = [cue for cue in cues if "final" in cue.group(0).lower()]
    cues = final_cues or cues or list(CONCLUSION_CUE.finditer(text))
    candidates = []
    for index, cue in enumerate(cues):
        end = cues[index + 1].start() if index + 1 < len(cues) else len(text)
        lines = [line.strip() for line in text[cue.end():end].splitlines() if line.strip()]
        if lines:
            candidates.append(clean_math(lines[0]))
    if not candidates or len(set(candidates)) != 1:
        return {"answer": None, "method": "ambiguous_answer_cues"}
    return extracted(candidates[0], method)


def standalone_math(text):
    """Lexical eligibility only; actual expression parsing stays in the sandbox."""
    answer = clean_math(text)
    normalized = numeric_answer_text(answer)
    if re.fullmatch(r"[+-]?(?:\d[\d,.]*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+/[1-9]\d*)", normalized or ""):
        return True
    if not answer or not re.fullmatch(r"[A-Za-z0-9\s\\{}()[\]+*/^_=.,:<>|%-]+", answer):
        return False
    functions = {"sqrt", "frac", "dfrac", "tfrac", "pi", "infty", "sin", "cos", "tan", "log", "ln", "exp", "left", "right", "cup", "cap", "mathbb", "cdot", "times", "pm", "le", "ge", "leq", "geq"}
    words = re.findall(r"[A-Za-z]+", answer)
    return all(len(word) == 1 or word in functions for word in words)


def final_meaningful_sentence(text):
    """Return only the last submitted sentence/line, never scan the reasoning."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    # Markdown/LaTeX may begin a sentence, so sentence boundaries are deliberately
    # narrower than a general prose tokenizer. Decimal points never split here.
    sentences = re.split(r"(?<=[.!?])\s+(?=(?:[A-Z]|[*_`$]|\\\())", lines[-1])
    return sentences[-1].strip() if sentences else None


def math_candidates(fragment):
    """Find candidates in one conclusion fragment without consulting the gold."""
    candidates, occupied = [], []

    def add(match, value=None):
        candidate = clean_math(match.group(1) if value is None else value)
        if candidate:
            candidates.append(candidate)
            occupied.append(match.span())

    # Preserve structured expressions before scanning their interior numbers.
    structured = (
        re.compile(r"\\\((.+?)\\\)"),
        re.compile(r"\\\[(.+?)\\\]"),
        re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)"),
        re.compile(r"\*\*(.+?)\*\*"),
        re.compile(r"(?<![\w])\(([^()]*,[^()]*)\)"),
    )
    for pattern in structured:
        for match in pattern.finditer(fragment):
            if not any(start < match.end() and match.start() < end for start, end in occupied):
                add(match)
    for match in NUMBER_CANDIDATE.finditer(fragment):
        if not any(start <= match.start() and match.end() <= end for start, end in occupied):
            add(match, match.group(0))
    # Repeated renderings of the same value are not ambiguous.
    return list(dict.fromkeys(candidates))


def final_sentence_submission(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        previous, last = clean_math(lines[-2]), clean_math(lines[-1])
        if all(re.fullmatch(r"[+-]?[\d,.]+", value) for value in (previous, last)) and previous != last:
            return {"answer": None, "method": "ambiguous_last_lines"}
    sentence = final_meaningful_sentence(text)
    if not sentence:
        return {"answer": None, "method": "no_clear_final_answer"}
    relations = list(FINAL_RELATION.finditer(sentence))
    fragment = sentence[relations[-1].end():].strip() if relations else sentence
    direct = clean_math(fragment)
    if standalone_math(direct):
        return extracted(direct, "final_sentence")
    candidates = math_candidates(fragment)
    if len(candidates) != 1:
        return {"answer": None, "method": "ambiguous_final_sentence" if candidates else "no_clear_final_answer"}
    return extracted(candidates[0], "final_sentence")


def extract_math_answer(completion):
    text = assistant_text(completion)
    if not text:
        return {"answer": None, "method": "missing_submission"}
    opens, closes = text.count("<final>"), text.count("</final>")
    if opens or closes:
        if opens != 1 or closes != 1:
            return {"answer": None, "method": "ambiguous_or_incomplete_final"}
        match = re.search(r"<final>(.*?)</final>", text, re.DOTALL)
        if match is None:
            return {"answer": None, "method": "incomplete_final"}
        candidate = match.group(1).strip()
        if re.search(r"\\(?:boxed|fbox)\s*\{", candidate):
            return boxed_submission(candidate, "final_tag+boxed")
        if ANSWER_CUE.search(candidate) or CONCLUSION_CUE.search(candidate):
            return cue_submission(candidate, "final_tag+answer_cue")
        return extracted(candidate, "final_tag")
    if re.search(r"\\(?:boxed|fbox)\s*\{", text):
        return boxed_submission(text, "boxed")
    if ANSWER_CUE.search(text) or CONCLUSION_CUE.search(text):
        return cue_submission(text, "answer_cue")
    return final_sentence_submission(text)


def numeric_answer_text(answer):
    """Permit one numeric answer plus currency/units, never scan a derivation.

Expressions such as LaTeX fractions are left for the bounded symbolic verifier.
"""
    if answer is None:
        return None
    text = clean_math(answer)
    units = r"(?:dollars?|USD|cents?|percent|hours?|minutes?|seconds?|days?|weeks?|years?|miles?|meters?|metres?|cm|km|liters?|litres?|people|students|apples?|items?|lawns?)"
    match = re.fullmatch(r"(?:\\?\$|USD\s*)?\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][+-]?\d+)?|[+-]?\.\d+(?:[eE][+-]?\d+)?|[+-]?\d+/[1-9]\d*)\s*(?:" + units + r")?", text, re.IGNORECASE)
    return match.group(1) if match else text
