"""Container-only entry point. Never run generated code directly on the host."""
import contextlib
import io
import json
import os
import sys
import traceback

MAX_OUTPUT = 4096


class OutputLimit(Exception):
    pass


class BoundedBuffer(io.StringIO):
    def write(self, text):
        if self.tell() + len(text) > MAX_OUTPUT:
            room = max(0, MAX_OUTPUT - self.tell())
            super().write(text[:room])
            raise OutputLimit("Printed output exceeded the limit")
        return super().write(text)


def handle(payload):
    if os.environ.get("TOOL_RL_SANDBOXED") != "1":
        raise RuntimeError("This worker must run inside the configured sandbox")
    mode = payload["mode"]
    if mode == "ping":
        return {"ok": True, "value": "pong"}
    if mode == "symbolic":
        from math_verify import LatexExtractionConfig, parse, verify
        config = [LatexExtractionConfig()]
        gold = parse("$" + payload["gold"] + "$", extraction_config=config, fallback_mode="no_fallback")
        answer = parse("$" + payload["answer"] + "$", extraction_config=config, fallback_mode="no_fallback")
        if payload.get("exact_numeric"):
            from sympy import Expr, simplify
            equivalent = any(isinstance(g, Expr) and isinstance(a, Expr) and not g.free_symbols and not a.free_symbols and simplify(g - a) == 0 for g in gold for a in answer)
            return {"ok": True, "value": bool(equivalent)}
        return {"ok": True, "value": bool(gold and answer and verify(gold, answer))}
    namespace = {"__name__": "__candidate__"}
    stdout, stderr = BoundedBuffer(), BoundedBuffer()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exec(compile(payload["code"], "<candidate>", "exec"), namespace)
            value = None
            if mode == "call":
                value = namespace[payload["entry_point"]](*payload["args"], **payload.get("kwargs", {}))
                value = json.loads(json.dumps(value, allow_nan=False))
            elif mode != "execute":
                raise ValueError("Unknown worker mode")
        return {"ok": True, "exit_code": 0, "value": value, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
    except BaseException as exc:
        return {"ok": False, "exit_code": 1, "error": type(exc).__name__, "output_limited": isinstance(exc, OutputLimit), "stderr": traceback.format_exc()[-MAX_OUTPUT:], "stdout": stdout.getvalue()}


if __name__ == "__main__":
    try:
        response = handle(json.loads(sys.stdin.buffer.read(65537)))
    except BaseException as exc:
        response = {"ok": False, "error": type(exc).__name__, "stderr": str(exc)}
    print(json.dumps(response, allow_nan=False))
