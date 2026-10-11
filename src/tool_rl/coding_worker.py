"""Container-only worker for coding scratch runs and hidden unit-test suites."""
import contextlib
import io
import json
import os
import sys

MAX_OUTPUT = 4096


class OutputLimit(Exception):
    pass


class BoundedBuffer(io.StringIO):
    def write(self, text):
        text = str(text)
        room = MAX_OUTPUT - self.tell()
        if len(text) > room:
            super().write(text[:max(0, room)])
            raise OutputLimit("Printed output exceeded the limit")
        return super().write(text)


def _failure(exc):
    return {"type": type(exc).__name__}


def _run_mbpp(namespace, spec):
    tests = list(spec.get("tests", [])) + list(spec.get("challenge_tests", []))
    if not tests or not all(isinstance(test, str) for test in tests):
        raise ValueError("Invalid MBPP test suite")
    passed, failures = 0, []
    for index, test in enumerate(tests):
        try:
            exec(compile(test, "<hidden-test>", "exec"), namespace)
            passed += 1
        except BaseException as exc:
            failures.append({"index": index, **_failure(exc)})
    return passed, len(tests), failures


def _run_humaneval(namespace, spec):
    entry_point, test_source = spec.get("entry_point"), spec.get("test")
    if not isinstance(entry_point, str) or not entry_point.isidentifier() or not isinstance(test_source, str):
        raise ValueError("Invalid HumanEval test suite")
    exec(compile(test_source, "<hidden-tests>", "exec"), namespace)
    candidate, check = namespace.get(entry_point), namespace.get("check")
    if not callable(candidate) or not callable(check):
        raise NameError("Required candidate/check function is missing")
    try:
        check(candidate)
        return 1, 1, []
    except BaseException as exc:
        return 0, 1, [{"index": 0, **_failure(exc)}]


def _run_kodcode(namespace, spec):
    source, functions = spec.get("source"), spec.get("functions")
    if not isinstance(source, str) or not isinstance(functions, list) or not functions:
        raise ValueError("Invalid KodCode test suite")
    if not all(isinstance(name, str) and name.isidentifier() for name in functions):
        raise ValueError("Invalid KodCode test names")
    exec(compile(source, "<hidden-tests>", "exec"), namespace)
    passed, failures = 0, []
    for index, name in enumerate(functions):
        test = namespace.get(name)
        if not callable(test):
            failures.append({"index": index, "type": "NameError"})
            continue
        try:
            test()
            passed += 1
        except BaseException as exc:
            failures.append({"index": index, **_failure(exc)})
    return passed, len(functions), failures


def handle(payload):
    if os.environ.get("TOOL_RL_SANDBOXED") != "1":
        raise RuntimeError("This worker must run inside the configured sandbox")
    mode = payload.get("mode")
    if mode == "ping":
        return {"ok": True, "value": "pong"}
    stdout, stderr = BoundedBuffer(), BoundedBuffer()
    try:
        namespace = {"__name__": "__candidate__"}
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            value = None
            if mode == "unit_tests":
                spec = payload.get("tests")
                if not isinstance(spec, dict):
                    raise ValueError("Missing unit-test specification")
                if spec.get("kind") == "mbpp":
                    setup = spec.get("setup", "")
                    if not isinstance(setup, str):
                        raise ValueError("Invalid MBPP setup")
                    if setup:
                        exec(compile(setup, "<hidden-setup>", "exec"), namespace)
                    exec(compile(payload["code"], "<candidate>", "exec"), namespace)
                    passed, total, failures = _run_mbpp(namespace, spec)
                elif spec.get("kind") == "humaneval":
                    exec(compile(payload["code"], "<candidate>", "exec"), namespace)
                    passed, total, failures = _run_humaneval(namespace, spec)
                elif spec.get("kind") == "kodcode":
                    exec(compile(payload["code"], "<candidate>", "exec"), namespace)
                    passed, total, failures = _run_kodcode(namespace, spec)
                else:
                    raise ValueError("Unsupported coding benchmark")
                value = {"tests_passed": passed, "tests_total": total, "failures": failures}
            elif mode == "execute":
                exec(compile(payload["code"], "<candidate>", "exec"), namespace)
            else:
                raise ValueError("Unknown worker mode")
        return {"ok": True, "exit_code": 0, "value": value, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
    except BaseException as exc:
        return {"ok": False, "exit_code": 1, "error": type(exc).__name__, "output_limited": isinstance(exc, OutputLimit), "stderr": stderr.getvalue(), "stdout": stdout.getvalue()}


if __name__ == "__main__":
    try:
        raw = sys.stdin.buffer.read(131073)
        if len(raw) > 131072:
            raise ValueError("Input limit exceeded")
        response = handle(json.loads(raw))
    except BaseException as exc:
        response = {"ok": False, "error": type(exc).__name__, "stderr": str(exc)[:MAX_OUTPUT]}
    print(json.dumps(response, allow_nan=False))
