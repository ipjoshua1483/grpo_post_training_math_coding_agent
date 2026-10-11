"""Coding-only scratch execution environment for model tool calls."""
import json


class CodingToolEnvironment:
    """Expose scratch Python execution only while solving a coding problem.

    The environment receives the public problem record, never hidden tests.
    Hidden unit tests are used later by the outcome verifier.
    """

    def __init__(self, problem, sandbox, max_calls=2):
        if problem.get("domain") != "code":
            raise ValueError("The coding tool is available only for domain=code problems")
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self._sandbox = sandbox
        self._max_calls = max_calls
        self._calls = []

    def reset(self, **kwargs):
        self._calls = []

    def execute_python(self, code: str) -> str:
        """Run scratch Python in a fresh restricted container.

        Args:
            code: Complete Python source. Include any examples you want to run.

        Returns:
            Bounded JSON with stdout, stderr, status, and timeout information.
        """
        if len(self._calls) >= self._max_calls:
            return json.dumps({"ok": False, "error": "tool_budget_exhausted"})
        if not isinstance(code, str):
            result = {"ok": False, "error": "code_must_be_string"}
        else:
            result = self._sandbox.execute_python(code)
        self._calls.append(result)
        public = {key: result.get(key) for key in (
            "ok", "stdout", "stderr", "exit_code", "timed_out", "output_limited", "error"
        )}
        public["stdout"] = (public["stdout"] or "")[:1024]
        public["stderr"] = (public["stderr"] or "")[:512]
        public["error"] = str(public["error"])[:128] if public["error"] is not None else None
        public["observation_truncated"] = (
            public["stdout"] != result.get("stdout", "")
            or public["stderr"] != result.get("stderr", "")
        )
        encoded = json.dumps(public, ensure_ascii=True)
        while len(encoded) > 2048:
            public["stdout"] = public["stdout"][: len(public["stdout"]) // 2]
            public["stderr"] = public["stderr"][: len(public["stderr"]) // 2]
            public["observation_truncated"] = True
            encoded = json.dumps(public, ensure_ascii=True)
        return encoded
