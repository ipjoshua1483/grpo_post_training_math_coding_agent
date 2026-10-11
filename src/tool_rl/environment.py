"""TRL creates one ToolEnvironment per rollout, isolating tool budgets."""

import json

from .sandbox import DockerSandbox


class ToolEnvironment:
    def __init__(self, sandbox=None, max_calls: int = 2):
        self._sandbox = sandbox or DockerSandbox()
        self._max_calls = max_calls
        self._calls = []

    def reset(self, **kwargs) -> None:
        self._calls = []

    def execute_python(self, code: str) -> str:
        """Run Python in a fresh sandbox and return printed output or an error.

        Args:
            code: Complete Python source. Use print to display results.

        Returns:
            A JSON string containing success, stdout, stderr, or an error.
        """
        if len(self._calls) >= self._max_calls:
            return json.dumps({"ok": False, "error": "tool_budget_exhausted"})
        if not isinstance(code, str):
            result = {"ok": False, "error": "code_must_be_string"}
        else:
            result = self._sandbox.execute_python(code)
        self._calls.append(result)
        public = {key: result.get(key) for key in ("ok", "stdout", "stderr", "exit_code", "timed_out", "output_limited", "error")}
        public["stdout"] = (public["stdout"] or "")[:1024]
        public["stderr"] = (public["stderr"] or "")[:512]
        public["error"] = str(public["error"])[:128] if public["error"] is not None else None
        public["observation_truncated"] = public["stdout"] != result.get("stdout", "") or public["stderr"] != result.get("stderr", "")
        encoded = json.dumps(public, ensure_ascii=True)
        while len(encoded) > 2048:
            public["stdout"] = public["stdout"][:len(public["stdout"]) // 2]
            public["stderr"] = public["stderr"][:len(public["stderr"]) // 2]
            public["observation_truncated"] = True
            encoded = json.dumps(public, ensure_ascii=True)
        return encoded
