"""Separate restricted image for coding tools and official unit-test rewards."""
import os

from .sandbox import DockerSandbox, SandboxConfig


class CodingSandbox(DockerSandbox):
    def __init__(self, config=None):
        config = config or SandboxConfig(
            image=os.environ.get("TOOL_RL_CODING_SANDBOX_IMAGE", "tool-rl-coding-sandbox:local"),
            timeout_s=10.0,
            max_output_bytes=16384,
            memory_mb=256,
            cpu_seconds=5,
            process_limit=32,
            runtime=os.environ.get("TOOL_RL_SANDBOX_RUNTIME") or None,
        )
        super().__init__(config)

    def run_unit_tests(self, code, tests):
        if not isinstance(code, str) or not isinstance(tests, dict):
            return self._result(False, error="invalid_unit_test_input")
        return self.run({"mode": "unit_tests", "code": code, "tests": tests})
