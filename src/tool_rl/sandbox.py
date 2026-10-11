"""Fresh CPU Docker containers; generated code never runs in the host process."""
import json
import math
import os
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass


class SandboxUnavailable(RuntimeError):
    """Infrastructure failures stop the experiment, rather than becoming reward 0."""


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "tool-rl-python-sandbox:local"
    timeout_s: float = 10.0
    max_output_bytes: int = 16384
    memory_mb: int = 256
    cpu_seconds: int = 5
    process_limit: int = 32
    runtime: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("A positive finite wall timeout is required")
        if self.memory_mb < 32 or self.max_output_bytes < 1024 or self.cpu_seconds < 1 or self.process_limit < 4:
            raise ValueError("Invalid sandbox resource limits")


class DockerSandbox:
    def __init__(self, config=None):
        self.config = config or SandboxConfig(image=os.environ.get("TOOL_RL_SANDBOX_IMAGE", "tool-rl-python-sandbox:local"), runtime=os.environ.get("TOOL_RL_SANDBOX_RUNTIME") or None)

    def command(self, name):
        c = self.config
        command = ["docker", "create", "--pull=never", "--name", name, "--interactive",
            "--network=none", "--read-only", "--user=65534:65534", "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true", f"--pids-limit={c.process_limit}",
            f"--memory={c.memory_mb}m", f"--memory-swap={c.memory_mb}m", "--cpus=1",
            f"--ulimit=cpu={c.cpu_seconds}:{c.cpu_seconds}", "--ulimit=nofile=64:64",
            "--ulimit=fsize=1048576:1048576", "--tmpfs=/tmp:rw,noexec,nosuid,size=64m,mode=1777",
            "--env=TOOL_RL_SANDBOXED=1", "--log-driver=none", "--label=tool-rl.sandbox=true"]
        if c.runtime:
            command.extend(["--runtime", c.runtime])
        # No host mounts, GPU devices, credentials, or Docker socket.
        return command + [c.image]

    def _remove(self, name):
        try:
            removed = subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable(f"Could not confirm cleanup of {name}") from exc
        if removed.returncode and b"No such container" not in removed.stderr:
            raise SandboxUnavailable(f"Container cleanup failed for {name}: {removed.stderr[:1024]!r}")

    def preflight(self):
        try:
            inspected = subprocess.run(["docker", "image", "inspect", self.config.image, "--format", "{{.Id}}"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SandboxUnavailable("Docker unavailable; start its engine and build the sandbox image") from exc
        if inspected.returncode:
            raise SandboxUnavailable("Docker engine/image unavailable. Build: docker build -f docker/Dockerfile.sandbox -t tool-rl-python-sandbox:local .")
        result = self.run({"mode": "ping"})
        if result.get("value") != "pong" or not result["ok"]:
            raise SandboxUnavailable(f"Sandbox health check failed: {result}")
        return {"image_id": inspected.stdout.strip(), "config": asdict(self.config)}

    def execute_python(self, code):
        if not isinstance(code, str):
            return self._result(False, error="code_must_be_string")
        return self.run({"mode": "execute", "code": code})

    @staticmethod
    def _result(ok, **kwargs):
        return {"ok": ok, "stdout": "", "stderr": "", "exit_code": None, "timed_out": False, "output_limited": False, **kwargs}

    def run(self, payload):
        encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
        if len(encoded) > 65536:
            return self._result(False, error="input_limit")
        name = "tool-rl-" + uuid.uuid4().hex
        started = time.monotonic()
        # Create before starting: timeout cleanup always targets a known container.
        # A delayed/failed create cannot start hostile code on its own.
        try:
            created = subprocess.run(self.command(name), capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._remove(name)
            raise SandboxUnavailable("Docker container creation failed") from exc
        if created.returncode:
            self._remove(name)
            raise SandboxUnavailable(f"Docker create failed: {created.stderr[:1024]!r}")
        try:
            proc = subprocess.Popen(["docker", "start", "--attach", "--interactive", name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            self._remove(name)
            raise SandboxUnavailable("Docker start unavailable") from exc
        chunks = {"stdout": bytearray(), "stderr": bytearray()}
        lock, overflow = threading.Lock(), threading.Event()
        def drain(stream, key):
            while True:
                block = stream.read(1024)
                if not block:
                    return
                with lock:
                    room = self.config.max_output_bytes - sum(map(len, chunks.values()))
                    chunks[key].extend(block[:max(0, room)])
                    if len(block) > room:
                        overflow.set()
        def write_input():
            try:
                proc.stdin.write(encoded)
                proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
        threads = [threading.Thread(target=drain, args=(proc.stdout, "stdout"), daemon=True), threading.Thread(target=drain, args=(proc.stderr, "stderr"), daemon=True), threading.Thread(target=write_input, daemon=True)]
        for thread in threads:
            thread.start()
        reason = None
        deadline = time.monotonic() + self.config.timeout_s
        try:
            while proc.poll() is None:
                if overflow.is_set():
                    reason = "output_limit"
                    break
                if time.monotonic() >= deadline:
                    reason = "timeout"
                    break
                time.sleep(0.01)
        finally:
            try:
                # Remove the whole container/process namespace, not only its CLI.
                self._remove(name)
            finally:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
                for thread in threads:
                    thread.join(timeout=1)
                for stream in (proc.stdout, proc.stderr):
                    stream.close()
        elapsed = time.monotonic() - started
        diagnostic = chunks["stderr"].decode("utf-8", errors="replace")[:4096]
        if reason or overflow.is_set():
            reason = reason or "output_limit"
            return self._result(False, error=reason, timed_out=reason == "timeout", output_limited=reason == "output_limit", exit_code=proc.returncode, elapsed_s=elapsed, stderr=diagnostic)
        if b"Error response from daemon" in chunks["stderr"] or b"Cannot connect to the Docker daemon" in chunks["stderr"]:
            raise SandboxUnavailable(diagnostic)
        if proc.returncode:
            return self._result(False, error="execution_failed", exit_code=proc.returncode, elapsed_s=elapsed, stderr=diagnostic)
        try:
            def reject_constant(value):
                raise ValueError(value)
            result = json.loads(chunks["stdout"], parse_constant=reject_constant)
            if not isinstance(result, dict) or type(result.get("ok")) is not bool:
                raise ValueError("Invalid result envelope")
            if not isinstance(result.get("stdout", ""), str) or not isinstance(result.get("stderr", ""), str):
                raise ValueError("Invalid output strings")
            code = result.get("exit_code", proc.returncode)
            if type(code) is not int:
                raise ValueError("Invalid exit code")
            return self._result(result["ok"], value=result.get("value"), stdout=result.get("stdout", ""), stderr=result.get("stderr", ""), error=result.get("error"), output_limited=result.get("output_limited") is True, exit_code=code, container_exit_code=proc.returncode, elapsed_s=elapsed)
        except (ValueError, UnicodeDecodeError):
            return self._result(False, error="invalid_output", exit_code=proc.returncode, elapsed_s=elapsed, stderr=diagnostic)
