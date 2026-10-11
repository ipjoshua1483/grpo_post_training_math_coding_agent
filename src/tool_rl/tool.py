"""Public execution API; importing it neither starts Docker nor executes code."""
from .sandbox import DockerSandbox, SandboxConfig, SandboxUnavailable


def execute_python(code: str, *, sandbox=None) -> dict:
    """Execute Python only through the configured restricted Docker backend."""
    return (sandbox or DockerSandbox()).execute_python(code)

