"""
Docker execution environment — inspired by mini-swe-agent.

Each command runs as an independent `docker exec ... bash -lc "command"`.
State persists via the container's filesystem, not via a shell session.
"""

import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Max output chars returned to the agent (to control LLM token cost)
DEFAULT_OUTPUT_LIMIT = 50_000


@dataclass
class DockerEnvironmentConfig:
    image: str = "crawl-agent:latest"
    cwd: str = "/workspace"
    timeout: int = 120
    container_timeout: str = "4h"
    output_limit: int = DEFAULT_OUTPUT_LIMIT
    env: Dict[str, str] = field(default_factory=dict)
    docker_executable: str = "docker"
    extra_run_args: list = field(default_factory=lambda: ["--rm"])


class DockerEnvironment:
    """
    Executes bash commands inside a Docker container via `docker exec`.

    Lifecycle:
        env = DockerEnvironment(config)  # starts container
        result = env.execute("curl -s https://example.com | head -20")
        env.cleanup()                    # stops & removes container
    """

    def __init__(self, config: Optional[DockerEnvironmentConfig] = None):
        self.config = config or DockerEnvironmentConfig()
        self.container_id: Optional[str] = None
        self.container_name: Optional[str] = None
        self._start_container()

    def _start_container(self):
        """Start a long-running container that waits for exec commands."""
        self.container_name = f"crawl-agent-{uuid.uuid4().hex[:8]}"
        cmd = [
            self.config.docker_executable, "run", "-d",
            "--name", self.container_name,
            "-w", self.config.cwd,
            *self.config.extra_run_args,
            self.config.image,
            "sleep", self.config.container_timeout,
        ]
        logger.info(f"Starting container: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=180, check=True,
            )
            self.container_id = result.stdout.strip()
            logger.info(f"Container started: {self.container_name} ({self.container_id[:12]})")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start container: {e.stderr}")
            raise RuntimeError(f"Docker container start failed: {e.stderr}") from e

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a bash command in the container.

        Returns:
            {"output": str, "returncode": int, "truncated": bool}
        """
        if not self.container_id:
            raise RuntimeError("Container not started")

        work_dir = cwd or self.config.cwd
        timeout = timeout or self.config.timeout

        cmd = [
            self.config.docker_executable, "exec",
            "-w", work_dir,
        ]
        # Inject environment variables
        for key, value in self.config.env.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([
            self.container_id,
            "bash", "-lc", command,
        ])

        try:
            result = subprocess.run(
                cmd,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = result.stdout or ""
            truncated = False
            if len(output) > self.config.output_limit:
                output = output[:self.config.output_limit] + f"\n... [truncated, {len(result.stdout)} chars total]"
                truncated = True

            return {
                "output": output,
                "returncode": result.returncode,
                "truncated": truncated,
            }

        except subprocess.TimeoutExpired as e:
            raw = ""
            if e.output:
                raw = e.output if isinstance(e.output, str) else e.output.decode("utf-8", errors="replace")
            return {
                "output": f"{raw}\n[TIMEOUT after {timeout}s]",
                "returncode": -1,
                "truncated": False,
            }

        except Exception as e:
            return {
                "output": f"[EXECUTION ERROR: {type(e).__name__}: {e}]",
                "returncode": -1,
                "truncated": False,
            }

    def is_alive(self) -> bool:
        """Check if the container is still running."""
        if not self.container_id:
            return False
        try:
            result = subprocess.run(
                [self.config.docker_executable, "inspect", "-f", "{{.State.Running}}", self.container_id],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() == "true"
        except Exception:
            return False

    def cleanup(self):
        """Stop and remove the container."""
        if self.container_id:
            logger.info(f"Cleaning up container {self.container_name}")
            try:
                subprocess.run(
                    [self.config.docker_executable, "stop", "-t", "5", self.container_id],
                    capture_output=True, timeout=30,
                )
            except Exception:
                # Force remove if stop fails
                try:
                    subprocess.run(
                        [self.config.docker_executable, "rm", "-f", self.container_id],
                        capture_output=True, timeout=15,
                    )
                except Exception:
                    pass
            self.container_id = None

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
