"""
Local execution environment — fallback when Docker is not available.

Same interface as DockerEnvironment but runs commands via local subprocess.
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_LIMIT = 50_000


@dataclass
class LocalEnvironmentConfig:
    cwd: str = ""
    timeout: int = 120
    output_limit: int = DEFAULT_OUTPUT_LIMIT
    env: Dict[str, str] = field(default_factory=dict)


class LocalEnvironment:
    """
    Executes commands locally via subprocess.run.
    Drop-in replacement for DockerEnvironment during development.
    """

    def __init__(self, config: Optional[LocalEnvironmentConfig] = None):
        self.config = config or LocalEnvironmentConfig()

    def execute(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a bash/shell command locally."""
        work_dir = cwd or self.config.cwd or os.getcwd()
        timeout = timeout or self.config.timeout

        # On Windows, use cmd; on Unix, use bash
        shell_cmd = command
        shell = True

        try:
            result = subprocess.run(
                shell_cmd,
                shell=shell,
                text=True,
                cwd=work_dir,
                env={**os.environ, **self.config.env},
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
        return True

    def cleanup(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
