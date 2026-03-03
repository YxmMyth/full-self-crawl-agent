"""
CrawlAgent — mini-swe-agent style ReAct loop for web data extraction.

Architecture:
    messages = [system_prompt, task_description]
    while not done:
        response = llm.chat(messages)         # LLM decides next action
        output = docker_env.execute(command)  # Execute in Docker
        messages.append(observation)          # Append result to history
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CrawlAgentConfig:
    """Configuration for the CrawlAgent."""
    step_limit: int = 25
    command_timeout: int = 120
    max_tokens: int = 4096
    temperature: float = 0.3
    max_output_chars: int = 15000  # truncate long command outputs for LLM


class CrawlAgent:
    """
    Autonomous web crawling agent using a ReAct loop.

    Inspired by mini-swe-agent's DefaultAgent:
    - System prompt encodes methodology
    - LLM generates bash commands
    - Docker container executes them
    - Observations feed back into history
    - LLM decides when to stop
    """

    COMPLETION_MARKER = "CRAWL_COMPLETE"
    RESULT_FILE = "/workspace/results.json"

    def __init__(
        self,
        llm_client,
        docker_env,
        config: Optional[CrawlAgentConfig] = None,
    ):
        self.llm = llm_client
        self.env = docker_env
        self.config = config or CrawlAgentConfig()
        self.messages: List[Dict[str, str]] = []
        self.step_count = 0
        self.total_tokens_used = 0

    async def run(self, url: str, spec: dict) -> Dict[str, Any]:
        """
        Run the crawl agent on a single URL.

        Args:
            url: Target URL to crawl
            spec: Extraction specification (targets, fields, etc.)

        Returns:
            Dict with success, extracted_data, metadata, messages history
        """
        from src.prompts.crawl_system import render_system_prompt, render_task_prompt

        # Build initial messages
        system_prompt = render_system_prompt(self.config.step_limit)
        task_prompt = render_task_prompt(url, spec)

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        self.step_count = 0

        start_time = time.time()
        logger.info(f"CrawlAgent starting: {url}")

        for step in range(self.config.step_limit):
            self.step_count = step + 1
            logger.info(f"Step {self.step_count}/{self.config.step_limit}")

            # 1. Query LLM for next action
            try:
                response = await self.llm.chat(
                    self.messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            except Exception as e:
                logger.error(f"LLM call failed at step {self.step_count}: {e}")
                break

            if not response:
                logger.warning("Empty LLM response, stopping")
                break

            self.messages.append({"role": "assistant", "content": response})

            # 2. Check for completion
            if self.COMPLETION_MARKER in response:
                logger.info(f"Agent signaled completion at step {self.step_count}")
                break

            # 3. Extract and execute command
            command = self._extract_command(response)
            if command is None:
                # No command found — LLM is thinking or confused
                # Prompt it to take action
                self.messages.append({
                    "role": "user",
                    "content": "Please provide a bash command to execute, or say CRAWL_COMPLETE if done.",
                })
                continue

            # 4. Execute in Docker
            result = self.env.execute(
                command,
                timeout=self.config.command_timeout,
            )

            # 5. Build observation and append to history
            observation = self._format_observation(result)
            self.messages.append({"role": "user", "content": observation})

            logger.info(
                f"Step {self.step_count}: rc={result['returncode']}, "
                f"output_len={len(result['output'])}"
            )

        elapsed = time.time() - start_time

        # Extract final results from container
        final_result = self._collect_results()

        return {
            "success": final_result.get("success", False),
            "extracted_data": final_result.get("extracted_data", []),
            "metadata": {
                **final_result.get("metadata", {}),
                "steps": self.step_count,
                "elapsed_seconds": round(elapsed, 1),
                "llm_calls": self.step_count,
                "source_url": url,
            },
            "messages": self.messages,  # full history for debugging
        }

    def _extract_command(self, response: str) -> Optional[str]:
        """
        Extract bash command from LLM response.

        Looks for fenced code blocks: ```bash ... ``` or ```sh ... ```
        Falls back to ``` ... ``` if no language specified.
        """
        # Try ```bash or ```sh blocks first
        patterns = [
            r"```(?:bash|sh)\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                cmd = match.group(1).strip()
                if cmd:
                    return cmd
        return None

    def _format_observation(self, result: Dict[str, Any]) -> str:
        """Format execution result as an observation for the LLM."""
        output = result["output"]
        rc = result["returncode"]

        # Truncate long output to control token costs
        truncated = False
        if len(output) > self.config.max_output_chars:
            half = self.config.max_output_chars // 2
            output = (
                output[:half]
                + f"\n\n... [{len(result['output'])} chars total, showing first and last {half}] ...\n\n"
                + output[-half:]
            )
            truncated = True

        header = f"[Exit code: {rc}]"
        if truncated:
            header += " [Output truncated]"
        if rc == -1:
            header += " [TIMEOUT]"

        return f"{header}\n{output}"

    def _collect_results(self) -> Dict[str, Any]:
        """Read results.json from the container."""
        try:
            result = self.env.execute(
                f"cat {self.RESULT_FILE}",
                timeout=10,
            )
            if result["returncode"] == 0 and result["output"].strip():
                data = json.loads(result["output"])
                if isinstance(data, dict):
                    return data
                # If it's a list, wrap it
                if isinstance(data, list):
                    return {"success": True, "extracted_data": data}
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to read results from container: {e}")

        # Fallback: try to extract data from the last assistant message
        return self._extract_results_from_history()

    def _extract_results_from_history(self) -> Dict[str, Any]:
        """Last resort: scan message history for JSON data."""
        for msg in reversed(self.messages):
            if msg["role"] != "user":
                continue
            # Look for JSON arrays or objects in observation outputs
            content = msg["content"]
            # Try to find JSON in the output
            json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', content)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                    if isinstance(data, list) and len(data) > 0:
                        return {"success": True, "extracted_data": data}
                    if isinstance(data, dict) and data.get("extracted_data"):
                        return data
                except json.JSONDecodeError:
                    continue
        return {"success": False, "extracted_data": [], "error": "No results collected"}
