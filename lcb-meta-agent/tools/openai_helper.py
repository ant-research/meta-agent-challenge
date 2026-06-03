#!/usr/bin/env python3
"""
OpenAI API Helper for LCB Meta-Agent

Provides utilities for interacting with OpenAI-compatible API endpoints.
Configure via environment variables:
  - TASK_MODEL_API_KEY: API key
  - TASK_MODEL_API_BASE: Base URL for API endpoint (optional)
  - TASK_MODEL_NAME: Model name (default: gpt-4)

⚠️ CRITICAL: This is the ONLY allowed way to access model APIs.
   Do not import or use any other LLM packages or services.
"""

import os
import json
import socket
import time
from typing import Any, Dict, List, Optional

import httpx

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    raise


class OpenAIHelper:
    """Helper class for OpenAI API interactions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: API key (defaults to TASK_MODEL_API_KEY env var)
            base_url: Base URL for API (defaults to TASK_MODEL_API_BASE env var)
            model: Model name (defaults to TASK_MODEL_NAME env var or "gpt-4")
        """
        self.api_key = api_key or os.environ.get("TASK_MODEL_API_KEY")
        self.base_url = base_url or os.environ.get("TASK_MODEL_API_BASE")
        self.model = model or os.environ.get("TASK_MODEL_NAME", "gpt-4")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set TASK_MODEL_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize client with TCP keepalive
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        keepalive_opts = [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30),
        ]
        if hasattr(socket, "TCP_KEEPIDLE"):
            keepalive_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30))
        elif hasattr(socket, "TCP_KEEPALIVE"):
            keepalive_opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30))
        transport = httpx.HTTPTransport(socket_options=keepalive_opts)
        client_kwargs["http_client"] = httpx.Client(transport=transport)

        self.client = OpenAI(**client_kwargs)
        self.call_count = 0
        self.total_latency = 0.0
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = f"/workspace/openai_helper_{timestamp}.log"

    def _to_loggable(self, value: Any) -> Any:
        """Convert values to a JSON-serializable form for request logging."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): self._to_loggable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_loggable(v) for v in value]
        return repr(value)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_completion_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (overrides default)
            temperature: Sampling temperature
            max_completion_tokens: Maximum completion-token budget to generate
            **kwargs: Additional arguments to pass to API

        Returns:
            Response content as string
        """
        self.call_count += 1
        t0 = time.time()
        request_kwargs = {
            "model": model or self.model,
            "messages": messages,
            **kwargs,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_completion_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_completion_tokens

        response = self.client.chat.completions.create(**request_kwargs)
        latency = time.time() - t0
        self.total_latency += latency
        usage = response.usage
        content = response.choices[0].message.content
        extra_body = kwargs.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        entry = {
            "call": self.call_count,
            "latency": round(latency, 2),
            "tokens_in": getattr(usage, "prompt_tokens", None),
            "tokens_out": getattr(usage, "completion_tokens", None),
            "model": model or self.model,
            "temperature": temperature,
            "top_p": kwargs.get("top_p"),
            "top_k": extra_body.get("top_k"),
            "min_p": extra_body.get("min_p"),
            "enable_thinking": extra_body.get("enable_thinking"),
            "timeout": self._to_loggable(kwargs.get("timeout")),
            "max_completion_tokens": max_completion_tokens,
            "messages": messages,
            "response": content,
        }
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass  # /workspace may be read-only in some runtime environments
        return content

    def solve_lcb_problem(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_completion_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Simple wrapper for code-generation calls in LiveCodeBench.
        Caller fully controls prompt construction.

        Args:
            user_prompt: Fully assembled user prompt
            system_prompt: System prompt (optional)
            model: Model name override (optional)
            temperature: Sampling temperature
            max_completion_tokens: Maximum completion-token budget to generate
            **kwargs: Additional arguments for OpenAI chat completion

        Returns:
            Generated Python solution code as string
        """
        if not user_prompt:
            raise ValueError("user_prompt must be a non-empty string.")

        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        return self.chat(
            messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            **kwargs
        )

if __name__ == "__main__":
    print("OpenAI API Helper for LCB Meta-Agent")
    print("=" * 60)
    print()

    if not os.environ.get("TASK_MODEL_API_KEY"):
        print("Warning: TASK_MODEL_API_KEY not set in environment")
        print("Set it before running this script:")
        print("  export TASK_MODEL_API_KEY='your-api-key'")
    else:
        helper = OpenAIHelper()
        print(f"Client initialized: model={helper.model}, base_url={helper.base_url}")
