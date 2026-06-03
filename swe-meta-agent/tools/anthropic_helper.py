#!/usr/bin/env python3
"""
Anthropic API Helper

Configure via environment variables:
  - TASK_MODEL_API_KEY
  - TASK_MODEL_API_BASE
  - TASK_MODEL_NAME (default: "claude-haiku-4-5-20251016")
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Dict, List, Optional

import httpx

from anthropic import Anthropic
from anthropic.types import Message, ContentBlock


class AnthropicHelper:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("TASK_MODEL_API_KEY")
        self.base_url = base_url or os.environ.get("TASK_MODEL_API_BASE")
        self.model = model or os.environ.get("TASK_MODEL_NAME", "claude-haiku-4-5-20251016")

        if not self.api_key:
            raise ValueError("Missing API key. Set TASK_MODEL_API_KEY or pass api_key=...")

        kwargs: Dict = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

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
        kwargs["http_client"] = httpx.Client(transport=transport)

        self.client = Anthropic(**kwargs)

        self.call_count = 0
        self.total_latency = 0.0
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = f"/workspace/anthropic_helper_{ts}.log"

    def _log(self, entry: Dict) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass
