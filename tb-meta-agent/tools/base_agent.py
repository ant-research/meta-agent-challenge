#!/usr/bin/env python3
"""
Compatibility shim for SWE Meta-Agent task.

This task is evaluated by a remote Harbor service which expects a Harbor-style agent
implementation (see `harbor.agents.base.BaseAgent`). Historically, this repo shipped
task-specific Base*Agent interfaces (e.g. solve(instances) -> predictions). For SWE,
the remote evaluator expects the Harbor BaseAgent interface.

Agent authors should implement their agent in /workspace/agent.py and inherit:
  - harbor.agents.base.BaseAgent

This module re-exports Harbor's interfaces so agents can do:
  from base_agent import BaseAgent, BaseEnvironment, AgentContext
"""

from __future__ import annotations

try:
    from harbor.agents.base import BaseAgent  # type: ignore
    from harbor.environments.base import BaseEnvironment  # type: ignore
    from harbor.models.agent.context import AgentContext  # type: ignore
    from harbor.models.trial.result import AgentInfo  # type: ignore
except Exception as e:  # pragma: no cover
    # Very small fallback so the workspace can be imported in environments where
    # Harbor is not installed. Remote evaluation is expected to have Harbor.
    from abc import ABC, abstractmethod
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    @dataclass
    class AgentInfo:
        name: str
        version: str

    @dataclass
    class AgentContext:  # minimal stub
        metadata: dict[str, Any] | None = None

    class BaseEnvironment:  # minimal stub
        pass

    class BaseAgent(ABC):
        SUPPORTS_ATIF: bool = False

        def __init__(
            self,
            logs_dir: Path,
            model_name: str | None = None,
            *args,
            **kwargs,
        ):
            self.logs_dir = logs_dir
            self.model_name = model_name

        @staticmethod
        @abstractmethod
        def name() -> str: ...

        @abstractmethod
        def version(self) -> str | None: ...

        async def setup(self, environment: BaseEnvironment) -> None:  # noqa: D401
            return None

        @abstractmethod
        async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> Any: ...

__all__ = ["BaseAgent", "BaseEnvironment", "AgentContext", "AgentInfo"]
