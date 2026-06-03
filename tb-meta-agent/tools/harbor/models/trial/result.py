from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelInfo:
    name: str
    provider: str


@dataclass
class AgentInfo:
    name: str
    version: str
    model_info: ModelInfo | None = None


__all__ = ["AgentInfo", "ModelInfo"]

