from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentContext:
    # Keep the commonly used field; the real Harbor object has more.
    metadata: dict[str, Any] | None = None

    def is_empty(self) -> bool:
        return self.metadata is None


__all__ = ["AgentContext"]

