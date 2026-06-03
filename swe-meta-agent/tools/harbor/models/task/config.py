from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "sse"  # "sse" | "streamable-http" | "stdio"
    url: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)


__all__ = ["MCPServerConfig"]

