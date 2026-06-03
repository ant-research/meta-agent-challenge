from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecResult:
    stdout: str | None
    stderr: str | None
    return_code: int


class BaseEnvironment(ABC):
    """
    Minimal stub for Harbor BaseEnvironment.

    External evaluation runs in Harbor proper; this class exists for typing and
    to help agent authors structure their code.
    """

    @abstractmethod
    async def exec(
        self,
        command: str,
        *,
        user: str | int | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
        cwd: str | None = None,
        **kwargs: Any,
    ) -> ExecResult:
        raise NotImplementedError


__all__ = ["BaseEnvironment", "ExecResult"]

