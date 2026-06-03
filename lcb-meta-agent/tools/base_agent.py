#!/usr/bin/env python3
"""LiveCodeBench meta-agent base interface."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Problem:
    """Problem structure exposed to agents (no hidden tests)."""

    idx: str
    question_title: str
    question_content: str
    starter_code: str
    platform: str
    difficulty: str
    contest_date: str
    question_id: str
    contest_id: str
    fn_name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Problem":
        return cls(
            idx=str(data["idx"]),
            question_title=data.get("question_title", ""),
            question_content=data.get("question_content", ""),
            starter_code=data.get("starter_code", ""),
            platform=data.get("platform", ""),
            difficulty=data.get("difficulty", ""),
            contest_date=data.get("contest_date", ""),
            question_id=data.get("question_id", ""),
            contest_id=data.get("contest_id", ""),
            fn_name=data.get("fn_name"),
        )


@dataclass
class Prediction:
    """Prediction format expected by evaluation API."""

    idx: str
    pred: str

    def __post_init__(self) -> None:
        self.idx = str(self.idx)
        if not isinstance(self.pred, str):
            raise TypeError(f"Prediction for idx={self.idx} must use string field `pred`.")
        if not self.pred.strip():
            raise ValueError(f"Prediction for idx={self.idx} must not use an empty `pred`.")

    def to_dict(self) -> Dict[str, Any]:
        return {"idx": self.idx, "pred": self.pred}


class BaseLCBAgent(ABC):
    """Base class for LiveCodeBench code-generation agents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.config = kwargs
        self.start_time: Optional[float] = None
        self.timeout_sec: Optional[int] = None

    @abstractmethod
    def solve(
        self,
        problems: List[Problem],
        timeout_sec: int = 21600,
    ) -> List[Prediction]:
        """Generate code for each problem and return predictions."""

    def _check_timeout(self) -> bool:
        if self.start_time is None or self.timeout_sec is None:
            return False
        return (time.time() - self.start_time) >= self.timeout_sec

    def _remaining_time(self) -> float:
        if self.start_time is None or self.timeout_sec is None:
            return float("inf")
        return max(0.0, self.timeout_sec - (time.time() - self.start_time))

    def _start_timer(self, timeout_sec: int) -> None:
        self.start_time = time.time()
        self.timeout_sec = timeout_sec

    @classmethod
    def load_problems(cls, filepath: str) -> List[Problem]:
        problems: List[Problem] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                problems.append(Problem.from_dict(json.loads(line)))
        return problems

    @classmethod
    def save_predictions(cls, predictions: List[Prediction], filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for pred in predictions:
                if not isinstance(pred.pred, str):
                    raise TypeError(
                        f"Prediction for idx={pred.idx} must use string field `pred`."
                    )
                if not pred.pred.strip():
                    raise ValueError(
                        f"Prediction for idx={pred.idx} must not use an empty `pred`."
                    )
                f.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


def load_problems(filepath: str) -> List[Problem]:
    return BaseLCBAgent.load_problems(filepath)


def save_predictions(predictions: List[Prediction], filepath: str) -> None:
    BaseLCBAgent.save_predictions(predictions, filepath)
