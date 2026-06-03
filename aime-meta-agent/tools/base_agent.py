#!/usr/bin/env python3
"""
AIME Agent Base Class

Defines the standard interface for AIME meta-agents.
All agents should inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import signal
import threading
import time
from dataclasses import dataclass
from functools import wraps


class AgentTimeoutError(BaseException):
    """Hard timeout used to force `solve()` to stop and return partial predictions."""


_COLLECT_LOCK = threading.Lock()
_COLLECT_ACTIVE = False
_COLLECTED: List["Prediction"] = []
_TPE_PATCHED = False


def _patch_threadpoolexecutor_exit_once() -> None:
    """
    Many agents use `with ThreadPoolExecutor(...) as ex: ...`.
    If we raise inside that block, Python will call `__exit__`, which shuts down
    the pool and can block waiting for workers. On timeout we want best-effort
    fast exit.
    """
    global _TPE_PATCHED
    if _TPE_PATCHED:
        return
    try:
        from concurrent.futures.thread import ThreadPoolExecutor as _TPE  # type: ignore
        orig_exit = _TPE.__exit__

        def _exit(self, exc_type, exc, tb):
            if exc_type is AgentTimeoutError:
                try:
                    self.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    self.shutdown(wait=False)
                return False
            return orig_exit(self, exc_type, exc, tb)

        _TPE.__exit__ = _exit
        _TPE_PATCHED = True
    except Exception:
        return


def _get_collected_predictions() -> List["Prediction"]:
    # Deduplicate by idx (keep latest).
    with _COLLECT_LOCK:
        by_idx: Dict[int, Prediction] = {}
        for p in _COLLECTED:
            try:
                by_idx[int(p.idx)] = p
            except Exception:
                continue
        return [by_idx[i] for i in sorted(by_idx.keys())]


@dataclass
class Problem:
    """AIME problem data structure"""
    idx: int
    question: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Problem":
        return cls(
            idx=data["idx"],
            question=data["question"]
        )


@dataclass
class Prediction:
    """Prediction result structure"""
    idx: int
    pred: str  # Final predicted answer

    def __post_init__(self) -> None:
        # Best-effort: as soon as a prediction is created, make it visible to the
        # base layer so we can return partial results on timeout.
        if not _COLLECT_ACTIVE:
            return
        with _COLLECT_LOCK:
            _COLLECTED.append(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.idx,
            "pred": self.pred
        }


class BaseAIMEAgent(ABC):
    """
    Base class for AIME meta-agents.

    All agents must implement the solve() method which takes a list of problems
    (without ground truth) and returns predictions within the time limit.

    Usage:
        agent = MyAgent(api_key="...", api_base="...", model="...")
        problems = load_problems("aime_eval.jsonl")
        predictions = agent.solve(problems, timeout_sec=21600)
        save_predictions(predictions, "predictions.jsonl")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the agent.

        Args:
            api_key: OpenAI API key (or compatible)
            api_base: API base URL
            model: Model name to use
            **kwargs: Additional agent-specific configuration
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.config = kwargs
        self.start_time = None
        self.timeout_sec = None
        self._base_timeout_active = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only wrap subclasses that define solve() themselves.
        if "solve" not in cls.__dict__:
            return
        orig = cls.__dict__["solve"]
        if getattr(orig, "_base_agent_timeout_wrapped", False):
            return

        @wraps(orig)
        def solve_wrapped(self, problems: List[Problem], timeout_sec: int = 21600) -> List[Prediction]:
            self._base_timeout_active = True
            self._start_timer(timeout_sec)
            try:
                return orig(self, problems, timeout_sec=timeout_sec)
            except AgentTimeoutError:
                return _get_collected_predictions()
            finally:
                self._stop_timer()
                self._base_timeout_active = False

        solve_wrapped._base_agent_timeout_wrapped = True  # type: ignore[attr-defined]
        cls.solve = solve_wrapped  # type: ignore[assignment]

    @abstractmethod
    def solve(
        self,
        problems: List[Problem],
        timeout_sec: int = 21600  # 6 hours default
    ) -> List[Prediction]:
        """
        Solve AIME problems and return predictions.

        This is the main entry point that must be implemented by all agents.

        Args:
            problems: List of Problem objects (without ground truth answers)
            timeout_sec: Maximum time allowed in seconds (default: 6 hours)

        Returns:
            List of Prediction objects with predicted answers

        Note:
            - The method should handle timeout internally and return partial results if needed
            - Predictions should be ordered by idx
            - Each prediction must have a single answer string in pred
            - If you cannot solve all problems within the time limit, return whatever predictions
              you have. Missing predictions are treated as incorrect.
              The evaluation API response includes a `missing_predictions` field telling you
              how many problems were not covered by your returned predictions.
        """
        pass

    def _check_timeout(self) -> bool:
        """Check if timeout has been reached"""
        if self.start_time is None or self.timeout_sec is None:
            return False
        elapsed = time.time() - self.start_time
        return elapsed >= self.timeout_sec

    def _remaining_time(self) -> float:
        """Get remaining time in seconds"""
        if self.start_time is None or self.timeout_sec is None:
            return float('inf')
        elapsed = time.time() - self.start_time
        return max(0, self.timeout_sec - elapsed)

    def _start_timer(self, timeout_sec: int):
        """Start the timeout timer"""
        global _COLLECT_ACTIVE, _COLLECTED

        # If base wrapper already started the timer, don't let agent code restart it.
        if getattr(self, "_base_timeout_active", False) and self.start_time is not None and self.timeout_sec is not None:
            return

        self.start_time = time.time()
        self.timeout_sec = timeout_sec

        with _COLLECT_LOCK:
            _COLLECTED = []
        _COLLECT_ACTIVE = True

        _patch_threadpoolexecutor_exit_once()

        # Hard timeout (main thread only).
        if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
            def _handler(signum, frame):
                raise AgentTimeoutError()

            try:
                signal.signal(signal.SIGALRM, _handler)
                if hasattr(signal, "setitimer") and hasattr(signal, "ITIMER_REAL"):
                    signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
                else:
                    signal.alarm(int(timeout_sec))
            except Exception:
                pass

    def _stop_timer(self) -> None:
        """Stop the hard timeout + prediction collection (best-effort)."""
        global _COLLECT_ACTIVE
        _COLLECT_ACTIVE = False
        try:
            if hasattr(signal, "setitimer") and hasattr(signal, "ITIMER_REAL"):
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            elif hasattr(signal, "alarm"):
                signal.alarm(0)
        except Exception:
            pass

        self.start_time = None
        self.timeout_sec = None

    @classmethod
    def load_problems(cls, filepath: str) -> List[Problem]:
        """
        Load problems from JSONL file (without ground truth).

        Args:
            filepath: Path to JSONL file with problems

        Returns:
            List of Problem objects
        """
        problems = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    problems.append(Problem.from_dict(data))
        return problems

    @classmethod
    def save_predictions(cls, predictions: List[Prediction], filepath: str):
        """
        Save predictions to JSONL file.

        Args:
            predictions: List of Prediction objects
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred.to_dict()) + '\n')

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model})"


def load_problems(filepath: str) -> List[Problem]:
    """Helper function to load problems"""
    return BaseAIMEAgent.load_problems(filepath)


def save_predictions(predictions: List[Prediction], filepath: str):
    """Helper function to save predictions"""
    BaseAIMEAgent.save_predictions(predictions, filepath)
