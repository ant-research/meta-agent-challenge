#!/usr/bin/env python3
"""Unified runner for LiveCodeBench meta-agents."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Type

TOOLS_DIR = Path(__file__).parent.parent / "tools"
sys.path.insert(0, str(TOOLS_DIR))
# Keep the parent path available so `from tools.xxx import ...` can resolve.
sys.path.insert(0, str(TOOLS_DIR.parent))

# Canonicalize helper module identities.
# Without this, importing the same file as `base_agent` vs `tools.base_agent`
# can create distinct class objects and break `issubclass` checks.
import base_agent as base_agent_module
import openai_helper as openai_helper_module

sys.modules["base_agent"] = base_agent_module
sys.modules["tools.base_agent"] = base_agent_module
sys.modules["openai_helper"] = openai_helper_module
sys.modules["tools.openai_helper"] = openai_helper_module

from base_agent import BaseLCBAgent


class TimeoutException(Exception):
    """Raised when agent execution exceeds the configured timeout."""


def timeout_handler(signum, frame):
    raise TimeoutException("Agent execution timed out")


def load_agent_class(agent_path: str) -> Type[BaseLCBAgent]:
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent file not found: {agent_path}")

    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent from {agent_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseLCBAgent) and obj is not BaseLCBAgent:
            return obj

    raise ValueError(
        f"No valid agent class found in {agent_path}. "
        "Agent must inherit from BaseLCBAgent."
    )


def run_agent(
    agent_path: str,
    input_file: str,
    output_file: str,
    timeout_sec: int = 21600,
    first_k: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    **agent_kwargs,
) -> dict:
    print("=" * 70)
    print("LiveCodeBench Agent Runner")
    print("=" * 70)
    print(f"Agent: {agent_path}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Timeout: {timeout_sec} seconds ({timeout_sec / 3600:.1f} hours)")
    print("=" * 70)
    print()

    start_time = time.time()
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)

    try:
        agent_class = load_agent_class(agent_path)
        print(f"✓ Loaded agent: {agent_class.__name__}")

        agent = agent_class(
            api_key=api_key,
            api_base=api_base,
            model=model,
            **agent_kwargs,
        )
        print(f"✓ Initialized: {agent}")

        problems = BaseLCBAgent.load_problems(input_file)
        if first_k is not None:
            problems = problems[:first_k]
            print(f"✓ Loaded {len(problems)} problems (first_k={first_k})")
        else:
            print(f"✓ Loaded {len(problems)} problems")

        predictions = agent.solve(problems, timeout_sec=timeout_sec)
        print(f"✓ Agent completed: {len(predictions)} predictions generated")

        BaseLCBAgent.save_predictions(predictions, output_file)
        print("✓ Predictions saved")

        elapsed_time = time.time() - start_time
        metrics = {
            "success": True,
            "total_problems": len(problems),
            "predictions_generated": len(predictions),
            "elapsed_time_sec": elapsed_time,
            "timeout_sec": timeout_sec,
            "agent_class": agent_class.__name__,
            "output_file": output_file,
        }

        print("=" * 70)
        print("Execution Summary")
        print("=" * 70)
        print(f"Success: ✓")
        print(f"Problems: {len(problems)}")
        print(f"Predictions: {len(predictions)}")
        print(f"Time: {elapsed_time:.1f}s ({elapsed_time / 60:.1f} min)")
        print(f"Output: {output_file}")
        print("=" * 70)

        return metrics
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run LiveCodeBench agent on JSONL problems")
    parser.add_argument("--agent", type=str, required=True, help="Path to agent Python file")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL predictions")
    parser.add_argument("--timeout", type=int, default=21600, help="Execution timeout in seconds")
    parser.add_argument(
        "--first-k",
        type=int,
        default=None,
        help="Only run the first k problems in input-file order",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--api-base", type=str, default=None, help="API base URL")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("TASK_MODEL_API_KEY")
    api_base = args.api_base or os.environ.get("TASK_MODEL_API_BASE")
    model = args.model or os.environ.get("TASK_MODEL_NAME")

    try:
        metrics = run_agent(
            agent_path=args.agent,
            input_file=args.input,
            output_file=args.output,
            timeout_sec=args.timeout,
            first_k=args.first_k,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )
        print("\nMETRICS_JSON:")
        print(json.dumps(metrics))
        return 0
    except Exception as exc:
        error_metrics = {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print("\nMETRICS_JSON:")
        print(json.dumps(error_metrics))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
