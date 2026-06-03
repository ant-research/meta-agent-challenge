#!/usr/bin/env python3
"""
Science Agent Runner

Unified entry point for running Science agents.
Loads agent implementation, runs it with timeout, and saves predictions.
"""

import sys
import os
import json
import time
import signal
import logging
import importlib.util
from pathlib import Path
from typing import Type, Optional
import traceback

# Add tools directory to path (base_agent.py is in the sibling tools/ dir)
sys.path.insert(0, str(Path(__file__).parent.parent / 'tools'))

from base_agent import BaseScienceAgent, Problem, Prediction


def _setup_logger(name: str = "agent_runner") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class TimeoutException(Exception):
    """Raised when agent execution times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Agent execution timed out")


def load_agent_class(agent_path: str) -> Type[BaseScienceAgent]:
    """
    Dynamically load agent class from Python file.

    Args:
        agent_path: Path to Python file containing agent class

    Returns:
        Agent class (subclass of BaseScienceAgent)

    Raises:
        ImportError: If agent file cannot be loaded
        ValueError: If no valid agent class found
    """
    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent file not found: {agent_path}")

    # Load module from file
    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent from {agent_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)

    # Find agent class (subclass of BaseScienceAgent, not BaseScienceAgent itself)
    agent_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and
            issubclass(obj, BaseScienceAgent) and
            obj is not BaseScienceAgent):
            agent_class = obj
            break

    if agent_class is None:
        raise ValueError(
            f"No valid agent class found in {agent_path}. "
            f"Agent must inherit from BaseScienceAgent."
        )

    return agent_class


def run_agent(
    agent_path: str,
    input_file: str,
    output_file: str,
    timeout_sec: int = 21600,
    first_k: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
    **agent_kwargs
) -> dict:
    """
    Run agent on Science problems.

    Args:
        agent_path: Path to Python file with agent implementation
        input_file: Path to input JSONL file (problems without ground truth)
        output_file: Path to output JSONL file (predictions)
        timeout_sec: Maximum execution time in seconds (default: 6 hours)
        api_key: OpenAI API key (optional, can use env var)
        api_base: API base URL (optional, can use env var)
        model: Model name (optional, can use env var)
        **agent_kwargs: Additional arguments passed to agent constructor

    Returns:
        Dictionary with execution metrics

    Raises:
        TimeoutException: If execution exceeds timeout
        Exception: If agent fails
    """
    logger = _setup_logger()

    logger.info("=" * 70)
    logger.info("Science Agent Runner")
    logger.info("=" * 70)
    logger.info(f"Agent: {agent_path}")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Timeout: {timeout_sec} seconds ({timeout_sec/3600:.1f} hours)")
    logger.info("=" * 70)

    start_time = time.time()

    # Set timeout signal (Unix only)
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_sec)

    try:
        # Load agent class
        logger.info("Loading agent class...")
        AgentClass = load_agent_class(agent_path)
        logger.info(f"Loaded agent: {AgentClass.__name__}")

        # Initialize agent
        logger.info("Initializing agent...")
        agent = AgentClass(
            api_key=api_key,
            api_base=api_base,
            model=model,
            **agent_kwargs
        )
        logger.info(f"Initialized: {agent}")

        # Load problems
        logger.info(f"Loading problems from {input_file}...")
        problems = BaseScienceAgent.load_problems(input_file)
        if first_k is not None:
            problems = problems[:first_k]
            logger.info(f"Loaded {len(problems)} problems (first_k={first_k})")
        else:
            logger.info(f"Loaded {len(problems)} problems")

        # Run agent
        logger.info("Starting agent execution...")
        predictions = agent.solve(problems, timeout_sec=timeout_sec)
        logger.info(f"Agent completed: {len(predictions)} predictions generated")

        # Save predictions
        logger.info(f"Saving predictions to {output_file}...")
        BaseScienceAgent.save_predictions(predictions, output_file)
        logger.info("Predictions saved")

        # Calculate metrics
        elapsed_time = time.time() - start_time
        metrics = {
            "success": True,
            "total_problems": len(problems),
            "predictions_generated": len(predictions),
            "elapsed_time_sec": elapsed_time,
            "timeout_sec": timeout_sec,
            "agent_class": AgentClass.__name__,
            "output_file": output_file
        }

        logger.info("=" * 70)
        logger.info("Execution Summary")
        logger.info("=" * 70)
        logger.info("Success: True")
        logger.info(f"Problems: {len(problems)}")
        logger.info(f"Predictions: {len(predictions)}")
        logger.info(f"Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 70)

        return metrics

    except TimeoutException:
        logger.info("=" * 70)
        logger.info("TIMEOUT: Agent execution exceeded time limit")
        logger.info("=" * 70)
        raise

    except Exception as e:
        logger.info("=" * 70)
        logger.info("ERROR: Agent execution failed")
        logger.info("=" * 70)
        logger.info(f"Error: {e}")
        logger.info(traceback.format_exc())
        raise

    finally:
        # Cancel timeout alarm
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Science agent with standardized interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run agent with default settings
  python agent_runner.py --agent my_agent.py

  # Run with custom timeout (3 hours)
  python agent_runner.py --agent my_agent.py --timeout 10800

  # Run with custom input/output
  python agent_runner.py --agent my_agent.py \\
      --input /data/problems.jsonl \\
      --output /workspace/predictions.jsonl

  # Run with specific API configuration (use environment variables)
  python agent_runner.py --agent my_agent.py \\
      --api-key $TASK_MODEL_API_KEY \\
      --api-base $TASK_MODEL_API_BASE \\
      --model $TASK_MODEL_NAME
        """
    )

    parser.add_argument(
        "--agent",
        required=True,
        help="Path to Python file with agent implementation (must inherit from BaseScienceAgent)"
    )

    parser.add_argument(
        "--input",
        default="/app/data/gpqa_eval.jsonl",
        help="Input JSONL file with problems (default: /app/data/gpqa_eval.jsonl)"
    )

    parser.add_argument(
        "--output",
        default="/workspace/predictions.jsonl",
        help="Output JSONL file for predictions (default: /workspace/predictions.jsonl)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=21600,
        help="Timeout in seconds (default: 21600 = 6 hours)"
    )

    parser.add_argument(
        "--first-k",
        type=int,
        default=None,
        help="Only evaluate first k problems (default: all)"
    )

    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (default: use TASK_MODEL_API_KEY env var)"
    )

    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL (default: use TASK_MODEL_API_BASE env var)"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: use TASK_MODEL_NAME env var or 'default')"
    )

    parser.add_argument(
        "--agent-kwargs",
        default="{}",
        help="Additional agent kwargs as JSON string (e.g., '{\"temperature\": 0.7}')"
    )

    args = parser.parse_args()

    # Parse agent kwargs
    agent_kwargs = {}
    if args.agent_kwargs:
        try:
            agent_kwargs = json.loads(args.agent_kwargs)
        except json.JSONDecodeError as e:
            logging.getLogger("agent_runner").error(f"Error parsing --agent-kwargs: {e}")
            sys.exit(1)

    try:
        metrics = run_agent(
            agent_path=args.agent,
            input_file=args.input,
            output_file=args.output,
            timeout_sec=args.timeout,
            first_k=args.first_k,
            api_key=args.api_key,
            api_base=args.api_base,
            model=args.model,
            **agent_kwargs

        )
        sys.exit(0)

    except Exception as e:
        logging.getLogger("agent_runner").error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
