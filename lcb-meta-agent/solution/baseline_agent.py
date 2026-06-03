#!/usr/bin/env python3
"""Baseline LiveCodeBench agent aligned with the latest AIME baseline design.

This baseline keeps the official ScaleBox/Qwen3-style LiveCodeBench prompt
content, but adopts the same execution philosophy as the latest AIME baseline:

- single worker by default to avoid upstream rate limits
- thinking mode enabled on every request
- streaming enabled for better observability on long generations
- 5 retries with exponential backoff
- reserve fixed overhead and early-stop when time is nearly exhausted

The LCB-specific parts that remain are:
- prompt construction from LiveCodeBench problem fields
- extracting Python code instead of a boxed numeric answer
- fallback code generation for unanswered problems
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, "/app/tools")

from base_agent import BaseLCBAgent, Prediction, Problem
from openai_helper import OpenAIHelper

QWEN3_SYSTEM_MESSAGE = (
    "You are a helpful and harmless assistant. You are Qwen developed by "
    "Alibaba. You should think step-by-step."
)

FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within "
    "delimiters as follows. Ensure that when the python program runs, it reads "
    "the inputs, runs the algorithm and writes output to STDOUT."
)

DEFAULT_STDIN_TEMPLATE = """def main():
    import sys

    data = sys.stdin.buffer.read().split()
    if not data:
        return


if __name__ == '__main__':
    main()
"""

# Keep sequential execution by default, mirroring AIME's latest baseline.
NUM_WORKERS = 4

# Stream responses so long generations expose progress in the agent logs.
USE_STREAMING = True


def build_qwen3_message_contents(problem: Problem) -> tuple[str, str]:
    """Build message contents equivalent to the ScaleBox Qwen3 LCB template."""
    user_prompt = (
        "You will be given a question (problem specification) and will "
        "generate a correct Python program that matches the specification "
        "and passes all tests.\n\nQuestion: "
    )

    user_prompt += problem.question_content + "\n\n"
    if problem.starter_code:
        user_prompt += f"{FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        user_prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        user_prompt += f"{FORMATTING_WITHOUT_STARTER_CODE}\n"
        user_prompt += "```python\n# YOUR CODE HERE\n```\n\n"

    return QWEN3_SYSTEM_MESSAGE, user_prompt.rstrip()


def build_messages(problem: Problem) -> list[dict[str, str]]:
    system_message, user_prompt = build_qwen3_message_contents(problem)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def _strip_reasoning_prefix(text: str) -> str:
    return re.split(r"</think>\s*", text)[-1]


def _looks_like_python_code(candidate: str) -> bool:
    lines = [line.strip() for line in candidate.splitlines() if line.strip()]
    if not lines:
        return False

    first = lines[0]
    code_prefixes = (
        "def ",
        "class ",
        "import ",
        "from ",
        "if ",
        "elif ",
        "else:",
        "for ",
        "while ",
        "try:",
        "except",
        "finally:",
        "with ",
        "return ",
        "raise ",
        "assert ",
        "print(",
        "async ",
        "@",
        "#",
    )
    if first.startswith(code_prefixes):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", first):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\(", first):
        return True
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\[[^\]]+\]\s*=", first):
        return True
    return False


def _extract_parseable_python_suffix(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    if stripped.startswith("python\n"):
        stripped = stripped.split("\n", 1)[1].strip()

    candidates = [stripped]
    lines = stripped.splitlines()
    for start in range(1, len(lines)):
        candidate = "\n".join(lines[start:]).strip()
        if candidate:
            candidates.append(candidate)

    for candidate in candidates:
        if not _looks_like_python_code(candidate):
            continue
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue

    return ""


def extract_code_from_response(model_output: str) -> str:
    """Extract code in a way compatible with ScaleBox's LCB evaluation flow.

    ScaleBox's logic (from _eval_one):
    1. Default to taking the last code block
    2. Then search from end to start for the first code block containing 'def'
    """
    if not model_output:
        return ""

    try:
        completion = _strip_reasoning_prefix(model_output)
        output_lines = completion.split("\n")
        fence_lines = [i for i, line in enumerate(output_lines) if "```" in line]

        if len(fence_lines) >= 2:
            # Default: take the last code block
            code = "\n".join(output_lines[fence_lines[-2] + 1 : fence_lines[-1]]).strip()
            if code:
                # Search from end to start for the first code block containing 'def'
                i = len(fence_lines) - 1
                while i >= 1 and i < len(fence_lines):
                    start = fence_lines[i - 1]
                    end = fence_lines[i]
                    if start < end:
                        candidate = "\n".join(output_lines[start + 1 : end]).strip()
                        if "def" in candidate:
                            return candidate
                    i -= 2
                return code
    except Exception:
        pass

    return _extract_parseable_python_suffix(completion)


def build_fallback_code(problem: Problem) -> str:
    if problem.starter_code.strip():
        return problem.starter_code.strip()
    if problem.fn_name:
        return (
            "class Solution:\n"
            f"    def {problem.fn_name}(self, *args, **kwargs):\n"
            "        return None\n"
        )
    return DEFAULT_STDIN_TEMPLATE.strip()


class BaselineAgent(BaseLCBAgent):
    """LCB baseline agent using the latest AIME-style timing and retry policy."""

    TEMPERATURE = 0.6
    TOP_P = 0.95
    TOP_K = 20
    MIN_P = 0.0
    MAX_COMPLETION_TOKENS = 32768
    REQUEST_TIMEOUT = 3600.0

    MAX_RETRIES = 5
    RETRY_BASE_DELAY = 5
    OVERHEAD_SEC = 300
    EARLY_STOP_SEC = 1800
    EXTRACTION_FAILURE_CODE = "# CODE EXTRACTION FAILED"

    def solve(
        self,
        problems: list[Problem],
        timeout_sec: int = 21600,
    ) -> list[Prediction]:
        self._start_timer(timeout_sec)
        helper = OpenAIHelper(
            api_key=self.api_key,
            base_url=self.api_base,
            model=self.model,
        )

        total = len(problems)
        usable_time = max(timeout_sec - self.OVERHEAD_SEC, total * 60)
        per_problem_sec = usable_time / total if total > 0 else 600

        print("=" * 60)
        if NUM_WORKERS == 1:
            print("LCB Baseline Agent - Qwen3 Thinking")
        else:
            print("LCB Baseline Agent - Qwen3 Thinking (Multi-threaded)")
        print("=" * 60)
        print(f"Problems: {total}")
        print(f"Timeout: {timeout_sec}s ({timeout_sec / 3600:.1f}h)")
        print(f"Per-problem budget: {per_problem_sec:.0f}s ({per_problem_sec / 60:.1f}m)")
        if NUM_WORKERS > 1:
            print(f"Workers: {NUM_WORKERS}")
        print(f"Model: {helper.model}")
        print("Prompt profile: fixed ScaleBox/Qwen3-style content")
        print(f"Streaming: {USE_STREAMING}")
        print(f"Temperature: {self.TEMPERATURE}")
        print(f"Top-p: {self.TOP_P}")
        print(f"Top-k: {self.TOP_K}")
        print(f"Min-p: {self.MIN_P}")
        print(f"Max completion tokens: {self.MAX_COMPLETION_TOKENS}")
        print(f"Request timeout: {self.REQUEST_TIMEOUT:.0f}s")
        print("=" * 60)
        print()

        predictions: list[Prediction] = []
        completed = 0

        if NUM_WORKERS == 1:
            for problem in problems:
                try:
                    code = self._solve_single(helper, problem)
                    source = (
                        "placeholder"
                        if code == self.EXTRACTION_FAILURE_CODE
                        else "model"
                    )
                except Exception as exc:
                    print(f"  [ERROR] Problem idx={problem.idx}: {exc}")
                    print(traceback.format_exc())
                    code = build_fallback_code(problem)
                    source = "fallback"

                predictions.append(Prediction(idx=problem.idx, pred=code))
                completed += 1
                remaining = self._remaining_time()
                print(
                    f"[{completed}/{total}] idx={problem.idx} difficulty={problem.difficulty} "
                    f"source={source} remaining={remaining:.0f}s"
                )

                if remaining < self.EARLY_STOP_SEC:
                    print(
                        f"  [Early stop] Remaining time {remaining:.0f}s < "
                        f"{self.EARLY_STOP_SEC}s, submitting now."
                    )
                    break
        else:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_problem = {
                    executor.submit(self._solve_single, helper, problem): problem
                    for problem in problems
                }

                for future in as_completed(future_to_problem):
                    problem = future_to_problem[future]
                    try:
                        code = future.result()
                        source = (
                            "placeholder"
                            if code == self.EXTRACTION_FAILURE_CODE
                            else "model"
                        )
                    except Exception as exc:
                        print(f"  [ERROR] Problem idx={problem.idx}: {exc}")
                        print(traceback.format_exc())
                        code = build_fallback_code(problem)
                        source = "fallback"

                    predictions.append(Prediction(idx=problem.idx, pred=code))
                    completed += 1
                    remaining = self._remaining_time()
                    print(
                        f"[{completed}/{total}] idx={problem.idx} difficulty={problem.difficulty} "
                        f"source={source} remaining={remaining:.0f}s"
                    )

        prediction_map = {prediction.idx: prediction for prediction in predictions}
        for problem in problems:
            if problem.idx not in prediction_map:
                prediction_map[problem.idx] = Prediction(
                    idx=problem.idx,
                    pred=build_fallback_code(problem),
                )

        ordered_predictions = [prediction_map[problem.idx] for problem in problems]

        print()
        print("=" * 60)
        print(f"Completed: {len(ordered_predictions)}/{total} predictions")
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        print(f"Time used: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
        print("=" * 60)

        return ordered_predictions

    def _solve_single(self, helper: OpenAIHelper, problem: Problem) -> str:
        """Solve a single LiveCodeBench problem with AIME-style retry logic."""
        messages = build_messages(problem)

        attempt = 0
        rate_limit_retries = 0
        while attempt < self.MAX_RETRIES:
            try:
                start = time.time()
                extra_body = {
                    "enable_thinking": True,
                    "top_k": self.TOP_K,
                    "min_p": self.MIN_P,
                }

                if USE_STREAMING:
                    stream = helper.client.chat.completions.create(
                        model=helper.model,
                        messages=messages,
                        temperature=self.TEMPERATURE,
                        max_completion_tokens=self.MAX_COMPLETION_TOKENS,
                        top_p=self.TOP_P,
                        timeout=self.REQUEST_TIMEOUT,
                        stream=True,
                        extra_body=extra_body,
                    )

                    response_text = ""
                    chunk_count = 0
                    first_chunk_time = None
                    total_tokens = None

                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - start
                                print(
                                    f"  [Stream] idx={problem.idx} "
                                    f"first_chunk={first_chunk_time:.1f}s",
                                    flush=True,
                                )
                            response_text += chunk.choices[0].delta.content
                            chunk_count += 1
                            if chunk_count % 50 == 0:
                                elapsed = time.time() - start
                                chunks_per_sec = chunk_count / elapsed if elapsed > 0 else 0.0
                                print(
                                    f"  [Stream] idx={problem.idx} chunks={chunk_count} "
                                    f"elapsed={elapsed:.1f}s chars={len(response_text)} "
                                    f"chunk/s={chunks_per_sec:.1f}",
                                    flush=True,
                                )

                        if hasattr(chunk, "usage") and chunk.usage:
                            total_tokens = chunk.usage.completion_tokens

                    elapsed = time.time() - start
                    if total_tokens:
                        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
                        print(
                            f"  [Stream] idx={problem.idx} DONE latency={elapsed:.1f}s "
                            f"chunks={chunk_count} tokens={total_tokens} "
                            f"token/s={tokens_per_sec:.1f}",
                            flush=True,
                        )
                    else:
                        chunks_per_sec = chunk_count / elapsed if elapsed > 0 else 0.0
                        print(
                            f"  [Stream] idx={problem.idx} DONE latency={elapsed:.1f}s "
                            f"chunks={chunk_count} chars={len(response_text)} "
                            f"chunk/s={chunks_per_sec:.1f}",
                            flush=True,
                        )

                    code = extract_code_from_response(response_text)
                    if code.strip():
                        return code.strip()
                    print(
                        f"  [No usable code] idx={problem.idx} "
                        "Returning placeholder without retry."
                    )
                    return self.EXTRACTION_FAILURE_CODE

                response = helper.chat(
                    messages=messages,
                    temperature=self.TEMPERATURE,
                    max_completion_tokens=self.MAX_COMPLETION_TOKENS,
                    top_p=self.TOP_P,
                    timeout=self.REQUEST_TIMEOUT,
                    extra_body=extra_body,
                )
                elapsed = time.time() - start
                print(
                    f"  [NonStream] idx={problem.idx} latency={elapsed:.1f}s "
                    f"chars={len(response)}"
                )
                code = extract_code_from_response(response)
                if code.strip():
                    return code.strip()
                print(
                    f"  [No usable code] idx={problem.idx} "
                    "Returning placeholder without retry."
                )
                return self.EXTRACTION_FAILURE_CODE

            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()

                if is_rate_limit:
                    rate_limit_retries += 1
                    delay = max(self.RETRY_BASE_DELAY * (2 ** min(rate_limit_retries, 5)), 30)
                    print(
                        f"  [Rate limit #{rate_limit_retries}] idx={problem.idx} "
                        f"Waiting {delay}s..."
                    )
                    time.sleep(delay)
                    continue

                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                print(
                    f"  [Retry {attempt + 1}/{self.MAX_RETRIES}] idx={problem.idx} "
                    f"Error: {exc}. Waiting {delay}s..."
                )
                attempt += 1
                if attempt < self.MAX_RETRIES:
                    time.sleep(delay)

        print(f"  [FAILED] idx={problem.idx} All retries exhausted, returning placeholder")
        return self.EXTRACTION_FAILURE_CODE



def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline LCB agent")
    parser.add_argument("--input", required=True, help="Input JSONL problems")
    parser.add_argument("--output", required=True, help="Output JSONL predictions")
    parser.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args()

    problems = BaseLCBAgent.load_problems(args.input)
    agent = BaselineAgent()
    predictions = agent.solve(problems, timeout_sec=args.timeout)
    BaseLCBAgent.save_predictions(predictions, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
