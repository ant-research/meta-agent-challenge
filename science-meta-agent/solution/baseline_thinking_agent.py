#!/usr/bin/env python3
"""
Baseline Thinking Agent for Science Questions - Qwen3-8B with Thinking Mode

Uses Qwen3-8B's built-in thinking/reasoning mode to solve graduate-level science questions.
This serves as a baseline to measure the improvement from meta-agent optimization.

Sampling parameters follow Qwen3 thinking mode best practices:
  - temperature=0.6, top_p=0.95, top_k=20
  - enable_thinking=True
  - Cannot use greedy decoding (temperature=0) with thinking mode

Rate limits (SiliconFlow): RPM=1,000, TPM=50,000
  - Uses NUM_WORKERS=1 for sequential execution to avoid rate limits
  - Per-request timeout=3600s to prevent httpx.ReadTimeout on long thinking
  - Exponential backoff with longer delay on 429 rate-limit errors
"""

import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, "/app/tools")

from base_agent import BaseScienceAgent, Problem, Prediction
from openai_helper import OpenAIHelper

SYSTEM_PROMPT = """You are an expert in graduate-level science (physics, chemistry, biology, and related fields).
Think step by step, analyze each option carefully, and select the best answer.
Provide your final answer as a single letter (A, B, C, or D)."""

# Sequential execution to avoid SiliconFlow rate limits (TPM=50,000).
# Qwen3-8B thinking mode generates ~10K-30K tokens per request,
# so even 2 concurrent requests can exceed the TPM limit.
NUM_WORKERS = 1

# EXPERIMENTAL: Use streaming mode for faster response with thinking
USE_STREAMING = True


def extract_answer(text: str, choices: List[str]) -> str:
    """
    Extract the final answer letter from model output.

    Strategy:
    1. Find the last occurrence of "answer is X" or "Answer: X" patterns
    2. Find the last standalone letter that matches available choices
    3. Default: return first choice letter (usually "A")
    """
    if not text:
        return "A"

    # Get valid choice letters from the choices list
    valid_letters = set()
    for choice in choices:
        match = re.match(r'^([A-Z])\)', choice)
        if match:
            valid_letters.add(match.group(1))

    if not valid_letters:
        valid_letters = {'A', 'B', 'C', 'D'}  # Default fallback

    # Strategy 1: Find "answer is X" or "Answer: X" patterns
    answer_patterns = [
        r'answer\s+is\s+([A-Z])',
        r'Answer:\s*([A-Z])',
        r'answer:\s*([A-Z])',
        r'select\s+([A-Z])',
        r'choose\s+([A-Z])',
        r'option\s+([A-Z])',
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Return the last match if it's valid
            last_match = matches[-1].upper()
            if last_match in valid_letters:
                return last_match

    # Strategy 2: Find the last standalone letter that's valid
    letter_matches = re.findall(r'\b([A-Z])\b', text)
    if letter_matches:
        # Check from the end
        for letter in reversed(letter_matches):
            if letter in valid_letters:
                return letter

    # Default: return first valid letter
    return sorted(valid_letters)[0] if valid_letters else "A"


class BaselineThinkingAgent(BaseScienceAgent):
    """
    Baseline agent using Qwen3-8B thinking mode for science questions.

    Sends each science problem to the model with thinking enabled,
    extracts the answer letter, and returns predictions.
    Supports both sequential (NUM_WORKERS=1) and multi-threaded execution.
    """

    # Retry config (includes 429 rate-limit retries)
    MAX_RETRIES = 5
    RETRY_BASE_DELAY = 5  # seconds

    # Time reserved for overhead (startup, saving, etc.)
    OVERHEAD_SEC = 300

    # If remaining time drops below this threshold, stop and submit
    EARLY_STOP_SEC = 1800  # 30 minutes

    def solve(
        self,
        problems: List[Problem],
        timeout_sec: int = 21600
    ) -> List[Prediction]:
        self._start_timer(timeout_sec)

        helper = OpenAIHelper()

        n = len(problems)
        usable_time = max(timeout_sec - self.OVERHEAD_SEC, n * 60)
        per_problem_sec = usable_time / n if n > 0 else 600

        print("=" * 60)
        if NUM_WORKERS == 1:
            print("Baseline Thinking Agent - Qwen3-8B (Science)")
        else:
            print("Baseline Thinking Agent - Qwen3-8B (Science, Multi-threaded)")
        print("=" * 60)
        print(f"Problems: {n}")
        print(f"Timeout: {timeout_sec}s ({timeout_sec/3600:.1f}h)")
        print(f"Per-problem budget: {per_problem_sec:.0f}s ({per_problem_sec/60:.1f}m)")
        if NUM_WORKERS > 1:
            print(f"Workers: {NUM_WORKERS}")
        print(f"Model: {helper.model}")
        print("=" * 60)
        print()

        predictions = []
        completed = 0

        if NUM_WORKERS == 1:
            # Sequential execution
            for problem in problems:
                try:
                    answer = self._solve_single(helper, problem)
                except Exception as e:
                    print(f"  [ERROR] Problem idx={problem.idx}: {e}")
                    answer = "A"

                predictions.append(Prediction(idx=problem.idx, pred=answer))
                completed += 1
                remaining = self._remaining_time()
                print(f"[{completed}/{n}] idx={problem.idx} -> {answer} "
                      f"(remaining: {remaining:.0f}s)")

                if remaining < self.EARLY_STOP_SEC:
                    print(f"  [Early stop] Remaining time {remaining:.0f}s < {self.EARLY_STOP_SEC}s, submitting now.")
                    break
        else:
            # Multi-threaded execution
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_problem = {
                    executor.submit(self._solve_single, helper, problem): problem
                    for problem in problems
                }

                for future in as_completed(future_to_problem):
                    problem = future_to_problem[future]
                    try:
                        answer = future.result()
                    except Exception as e:
                        print(f"  [ERROR] Problem idx={problem.idx}: {e}")
                        answer = "A"

                    predictions.append(Prediction(idx=problem.idx, pred=answer))
                    completed += 1
                    remaining = self._remaining_time()
                    print(f"[{completed}/{n}] idx={problem.idx} -> {answer} "
                          f"(remaining: {remaining:.0f}s)")

        # Fill any missing predictions
        answered_idxs = {p.idx for p in predictions}
        for problem in problems:
            if problem.idx not in answered_idxs:
                predictions.append(Prediction(idx=problem.idx, pred="A"))

        predictions.sort(key=lambda p: p.idx)

        print()
        print("=" * 60)
        print(f"Completed: {len(predictions)}/{n} predictions")
        elapsed = time.time() - self.start_time
        print(f"Time used: {elapsed:.0f}s ({elapsed/60:.1f}m)")
        print("=" * 60)

        return predictions

    def _solve_single(self, helper: OpenAIHelper, problem: Problem) -> str:
        """Solve a single problem with retry logic. Thread-safe."""
        # Format the question with choices
        choices_text = "\n".join(problem.choices)
        full_question = f"{problem.question}\n\nChoices:\n{choices_text}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_question},
        ]

        attempt = 0
        rate_limit_retries = 0
        while attempt < self.MAX_RETRIES:
            try:
                start = time.time()

                if USE_STREAMING:
                    # Streaming mode: faster response, especially with thinking
                    stream = helper.client.chat.completions.create(
                        model=helper.model,
                        messages=messages,
                        temperature=0.6,
                        max_tokens=32768,
                        top_p=0.95,
                        timeout=3600.0,
                        stream=True,
                        extra_body={
                            "enable_thinking": True,
                            "top_k": 20,
                        },
                    )

                    # Collect streamed response and log progress in real-time
                    response_text = ""
                    chunk_count = 0
                    first_chunk_time = None
                    total_tokens = None

                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            if first_chunk_time is None:
                                first_chunk_time = time.time() - start
                                print(f"  [Stream] idx={problem.idx} first_chunk={first_chunk_time:.1f}s", flush=True)
                            response_text += chunk.choices[0].delta.content
                            chunk_count += 1
                            # Log every 50 chunks for real-time monitoring
                            if chunk_count % 50 == 0:
                                elapsed = time.time() - start
                                chunks_per_sec = chunk_count / elapsed if elapsed > 0 else 0
                                print(f"  [Stream] idx={problem.idx} chunks={chunk_count} elapsed={elapsed:.1f}s chars={len(response_text)} chunk/s={chunks_per_sec:.1f}", flush=True)

                        # Try to get token count from usage (usually in last chunk)
                        if hasattr(chunk, 'usage') and chunk.usage:
                            total_tokens = chunk.usage.completion_tokens

                    elapsed = time.time() - start
                    chunks_per_sec = chunk_count / elapsed if elapsed > 0 else 0

                    if total_tokens:
                        tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
                        print(f"  [Stream] idx={problem.idx} DONE latency={elapsed:.1f}s chunks={chunk_count} tokens={total_tokens} token/s={tokens_per_sec:.1f}", flush=True)
                    else:
                        print(f"  [Stream] idx={problem.idx} DONE latency={elapsed:.1f}s chunks={chunk_count} chars={len(response_text)} chunk/s={chunks_per_sec:.1f}", flush=True)

                    return extract_answer(response_text, problem.choices)
                else:
                    # Non-streaming mode: wait for full response
                    response = helper.chat(
                        messages=messages,
                        temperature=0.6,
                        max_tokens=32768,
                        top_p=0.95,
                        timeout=3600.0,
                        extra_body={
                            "enable_thinking": True,
                            "top_k": 20,
                        },
                    )
                    elapsed = time.time() - start
                    print(f"  [NonStream] idx={problem.idx} latency={elapsed:.1f}s chars={len(response)}")
                    return extract_answer(response, problem.choices)

            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower()

                if is_rate_limit:
                    rate_limit_retries += 1
                    delay = max(self.RETRY_BASE_DELAY * (2 ** min(rate_limit_retries, 5)), 30)
                    print(f"  [Rate limit #{rate_limit_retries}] idx={problem.idx} "
                          f"Waiting {delay}s...")
                    time.sleep(delay)
                    # Don't increment attempt — retry forever on 429
                    continue

                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  [Retry {attempt+1}/{self.MAX_RETRIES}] "
                      f"idx={problem.idx} Error: {e}. Waiting {delay}s...")
                attempt += 1

                if attempt < self.MAX_RETRIES:
                    time.sleep(delay)

        print(f"  [FAILED] idx={problem.idx} All retries exhausted, returning 'A'")
        return "A"
