#!/usr/bin/env python3
"""
Timeout test agent.

Purpose:
- Produce a couple of predictions quickly.
- Then block inside a ThreadPoolExecutor context to simulate a "stuck" agent.
- With BaseAIMEAgent hard-timeout + partial-return enabled, evaluation should
  return the already-produced predictions when timeout_sec is reached.

Example (from repo root):
  python aime-meta-agent/eval_utils/agent_runner.py \
    --agent aime-meta-agent/tests/timeout_agent.py \
    --input aime-meta-agent/data/aime_eval.jsonl \
    --output /tmp/preds.jsonl \
    --timeout 2 \
    --first-k 10
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from base_agent import BaseAIMEAgent, Problem, Prediction


class AIMEAgent(BaseAIMEAgent):
    def _slow_solve(self, p: Problem, sleep_sec: float) -> Prediction:
        time.sleep(sleep_sec)
        # Any deterministic value is fine for timeout testing.
        return Prediction(idx=p.idx, pred="0")

    def solve(self, problems: list[Problem], timeout_sec: int = 21600) -> list[Prediction]:
        # Intentionally do NOT call _check_timeout() frequently.
        # This agent is designed to rely on base_agent hard timeout.
        preds: list[Prediction] = []

        # Produce a batch of predictions at a controlled rate so that a 60s timeout
        # returns a non-trivial partial set (but not all problems).
        #
        # With timeout_sec=60:
        # - First 8 problems take ~8 * 5s = 40s.
        # - Then we block inside a ThreadPoolExecutor on very slow tasks.
        fast_n = min(8, len(problems))
        for p in problems[:fast_n]:
            preds.append(self._slow_solve(p, sleep_sec=5.0))

        # Then get "stuck" in a thread pool waiting for slow tasks.
        rest = problems[fast_n:]
        if not rest:
            return preds

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(self._slow_solve, p, 120.0) for p in rest]
            for f in as_completed(futs):
                preds.append(f.result())

        return preds
