#!/usr/bin/env python3
"""
Base-agent timeout test.

This validates that:
1) A hard timeout (SIGALRM/setitimer) stops an agent even if it blocks waiting
   on futures.
2) Partial predictions created before timeout are returned (no padding).

Run (from repo root):
  python3 aime-meta-agent/tests/test_timeout_agent.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def ok(label: str) -> None:
    print(f"  [PASS] {label}")


def fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}" + (f": {detail}" if detail else ""))
    raise SystemExit(1)


def main() -> None:
    tests_dir = Path(__file__).resolve().parent
    task_dir = tests_dir.parent  # aime-meta-agent/
    tools_dir = task_dir / "tools"

    # Ensure `from base_agent import ...` resolves to the task's base_agent.
    sys.path.insert(0, str(tools_dir))
    sys.path.insert(0, str(tests_dir))

    from base_agent import Problem  # noqa: E402
    import timeout_agent  # noqa: E402

    agent = timeout_agent.AIMEAgent()
    problems = [Problem(idx=i, question="q") for i in range(10)]

    t0 = time.time()
    res = agent.solve(problems, timeout_sec=60)
    elapsed = time.time() - t0

    if elapsed > 65.0:
        fail("returns quickly after timeout", f"elapsed={elapsed:.3f}s (expected <=65.0s)")
    ok("returns quickly after timeout")

    idxs = [p.idx for p in res]
    want = list(range(8))
    if idxs != want:
        fail("returns partial predictions", f"idxs={idxs!r} (expected {want!r})")
    ok(f"returns partial predictions ({want!r})")


if __name__ == "__main__":
    main()
