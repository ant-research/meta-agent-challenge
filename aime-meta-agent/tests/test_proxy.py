#!/usr/bin/env python3
"""
Tests for the AIME evaluation-api:
  - OpenAI-compatible model proxy (/v1/chat/completions, /v1/models, /v1/stats)
  - /evaluate/agent split access control (test split requires X-Verifier-Secret)

Run against a live evaluation_api server:
    python3 tests/test_proxy.py --base-url http://evaluation-api:8080

Optionally verify test split works when you have the verifier secret:
    python3 tests/test_proxy.py --verifier-secret "$VERIFIER_SECRET"
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import requests

DEFAULT_BASE = "http://localhost:8080"


def ok(label: str):
    print(f"  [PASS] {label}")


def fail(label: str, detail: str = ""):
    print(f"  [FAIL] {label}" + (f": {detail}" if detail else ""))
    sys.exit(1)


def assert_eq(label, got, want):
    if got != want:
        fail(label, f"got {got!r}, want {want!r}")
    ok(label)


def assert_status(label, resp, want_status):
    if resp.status_code != want_status:
        fail(label, f"HTTP {resp.status_code} (want {want_status}): {resp.text[:300]}")
    ok(label)


# ============================================================================
# Proxy tests
# ============================================================================

def test_proxy(base: str):
    print("\n=== LLM Proxy Tests ===")

    # /v1/models — returns a list with the forced model
    resp = requests.get(f"{base}/v1/models")
    assert_status("/v1/models returns 200", resp, 200)
    data = resp.json()
    if "data" not in data:
        fail("/v1/models has 'data' key")
    ok("/v1/models has 'data' key")
    if len(data["data"]) != 1:
        fail("/v1/models has exactly one model", str(data["data"]))
    ok("/v1/models has exactly one model")
    model_id = data["data"][0]["id"]
    print(f"  Forced model: {model_id}")

    # /v1/stats — returns proxy stats dict
    resp = requests.get(f"{base}/v1/stats")
    assert_status("/v1/stats returns 200", resp, 200)
    stats = resp.json()
    for key in ("total_requests", "total_input_tokens", "total_output_tokens", "errors"):
        if key not in stats:
            fail(f"/v1/stats has key '{key}'")
        ok(f"/v1/stats has key '{key}'")
    baseline_requests = stats["total_requests"]

    # /v1/chat/completions — proxy must route (not return 500 config error)
    resp = requests.post(
        f"{base}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Say hi."}], "model": "any"},
        headers={"Authorization": "Bearer proxy-token"},
    )
    if resp.status_code == 500 and "Model proxy not configured" in resp.text:
        fail("/v1/chat/completions proxy configured",
             "TASK_MODEL_API_BASE or TASK_MODEL_API_KEY not set in server env")
    ok("/v1/chat/completions routes through proxy (not 500 config error)")
    print(f"  Upstream status: {resp.status_code}")

    # /v1/stats — total_requests should have incremented
    resp2 = requests.get(f"{base}/v1/stats")
    stats2 = resp2.json()
    if stats2["total_requests"] < baseline_requests:
        fail("/v1/stats total_requests non-decreasing")
    ok("/v1/stats total_requests non-decreasing after proxy call")

    # Model name is forced regardless of what we sent
    if resp.status_code == 200:
        result = resp.json()
        used_model = result.get("model", "")
        if used_model and used_model != model_id:
            fail(f"proxy forces model name: got {used_model!r}, want {model_id!r}")
        ok(f"proxy forces model name to {model_id!r}")

    # Concurrent proxy calls — all should get routed
    print("\n  Concurrency test (5 parallel requests)...")
    results = []
    lock = threading.Lock()

    def do_request():
        r = requests.post(
            f"{base}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "1+1=?"}], "model": "any"},
            headers={"Authorization": "Bearer proxy-token"},
            timeout=60,
        )
        with lock:
            results.append(r.status_code)

    threads = [threading.Thread(target=do_request) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All responses should be something (not connection errors)
    if len(results) != 5:
        fail(f"concurrent 5 requests: got {len(results)} responses, want 5")
    ok(f"concurrent 5 requests: all received responses {results}")

    print("\n  LLM Proxy tests done.")


def _write_temp_agent(workspace_path: Path) -> Path:
    """
    Write a minimal agent into /workspace so evaluation-api (mounted /workspace:ro)
    can read it when running /evaluate/agent.
    """
    workspace_path.mkdir(parents=True, exist_ok=True)
    agent_path = workspace_path / f"_proxy_test_agent_{os.getpid()}.py"
    agent_path.write_text(
        "\n".join(
            [
                "from base_agent import BaseAIMEAgent, Prediction",
                "",
                "class ProxyTestAgent(BaseAIMEAgent):",
                "    def solve(self, problems, timeout_sec=21600):",
                "        # Fast deterministic output; avoids any model calls.",
                "        return [Prediction(idx=p.idx, pred='0') for p in problems]",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return agent_path


def test_split_access_control(base: str, verifier_secret: str | None):
    print("\n=== /evaluate/agent Split Access Control ===")

    # Create a tiny agent in /workspace (shared with evaluation-api).
    try:
        agent_path = _write_temp_agent(Path("/workspace"))
    except PermissionError:
        ok("split tests skipped (/workspace not writable in this environment)")
        return

    try:
        # eval split should be allowed without verifier secret.
        r = requests.post(
            f"{base}/evaluate/agent",
            json={
                "agent_file": str(agent_path),
                "split": "eval",
                "timeout": 60,
                "first_k": 1,
            },
            timeout=120,
        )
        assert_status("eval split allowed", r, 200)
        data = r.json()
        if not data.get("success", False):
            fail("eval split success", str(data)[:300])
        ok("eval split success")

        # test split must be denied without secret.
        r = requests.post(
            f"{base}/evaluate/agent",
            json={
                "agent_file": str(agent_path),
                "split": "test",
                "timeout": 60,
                "first_k": 1,
            },
            timeout=30,
        )
        assert_status("test split denied without secret", r, 403)

        # test split must be denied with a wrong secret.
        r = requests.post(
            f"{base}/evaluate/agent",
            json={
                "agent_file": str(agent_path),
                "split": "test",
                "timeout": 60,
                "first_k": 1,
            },
            headers={"X-Verifier-Secret": "wrong-secret"},
            timeout=30,
        )
        assert_status("test split denied with wrong secret", r, 403)

        # If secret is provided, verify test split can run.
        if verifier_secret:
            r = requests.post(
                f"{base}/evaluate/agent",
                json={
                    "agent_file": str(agent_path),
                    "split": "test",
                    "timeout": 60,
                    "first_k": 1,
                },
                headers={"X-Verifier-Secret": verifier_secret},
                timeout=180,
            )
            assert_status("test split allowed with secret", r, 200)
            data = r.json()
            if not data.get("success", False):
                fail("test split success", str(data)[:300])
            ok("test split success")
        else:
            ok("test split success test skipped (no verifier secret provided)")
    finally:
        try:
            agent_path.unlink(missing_ok=True)
        except Exception:
            pass


# ============================================================================
# RPM rate-limit tests
# ============================================================================

def test_rpm_limit(base: str, rpm_limit: int):
    """Send rpm_limit+1 concurrent requests and verify that >=1 gets a 429.

    Requires the server to be started with MODEL_PROXY_RPM=<rpm_limit>.
    Pass --rpm-limit N on the command line (or set MODEL_PROXY_RPM in env).
    """
    print(f"\n=== RPM Rate-Limit Tests (limit={rpm_limit}) ===")

    # Test 1: fire rpm_limit+1 requests concurrently, expect >=1 rejected.
    n = rpm_limit + 1
    statuses: list = []
    lock = threading.Lock()

    def fire():
        r = requests.post(
            f"{base}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "model": "any"},
            headers={"Authorization": "Bearer proxy-token"},
            timeout=30,
        )
        with lock:
            statuses.append(r.status_code)

    threads = [threading.Thread(target=fire) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    count_429 = statuses.count(429)
    count_ok = sum(1 for s in statuses if s != 429)
    print(f"  Sent {n} requests: {count_ok} passed, {count_429} rate-limited (429)")
    print(f"  All statuses: {sorted(statuses)}")

    if count_429 < 1:
        fail("RPM limit enforced: expected >=1 rejected (429)", f"statuses={statuses}")
    ok(f"RPM limit enforced: {count_429} request(s) correctly rejected with 429")

    if count_ok > rpm_limit:
        fail(
            f"RPM window respected: at most {rpm_limit} should pass",
            f"but {count_ok} passed",
        )
    ok(f"RPM window respected: {count_ok} <= {rpm_limit} requests passed")

    # Test 2: one more request within the same window should still be rejected.
    r = requests.post(
        f"{base}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "another"}], "model": "any"},
        headers={"Authorization": "Bearer proxy-token"},
        timeout=30,
    )
    print(f"  Extra request within window: HTTP {r.status_code}")
    if r.status_code != 429:
        fail("Extra request within window rejected", f"got {r.status_code}, want 429")
    ok("Extra request within window correctly rejected (429)")

    # Test 3: wait for the 60-second sliding window to expire, then one request passes.
    print("  Waiting 61s for RPM window to reset…")
    time.sleep(61)

    r = requests.post(
        f"{base}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "after reset"}], "model": "any"},
        headers={"Authorization": "Bearer proxy-token"},
        timeout=30,
    )
    print(f"  Post-reset request: HTTP {r.status_code}")
    if r.status_code == 429:
        fail("RPM window resets after 60s", "still got 429 after waiting")
    ok("RPM window resets after 60s (request accepted)")

    print("\n  RPM tests done.")


# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--verifier-secret", default=os.environ.get("VERIFIER_SECRET", ""))
    parser.add_argument("--skip-proxy", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    verifier_secret = args.verifier_secret.strip() or None

    # Health check
    try:
        resp = requests.get(f"{base}/health", timeout=5)
        resp.raise_for_status()
        print(f"Server healthy: {base}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {base}: {e}")
        sys.exit(1)

    if not args.skip_proxy:
        test_proxy(base)
    if not args.skip_split:
        test_split_access_control(base, verifier_secret=verifier_secret)

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
