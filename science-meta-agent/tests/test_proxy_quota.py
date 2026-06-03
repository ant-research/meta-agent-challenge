#!/usr/bin/env python3
"""
Tests for the LLM proxy endpoint (/v1/chat/completions, /v1/models, /v1/stats),
the search proxy endpoint (/search/query), and search quota stats (/search/quota/stats).

Designed to run against a server backed by mock LLM + search servers,
so no real API calls are made.

Run inside the evaluation-api container:
    python3 /tmp/test_proxy_quota.py [--base-url http://localhost:8080]
"""

import argparse
import sys
import threading
import time

import requests

DEFAULT_BASE = "http://localhost:8080"
DEFAULT_TIMEOUT = 15


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
# LLM Proxy 吞吐量测试
# 验证：响应内容正确、stats token 更新、延迟合理、并发请求全部成功
# ============================================================================

def test_proxy(base: str):
    print("\n=== LLM Proxy Throughput Tests ===")

    # /v1/models — 返回单一强制模型
    resp = requests.get(f"{base}/v1/models", timeout=DEFAULT_TIMEOUT)
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

    # /v1/stats — 读取基线
    resp = requests.get(f"{base}/v1/stats", timeout=DEFAULT_TIMEOUT)
    assert_status("/v1/stats returns 200", resp, 200)
    stats0 = resp.json()
    for key in ("total_requests", "total_input_tokens", "total_output_tokens", "errors"):
        if key not in stats0:
            fail(f"/v1/stats missing key '{key}'")
        ok(f"/v1/stats has key '{key}'")
    baseline_req = stats0["total_requests"]
    baseline_in  = stats0["total_input_tokens"]
    baseline_out = stats0["total_output_tokens"]

    # /v1/chat/completions — 单次请求，验证实际响应内容和延迟
    t0 = time.time()
    resp = requests.post(
        f"{base}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "What is 2+2?"}], "model": "any"},
        headers={"Authorization": "Bearer proxy-token"},
        timeout=DEFAULT_TIMEOUT,
    )
    latency = time.time() - t0

    if resp.status_code == 500 and "Model proxy not configured" in resp.text:
        fail("proxy configured", "TASK_MODEL_API_BASE or KEY not set in server env")
    assert_status("single chat/completions returns 200", resp, 200)

    result = resp.json()
    if "choices" not in result or not result["choices"]:
        fail("response has choices", str(result)[:200])
    reply_content = result["choices"][0]["message"]["content"]
    ok(f"response has choices: {reply_content[:60]!r}")

    # 验证 proxy 强制替换了 model 名
    used_model = result.get("model", "")
    if used_model and used_model != model_id:
        fail(f"proxy forces model name", f"got {used_model!r}, want {model_id!r}")
    ok(f"proxy forces model name to {model_id!r}")

    # 延迟合理（mock server 应 <3s）
    if latency > 3.0:
        fail(f"latency acceptable (<3s)", f"got {latency:.2f}s")
    ok(f"latency acceptable: {latency:.3f}s")

    # stats 应正确更新
    resp = requests.get(f"{base}/v1/stats", timeout=DEFAULT_TIMEOUT)
    assert_status("/v1/stats returns 200 (after 1 req)", resp, 200)
    stats1 = resp.json()
    if stats1["total_requests"] != baseline_req + 1:
        fail("total_requests incremented by 1",
             f"got {stats1['total_requests']}, want {baseline_req + 1}")
    ok("total_requests incremented by 1")
    if stats1["total_input_tokens"] <= baseline_in:
        fail("total_input_tokens increased",
             f"was {baseline_in}, now {stats1['total_input_tokens']}")
    ok(f"total_input_tokens increased: +{stats1['total_input_tokens'] - baseline_in}")
    if stats1["total_output_tokens"] <= baseline_out:
        fail("total_output_tokens increased",
             f"was {baseline_out}, now {stats1['total_output_tokens']}")
    ok(f"total_output_tokens increased: +{stats1['total_output_tokens'] - baseline_out}")

    # 并发测试 — N 个请求全部成功，stats 正确累计
    n = 5
    print(f"\n  Concurrency test ({n} parallel requests)...")
    results = []
    lock = threading.Lock()

    def do_request():
        r = requests.post(
            f"{base}/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello world"}], "model": "any"},
            headers={"Authorization": "Bearer proxy-token"},
            timeout=DEFAULT_TIMEOUT,
        )
        with lock:
            results.append(r.status_code)

    threads = [threading.Thread(target=do_request) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = sum(1 for s in results if s == 200)
    if successes != n:
        fail(f"concurrent {n} requests all succeed", f"{successes}/{n} returned 200")
    ok(f"concurrent {n} requests all returned 200")

    resp = requests.get(f"{base}/v1/stats", timeout=DEFAULT_TIMEOUT)
    assert_status("/v1/stats returns 200 (after concurrency)", resp, 200)
    stats2 = resp.json()
    expected_total = baseline_req + 1 + n
    if stats2["total_requests"] != expected_total:
        fail("total_requests after concurrent test",
             f"got {stats2['total_requests']}, want {expected_total}")
    ok(f"total_requests correctly updated to {stats2['total_requests']}")

    avg_lat = stats2.get("avg_latency_seconds", 0)
    ok(f"avg_latency_seconds: {avg_lat:.3f}s")

    print("\n  LLM Proxy throughput tests done.")


# ============================================================================
# Search Quota 测试
# 验证：quota stats endpoint 返回结构正确、limit 正确、used + remaining == limit
# ============================================================================

def test_quota(base: str):
    print("\n=== Search Quota Tests ===")

    # /search/quota/stats — 两个 split 都存在
    resp = requests.get(f"{base}/search/quota/stats", timeout=DEFAULT_TIMEOUT)
    assert_status("/search/quota/stats returns 200", resp, 200)
    data = resp.json()
    for split in ("eval", "test"):
        if split not in data:
            fail(f"/search/quota/stats has key '{split}'")
        ok(f"/search/quota/stats has key '{split}'")
        stats = data[split]
        for key in ("split", "used", "limit", "remaining"):
            if key not in stats:
                fail(f"/search/quota/stats[{split}] has key '{key}'")
            ok(f"/search/quota/stats[{split}] has key '{key}'")

    # /search/quota/stats?split=eval
    resp = requests.get(
        f"{base}/search/quota/stats",
        params={"split": "eval"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/search/quota/stats?split=eval returns 200", resp, 200)
    s = resp.json()
    assert_eq("/search/quota/stats?split=eval .split", s["split"], "eval")
    assert_eq("/search/quota/stats?split=eval .limit", s["limit"], 2500)

    baseline_used      = s["used"]
    baseline_remaining = s["remaining"]
    assert_eq("baseline used + remaining == limit", baseline_used + baseline_remaining, 2500)

    print("\n  Search Quota tests done.")


# ============================================================================
# Test Split 密钥验证测试
# 验证：/evaluate/agent split=test 需要 X-Verifier-Secret，
#       无密钥或错误密钥返回 403；禁用端点始终返回 403
# ============================================================================

def test_verifier_secret(base: str):
    print("\n=== Verifier Secret Tests ===")

    # --- 禁用端点始终返回 403 ---
    for endpoint in ("/evaluate", "/evaluate/file", "/score/single"):
        resp = requests.post(
            f"{base}{endpoint}",
            json={},
            timeout=DEFAULT_TIMEOUT,
        )
        assert_status(f"POST {endpoint} returns 403 (disabled)", resp, 403)
        body = resp.json()
        if "disabled" not in body.get("error", "").lower() and "disabled" not in body.get("reason", "").lower():
            fail(f"{endpoint} error mentions 'disabled'", str(body)[:200])
        ok(f"{endpoint} error mentions 'disabled'")

    # --- /evaluate/agent split=test 无密钥 → 403 ---
    resp = requests.post(
        f"{base}/evaluate/agent",
        json={"agent_file": "/workspace/agent.py", "split": "test"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/evaluate/agent split=test without secret returns 403", resp, 403)
    body = resp.json()
    assert_eq(
        "/evaluate/agent no-secret error is 'Unauthorized'",
        body.get("error"), "Unauthorized",
    )

    # --- /evaluate/agent split=test 错误密钥 → 403 ---
    resp = requests.post(
        f"{base}/evaluate/agent",
        json={"agent_file": "/workspace/agent.py", "split": "test"},
        headers={"X-Verifier-Secret": "wrong-secret-12345"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/evaluate/agent split=test with wrong secret returns 403", resp, 403)
    body = resp.json()
    assert_eq(
        "/evaluate/agent wrong-secret error is 'Unauthorized'",
        body.get("error"), "Unauthorized",
    )

    # --- /evaluate/agent split=eval 不需要密钥（应返回非 403） ---
    # 注意：agent_file 不存在时可能返回 500，但不应返回 403
    resp = requests.post(
        f"{base}/evaluate/agent",
        json={"agent_file": "/nonexistent/agent.py", "split": "eval"},
        timeout=DEFAULT_TIMEOUT,
    )
    if resp.status_code == 403:
        fail("/evaluate/agent split=eval should not require secret", "got 403")
    ok(f"/evaluate/agent split=eval does not require secret (status={resp.status_code})")

    # --- /evaluate/agent 无效 split → 400 ---
    resp = requests.post(
        f"{base}/evaluate/agent",
        json={"agent_file": "/workspace/agent.py", "split": "invalid"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/evaluate/agent invalid split returns 400", resp, 400)

    # --- /evaluate/agent 缺少 agent_file → 400 ---
    resp = requests.post(
        f"{base}/evaluate/agent",
        json={"split": "eval"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/evaluate/agent missing agent_file returns 400", resp, 400)

    print("\n  Verifier secret tests done.")


# ============================================================================
# Search Query Proxy 测试
# 验证：/search/query 原子消耗 quota + 调用 mock search server 并返回结果
# ============================================================================

def test_search_query(base: str):
    print("\n=== Search Query Proxy Tests ===")

    # 读取基线 quota
    resp = requests.get(
        f"{base}/search/quota/stats",
        params={"split": "eval"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/search/quota/stats?split=eval returns 200 (baseline)", resp, 200)
    s0 = resp.json()
    baseline_used = s0["used"]

    # 正常搜索请求
    resp = requests.post(
        f"{base}/search/query",
        json={"split": "eval", "query": "what causes lightning?"},
        timeout=DEFAULT_TIMEOUT,
    )
    if resp.status_code == 500 and "not configured" in resp.text:
        fail("/search/query: SEARCH_API_KEY or SEARCH_API_BASE not set", resp.text[:200])
    assert_status("/search/query returns 200", resp, 200)
    data = resp.json()
    ok(f"/search/query returned data: {str(data)[:80]}")

    # quota 应增加 1
    resp = requests.get(
        f"{base}/search/quota/stats",
        params={"split": "eval"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/search/quota/stats?split=eval returns 200 (after 1 query)", resp, 200)
    s1 = resp.json()
    assert_eq("/search/query increments quota.used by 1", s1["used"], baseline_used + 1)

    # 空 query 返回 400
    resp = requests.post(
        f"{base}/search/query",
        json={"split": "eval", "query": ""},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("empty query returns 400", resp, 400)

    # 无效 split 返回 400
    resp = requests.post(
        f"{base}/search/query",
        json={"split": "invalid", "query": "test"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("invalid split returns 400", resp, 400)

    # 空 query 和 invalid split 不消耗 quota
    resp = requests.get(
        f"{base}/search/quota/stats",
        params={"split": "eval"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/search/quota/stats?split=eval returns 200 (after invalid reqs)", resp, 200)
    s2 = resp.json()
    assert_eq("invalid requests do not consume quota", s2["used"], baseline_used + 1)

    # 并发搜索 — 每个请求独立消耗一个 slot
    n = 5
    print(f"\n  Concurrent search test ({n} parallel requests)...")
    results = []
    lock = threading.Lock()

    def do_search():
        r = requests.post(
            f"{base}/search/query",
            json={"split": "eval", "query": "concurrent test query"},
            timeout=DEFAULT_TIMEOUT,
        )
        with lock:
            results.append(r.status_code)

    threads = [threading.Thread(target=do_search) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = sum(1 for s in results if s == 200)
    if successes != n:
        fail(f"concurrent {n} search requests all succeed", f"{successes}/{n} succeeded")
    ok(f"concurrent {n} search requests all returned 200")

    resp = requests.get(
        f"{base}/search/quota/stats",
        params={"split": "eval"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/search/quota/stats?split=eval returns 200 (after concurrency)", resp, 200)
    s3 = resp.json()
    assert_eq(f"quota.used increased by {n} after concurrent search",
              s3["used"], baseline_used + 1 + n)

    print("\n  Search query proxy tests done.")


# ============================================================================
# Quota Exhaustion 测试
# ============================================================================

def test_quota_exhaustion(base: str):
    """验证 quota 耗尽后返回 429。"""
    print("\n=== Quota Exhaustion Test (uses 'test' split) ===")

    resp = requests.get(
        f"{base}/search/quota/stats",
        params={"split": "test"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("/search/quota/stats?split=test returns 200", resp, 200)
    s = resp.json()
    remaining = s["remaining"]
    print(f"  Remaining quota: {remaining}")

    # 并发耗尽剩余 quota（batch_size 个一批）
    batch_size = 50
    sent = 0
    while sent < remaining:
        n = min(batch_size, remaining - sent)
        threads = []
        results_drain = []
        lock = threading.Lock()

        def do_drain():
            r = requests.post(
                f"{base}/search/query",
                json={"split": "test", "query": "drain quota"},
                timeout=DEFAULT_TIMEOUT,
            )
            with lock:
                results_drain.append(r.status_code)

        for _ in range(n):
            t = threading.Thread(target=do_drain)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        for sc in results_drain:
            if sc == 500:
                fail("/search/query: SEARCH_API_KEY or SEARCH_API_BASE not set")
            if sc not in (200, 429):
                fail("unexpected status while draining quota", f"{sc}")

        sent += n
        if sent % 500 == 0 or sent == remaining:
            print(f"  Drained {sent}/{remaining}...")

    # 下一次调用必须被拒绝
    resp = requests.post(
        f"{base}/search/query",
        json={"split": "test", "query": "one more after exhaustion"},
        timeout=DEFAULT_TIMEOUT,
    )
    assert_status("exhausted quota returns 429", resp, 429)
    d = resp.json()
    assert_eq("exhausted quota: error=quota_exceeded", d.get("error"), "quota_exceeded")
    assert_eq("exhausted quota: remaining=0", d.get("remaining"), 0)
    assert_eq("exhausted quota: used==limit", d.get("used"), 2500)

    print("\n  Quota exhaustion test done.")


# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--skip-proxy",  action="store_true", help="跳过 LLM proxy 测试")
    parser.add_argument("--skip-quota",  action="store_true", help="跳过 quota 测试")
    parser.add_argument("--skip-search", action="store_true", help="跳过 search query 测试")
    parser.add_argument("--skip-secret", action="store_true", help="跳过 verifier secret 测试")
    parser.add_argument("--exhaustion",  action="store_true", help="运行 quota exhaustion 测试")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

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

    if not args.skip_quota:
        test_quota(base)

    if not args.skip_search:
        test_search_query(base)

    if not args.skip_secret:
        test_verifier_secret(base)

    if args.exhaustion:
        test_quota_exhaustion(base)

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
