#!/usr/bin/env python3
"""
Science Evaluation API Server

Provides HTTP API endpoints for agents to submit predictions and get evaluation scores.
"""

import json
import traceback
import subprocess
import os
import hashlib
import base64
import tempfile
import time
import threading
from typing import Dict, Any, List
from pathlib import Path
import sys
import signal
import shutil
import fcntl
from contextlib import contextmanager

# Add eval_utils directory to path
sys.path.insert(0, str(Path(__file__).parent))

import socket

import requests as http_requests
from requests.adapters import HTTPAdapter
from flask import Flask, request, jsonify, Response
from grader import choice_equal
import numpy as np


class _KeepAliveAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        opts = [
            (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3),
            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30),
        ]
        if hasattr(socket, "TCP_KEEPIDLE"):
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30))
        elif hasattr(socket, "TCP_KEEPALIVE"):
            opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30))
        kwargs["socket_options"] = opts
        super().init_poolmanager(*args, **kwargs)


_proxy_session = http_requests.Session()
_proxy_session.mount("http://", _KeepAliveAdapter())
_proxy_session.mount("https://", _KeepAliveAdapter())

app = Flask(__name__)

# ============================================================================
# Evaluation-run Isolation
#
# In this benchmark setup, dev and eval share the same evaluation-api container.
# Dev calls to /evaluate/agent can leave behind hung agent_runner processes (or
# overlap with verifier runs). That can cause timeouts and log/output pollution
# during real evaluation. We therefore:
# 1) serialize /evaluate/agent with a process-wide lock
# 2) best-effort terminate stale agent_runner / workspace agent processes
# 3) write per-run logs keyed by a timestamp run_id (and also update stdout.log
#    / stderr.log as "latest" for compatibility with existing log collectors)
# ============================================================================

_EVAL_LOCK_PATH = "/tmp/evaluate_agent_science.lock"
_EVAL_STATE_PATH = "/tmp/evaluate_agent_science.state.json"


@contextmanager
def _eval_flock(nonblocking: bool):
    """Cross-process lock for /evaluate/agent. Held for the entire evaluation."""
    fh = open(_EVAL_LOCK_PATH, "w")
    try:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        fcntl.flock(fh.fileno(), flags)
    except Exception:
        fh.close()
        raise
    try:
        yield fh
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _atomic_write_json(path: str, obj: dict) -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def _read_eval_state() -> dict | None:
    try:
        return json.loads(Path(_EVAL_STATE_PATH).read_text(encoding="utf-8"))
    except Exception:
        return None


def _clear_eval_state() -> None:
    Path(_EVAL_STATE_PATH).unlink(missing_ok=True)


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except Exception:
        # If we cannot probe, assume alive.
        return True


def _kill_pgid(pgid: int) -> None:
    # Best-effort: terminate then kill.
    for sig, wait_sec in [(signal.SIGTERM, 1.0), (signal.SIGKILL, 0.5)]:
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        except Exception:
            pass
        deadline = time.time() + wait_sec
        while time.time() < deadline:
            if not _pgid_alive(pgid):
                return
            time.sleep(0.05)


def _iter_process_table() -> List[tuple[int, str]]:
    """Return (pid, cmdline) pairs from `ps` (best-effort)."""
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,args"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    lines = out.splitlines()
    pairs: List[tuple[int, str]] = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        pairs.append((pid, parts[1]))
    return pairs


def _kill_pids(pids: List[int], sig: int, wait_sec: float) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except Exception:
            continue

    deadline = time.time() + wait_sec
    while time.time() < deadline:
        alive = 0
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive += 1
            except ProcessLookupError:
                pass
            except Exception:
                # If we cannot probe, assume alive and continue waiting.
                alive += 1
        if alive == 0:
            return
        time.sleep(0.05)


def cleanup_stale_agent_tasks() -> Dict[str, Any]:
    """Best-effort: terminate stale agent evaluation processes.

    We keep matching conservative to avoid killing the Flask server itself.
    Primary target is orphaned agent_runner subprocesses from prior
    /evaluate/agent calls.
    """
    me = os.getpid()
    candidates: List[int] = []

    for pid, cmd in _iter_process_table():
        if pid == me:
            continue

        # The canonical evaluation subprocess.
        if "/app/eval_utils/agent_runner.py" in cmd:
            candidates.append(pid)
            continue

        # Some workflows may run the submitted agent directly in the eval container.
        # Keep this narrow: only /workspace/agent.py.
        if "python" in cmd and "/workspace/" in cmd and cmd.strip().endswith(".py"):
            base = os.path.basename(cmd.split()[-1])
            if base == "agent.py":
                candidates.append(pid)

    unique = sorted(set(candidates))
    if not unique:
        return {"killed": 0, "pids": []}

    # Graceful then forceful.
    _kill_pids(unique, signal.SIGTERM, wait_sec=1.0)
    _kill_pids(unique, signal.SIGKILL, wait_sec=0.5)
    return {"killed": len(unique), "pids": unique}


def _make_run_id() -> str:
    # Timestamp-only (with milliseconds) so it remains easy to correlate with logs.
    sec = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    ms = int((time.time() % 1.0) * 1000)
    return f"{sec}_{ms:03d}"


# ============================================================================
# Model Proxy — transparent proxy that forces TASK_MODEL_NAME
# ============================================================================

class ModelProxy:
    """Transparent proxy that forwards OpenAI-compatible requests to the real
    model API while forcing the model name to TASK_MODEL_NAME."""

    def __init__(self):
        self._lock = threading.Lock()
        # Sliding window for RPM limiting
        self._request_timestamps: List[float] = []
        # Stats
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.errors = 0
        self._rpm_log_timestamps: List[float] = []
        threading.Thread(target=self._periodic_log, daemon=True).start()

    @property
    def rpm_limit(self) -> int:
        return int(os.environ.get('MODEL_PROXY_RPM', '0'))  # 0 = unlimited

    @property
    def tpm_limit(self) -> int:
        return int(os.environ.get('MODEL_PROXY_TPM', '0'))  # 0 = unlimited

    def _check_rpm(self) -> bool:
        """Return True if request is within RPM limit."""
        limit = self.rpm_limit
        if limit <= 0:
            return True
        now = time.time()
        with self._lock:
            self._request_timestamps = [
                t for t in self._request_timestamps if now - t < 60
            ]
            if len(self._request_timestamps) >= limit:
                return False
            self._request_timestamps.append(now)
        return True

    def _record_usage(self, latency: float, usage: dict):
        with self._lock:
            self.total_requests += 1
            self.total_latency += latency
            self.total_input_tokens += usage.get('prompt_tokens', 0)
            self.total_output_tokens += usage.get('completion_tokens', 0)
            self._rpm_log_timestamps.append(time.time())

    def get_stats(self) -> dict:
        with self._lock:
            avg_latency = (
                self.total_latency / self.total_requests
                if self.total_requests > 0 else 0
            )
            return {
                'total_requests': self.total_requests,
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens,
                'total_tokens': self.total_input_tokens + self.total_output_tokens,
                'avg_latency_seconds': round(avg_latency, 3),
                'errors': self.errors,
            }

    def _periodic_log(self):
        while True:
            time.sleep(60)
            now = time.time()
            with self._lock:
                self._rpm_log_timestamps = [t for t in self._rpm_log_timestamps if now - t < 60]
                rpm = len(self._rpm_log_timestamps)
            print(f"[PROXY STATS] RPM: {rpm} | total_requests: {self.total_requests} | errors: {self.errors}", flush=True)


model_proxy = ModelProxy()


# ============================================================================
# Server-side Search Quota
# Tracks per-split search call counts inside the Flask process so agent code
# cannot manipulate them via file writes.  search_helper.py calls
# /search/query which atomically consumes one quota slot and makes the real
# search API call in a single server-side operation.
# ============================================================================

SEARCH_CALL_LIMIT = 2500


class SearchQuotaService:
    def __init__(self):
        self._lock = threading.Lock()
        self._counts: Dict[str, int] = {'eval': 0, 'test': 0}

    def check_and_increment(self, split: str) -> tuple:
        """Atomically check quota and increment if allowed.

        Returns (new_count, allowed).
        """
        with self._lock:
            current = self._counts.get(split, 0)
            if current >= SEARCH_CALL_LIMIT:
                return current, False
            self._counts[split] = current + 1
            return current + 1, True

    def get_stats(self, split: str) -> dict:
        with self._lock:
            count = self._counts.get(split, 0)
            return {
                'split': split,
                'used': count,
                'limit': SEARCH_CALL_LIMIT,
                'remaining': max(0, SEARCH_CALL_LIMIT - count),
            }

    def reset(self, split: str):
        with self._lock:
            self._counts[split] = 0


search_quota = SearchQuotaService()


def _decrypt_file(encrypted_path: str, secret: str) -> bytes:
    """Decrypt a Fernet-encrypted file using the given secret."""
    from cryptography.fernet import Fernet, InvalidToken
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    f = Fernet(key)
    with open(encrypted_path, 'rb') as fp:
        return f.decrypt(fp.read())


class EvaluationService:
    """Evaluation service core class"""

    def __init__(self):
        """Initialize service and load ground truth data"""
        self.ground_truth_cache = {
            'eval': {},
            'test': {}
        }
        self._load_ground_truth()

    def _load_ground_truth(self):
        """
        Load eval ground truth from data files.
        Test ground truth is encrypted and loaded on demand via decrypt_test_data().
        """
        data_dir = Path("/app/data")

        # Only load eval ground truth at startup (unencrypted)
        gt_file = data_dir / "hle_mc_full.jsonl"
        if not gt_file.exists():
            print(f"WARNING: Eval ground truth file not found: {gt_file}")
        else:
            count = 0
            with open(gt_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    idx = item.get('idx')
                    answer = item.get('answer')
                    if idx is not None and answer is not None:
                        self.ground_truth_cache['eval'][idx] = str(answer)
                        count += 1
            print(f"Loaded {count} ground truth answers for split 'eval'")

        # Test ground truth is encrypted — not loaded at startup
        enc_file = data_dir / "gpqa_full.jsonl.enc"
        if enc_file.exists():
            print(f"Test ground truth is encrypted: {enc_file}")
        else:
            print(f"WARNING: Encrypted test ground truth not found: {enc_file}")

        print("Service ready to evaluate predictions")

    def decrypt_test_data(self, secret: str):
        """
        Decrypt and load test ground truth using the verifier secret.
        Also decrypts the test input file for agent execution.
        Returns the path to the decrypted test input file.
        """
        data_dir = Path("/app/data")

        # Decrypt test ground truth
        if not self.ground_truth_cache['test']:
            gt_data = _decrypt_file(str(data_dir / "gpqa_full.jsonl.enc"), secret)
            count = 0
            for line in gt_data.decode('utf-8').splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                idx = item.get('idx')
                answer = item.get('answer')
                if idx is not None and answer is not None:
                    self.ground_truth_cache['test'][idx] = str(answer)
                    count += 1
            print(f"Decrypted and loaded {count} test ground truth answers")

        # Decrypt test input file to a temp file
        input_data = _decrypt_file(str(data_dir / "gpqa_test.jsonl.enc"), secret)
        tmp = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.jsonl', prefix='gpqa_test_',
            dir='/tmp', delete=False
        )
        tmp.write(input_data)
        tmp.close()
        return tmp.name

    def get_ground_truth(self, idx: int, split: str = 'test') -> str:
        """Get ground truth for specified idx"""
        if split not in self.ground_truth_cache:
            raise ValueError(f"Unknown split: {split}. Available: {list(self.ground_truth_cache.keys())}")
        if idx not in self.ground_truth_cache[split]:
            raise ValueError(f"No ground truth found for idx={idx} in split '{split}'")
        return self.ground_truth_cache[split][idx]

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        split: str = 'test',
        first_k: int = None
    ) -> Dict[str, Any]:
        """Evaluate predictions against ground truth."""
        if split not in self.ground_truth_cache:
            raise ValueError(f"Unknown split: {split}")

        all_gt_indices = set(self.ground_truth_cache[split].keys())
        if first_k is not None:
            eval_indices = set(sorted(all_gt_indices)[:first_k])
        else:
            eval_indices = all_gt_indices
        total_count = len(eval_indices)

        if total_count == 0:
            raise ValueError(f"No ground truth loaded for split '{split}'")

        # 验证并建立预测映射
        pred_map: Dict[int, Dict] = {}
        for sample in predictions:
            idx = sample.get('idx')
            if idx is None:
                continue

            pred = sample.get('pred')
            if not isinstance(pred, str):
                return {
                    'split': split, 'total': total_count,
                    'correct': 0, 'accuracy': 0.0,
                    'error': f"'pred' must be a string in sample idx={idx}",
                    'results': []
                }

            if idx not in eval_indices:
                continue  # 忽略不在 GT 中的 idx

            pred_map[idx] = sample

        # 对所有 GT idx 评分，未预测的自动判错
        results = []
        correct_count = 0

        for idx in sorted(eval_indices):
            if idx not in pred_map:
                results.append({'idx': idx, 'is_correct': False})
                continue

            sample = pred_map[idx]
            gt = self.ground_truth_cache[split][idx]
            is_correct = choice_equal(sample['pred'], gt)
            sample['score'] = is_correct
            results.append({'idx': idx, 'is_correct': is_correct})
            if is_correct:
                correct_count += 1

        accuracy = correct_count / total_count

        return {
            'split': split,
            'total': total_count,
            'correct': correct_count,
            'accuracy': accuracy,
            'missing_predictions': total_count - len(pred_map),
            'results': results
        }


# Initialize service
service = EvaluationService()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'science-evaluation-api',
        'splits': list(service.ground_truth_cache.keys()),
        'eval_count': len(service.ground_truth_cache.get('eval', {})),
        'test_count': len(service.ground_truth_cache.get('test', {}))
    })


@app.route('/v1/chat/completions', methods=['POST'])
def proxy_chat_completions():
    """Transparent model proxy — forces TASK_MODEL_NAME and forwards to real API."""
    real_api_base = os.environ.get('TASK_MODEL_API_BASE', '')
    real_api_key = os.environ.get('TASK_MODEL_API_KEY', '')
    forced_model = os.environ.get('TASK_MODEL_NAME', '')

    # Log proxy env for debugging
    proxy_vars = {k: v for k, v in os.environ.items()
                  if 'proxy' in k.lower() or 'PROXY' in k}
    if proxy_vars:
        print(f"[PROXY DEBUG] Proxy env vars detected: {proxy_vars}", flush=True)
    print(f"[PROXY DEBUG] target={real_api_base} model={forced_model}", flush=True)

    if not real_api_base or not real_api_key:
        return jsonify({'error': 'Model proxy not configured'}), 500

    # RPM check
    if not model_proxy._check_rpm():
        return jsonify({
            'error': {
                'message': 'Rate limit exceeded',
                'type': 'rate_limit_error',
            }
        }), 429

    body = request.get_json(force=True)
    # Force model name regardless of what agent requested
    body['model'] = forced_model

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {real_api_key}',
    }

    # Check if streaming is requested
    is_stream = body.get('stream', False)

    target_url = f"{real_api_base.rstrip('/')}/chat/completions"
    start = time.time()

    try:
        if is_stream:
            # Streaming: forward the SSE stream transparently
            # Timeout: (connect=30s, read=3700s) to prevent thread leakage
            # Read timeout slightly longer than agent-side timeout (3600s)
            resp = _proxy_session.post(
                target_url, json=body, headers=headers,
                stream=True, timeout=(30, 3700),
            )
            latency = time.time() - start
            print(f"[PROXY DEBUG] stream connect: status={resp.status_code} latency={latency:.1f}s", flush=True)

            if resp.status_code != 200:
                with model_proxy._lock:
                    model_proxy.errors += 1
                return Response(
                    resp.content,
                    status=resp.status_code,
                    content_type=resp.headers.get('Content-Type', 'application/json'),
                )

            # Record basic stats (tokens not available until stream ends)
            model_proxy._record_usage(latency, {})

            def generate():
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk

            return Response(
                generate(),
                status=resp.status_code,
                content_type=resp.headers.get('Content-Type', 'text/event-stream'),
            )
        else:
            # Non-streaming: forward and record full usage
            # Timeout: (connect=30s, read=3700s) to prevent thread leakage
            print(f"[BODY]={body}")
            resp = _proxy_session.post(
                target_url, json=body, headers=headers, timeout=(30, 3700),
            )
            latency = time.time() - start
            print(f"[RESP]={resp}")
            print(f"[PROXY DEBUG] non-stream response: status={resp.status_code} latency={latency:.1f}s", flush=True)

            if resp.status_code != 200:
                with model_proxy._lock:
                    model_proxy.errors += 1
                return Response(
                    resp.content,
                    status=resp.status_code,
                    content_type=resp.headers.get('Content-Type', 'application/json'),
                )

            result = resp.json()
            usage = result.get('usage', {})
            model_proxy._record_usage(latency, usage)

            return jsonify(result), 200

    except http_requests.Timeout:
        print(f"[PROXY DEBUG] TIMEOUT after {time.time()-start:.1f}s", flush=True)
        with model_proxy._lock:
            model_proxy.errors += 1
        return jsonify({
            'error': {'message': 'Upstream timeout', 'type': 'timeout_error'}
        }), 504
    except Exception as e:
        print(f"[PROXY DEBUG] ERROR after {time.time()-start:.1f}s: {type(e).__name__}: {e}", flush=True)
        with model_proxy._lock:
            model_proxy.errors += 1
        return jsonify({
            'error': {'message': str(e), 'type': 'proxy_error'}
        }), 502


@app.route('/v1/models', methods=['GET'])
def proxy_list_models():
    """Return the single allowed model so OpenAI SDK's model listing works."""
    forced_model = os.environ.get('TASK_MODEL_NAME', 'default')
    return jsonify({
        'object': 'list',
        'data': [{
            'id': forced_model,
            'object': 'model',
            'owned_by': 'proxy',
        }]
    })


@app.route('/v1/stats', methods=['GET'])
def proxy_stats():
    """Return cumulative proxy usage statistics."""
    return jsonify(model_proxy.get_stats())


@app.route('/search/quota/stats', methods=['GET'])
def search_quota_stats():
    """Return search quota usage for all splits."""
    split = request.args.get('split')
    if split:
        return jsonify(search_quota.get_stats(split))
    return jsonify({
        'eval': search_quota.get_stats('eval'),
        'test': search_quota.get_stats('test'),
    })


@app.route('/search/query', methods=['POST'])
def search_proxy_query():
    """Proxy a search request: atomic quota check + real API call in one step.

    Credentials (SEARCH_API_KEY / SEARCH_API_BASE) live only in the Flask
    process environment and are never exposed to agent subprocess code.

    Request body: {"split": "eval"|"test", "query": "...", ...extra params}
    Response 200: raw JSON from search API
    Response 429: quota exceeded
    Response 500: search API not configured
    """
    data = request.get_json(force=True) or {}
    split = data.get('split', 'eval')
    if split not in ('eval', 'test'):
        return jsonify({'error': f'Invalid split: {split}'}), 400

    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'Missing query parameter'}), 400

    # Atomic quota check-and-increment
    new_count, allowed = search_quota.check_and_increment(split)
    if not allowed:
        return jsonify({
            'error': 'quota_exceeded',
            'message': f'Search quota exceeded ({SEARCH_CALL_LIMIT} calls per split)',
            'used': new_count,
            'remaining': 0,
        }), 429

    # Real search call using server-side credentials (never exposed to agent)
    search_api_key = os.environ.get('SEARCH_API_KEY', '')
    search_api_base = os.environ.get('SEARCH_API_BASE', '')
    if not search_api_key or not search_api_base:
        return jsonify({'error': 'Search API not configured on server'}), 500

    # Build params; allow caller to override defaults
    excluded_keys = {'split', 'query'}
    params = {
        'api_key': search_api_key,
        'query': query,
        'country': data.get('country', 'us'),
        'advance_search': data.get('advance_search', 'false'),
        'domain': data.get('domain', 'google.com'),
        'language': data.get('language', 'en'),
    }
    for k, v in data.items():
        if k not in excluded_keys and k not in params:
            params[k] = v

    try:
        resp = http_requests.get(search_api_base, params=params, timeout=30)
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=resp.headers.get('Content-Type', 'application/json'),
        )
    except http_requests.Timeout:
        return jsonify({'error': 'Search API timeout'}), 504
    except Exception as e:
        return jsonify({'error': str(e)}), 502


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """DISABLED: Direct evaluation endpoint"""
    return jsonify({
        'error': 'This endpoint is disabled for security reasons',
        'reason': 'Multiple-choice answers can be brute-forced',
        'alternative': 'Use /evaluate/agent endpoint instead'
    }), 403


@app.route('/evaluate/file', methods=['POST'])
def evaluate_file():
    """DISABLED: File evaluation endpoint"""
    return jsonify({
        'error': 'This endpoint is disabled for security reasons',
        'reason': 'Multiple-choice answers can be brute-forced',
        'alternative': 'Use /evaluate/agent endpoint instead'
    }), 403


@app.route('/score/single', methods=['POST'])
def score_single():
    """DISABLED: Single answer scoring"""
    return jsonify({
        'error': 'This endpoint is disabled for security reasons',
        'reason': 'Multiple-choice answers can be brute-forced',
        'alternative': 'Use /evaluate/agent endpoint instead'
    }), 403


@app.route('/evaluate/agent', methods=['POST'])
def evaluate_agent():
    """
    Agent evaluation endpoint - runs agent code and evaluates results.

    For split="test", requires X-Verifier-Secret header.
    The secret is used to decrypt test data (encrypted at build time).
    """
    try:
        data = request.get_json() or {}

        kill_running = bool(data.get("kill_running", False))
        agent_file = data.get("agent_file")

        # Agent-facing kill switch. If agent_file is provided, kill_running acts as
        # a "kill then run" flag (best-effort) so callers can submit a single request.
        if kill_running and not agent_file:
            state = _read_eval_state() or {}
            killed = False
            pgid = state.get("runner_pgid")
            if isinstance(pgid, int) and _pgid_alive(pgid):
                _kill_pgid(pgid)
                killed = True
            cleanup_info = cleanup_stale_agent_tasks()
            if cleanup_info.get("killed", 0) > 0:
                killed = True
            _clear_eval_state()
            return jsonify({
                "success": True,
                "killed": killed,
                "previous_state": state,
                "cleanup": cleanup_info,
            })

        kill_attempted = False
        kill_wait_deadline = 0.0
        while True:
            try:
                with _eval_flock(nonblocking=True):
                    agent_file = data.get('agent_file')
                    split = data.get('split', 'eval')
                    timeout = data.get('timeout', 60)
                    first_k = data.get('first_k')

                    if not agent_file:
                        return jsonify({'error': 'Missing agent_file parameter'}), 400

                    if split not in ['eval', 'test']:
                        return jsonify({'error': f'Invalid split: {split}. Must be \"eval\" or \"test\"'}), 400

                    # Every new evaluation must clean up residual stale evaluations.
                    state = _read_eval_state() or {}
                    pgid = state.get("runner_pgid")
                    if isinstance(pgid, int) and _pgid_alive(pgid):
                        _kill_pgid(pgid)
                    cleanup_info = cleanup_stale_agent_tasks()
                    if cleanup_info.get("killed", 0) > 0:
                        print(f"cleanup_stale_agent_tasks: killed={cleanup_info['killed']} pids={cleanup_info['pids']}")
                    _clear_eval_state()

                    # Determine input file
                    if split == 'test':
                        secret = request.headers.get('X-Verifier-Secret')
                        if not secret:
                            return jsonify({
                                'error': 'Unauthorized',
                                'message': 'Test split requires X-Verifier-Secret header'
                            }), 403
                        try:
                            input_file = service.decrypt_test_data(secret)
                        except Exception:
                            return jsonify({
                                'error': 'Unauthorized',
                                'message': 'Invalid verifier secret'
                            }), 403
                    else:
                        input_file = '/app/data/hle_mc_eval.jsonl'

                    if not Path(input_file).exists():
                        return jsonify({'error': f'Input file not found: {input_file}'}), 500

                    # Check and install agent dependencies (if pyproject.toml exists)
                    pyproject_path = Path('/workspace/pyproject.toml')
                    if pyproject_path.exists():
                        print(f"Installing agent dependencies from {pyproject_path}...")
                        try:
                            import tomllib
                            with open(pyproject_path, 'rb') as f:
                                pyproject_data = tomllib.load(f)
                            deps = pyproject_data.get('project', {}).get('dependencies', [])
                        except Exception:
                            deps = []
                        if deps:
                            install_result = subprocess.run(
                                ['uv', 'pip', 'install', '--system'] + deps,
                                capture_output=True, text=True, timeout=300
                            )
                            if install_result.returncode != 0:
                                return jsonify({
                                    'success': False,
                                    'error': 'Failed to install agent dependencies from pyproject.toml',
                                    'stderr': install_result.stderr,
                                    'stdout': install_result.stdout
                                }), 500
                            print("Agent dependencies installed successfully")
                        else:
                            print("No dependencies found in pyproject.toml, skipping install")

                    print(f"Running agent: {agent_file} on split={split} (timeout={timeout}s)")

                    run_id = _make_run_id()
                    output_file = f'/tmp/agent_predictions_{run_id}.jsonl'

                    runner_cmd = [
                        'python3', '/app/eval_utils/agent_runner.py',
                        '--agent', agent_file,
                        '--input', input_file,
                        '--output', output_file,
                        '--timeout', str(timeout)
                    ]
                    if first_k is not None:
                        runner_cmd.extend(['--first-k', str(first_k)])

                    agent_env = os.environ.copy()
                    agent_env['PYTHONUNBUFFERED'] = '1'
                    agent_env['SEARCH_SPLIT'] = split
                    agent_env.pop('SEARCH_API_KEY', None)
                    agent_env.pop('SEARCH_API_BASE', None)
                    agent_env.pop('TASK_MODEL_API_KEY', None)
                    agent_env.pop('TASK_MODEL_API_BASE', None)
                    agent_env.pop('MODEL_PROXY_RPM', None)
                    agent_env.pop('MODEL_PROXY_TPM', None)
                    agent_env.pop('VERIFIER_SECRET', None)
                    agent_env['TASK_MODEL_API_BASE'] = 'http://127.0.0.1:8080/v1'
                    agent_env['TASK_MODEL_API_KEY'] = 'proxy-token'

                    log_dir = Path('/evaluation_logs')
                    log_dir.mkdir(parents=True, exist_ok=True)
                    stdout_log = log_dir / f'{run_id}.stdout.log'
                    stderr_log = log_dir / f'{run_id}.stderr.log'
                    stdout_latest = log_dir / 'stdout.log'
                    stderr_latest = log_dir / 'stderr.log'

                    def _update_latest_logs() -> None:
                        try:
                            shutil.copyfile(stdout_log, stdout_latest)
                        except Exception:
                            pass
                        try:
                            shutil.copyfile(stderr_log, stderr_latest)
                        except Exception:
                            pass

                    with open(stdout_log, 'w') as f_out, open(stderr_log, 'w') as f_err:
                        proc = subprocess.Popen(
                            runner_cmd,
                            stdout=f_out,
                            stderr=f_err,
                            env=agent_env,
                            preexec_fn=os.setsid,
                        )
                        runner_pgid = os.getpgid(proc.pid)
                        _atomic_write_json(_EVAL_STATE_PATH, {
                            "run_id": run_id,
                            "pid": os.getpid(),
                            "runner_pid": proc.pid,
                            "runner_pgid": runner_pgid,
                            "agent_file": agent_file,
                            "split": split,
                            "timeout": timeout,
                            "started_at": time.time(),
                        })
                        try:
                            result_returncode = proc.wait(timeout=timeout + 60)
                        except subprocess.TimeoutExpired:
                            _kill_pgid(runner_pgid)
                            _update_latest_logs()
                            agent_stdout = stdout_log.read_text(errors='replace')[-50000:]
                            agent_stderr = stderr_log.read_text(errors='replace')[-10000:]

                            # Salvage any predictions agent_runner already wrote to output_file
                            # before the outer wait fired. Missing indices are counted as 0 by
                            # evaluate_predictions, so the score is a fair lower bound.
                            partial_predictions = []
                            if Path(output_file).exists():
                                with open(output_file, 'r') as _f:
                                    for _line in _f:
                                        _line = _line.strip()
                                        if not _line:
                                            continue
                                        try:
                                            partial_predictions.append(json.loads(_line))
                                        except json.JSONDecodeError:
                                            continue

                            Path(output_file).unlink(missing_ok=True)
                            _clear_eval_state()

                            if partial_predictions:
                                try:
                                    eval_result = service.evaluate_predictions(
                                        partial_predictions, split=split, first_k=first_k
                                    )
                                    eval_result['partial'] = True
                                    eval_result['timed_out'] = True
                                    return jsonify({
                                        'success': True,
                                        'split': split,
                                        'agent_file': agent_file,
                                        'evaluation': eval_result,
                                        'run_id': run_id,
                                        'warning': (
                                            f'Agent execution exceeded {timeout}s limit; '
                                            f'scored {len(partial_predictions)} partial predictions '
                                            f'(missing counted as 0)'
                                        ),
                                        'agent_output': agent_stdout,
                                        'agent_stderr': agent_stderr,
                                        'cleanup': cleanup_info,
                                    }), 200
                                except Exception as _e:
                                    agent_stderr = (
                                        agent_stderr
                                        + f"\n[partial-score-error] {type(_e).__name__}: {_e}"
                                    )[-10000:]

                            return jsonify({
                                'error': 'Agent execution timed out',
                                'message': f'Agent execution exceeded {timeout}s limit',
                                'agent_output': agent_stdout,
                                'agent_stderr': agent_stderr,
                                'run_id': run_id,
                                'cleanup': cleanup_info,
                            }), 408
                        finally:
                            if split == 'test' and input_file.startswith('/tmp/'):
                                Path(input_file).unlink(missing_ok=True)

                    _update_latest_logs()

                    if result_returncode != 0:
                        agent_stdout = stdout_log.read_text(errors='replace')[-50000:]
                        agent_stderr = stderr_log.read_text(errors='replace')[-10000:]
                        Path(output_file).unlink(missing_ok=True)
                        _clear_eval_state()
                        return jsonify({
                            'error': 'Agent execution failed',
                            'message': 'Agent process exited with non-zero status',
                            'agent_output': agent_stdout,
                            'agent_stderr': agent_stderr,
                            'run_id': run_id,
                            'cleanup': cleanup_info,
                        }), 500

                    predictions = []
                    with open(output_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                predictions.append(json.loads(line))

                    eval_result = service.evaluate_predictions(predictions, split=split, first_k=first_k)
                    Path(output_file).unlink(missing_ok=True)
                    _clear_eval_state()

                    return jsonify({
                        'success': True,
                        'split': split,
                        'agent_file': agent_file,
                        'evaluation': eval_result,
                        'run_id': run_id,
                        'cleanup': cleanup_info,
                    })
            except BlockingIOError:
                if kill_running:
                    # Best-effort preemption: attempt to kill the currently running eval,
                    # then wait briefly for the lock to be released and retry.
                    if not kill_attempted:
                        state = _read_eval_state() or {}
                        pgid = state.get("runner_pgid")
                        if isinstance(pgid, int) and _pgid_alive(pgid):
                            _kill_pgid(pgid)
                        cleanup_stale_agent_tasks()
                        _clear_eval_state()
                        kill_attempted = True
                        kill_wait_deadline = time.time() + 30.0
                    if time.time() < kill_wait_deadline:
                        time.sleep(0.2)
                        continue

                state = _read_eval_state() or {}
                return jsonify({
                    "success": False,
                    "error": "another eval is running",
                    "state": state,
                    "hint": "POST /evaluate/agent with {\"kill_running\": true} then retry",
                }), 409

    except Exception as e:
        return jsonify({
            'error': 'Evaluation failed',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/monitor', methods=['POST'])
def monitor():
    """API usage monitoring endpoint."""
    try:
        data = request.get_json()
        agent_file = data.get('agent_file')

        if not agent_file:
            return jsonify({'error': 'Missing agent_file parameter'}), 400

        if not Path(agent_file).exists():
            return jsonify({'error': f'Agent file not found: {agent_file}'}), 404

        from monitor import check_agent_code
        violations = check_agent_code(agent_file)

        return jsonify({
            'agent_file': agent_file,
            'violations': violations,
            'is_valid': len(violations) == 0
        })

    except Exception as e:
        return jsonify({
            'error': 'Monitoring failed',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Science Evaluation API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print("=" * 60)
    print("Science Evaluation API Server")
    print("=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print("\nAvailable endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions  (model proxy)")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  GET  http://{args.host}:{args.port}/v1/stats")
    print(f"  GET  http://{args.host}:{args.port}/search/quota/stats")
    print(f"  POST http://{args.host}:{args.port}/evaluate/agent")
    print(f"  POST http://{args.host}:{args.port}/monitor")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
