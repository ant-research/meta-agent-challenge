#!/usr/bin/env python3
"""
Math Evaluation API Server

提供HTTP API接口，Agent可以提交预测数据获取评分结果。
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

# 添加eval_utils目录到路径（grader.py, parser.py等在同目录下）
sys.path.insert(0, str(Path(__file__).parent))

import socket

import requests as http_requests
from requests.adapters import HTTPAdapter
from flask import Flask, request, jsonify, Response
from grader import math_equal, math_equal_process
from parser import parse_ground_truth
import numpy as np
from pebble import ProcessPool
from concurrent.futures import TimeoutError


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

_EVAL_LOCK_PATH = "/tmp/evaluate_agent_aime.lock"
_EVAL_STATE_PATH = "/tmp/evaluate_agent_aime.state.json"


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
        return True


def _kill_pgid(pgid: int) -> None:
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
                alive += 1
        if alive == 0:
            return
        time.sleep(0.05)


def cleanup_stale_agent_tasks() -> Dict[str, Any]:
    """Best-effort: terminate stale agent evaluation processes.

    Keep matching conservative to avoid killing the Flask server itself.
    Primary target is orphaned agent_runner subprocesses from prior
    /evaluate/agent calls.
    """
    me = os.getpid()
    candidates: List[int] = []

    for pid, cmd in _iter_process_table():
        if pid == me:
            continue
        if "/app/eval_utils/agent_runner.py" in cmd:
            candidates.append(pid)
            continue
        # Narrow: only /workspace/agent.py if someone ran it directly.
        if "python" in cmd and "/workspace/" in cmd and cmd.strip().endswith(".py"):
            base = os.path.basename(cmd.split()[-1])
            if base == "agent.py":
                candidates.append(pid)

    unique = sorted(set(candidates))
    if not unique:
        return {"killed": 0, "pids": []}

    _kill_pids(unique, signal.SIGTERM, wait_sec=1.0)
    _kill_pids(unique, signal.SIGKILL, wait_sec=0.5)
    return {"killed": len(unique), "pids": unique}


def _make_run_id() -> str:
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


# ============================================================================


def _decrypt_file(encrypted_path: str, secret: str) -> bytes:
    """Decrypt a Fernet-encrypted file using the given secret."""
    from cryptography.fernet import Fernet
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    f = Fernet(key)
    with open(encrypted_path, 'rb') as fp:
        return f.decrypt(fp.read())


class EvaluationService:
    """评估服务核心类"""

    def __init__(self):
        """初始化服务并加载ground truth数据"""
        # 支持多个 split 的 ground truth
        self.ground_truth_cache = {
            'eval': {},  # aime_2022_2023.jsonl (idx 0-59)
            'test': {}   # aime_2024_2025.jsonl (idx 0-59)
        }
        self._load_ground_truth()

    def _load_ground_truth(self):
        """
        Load eval ground truth from data files.
        Test ground truth is encrypted and loaded on demand via decrypt_test_data().
        """
        tools_dir = Path(__file__).parent
        task_dir = tools_dir.parent
        data_dir = task_dir / "data"

        # Only load eval ground truth at startup (unencrypted)
        gt_file = data_dir / "aime_2022_2023.jsonl"
        if not gt_file.exists():
            print(f"WARNING: Eval ground truth file not found: {gt_file}")
        else:
            count = 0
            with open(gt_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    idx = item.get('id')
                    if idx is None:
                        idx = item.get('idx')
                    answer = item.get('answer')
                    if answer is None:
                        answer = item.get('gt')
                    if idx is not None and answer is not None:
                        self.ground_truth_cache['eval'][idx] = str(answer)
                        count += 1
            print(f"Loaded {count} ground truth answers for split 'eval'")

        # Test ground truth is encrypted — not loaded at startup
        enc_file = data_dir / "aime_2024_2025.jsonl.enc"
        if enc_file.exists():
            print(f"Test ground truth is encrypted: {enc_file}")
        else:
            print(f"WARNING: Encrypted test ground truth not found: {enc_file}")

        print("Service ready to evaluate predictions from container")

    def decrypt_test_data(self, secret: str):
        """
        Decrypt and load test ground truth using the verifier secret.
        Also decrypts the test input file for agent execution.
        Returns the path to the decrypted test input file.
        """
        tools_dir = Path(__file__).parent
        task_dir = tools_dir.parent
        data_dir = task_dir / "data"

        # Decrypt test ground truth
        if not self.ground_truth_cache['test']:
            gt_data = _decrypt_file(str(data_dir / "aime_2024_2025.jsonl.enc"), secret)
            count = 0
            for line in gt_data.decode('utf-8').splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                idx = item.get('id')
                if idx is None:
                    idx = item.get('idx')
                answer = item.get('answer')
                if answer is None:
                    answer = item.get('gt')
                if idx is not None and answer is not None:
                    self.ground_truth_cache['test'][idx] = str(answer)
                    count += 1
            print(f"Decrypted and loaded {count} test ground truth answers")

        # Decrypt test input file to a temp file
        input_data = _decrypt_file(str(data_dir / "aime_test.jsonl.enc"), secret)
        tmp = tempfile.NamedTemporaryFile(
            mode='wb', suffix='.jsonl', prefix='aime_test_',
            dir='/tmp', delete=False
        )
        tmp.write(input_data)
        tmp.close()
        return tmp.name

    def get_ground_truth(self, idx: int, split: str = 'test') -> str:
        """获取指定idx的ground truth"""
        if split not in self.ground_truth_cache:
            raise ValueError(f"Unknown split: {split}. Available splits: {list(self.ground_truth_cache.keys())}")

        if idx not in self.ground_truth_cache[split]:
            raise ValueError(f"No ground truth found for idx={idx} in split '{split}'")

        return self.ground_truth_cache[split][idx]

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        data_name: str = 'aime',
        split: str = 'test',
        first_k: int = None
    ) -> Dict[str, Any]:
        """
        评估预测数据

        Args:
            predictions: 预测列表，每个包含 idx, pred 字段
                        注意：不需要提供gt字段，会自动从后端数据补全
            data_name: 数据集名称
            first_k: 若指定，只对前 k 题评分（分母也是 k）；否则全量评分

        Returns:
            评估结果字典
        """
        try:
            all_gt_indices = set(self.ground_truth_cache[split].keys())

            # 若指定 first_k，只取排序后前 k 个 idx 参与评分（分母也是 k）
            if first_k is not None:
                eval_indices = set(sorted(all_gt_indices)[:first_k])
            else:
                eval_indices = all_gt_indices

            total_count = len(eval_indices)

            if total_count == 0:
                return {
                    'success': False,
                    'error': f"No ground truth loaded for split '{split}'"
                }

            # 验证并补全 ground truth
            pred_map: Dict[int, Dict] = {}
            for sample in predictions:
                # 检查 idx 字段
                if 'idx' not in sample:
                    return {
                        'success': False,
                        'error': f"Missing 'idx' field in sample: {sample}"
                    }

                # 检查 pred 字段
                if 'pred' not in sample:
                    return {
                        'success': False,
                        'error': f"Missing 'pred' field in sample {sample.get('idx', '?')}"
                    }
                if not isinstance(sample['pred'], str):
                    return {
                        'success': False,
                        'error': f"'pred' must be a string in sample {sample.get('idx', '?')}"
                    }

                idx = sample['idx']
                if idx not in eval_indices:
                    # 超出评分范围的预测直接忽略
                    continue

                # 补全 ground truth
                answer = self.get_ground_truth(idx, split=split)
                sample['gt_cot'] = None
                sample['gt'] = answer
                sample['gt_cot'], sample['gt'] = parse_ground_truth(sample, data_name)
                pred_map[idx] = sample

            # 对 eval_indices 内的 idx 评分，未预测的自动判错
            params = []
            eval_order = sorted(eval_indices)
            for idx in eval_order:
                if idx in pred_map:
                    params.append((idx, pred_map[idx]['pred'], pred_map[idx]['gt']))
                else:
                    params.append((idx, None, None))  # 未预测，直接判错

            scores = []
            timeout_cnt = 0

            with ProcessPool(max_workers=4) as pool:
                # 只对实际有预测的样本并行评分
                valid_params = [(i, p, g) for i, p, g in params if p is not None]
                if valid_params:
                    future = pool.map(math_equal_process, valid_params, timeout=3)
                    iterator = future.result()
                    valid_scores = []
                    while True:
                        try:
                            result = next(iterator)
                            valid_scores.append(result)
                        except StopIteration:
                            break
                        except TimeoutError:
                            valid_scores.append(False)
                            timeout_cnt += 1
                        except Exception:
                            valid_scores.append(False)
                else:
                    valid_scores = []

            # 按 eval_order 重建完整 scores（未预测的填 False）
            valid_iter = iter(valid_scores)
            for idx in eval_order:
                if idx in pred_map:
                    scores.append(next(valid_iter))
                else:
                    scores.append(False)

            # 回写 score 到各 sample
            for i, idx in enumerate(eval_order):
                if idx in pred_map:
                    pred_map[idx]['score'] = scores[i]

            correct_count = sum(scores)
            accuracy = correct_count / total_count * 100

            return {
                'success': True,
                'accuracy': round(accuracy, 1),
                'correct': correct_count,
                'total': total_count,
                'num_samples': len(predictions),
                'missing_predictions': total_count - len(pred_map),
                'timeout_samples': timeout_cnt,
                'scores': scores
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }


# 创建全局服务实例
evaluation_service = EvaluationService()


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'service': 'Math Evaluation API',
        'version': '1.0.0'
    })


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    DISABLED: Direct prediction evaluation is disabled to prevent
    brute-force answer enumeration (AIME answers are integers 0-999).
    Use /evaluate/agent to submit an agent file instead.
    """
    return jsonify({
        'success': False,
        'error': 'Direct prediction evaluation is disabled. '
                 'Use /evaluate/agent to submit an agent file for evaluation.'
    }), 403

    # --- Original implementation below (disabled for security) ---
    try:
        # 获取请求数据
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        predictions = data.get('predictions', [])
        data_name = data.get('data_name', 'aime')
        split = data.get('split', 'test')  # 默认使用 test split

        # Map split to correct data_name for parser compatibility
        if data_name == 'aime':
            data_name = f'aime_{split}'  # 转换为 aime_test 或 aime_eval

        if not predictions:
            return jsonify({
                'success': False,
                'error': 'predictions field is required and cannot be empty'
            }), 400

        # 执行评估
        result = evaluation_service.evaluate_predictions(predictions, data_name, split=split)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/evaluate/file', methods=['POST'])
def evaluate_file():
    """
    DISABLED: File-based prediction evaluation is disabled to prevent
    brute-force answer enumeration. Use /evaluate/agent instead.
    """
    return jsonify({
        'success': False,
        'error': 'File-based prediction evaluation is disabled. '
                 'Use /evaluate/agent to submit an agent file for evaluation.'
    }), 403

    # --- Original implementation below (disabled for security) ---
    try:
        # 检查文件
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Empty filename'
            }), 400

        data_name = request.form.get('data_name', 'aime')

        # 读取文件内容
        predictions = []
        for line in file:
            line = line.decode('utf-8').strip()
            if line:
                predictions.append(json.loads(line))

        # 执行评估
        result = evaluation_service.evaluate_predictions(predictions, data_name)

        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/score/single', methods=['POST'])
def score_single():
    """
    DISABLED: Single prediction scoring is disabled to prevent
    brute-force answer enumeration. Use /evaluate/agent instead.
    """
    return jsonify({
        'success': False,
        'error': 'Single prediction scoring is disabled. '
                 'Use /evaluate/agent to submit an agent file for evaluation.'
    }), 403

    # --- Original implementation below (disabled for security) ---
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400

        prediction = data.get('prediction')
        ground_truth = data.get('ground_truth')

        if prediction is None or ground_truth is None:
            return jsonify({
                'success': False,
                'error': 'prediction and ground_truth are required'
            }), 400

        # 评分
        is_correct = math_equal(prediction, ground_truth)

        return jsonify({
            'success': True,
            'correct': is_correct
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/evaluate/agent', methods=['POST'])
def evaluate_agent():
    """
    调用 Agent 文件生成预测并评分

    请求格式:
    {
        "agent_file": "/workspace/agent.py",
        "split": "eval" | "test",
        "timeout": 21600,
        "first_k": 10,
        "kill_running": true
    }
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
                    if not data:
                        return jsonify({'success': False, 'error': 'No JSON data provided'}), 400

                    agent_file = data.get('agent_file')
                    split = data.get('split', 'eval')
                    timeout = data.get('timeout', 21600)
                    first_k = data.get('first_k', None)
                    if not agent_file:
                        return jsonify({'success': False, 'error': 'agent_file is required'}), 400

                    if split not in ['eval', 'test']:
                        return jsonify({'success': False, 'error': 'split must be \"eval\" or \"test\"'}), 400

                    if first_k is not None:
                        if not isinstance(first_k, int) or first_k < 1:
                            return jsonify({'success': False, 'error': 'first_k must be a positive integer'}), 400

                    # Every new evaluation must clean up residual stale evaluations.
                    state = _read_eval_state() or {}
                    pgid = state.get("runner_pgid")
                    if isinstance(pgid, int) and _pgid_alive(pgid):
                        _kill_pgid(pgid)
                    cleanup_info = cleanup_stale_agent_tasks()
                    if cleanup_info.get("killed", 0) > 0:
                        print(f"cleanup_stale_agent_tasks: killed={cleanup_info['killed']} pids={cleanup_info['pids']}")
                    _clear_eval_state()

                    # Select input file
                    if split == 'test':
                        provided_secret = request.headers.get('X-Verifier-Secret', '')
                        if not provided_secret:
                            return jsonify({
                                'success': False,
                                'error': 'Access denied: test split requires X-Verifier-Secret header.'
                            }), 403
                        try:
                            input_file = evaluation_service.decrypt_test_data(provided_secret)
                        except Exception:
                            return jsonify({'success': False, 'error': 'Access denied: invalid verifier secret.'}), 403
                    else:
                        input_file = '/app/data/aime_eval.jsonl'

                    if not Path(agent_file).exists():
                        return jsonify({'success': False, 'error': f'Agent file not found: {agent_file}'}), 404
                    if not Path(input_file).exists():
                        return jsonify({'success': False, 'error': f'Input file not found: {input_file}'}), 404

                    run_id = _make_run_id()
                    output_file = f'/tmp/agent_predictions_{run_id}.jsonl'

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
                            # before the outer wait fired. Missing indices are counted as wrong
                            # by evaluate_predictions, so the score is a fair lower bound.
                            partial_predictions = []
                            predictions_content = ''
                            if Path(output_file).exists():
                                try:
                                    predictions_content = Path(output_file).read_text(errors='replace')
                                except Exception:
                                    predictions_content = ''
                                for _line in predictions_content.splitlines():
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
                                    eval_result = evaluation_service.evaluate_predictions(
                                        partial_predictions,
                                        data_name='aime',
                                        split=split,
                                        first_k=first_k,
                                    )
                                    detailed_results = []
                                    for pred in partial_predictions:
                                        detailed_results.append({
                                            'idx': pred.get('idx'),
                                            'correct': pred['score'] if pred.get('score') is not None else False
                                        })
                                    eval_result['detailed_results'] = detailed_results
                                    eval_result['predictions_content'] = predictions_content
                                    eval_result['partial'] = True
                                    eval_result['timed_out'] = True
                                    eval_result['run_id'] = run_id
                                    eval_result['cleanup'] = cleanup_info
                                    eval_result['warning'] = (
                                        f'Agent execution exceeded {timeout}s limit; '
                                        f'scored {len(partial_predictions)} partial predictions '
                                        f'(missing counted as wrong)'
                                    )
                                    return jsonify(eval_result)
                                except Exception as _e:
                                    agent_stderr = (
                                        agent_stderr
                                        + f"\n[partial-score-error] {type(_e).__name__}: {_e}"
                                    )[-10000:]

                            return jsonify({
                                'success': False,
                                'error': f'Agent execution timed out (limit: {timeout}s)',
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
                            'success': False,
                            'error': 'Agent execution failed',
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

                    eval_result = evaluation_service.evaluate_predictions(
                        predictions,
                        data_name='aime',
                        split=split,
                        first_k=first_k
                    )

                    detailed_results = []
                    for pred in predictions:
                        detailed_results.append({
                            'idx': pred['idx'],
                            'correct': pred['score'] if pred.get('score') is not None else False
                        })
                    eval_result['detailed_results'] = detailed_results

                    with open(output_file, 'r') as f:
                        eval_result['predictions_content'] = f.read()

                    Path(output_file).unlink(missing_ok=True)
                    _clear_eval_state()

                    eval_result['run_id'] = run_id
                    eval_result['cleanup'] = cleanup_info

                    return jsonify(eval_result)
            except BlockingIOError:
                if kill_running:
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
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/monitor', methods=['POST'])
def run_monitor():
    """
    Run API usage monitor on agent workspace.
    """
    try:
        result = subprocess.run(
            ['python3', '/app/eval_utils/monitor.py'],
            capture_output=True, text=True, timeout=60
        )
        return jsonify({
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }), 200 if result.returncode == 0 else 422
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/info', methods=['GET'])
def api_info():
    """API信息接口"""
    return jsonify({
        'service': 'Math Evaluation API',
        'version': '1.2.0',
        'endpoints': {
            '/health': {
                'method': 'GET',
                'description': 'Health check'
            },
            '/evaluate': {
                'method': 'POST',
                'status': 'DISABLED',
                'description': 'Direct prediction evaluation (disabled for security)'
            },
            '/evaluate/file': {
                'method': 'POST',
                'status': 'DISABLED',
                'description': 'File-based prediction evaluation (disabled for security)'
            },
            '/score/single': {
                'method': 'POST',
                'status': 'DISABLED',
                'description': 'Single prediction scoring (disabled for security)'
            },
            '/v1/chat/completions': {
                'method': 'POST',
                'description': 'Model proxy — forwards to upstream API with forced model name'
            },
            '/v1/models': {
                'method': 'GET',
                'description': 'List allowed models (returns only TASK_MODEL_NAME)'
            },
            '/v1/stats': {
                'method': 'GET',
                'description': 'Proxy usage statistics'
            },
            '/evaluate/agent': {
                'method': 'POST',
                'description': 'Run agent file and evaluate predictions',
                'input': {
                    'agent_file': 'Path to agent Python file (required)',
                    'split': '"eval" only (test split requires verifier secret)',
                    'timeout': 'Timeout in seconds (optional, default: 21600)',
                    'first_k': 'Only evaluate first k problems (optional, default: all)'
                },
                'output': {
                    'success': 'Boolean',
                    'accuracy': 'Float (percentage)',
                    'correct': 'Integer',
                    'total': 'Integer',
                    'missing_predictions': 'Integer — number of problems not covered by agent predictions (treated as incorrect)',
                    'scores': 'List of booleans'
                }
            }
        }
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Math Evaluation API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind (default: 8080)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print("="*60)
    print("Math Evaluation API Server")
    print("="*60)
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print("\nAvailable endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  GET  http://{args.host}:{args.port}/info")
    # print(f"  POST http://{args.host}:{args.port}/evaluate")
    # print(f"  POST http://{args.host}:{args.port}/evaluate/file")
    print(f"  POST http://{args.host}:{args.port}/evaluate/agent")
    # print(f"  POST http://{args.host}:{args.port}/score/single")
    print(f"\nModel proxy endpoints:")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  GET  http://{args.host}:{args.port}/v1/stats")
    print("="*60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
