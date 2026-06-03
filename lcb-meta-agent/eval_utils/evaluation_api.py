#!/usr/bin/env python3
"""LiveCodeBench evaluation API for meta-agent task."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
import pickle
import re
import signal
import shutil
import struct
import subprocess
import tempfile
import threading
import time
import tomllib
import traceback
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import socket

import numpy as np
import requests as http_requests
from requests.adapters import HTTPAdapter
from flask import Flask, Response, jsonify, request

from local_unified_executor import check_correctness


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
PROBLEM_EVAL_WORKERS = 2
CHUNKED_FERNET_MAGIC = b"LCBFERNETv1\n"

_cancel_scoring = threading.Event()

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

_EVAL_LOCK_PATH = "/tmp/evaluate_agent_lcb.lock"
_EVAL_STATE_PATH = "/tmp/evaluate_agent_lcb.state.json"


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


def _build_fernet(secret: str):
    from cryptography.fernet import Fernet

    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def _decrypt_file(encrypted_path: str, secret: str) -> bytes:
    """Decrypt a Fernet-encrypted file using the provided secret."""
    f = _build_fernet(secret)
    with open(encrypted_path, "rb") as fp:
        return f.decrypt(fp.read())


def _decrypt_file_to_path(encrypted_path: str, secret: str, output_path: str) -> None:
    """
    Decrypt encrypted_path into output_path.
    Supports:
    1) Chunked Fernet records (preferred, low-memory).
    2) Legacy single Fernet blob (fallback).
    """
    f = _build_fernet(secret)
    with open(encrypted_path, "rb") as src:
        prefix = src.read(len(CHUNKED_FERNET_MAGIC))
        if prefix == CHUNKED_FERNET_MAGIC:
            with open(output_path, "wb") as dst:
                while True:
                    length_bytes = src.read(4)
                    if not length_bytes:
                        break
                    if len(length_bytes) != 4:
                        raise ValueError(f"Corrupted encrypted file: {encrypted_path}")
                    token_length = struct.unpack(">I", length_bytes)[0]
                    if token_length <= 0:
                        raise ValueError(f"Invalid encrypted chunk length in: {encrypted_path}")
                    token = src.read(token_length)
                    if len(token) != token_length:
                        raise ValueError(f"Truncated encrypted chunk in: {encrypted_path}")
                    dst.write(f.decrypt(token))
            return

    # Backward compatibility for existing single-token Fernet files.
    decrypted = _decrypt_file(encrypted_path, secret)
    with open(output_path, "wb") as dst:
        dst.write(decrypted)


def _extract_idx_fast(line: str) -> Optional[str]:
    match = re.search(r'"idx"\s*:\s*"([^"\\]+)"', line)
    if match:
        return match.group(1)

    match = re.search(r'"idx"\s*:\s*(\d+)', line)
    if match:
        return match.group(1)
    return None


def _load_json_maybe(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return default
    return default


def parse_metadata(raw_metadata: Any) -> Dict[str, Any]:
    metadata = _load_json_maybe(raw_metadata, {})
    return metadata if isinstance(metadata, dict) else {}


def decode_test_cases(raw_test_cases: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_test_cases, list):
        return [x for x in raw_test_cases if isinstance(x, dict)]

    if isinstance(raw_test_cases, str):
        parsed = _load_json_maybe(raw_test_cases, None)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]

        try:
            decompressed = zlib.decompress(base64.b64decode(raw_test_cases.encode("utf-8")))
            payload = pickle.loads(decompressed)
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            parsed_payload = _load_json_maybe(payload, None)
            if isinstance(parsed_payload, list):
                return [x for x in parsed_payload if isinstance(x, dict)]
        except Exception:
            return []

    return []


def build_input_output(problem: Dict[str, Any]) -> Dict[str, Any]:
    public_tests = decode_test_cases(problem.get("public_test_cases", []))
    private_tests = decode_test_cases(problem.get("private_test_cases", []))
    all_tests = public_tests + private_tests

    metadata = parse_metadata(problem.get("metadata", "{}"))
    fn_name = problem.get("fn_name")
    if fn_name is None:
        fn_name = metadata.get("func_name")

    return {
        "inputs": [str(t.get("input", "")) for t in all_tests],
        "outputs": [str(t.get("output", "")) for t in all_tests],
        "fn_name": fn_name,
    }


def is_all_tests_passed(result_list: List[Any]) -> bool:
    if not isinstance(result_list, list) or not result_list:
        return False
    return all(bool(x is True or x == True) for x in result_list)


class ModelProxy:
    """OpenAI-compatible transparent proxy with forced model name."""

    def __init__(self):
        self._lock = threading.Lock()
        self._request_timestamps: List[float] = []
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.errors = 0
        self._rpm_log_timestamps: List[float] = []
        threading.Thread(target=self._periodic_log, daemon=True).start()

    @property
    def rpm_limit(self) -> int:
        return int(os.environ.get("MODEL_PROXY_RPM", "0"))  # 0 = unlimited

    @property
    def tpm_limit(self) -> int:
        return int(os.environ.get("MODEL_PROXY_TPM", "0"))  # 0 = unlimited

    def _check_rpm(self) -> bool:
        """Return True if request is within RPM limit."""
        limit = self.rpm_limit
        if limit <= 0:
            return True

        now = time.time()
        with self._lock:
            self._request_timestamps = [t for t in self._request_timestamps if now - t < 60]
            if len(self._request_timestamps) >= limit:
                return False
            self._request_timestamps.append(now)
        return True

    def _record_usage(self, latency: float, usage: dict) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_latency += latency
            self.total_input_tokens += usage.get("prompt_tokens", 0)
            self.total_output_tokens += usage.get("completion_tokens", 0)
            self._rpm_log_timestamps.append(time.time())

    def get_stats(self) -> dict:
        with self._lock:
            avg_latency = self.total_latency / self.total_requests if self.total_requests else 0.0
            return {
                "total_requests": self.total_requests,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "avg_latency_seconds": round(avg_latency, 3),
                "errors": self.errors,
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


@dataclass
class IndexedSplitDataset:
    split: str
    full_file: Path
    index: Dict[str, int] = field(default_factory=dict)
    header_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    io_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def load_index(self) -> None:
        self.index.clear()
        self.header_cache.clear()
        self.io_cache.clear()

        if not self.full_file.exists():
            print(f"WARNING: split '{self.split}' file not found: {self.full_file}")
            return

        with open(self.full_file, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break

                stripped = line.strip()
                if not stripped:
                    continue

                idx = _extract_idx_fast(stripped)
                if idx is None:
                    obj = json.loads(stripped)
                    idx = str(obj["idx"])
                    self.header_cache[idx] = self._build_header(obj)

                self.index[idx] = offset

        print(
            f"Loaded split '{self.split}': {len(self.index)} problems indexed from {self.full_file}"
        )

    def _build_header(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "idx": str(obj.get("idx", "")),
            "question_id": obj.get("question_id", ""),
            "question_title": obj.get("question_title", ""),
            "platform": obj.get("platform", ""),
            "difficulty": obj.get("difficulty", ""),
            "contest_date": obj.get("contest_date", ""),
        }

    def _read_problem(self, idx: str) -> Dict[str, Any]:
        if idx not in self.index:
            raise ValueError(f"No problem found for idx={idx} in split '{self.split}'")

        with open(self.full_file, "r", encoding="utf-8") as f:
            f.seek(self.index[idx])
            line = f.readline()

        if not line.strip():
            raise ValueError(f"Empty problem line for idx={idx} in split '{self.split}'")

        obj = json.loads(line)
        if "idx" not in obj:
            obj["idx"] = idx
        return obj

    def get_header(self, idx: str) -> Dict[str, Any]:
        if idx in self.header_cache:
            return self.header_cache[idx]

        obj = self._read_problem(idx)
        header = self._build_header(obj)
        self.header_cache[idx] = header
        return header

    def get_input_output(self, idx: str) -> Dict[str, Any]:
        if idx in self.io_cache:
            return self.io_cache[idx]

        obj = self._read_problem(idx)
        io_obj = build_input_output(obj)
        if not io_obj["inputs"]:
            raise ValueError(
                f"No test cases decoded for idx={idx} in split '{self.split}'. "
                "Please check data preparation output."
            )

        self.io_cache[idx] = io_obj
        if idx not in self.header_cache:
            self.header_cache[idx] = self._build_header(obj)
        return io_obj

    def get_first_k_indices(self, first_k: int) -> List[str]:
        """Return the first k problem ids in original file order."""
        return list(self.index.keys())[:first_k]


class EvaluationService:
    """Core evaluation service for LiveCodeBench code generation."""

    def __init__(self):
        # Only eval split is loaded at startup.
        # Test split is encrypted and loaded on-demand per verifier request.
        self.datasets: Dict[str, IndexedSplitDataset] = {
            "eval": IndexedSplitDataset("eval", Path("/app/data/lcb_eval_full.jsonl")),
        }
        self.test_full_enc = Path("/app/data/lcb_test_full.jsonl.enc")
        self.test_input_enc = Path("/app/data/lcb_test.jsonl.enc")
        self._load_indexes()

    def _load_indexes(self) -> None:
        for dataset in self.datasets.values():
            dataset.load_index()

    def get_available_count(self, split: str) -> int:
        if split == "test":
            # Test count is not exposed directly when encrypted at rest.
            return 0
        dataset = self.datasets.get(split)
        return len(dataset.index) if dataset else 0

    def is_test_data_encrypted(self) -> bool:
        return self.test_full_enc.exists() and self.test_input_enc.exists()

    def decrypt_test_data(self, secret: str) -> tuple[str, str]:
        """
        Decrypt test input/full data to temporary files and return:
          (test_input_path, test_full_path)
        """
        if not secret:
            raise ValueError("Verifier secret is required")
        if not self.is_test_data_encrypted():
            raise FileNotFoundError(
                "Encrypted test data files are missing. "
                "Expected lcb_test.jsonl.enc and lcb_test_full.jsonl.enc"
            )

        tmp_files: List[str] = []
        try:
            tmp_input = tempfile.NamedTemporaryFile(
                mode="wb", suffix=".jsonl", prefix="lcb_test_", dir="/tmp", delete=False
            )
            tmp_input.close()
            tmp_files.append(tmp_input.name)
            _decrypt_file_to_path(str(self.test_input_enc), secret, tmp_input.name)

            tmp_full = tempfile.NamedTemporaryFile(
                mode="wb", suffix=".jsonl", prefix="lcb_test_full_", dir="/tmp", delete=False
            )
            tmp_full.close()
            tmp_files.append(tmp_full.name)
            _decrypt_file_to_path(str(self.test_full_enc), secret, tmp_full.name)
            return tmp_input.name, tmp_full.name
        except Exception:
            for path in tmp_files:
                Path(path).unlink(missing_ok=True)
            raise

    def evaluate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        split: str = "test",
        case_timeout: int = 10,
        dataset_override: Optional[IndexedSplitDataset] = None,
        allowed_indices: Optional[List[str]] = None,
        scoring_timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        try:
            _cancel_scoring.clear()
            if not predictions:
                return {"success": False, "error": "Empty predictions list"}

            if dataset_override is not None:
                dataset = dataset_override
            else:
                if split not in self.datasets:
                    return {
                        "success": False,
                        "error": f"Unknown split: {split}. Available splits: {list(self.datasets.keys())}",
                    }
                dataset = self.datasets[split]

            if not dataset.index:
                return {
                    "success": False,
                    "error": f"No indexed data for split '{split}'. Please prepare data first.",
                }

            if allowed_indices is None:
                target_indices = list(dataset.index.keys())
            else:
                target_indices = [str(idx) for idx in allowed_indices if str(idx) in dataset.index]

            allowed_index_set = set(target_indices)
            total_count = len(target_indices)
            if total_count == 0:
                return {
                    "success": False,
                    "error": "No problems selected for evaluation.",
                }

            sample_passes: List[bool] = []
            detailed_results: List[Dict[str, Any]] = []
            timeout_samples = 0
            prepared_samples: List[Dict[str, Any]] = []
            seen_indices: set[str] = set()

            for sample in predictions:
                if "idx" not in sample:
                    return {
                        "success": False,
                        "error": f"Missing 'idx' field in sample: {sample}",
                    }

                idx = str(sample["idx"])
                if idx not in allowed_index_set:
                    continue
                if idx in seen_indices:
                    return {
                        "success": False,
                        "error": f"Duplicate prediction for idx={idx}. Exactly one final pred is allowed per problem.",
                    }
                seen_indices.add(idx)

                code_str = sample.get("pred")
                if not isinstance(code_str, str):
                    return {
                        "success": False,
                        "error": (
                            f"'pred' must be a non-empty string code "
                            f"in sample idx={idx}"
                        ),
                    }
                if not code_str.strip():
                    return {
                        "success": False,
                        "error": f"'pred' must not be empty in sample idx={idx}",
                    }

                try:
                    input_output = dataset.get_input_output(idx)
                    header = dataset.get_header(idx)
                except Exception as exc:
                    return {"success": False, "error": str(exc)}

                prepared_samples.append(
                    {
                        "sample": sample,
                        "idx": idx,
                        "code_str": code_str,
                        "input_output": input_output,
                        "header": header,
                    }
                )

            def _evaluate_one_problem(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                if _cancel_scoring.is_set():
                    return None
                passed = False
                sample_meta: Dict[str, Any] = {}
                timeout_hit = False
                try:
                    result_list, metadata = check_correctness(
                        in_outs=item["input_output"],
                        generation=item["code_str"],
                        timeout=case_timeout,
                        debug=False,
                    )
                    passed = is_all_tests_passed(result_list)

                    if not passed and isinstance(metadata, dict):
                        if metadata.get("error_code") in {-3}:
                            timeout_hit = True

                    sample_meta = metadata if isinstance(metadata, dict) else {}
                except Exception as exc:
                    passed = False
                    sample_meta = {
                        "error_code": -5,
                        "error_message": "EvaluationInternalError",
                        "error": repr(exc),
                    }

                item["sample"]["score"] = passed
                item["sample"]["judge_metadata"] = sample_meta
                return {
                    "idx": item["idx"],
                    "header": item["header"],
                    "passed": passed,
                    "timeout_hit": timeout_hit,
                }

            evaluated: List[Dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=PROBLEM_EVAL_WORKERS) as executor:
                futures = {
                    executor.submit(_evaluate_one_problem, item): item
                    for item in prepared_samples
                }
                try:
                    for future in as_completed(futures, timeout=scoring_timeout):
                        result = future.result()
                        if result is not None:
                            evaluated.append(result)
                except TimeoutError:
                    pass
                for f in futures:
                    f.cancel()

            for item in evaluated:
                passed = item["passed"]
                if item["timeout_hit"]:
                    timeout_samples += 1

                sample_passes.append(passed)
                detailed_results.append(
                    {
                        "idx": item["idx"],
                        "correct": passed,
                    }
                )

            correct_count = sum(1 for passed in sample_passes if passed)
            covered_count = len(sample_passes)
            coverage = float((covered_count / total_count) * 100.0) if total_count > 0 else 0.0
            accuracy = float((correct_count / total_count) * 100.0) if total_count > 0 else 0.0

            return {
                "success": True,
                "accuracy": accuracy,
                "correct": correct_count,
                "total": total_count,
                "covered": covered_count,
                "coverage": coverage,
                "num_samples": len(predictions),
                "timeout_samples": timeout_samples,
                "scores": sample_passes,
                "detailed_results": detailed_results,
            }

        except Exception as exc:
            return {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

evaluation_service = EvaluationService()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "ok",
            "service": "LiveCodeBench Evaluation API",
            "version": "1.0.0",
            "splits": {
                "eval": evaluation_service.get_available_count("eval"),
                "test": evaluation_service.get_available_count("test"),
            },
            "test_data_encrypted": evaluation_service.is_test_data_encrypted(),
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def proxy_chat_completions():
    """Transparent model proxy that forces TASK_MODEL_NAME."""
    real_api_base = os.environ.get("TASK_MODEL_API_BASE", "")
    real_api_key = os.environ.get("TASK_MODEL_API_KEY", "")
    forced_model = os.environ.get("TASK_MODEL_NAME", "")

    print(f"[PROXY DEBUG] target={real_api_base} model={forced_model}", flush=True)

    if not real_api_base or not real_api_key:
        return jsonify({"error": "Model proxy not configured"}), 500
    if not forced_model:
        return jsonify({"error": "TASK_MODEL_NAME is not configured"}), 500

    if not model_proxy._check_rpm():
        return jsonify(
            {
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                }
            }
        ), 429

    body = request.get_json(force=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Invalid request body"}), 400

    # Force model name regardless of what the agent requested.
    body["model"] = forced_model

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {real_api_key}",
    }
    is_stream = bool(body.get("stream", False))
    target_url = f"{real_api_base.rstrip('/')}/chat/completions"
    start = time.time()

    try:
        if is_stream:
            resp = _proxy_session.post(
                target_url,
                json=body,
                headers=headers,
                stream=True,
                timeout=(30, 3700),
            )
            latency = time.time() - start
            print(
                f"[PROXY DEBUG] stream connect: status={resp.status_code} latency={latency:.1f}s",
                flush=True,
            )

            if resp.status_code != 200:
                with model_proxy._lock:
                    model_proxy.errors += 1
                return Response(
                    resp.content,
                    status=resp.status_code,
                    content_type=resp.headers.get("Content-Type", "application/json"),
                )

            # Streaming token usage is usually available only in final chunks.
            model_proxy._record_usage(latency, {})

            def generate():
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk

            return Response(
                generate(),
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type", "text/event-stream"),
            )

        print(f"[BODY]={body}", flush=True)
        resp = _proxy_session.post(
            target_url,
            json=body,
            headers=headers,
            timeout=(30, 3700),
        )
        latency = time.time() - start
        print(f"[RESP]={resp}", flush=True)
        print(
            f"[PROXY DEBUG] non-stream response: status={resp.status_code} latency={latency:.1f}s",
            flush=True,
        )

        if resp.status_code != 200:
            with model_proxy._lock:
                model_proxy.errors += 1
            return Response(
                resp.content,
                status=resp.status_code,
                content_type=resp.headers.get("Content-Type", "application/json"),
            )

        result = resp.json()
        usage = result.get("usage", {})
        model_proxy._record_usage(latency, usage)
        return jsonify(result), 200

    except http_requests.Timeout:
        print(f"[PROXY DEBUG] TIMEOUT after {time.time() - start:.1f}s", flush=True)
        with model_proxy._lock:
            model_proxy.errors += 1
        return jsonify(
            {
                "error": {
                    "message": "Upstream timeout",
                    "type": "timeout_error",
                }
            }
        ), 504
    except Exception as exc:
        print(
            f"[PROXY DEBUG] ERROR after {time.time() - start:.1f}s: {type(exc).__name__}: {exc}",
            flush=True,
        )
        with model_proxy._lock:
            model_proxy.errors += 1
        return jsonify(
            {
                "error": {
                    "message": str(exc),
                    "type": "proxy_error",
                }
            }
        ), 502


@app.route("/v1/models", methods=["GET"])
def proxy_list_models():
    forced_model = os.environ.get("TASK_MODEL_NAME", "default")
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": forced_model,
                    "object": "model",
                    "owned_by": "proxy",
                }
            ],
        }
    )


@app.route("/v1/stats", methods=["GET"])
def proxy_stats():
    return jsonify(model_proxy.get_stats())


@app.route("/evaluate", methods=["POST"])
def evaluate_direct_disabled():
    return jsonify(
        {
            "success": False,
            "error": "Direct prediction evaluation is disabled. Use /evaluate/agent.",
        }
    ), 403


@app.route("/evaluate/file", methods=["POST"])
def evaluate_file_disabled():
    return jsonify(
        {
            "success": False,
            "error": "File-based evaluation is disabled. Use /evaluate/agent.",
        }
    ), 403


@app.route("/score/single", methods=["POST"])
def score_single_disabled():
    return jsonify(
        {
            "success": False,
            "error": "Single-sample scoring is disabled. Use /evaluate/agent.",
        }
    ), 403


@app.route("/evaluate/agent", methods=["POST"])
def evaluate_agent():
    """
    Execute an agent file and score predictions.

    Request JSON:
    {
      "agent_file": "/workspace/agent.py",
      "split": "eval" | "test",
      "timeout": 3600,
      "case_timeout": 10,
      "first_k": 10,
      "kill_running": true
    }
    """
    try:
        data = request.get_json() or {}

        kill_running = bool(data.get("kill_running", False))
        agent_file = data.get("agent_file")

        # Pure kill mode: kill_running=true without agent_file
        # just kills the running evaluation and returns.
        if kill_running and not agent_file:
            _cancel_scoring.set()
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
                    temp_files: List[str] = []
                    try:
                        if not data:
                            return jsonify({"success": False, "error": "No JSON data provided"}), 400

                        agent_file = data.get("agent_file")
                        split = data.get("split", "eval")
                        timeout = int(data.get("timeout", 3600))
                        case_timeout = int(data.get("case_timeout", 10))
                        first_k = data.get("first_k")

                        if not agent_file:
                            return jsonify({"success": False, "error": "agent_file is required"}), 400
                        if split not in {"eval", "test"}:
                            return jsonify({"success": False, "error": 'split must be "eval" or "test"'}), 400
                        if first_k is not None:
                            if isinstance(first_k, bool) or not isinstance(first_k, int) or first_k < 1:
                                return jsonify({"success": False, "error": "first_k must be a positive integer"}), 400
                            if split != "eval":
                                return jsonify(
                                    {
                                        "success": False,
                                        "error": "first_k is only supported for split='eval'.",
                                    }
                                ), 400

                        # Every new evaluation must clean up residual stale evaluations.
                        state = _read_eval_state() or {}
                        pgid = state.get("runner_pgid")
                        if isinstance(pgid, int) and _pgid_alive(pgid):
                            _kill_pgid(pgid)
                        cleanup_info = cleanup_stale_agent_tasks()
                        if cleanup_info.get("killed", 0) > 0:
                            print(f"cleanup_stale_agent_tasks: killed={cleanup_info['killed']} pids={cleanup_info['pids']}")
                        _clear_eval_state()

                        # Select input file / dataset
                        dataset_override: Optional[IndexedSplitDataset] = None
                        allowed_indices: Optional[List[str]] = None
                        full_file = ""
                        if split == "test":
                            provided_secret = request.headers.get("X-Verifier-Secret", "")
                            if not provided_secret:
                                return jsonify(
                                    {
                                        "success": False,
                                        "error": "Access denied: test split requires X-Verifier-Secret header.",
                                    }
                                ), 403
                            try:
                                input_file, full_file = evaluation_service.decrypt_test_data(provided_secret)
                            except FileNotFoundError as exc:
                                return jsonify(
                                    {
                                        "success": False,
                                        "error": str(exc),
                                    }
                                ), 500
                            except Exception:
                                return jsonify(
                                    {
                                        "success": False,
                                        "error": "Access denied: invalid verifier secret.",
                                    }
                                ), 403
                            temp_files.extend([input_file, full_file])
                            dataset_override = IndexedSplitDataset("test", Path(full_file))
                            dataset_override.load_index()
                            if not dataset_override.index:
                                return jsonify(
                                    {
                                        "success": False,
                                        "error": "Decrypted test dataset is empty or invalid.",
                                    }
                                ), 500
                        else:
                            input_file = "/app/data/lcb_eval.jsonl"
                            if first_k is not None:
                                allowed_indices = evaluation_service.datasets["eval"].get_first_k_indices(first_k)

                        if not Path(agent_file).exists():
                            return jsonify({"success": False, "error": f"Agent file not found: {agent_file}"}), 404
                        if not Path(input_file).exists():
                            return jsonify({"success": False, "error": f"Input file not found: {input_file}"}), 404

                        run_id = _make_run_id()
                        output_file = f"/tmp/agent_predictions_{run_id}.jsonl"

                        pyproject_path = Path("/workspace/pyproject.toml")
                        if pyproject_path.exists():
                            print(f"Installing agent dependencies from {pyproject_path}...")
                            try:
                                with open(pyproject_path, "rb") as f:
                                    pyproject_data = tomllib.load(f)
                                deps = pyproject_data.get("project", {}).get("dependencies", [])
                            except Exception:
                                deps = []
                            if deps:
                                install_result = subprocess.run(
                                    ["uv", "pip", "install", "--system"] + deps,
                                    capture_output=True,
                                    text=True,
                                    timeout=300,
                                )
                                if install_result.returncode != 0:
                                    return jsonify(
                                        {
                                            "success": False,
                                            "error": "Failed to install dependencies from pyproject.toml",
                                            "stdout": install_result.stdout,
                                            "stderr": install_result.stderr,
                                        }
                                    ), 500
                                print("Agent dependencies installed successfully")
                            else:
                                print("No dependencies found in pyproject.toml, skipping install")

                        runner_cmd = [
                            "python3",
                            "/app/eval_utils/agent_runner.py",
                            "--agent",
                            agent_file,
                            "--input",
                            input_file,
                            "--output",
                            output_file,
                            "--timeout",
                            str(timeout),
                        ]
                        if first_k is not None:
                            runner_cmd.extend(["--first-k", str(first_k)])

                        agent_env = os.environ.copy()
                        agent_env["PYTHONUNBUFFERED"] = "1"
                        agent_env.pop("TASK_MODEL_API_KEY", None)
                        agent_env.pop("TASK_MODEL_API_BASE", None)
                        agent_env.pop("MODEL_PROXY_RPM", None)
                        agent_env.pop("MODEL_PROXY_TPM", None)
                        agent_env.pop("VERIFIER_SECRET", None)
                        agent_env["TASK_MODEL_API_BASE"] = "http://127.0.0.1:8080/v1"
                        agent_env["TASK_MODEL_API_KEY"] = "proxy-token"
                        if split == "test" and full_file:
                            agent_env["LCB_ORACLE_TEST_FULL_FILE"] = full_file

                        print("Agent env - TASK_MODEL_API_BASE:", agent_env["TASK_MODEL_API_BASE"])
                        print("Agent env - TASK_MODEL_API_KEY:", agent_env["TASK_MODEL_API_KEY"])

                        log_dir = Path("/evaluation_logs")
                        log_dir.mkdir(parents=True, exist_ok=True)
                        stdout_log = log_dir / f"{run_id}.stdout.log"
                        stderr_log = log_dir / f"{run_id}.stderr.log"
                        stdout_latest = log_dir / "stdout.log"
                        stderr_latest = log_dir / "stderr.log"

                        def _update_latest_logs() -> None:
                            try:
                                shutil.copyfile(stdout_log, stdout_latest)
                            except Exception:
                                pass
                            try:
                                shutil.copyfile(stderr_log, stderr_latest)
                            except Exception:
                                pass

                        with open(stdout_log, "w", encoding="utf-8") as f_out, open(
                            stderr_log, "w", encoding="utf-8"
                        ) as f_err:
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
                            runner_start = time.time()
                            try:
                                result_returncode = proc.wait(timeout=timeout + 60)
                            except subprocess.TimeoutExpired:
                                _kill_pgid(runner_pgid)
                                _update_latest_logs()
                                agent_stdout = stdout_log.read_text(errors="replace")[-50000:]
                                agent_stderr = stderr_log.read_text(errors="replace")[-10000:]
                                Path(output_file).unlink(missing_ok=True)
                                _clear_eval_state()
                                return jsonify(
                                    {
                                        "success": False,
                                        "error": f"Agent execution timed out (limit: {timeout}s)",
                                        "agent_output": agent_stdout,
                                        "agent_stderr": agent_stderr,
                                        "run_id": run_id,
                                        "cleanup": cleanup_info,
                                    }
                                ), 408

                        _update_latest_logs()

                        if result_returncode != 0:
                            agent_stdout = stdout_log.read_text(errors="replace")[-50000:]
                            agent_stderr = stderr_log.read_text(errors="replace")[-10000:]
                            Path(output_file).unlink(missing_ok=True)
                            _clear_eval_state()
                            return jsonify(
                                {
                                    "success": False,
                                    "error": "Agent execution failed",
                                    "agent_output": agent_stdout,
                                    "agent_stderr": agent_stderr,
                                    "stdout": agent_stdout,
                                    "stderr": agent_stderr,
                                    "run_id": run_id,
                                    "cleanup": cleanup_info,
                                }
                            ), 500

                        predictions = []
                        with open(output_file, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    predictions.append(json.loads(line))

                        runner_elapsed = time.time() - runner_start
                        remaining = max(timeout - runner_elapsed, 60.0)
                        eval_result = evaluation_service.evaluate_predictions(
                            predictions=predictions,
                            split=split,
                            case_timeout=case_timeout,
                            dataset_override=dataset_override,
                            allowed_indices=allowed_indices,
                            scoring_timeout=remaining,
                        )

                        if not eval_result.get("success"):
                            Path(output_file).unlink(missing_ok=True)
                            _clear_eval_state()
                            return jsonify(eval_result), 400

                        Path(output_file).unlink(missing_ok=True)
                        _clear_eval_state()

                        eval_result["run_id"] = run_id
                        eval_result["cleanup"] = cleanup_info

                        return jsonify(eval_result), 200
                    finally:
                        for path in temp_files:
                            Path(path).unlink(missing_ok=True)
            except BlockingIOError:
                if kill_running:
                    if not kill_attempted:
                        _cancel_scoring.set()
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
                    "hint": 'POST /evaluate/agent with {"kill_running": true} then retry',
                }), 409

    except Exception as exc:
        return jsonify(
            {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        ), 500


@app.route("/monitor", methods=["POST"])
def run_monitor():
    try:
        result = subprocess.run(
            ["python3", "/app/eval_utils/monitor.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return jsonify(
            {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        ), (200 if result.returncode == 0 else 422)
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/info", methods=["GET"])
def api_info():
    return jsonify(
        {
            "service": "LiveCodeBench Evaluation API",
            "version": "1.0.0",
            "endpoints": {
                "/health": {
                    "method": "GET",
                    "description": "Health check",
                },
                "/evaluate": {
                    "method": "POST",
                    "status": "DISABLED",
                },
                "/evaluate/file": {
                    "method": "POST",
                    "status": "DISABLED",
                },
                "/score/single": {
                    "method": "POST",
                    "status": "DISABLED",
                },
                "/v1/chat/completions": {
                    "method": "POST",
                    "description": "Model proxy endpoint (forces TASK_MODEL_NAME)",
                },
                "/v1/models": {
                    "method": "GET",
                    "description": "List allowed models (returns TASK_MODEL_NAME only)",
                },
                "/v1/stats": {
                    "method": "GET",
                    "description": "Proxy usage statistics",
                },
                "/evaluate/agent": {
                    "method": "POST",
                    "description": "Run agent and evaluate predictions",
                    "input": {
                        "agent_file": "Path to agent Python file (required)",
                        "split": "eval or test (test requires verifier secret)",
                        "timeout": "Agent timeout in seconds",
                        "case_timeout": "Per-testcase timeout in seconds",
                        "first_k": "Optional eval-only prefix size for faster iteration",
                        "kill_running": "If true, kill any running evaluation before starting (or just kill if no agent_file)",
                    },
                    "response": {
                        "success": "Whether evaluation completed successfully",
                        "accuracy": "Accuracy percentage over the full split (correct / total * 100)",
                        "correct": "Number of solved problems",
                        "total": "Total number of problems in the split",
                        "covered": "Number of problems actually evaluated from submitted predictions",
                        "coverage": "Coverage percentage over the full split (covered / total * 100)",
                        "scores": "Per-covered-problem pass/fail booleans",
                        "detailed_results": "Per-covered-problem correctness summary",
                        "note": "Prediction source code is never returned in HTTP responses",
                        "first_k_note": "When first_k is provided, totals are computed over the eval prefix only",
                    },
                },
            },
        }
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LiveCodeBench Evaluation API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("=" * 60)
    print("LiveCodeBench Evaluation API Server")
    print("=" * 60)
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print("\nAvailable endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  GET  http://{args.host}:{args.port}/info")
    print(f"  POST http://{args.host}:{args.port}/evaluate/agent")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  GET  http://{args.host}:{args.port}/v1/stats")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
