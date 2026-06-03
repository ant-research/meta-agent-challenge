#!/usr/bin/env python3
"""
TB Meta-Agent Evaluation Proxy API

This task intentionally does NOT run a separate evaluation-api container.
Instead, this Flask server provides a familiar /evaluate/agent interface and
drives a Harbor CLI HTTP service ("cli_service") to run evaluation.

Env vars:
  - HARBOR_CLI_SERVICE_URL (required): external Harbor CLI service base URL (e.g. http://<ip>:8899)
  - HARBOR_CLI_SERVICE_API_KEY (optional): sent as Authorization: Bearer <key>
  - HOST_IP (required): IP address of the Docker host, used to build URLs reachable from inner containers

  - HARBOR_DEV_DATASET_NAME (required): Harbor dataset to use for development evaluation
  - HARBOR_TEST_DATASET_NAME (required): Harbor dataset to use for protected test evaluation

  - HARBOR_AGENT_IMPORT_PATH (optional): custom agent import path (default: "agent:TBAgent")
  - HARBOR_JOBS_DIR (optional): jobs output directory on the cli_service host
  - HARBOR_CLI_CWD (optional): working directory on the cli_service host; defaults to workspace_dir
"""

from __future__ import annotations

import ast
import os
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict


import socket

import requests
from requests.adapters import HTTPAdapter
from flask import Flask, Response, jsonify, request


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


_proxy_session = requests.Session()
_proxy_session.mount("http://", _KeepAliveAdapter())
_proxy_session.mount("https://", _KeepAliveAdapter())

app = Flask(__name__)

_eval_lock = threading.Lock()
_eval_session: requests.Session | None = None
_eval_session_guard = threading.Lock()


@contextmanager
def _eval_busy(nonblocking: bool = False):
    """Serialize evaluation calls.  Uses a threading.Lock so that kill_running
    can force-release it from another request thread."""
    if nonblocking:
        if not _eval_lock.acquire(blocking=False):
            raise BlockingIOError("another eval is running")
    else:
        _eval_lock.acquire()
    try:
        yield
    finally:
        try:
            _eval_lock.release()
        except RuntimeError:
            pass


def _force_release_eval():
    """Force-release the eval lock and abort the in-flight HTTP request to
    cli_service so the holding thread unblocks immediately."""
    with _eval_session_guard:
        sess = _eval_session
    if sess is not None:
        try:
            sess.close()
        except Exception:
            pass
    try:
        _eval_lock.release()
    except RuntimeError:
        pass


def _now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _harbor_headers() -> Dict[str, str]:
    headers = {
        "User-Agent": "meta-agent-challenge/tb-meta-agent-eval-proxy",
    }
    key = os.environ.get("HARBOR_CLI_SERVICE_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _cli_service_base_url() -> str:
    base = os.environ.get("HARBOR_CLI_SERVICE_URL", "").strip()
    if not base:
        raise RuntimeError("HARBOR_CLI_SERVICE_URL is not set")
    return base.rstrip("/")


def _cli_cancel() -> Dict[str, Any]:
    """POST to cli_service /v1/cli/cancel to abort the running exec."""
    url = f"{_cli_service_base_url()}/v1/cli/cancel"
    try:
        resp = _proxy_session.post(url, headers=_harbor_headers(), json={}, timeout=(10, 30))
        try:
            return resp.json()
        except Exception:
            return {"raw_text": resp.text[:2000], "_http_status": resp.status_code}
    except Exception as e:
        return {"error": str(e)}


def _cli_exec(*, args: list[str], cwd: str | None, timeout_sec: float | None) -> Dict[str, Any]:
    global _eval_session
    url = f"{_cli_service_base_url()}/v1/cli/exec"

    session = requests.Session()
    session.mount("http://", _KeepAliveAdapter())
    session.mount("https://", _KeepAliveAdapter())
    with _eval_session_guard:
        _eval_session = session

    env: Dict[str, str] = {}

    # The proxy runs inside this container on :8080, but inner harbor containers
    # are on a different Docker network and cannot reach us by container hostname.
    # Use the host-mapped port (EVALUATION_API_HOST_PORT) + HOST_IP instead.
    host_ip = os.environ.get("HOST_IP", "127.0.0.1")
    proxy_port = os.environ.get("EVALUATION_API_HOST_PORT", "8080")
    proxy_base = f"http://{host_ip}:{proxy_port}"
    proxy_anthropic = proxy_base + "/anthropic"

    env["TASK_MODEL_API_BASE"] = proxy_anthropic
    env["TASK_MODEL_API_KEY"] = "proxy-token"
    env["TASK_MODEL_NAME"] = os.environ.get("TASK_MODEL_NAME", "")

    # Prevent inner agents from bypassing the proxy via leaked credentials.
    # All agents (including built-in claude-code/openhands) go through the proxy.
    env["OPENAI_API_KEY"] = "proxy-token"
    env["OPENAI_BASE_URL"] = proxy_base
    env["ANTHROPIC_API_KEY"] = "proxy-token"
    env["ANTHROPIC_BASE_URL"] = proxy_anthropic
    env["ANTHROPIC_API_BASE"] = proxy_anthropic
    env["LLM_API_KEY"] = "proxy-token"
    env["LLM_BASE_URL"] = proxy_anthropic

    # Forward dataset names (even though we also pass dataset as CLI args) to
    # make the full configuration visible for debugging on the cli_service host.
    for k in ("HARBOR_DEV_DATASET_NAME", "HARBOR_TEST_DATASET_NAME"):
        v = os.environ.get(k, "").strip()
        if v:
            env[k] = v

    payload: Dict[str, Any] = {"args": args, "env": env}
    if cwd:
        payload["cwd"] = cwd
    if timeout_sec is not None:
        payload["timeout_sec"] = float(timeout_sec)

    connect_timeout = float(os.environ.get("HARBOR_EVAL_CONNECT_TIMEOUT_SEC", "30"))
    read_timeout = float(os.environ.get("HARBOR_EVAL_READ_TIMEOUT_SEC", str(max(600.0, (timeout_sec or 600.0) + 300.0))))
    try:
        resp = session.post(url, headers=_harbor_headers(), json=payload, timeout=(connect_timeout, read_timeout))
    finally:
        with _eval_session_guard:
            _eval_session = None
    try:
        data = resp.json()
    except Exception:
        data = {"raw_text": resp.text[:20000]}
    data["_http_status"] = resp.status_code
    return data


def _parse_rewards(stdout: str) -> Dict[str, Any] | None:
    if not stdout:
        return None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("Rewards:"):
            continue
        tail = line[len("Rewards:") :].strip()
        if not tail:
            continue
        try:
            obj = ast.literal_eval(tail)
        except Exception:
            return None
        if isinstance(obj, dict):
            return obj
    return None


def _strip_ansi(text: str) -> str:
    # Best-effort ANSI escape stripping.
    import re

    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def _parse_job_metrics(stdout: str) -> Dict[str, Any]:
    """
    Parse Harbor job summary table output (Rich) to extract metrics like resolved_rate.

    `harbor job start` prints a table with columns starting with Trials/Exceptions and
    then metrics columns. We extract the first data row.
    """
    import re

    out = _strip_ansi(stdout or "")
    lines = [ln.rstrip("\n") for ln in out.splitlines()]

    # Find the first header row containing Trials and Exceptions.
    header_idx = None
    for i, ln in enumerate(lines):
        if "Trials" in ln and "Exceptions" in ln and ("│" in ln or "┃" in ln):
            header_idx = i
            break
    if header_idx is None:
        return {}

    # Normalize separators to a single pipe.
    def _split_row(ln: str) -> list[str]:
        ln = ln.replace("┃", "│")
        parts = [p.strip() for p in ln.split("│")]
        # Rich tables usually have empty leading/trailing cells.
        parts = [p for p in parts if p != ""]
        return parts

    headers = _split_row(lines[header_idx])

    # Find the next data row with digits and separators.
    data_row = None
    for j in range(header_idx + 1, min(header_idx + 20, len(lines))):
        ln = lines[j]
        if ("│" in ln or "┃" in ln) and re.search(r"\d", ln):
            cells = _split_row(ln)
            # Heuristic: expect same number of cells as headers.
            if len(cells) == len(headers):
                data_row = cells
                break

    if data_row is None:
        return {}

    metrics: Dict[str, Any] = {}
    for h, v in zip(headers, data_row):
        key = h.strip()
        if not key:
            continue
        # Keep original header; also emit a normalized key for common metric parsing.
        metrics[key] = v

        norm = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        # Best-effort numeric coercion.
        try:
            if v.endswith("%"):
                metrics[norm] = float(v[:-1].strip()) / 100.0
            else:
                metrics[norm] = float(v)
        except Exception:
            metrics[norm] = v

    return metrics


def _docker_network_prune():
    """Remove unused Docker networks via cli_service (host-side)."""
    try:
        url = f"{_cli_service_base_url()}/v1/shell/exec"
        _proxy_session.post(
            url,
            headers=_harbor_headers(),
            json={"cmd": ["docker", "network", "prune", "-f"], "timeout_sec": 30},
            timeout=(5, 35),
        )
    except Exception:
        pass


@contextmanager
def _periodic_network_prune(interval_sec: int = 300):
    """Run ``docker network prune -f`` every *interval_sec* seconds in a
    background thread.  The thread stops when the context manager exits."""
    stop_event = threading.Event()

    def _loop():
        while not stop_event.wait(interval_sec):
            _docker_network_prune()

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop_event.set()
        t.join(timeout=5)


def _check_verifier_secret(req) -> bool:
    """Return True if the request carries a valid verifier secret."""
    secret = os.environ.get("VERIFIER_SECRET", "").strip()
    if not secret:
        return False
    header_val = req.headers.get("X-Verifier-Secret", "").strip()
    return header_val == secret


# ============================================================================
# Model Proxy — transparent proxy that forces TASK_MODEL_NAME
#
# Custom agents (agent.py) call the Anthropic Messages API through this proxy.
# The proxy overwrites the ``model`` field in every request with
# TASK_MODEL_NAME so the agent cannot use a stronger model than assigned.
#
# All agents (including built-in claude-code/openhands) go through this proxy.
# ============================================================================


class ModelProxy:
    """Tracks stats for the model-locking proxy."""

    def __init__(self):
        self._lock = threading.Lock()
        self._request_timestamps: list[float] = []
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_latency = 0.0
        self.errors = 0
        self._rpm_log_timestamps: list[float] = []
        threading.Thread(target=self._periodic_log, daemon=True).start()

    @property
    def rpm_limit(self) -> int:
        return int(os.environ.get("MODEL_PROXY_RPM", "0"))

    def _check_rpm(self) -> bool:
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

    def _record_usage(self, latency: float, usage: dict):
        with self._lock:
            self.total_requests += 1
            self.total_latency += latency
            self.total_input_tokens += usage.get("input_tokens", 0)
            self.total_output_tokens += usage.get("output_tokens", 0)
            self._rpm_log_timestamps.append(time.time())

    def get_stats(self) -> dict:
        with self._lock:
            avg = self.total_latency / self.total_requests if self.total_requests else 0
            return {
                "total_requests": self.total_requests,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "avg_latency_seconds": round(avg, 3),
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


@app.route("/anthropic/v1/messages", methods=["POST"])
def proxy_messages():
    """Anthropic Messages API proxy — forces TASK_MODEL_NAME."""
    real_base = os.environ.get("TASK_MODEL_API_BASE", "").rstrip("/")
    real_key = os.environ.get("TASK_MODEL_API_KEY", "")
    forced_model = os.environ.get("TASK_MODEL_NAME", "")
    if "/" in forced_model:
        forced_model = forced_model.split("/", 1)[1]

    if not real_base or not real_key:
        return jsonify({"error": "Model proxy not configured"}), 500

    if not model_proxy._check_rpm():
        return jsonify({"type": "error", "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"}}), 429

    body = request.get_json(force=True)
    body["model"] = forced_model

    headers = {
        "Content-Type": "application/json",
        "x-api-key": real_key,
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
    }

    is_stream = body.get("stream", False)
    target_url = f"{real_base}/v1/messages"
    start = time.time()

    try:
        if is_stream:
            resp = _proxy_session.post(target_url, json=body, headers=headers, stream=True, timeout=(30, 3700))
            latency = time.time() - start
            if resp.status_code != 200:
                with model_proxy._lock:
                    model_proxy.errors += 1
                return Response(resp.content, status=resp.status_code, content_type=resp.headers.get("Content-Type", "application/json"))
            model_proxy._record_usage(latency, {})

            def generate():
                for chunk in resp.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk

            return Response(generate(), status=resp.status_code, content_type=resp.headers.get("Content-Type", "text/event-stream"))
        else:
            resp = _proxy_session.post(target_url, json=body, headers=headers, timeout=(30, 3700))
            latency = time.time() - start
            if resp.status_code != 200:
                with model_proxy._lock:
                    model_proxy.errors += 1
                return Response(resp.content, status=resp.status_code, content_type=resp.headers.get("Content-Type", "application/json"))
            result = resp.json()
            model_proxy._record_usage(latency, result.get("usage", {}))
            return jsonify(result), 200
    except requests.Timeout:
        with model_proxy._lock:
            model_proxy.errors += 1
        return jsonify({"type": "error", "error": {"type": "timeout_error", "message": "Upstream timeout"}}), 504
    except Exception as e:
        with model_proxy._lock:
            model_proxy.errors += 1
        return jsonify({"type": "error", "error": {"type": "proxy_error", "message": str(e)}}), 502


@app.route("/anthropic/v1/models", methods=["GET"])
def proxy_list_models():
    forced_model = os.environ.get("TASK_MODEL_NAME", "default")
    return jsonify({"object": "list", "data": [{"id": forced_model, "object": "model", "owned_by": "proxy"}]})


@app.get("/proxy/stats")
def proxy_stats():
    return jsonify(model_proxy.get_stats())


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/info")
def info():
    return jsonify(
        {
            "task": "tb-meta-agent",
            "harbor_cli_service_url_set": bool(os.environ.get("HARBOR_CLI_SERVICE_URL", "").strip()),
        }
    )


@app.post("/evaluate/agent")
def evaluate_agent():
    """
    Proxy evaluation call.

    Body (JSON):
      - agent_file: path to /workspace/agent.py
      - workspace_dir: (optional) defaults to parent of agent_file; used as cwd for Harbor CLI
      - timeout: seconds
      - first_k: (optional) only evaluate the first K tasks; useful for fast iteration

    The proxy does NOT execute the agent locally.
    """
    run_id = _now_run_id()
    try:
        body = request.get_json(force=True, silent=False) or {}
        kill_running = body.get("kill_running", False)

        # Handle kill_running: signal the current eval to stop, then wait briefly
        # for the lock to become available.
        if kill_running:
            cancel_result = _cli_cancel()
            app.logger.info(f"cli cancel result: {cancel_result}")
            _force_release_eval()
            time.sleep(2)
            if not body.get("agent_file") and not os.environ.get("HARBOR_AGENT", "").strip():
                return jsonify({"success": True, "message": "kill signal sent", "cancel_result": cancel_result}), 200

        if "split" in body:
            return jsonify({"success": False, "error": "split is not supported; configure datasets via env vars"}), 400
        agent_file = body.get("agent_file")
        workspace_dir = body.get("workspace_dir")
        timeout = int(body.get("timeout", 21600))
        first_k = body.get("first_k")  # optional: limit to first K tasks (dev split only)
        if first_k is not None:
            try:
                first_k = int(first_k)
            except (TypeError, ValueError):
                return jsonify({"success": False, "error": "first_k must be an integer"}), 400
            if first_k <= 0:
                return jsonify({"success": False, "error": "first_k must be a positive integer"}), 400

        if not agent_file:
            # agent_file is only required when using --agent-import-path (custom agents).
            # Built-in agents (via HARBOR_AGENT) don't need an agent file.
            harbor_agent = os.environ.get("HARBOR_AGENT", "").strip()
            if not harbor_agent:
                return jsonify({"success": False, "error": "agent_file is required (or set HARBOR_AGENT for built-in agents)"}), 400

        if agent_file and not Path(agent_file).exists():
            return jsonify({"success": False, "error": f"agent_file not found: {agent_file}"}), 400

        if not workspace_dir:
            if agent_file:
                workspace_dir = str(Path(agent_file).resolve().parent)
            else:
                workspace_dir = "/workspace"

        ws_path = Path(workspace_dir)
        if not ws_path.exists() or not ws_path.is_dir():
            return jsonify({"success": False, "error": f"workspace_dir is not a directory: {workspace_dir}"}), 400

        dev_dataset = os.environ.get("HARBOR_DEV_DATASET_NAME", "").strip()
        test_dataset = os.environ.get("HARBOR_TEST_DATASET_NAME", "").strip()
        if not dev_dataset:
            return jsonify({"success": False, "error": "HARBOR_DEV_DATASET_NAME is not set"}), 500
        if not test_dataset:
            return jsonify({"success": False, "error": "HARBOR_TEST_DATASET_NAME is not set"}), 500

        # Proxy uses the dev dataset by default. When the verifier secret is
        # provided (test.sh, injected after the agent finishes), use the test
        # dataset instead.
        is_verifier = _check_verifier_secret(request)
        if is_verifier and first_k is not None:
            return jsonify({"success": False, "error": "first_k is not allowed on the test split"}), 403
        dataset_name = test_dataset if is_verifier else dev_dataset

        agent_import_path = os.environ.get("HARBOR_AGENT_IMPORT_PATH", "").strip() or "agent:TBAgent"
        jobs_dir = os.environ.get("HARBOR_JOBS_DIR", "").strip() or str(Path(workspace_dir) / "evaluation_logs")

        cli_cwd = os.environ.get("HARBOR_CLI_CWD", "").strip() or str(ws_path)

        # Serialize evaluate calls to avoid overlapping long remote runs.
        with _eval_busy(nonblocking=True), _periodic_network_prune(interval_sec=300):
            task_model_name = os.environ.get("TASK_MODEL_NAME", "").strip()
            harbor_agent = os.environ.get("HARBOR_AGENT", "").strip()

            # Built-in agent (--agent) takes precedence over custom import path.
            if harbor_agent:
                agent_args = ["--agent", harbor_agent]
            else:
                agent_args = ["--agent-import-path", agent_import_path]

            cli_args: list[str] = [
                "run",
                "-d",
                dataset_name,
                "--job-name",
                run_id,
                "--jobs-dir",
                jobs_dir,
                *agent_args,
                # Populate BaseAgent.model_name (Harbor passes this into the agent).
                # The proxy also forwards TASK_MODEL_NAME as an env var for agent code
                # that reads model selection from the environment.
                *(
                    ["-m", task_model_name]
                    if task_model_name
                    else []
                ),
                # Avoid interactive prompts (env var confirmations).
                "--yes",
                # Keep it deterministic and cheap unless caller overrides via env.
                "-k",
                os.environ.get("HARBOR_N_ATTEMPTS", "1").strip() or "1",
                "-n",
                os.environ.get("HARBOR_N_CONCURRENT", "1").strip() or "1",
                # Limit number of tasks via first_k (useful for fast iteration on a subset).
                # Rejected on the test split — see verifier check above.
                *(
                    ["-l", str(first_k)]
                    if first_k is not None
                    else []
                ),
                # Agent setup timeout multiplier (e.g. for slow tmux installation).
                *(
                    ["--agent-setup-timeout-multiplier", os.environ.get("HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER", "").strip()]
                    if os.environ.get("HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER", "").strip()
                    else []
                ),
            ]
            cli = _cli_exec(args=cli_args, cwd=cli_cwd, timeout_sec=float(timeout) + 600.0)

            # Resume failed trials to handle transient errors.
            job_path = f"{jobs_dir}/{run_id}"
            resume_error_filters = [
                "ImageNotFound", "APIError", "CancelledError",
                "VerifierTimeoutError", "RewardFileNotFoundError",
                "RuntimeError", "AddTestsDirError", "AgentSetupTimeoutError",
                "NotFoundError", "EnvironmentStartTimeoutError",
                "InternalServerError", "ReadTimeout",
                "DownloadVerifierDirError", "BuildError", "BadRequestError",
            ]
            resume_filter_args: list[str] = []
            for err in resume_error_filters:
                resume_filter_args.extend(["-f", err])

            n_resumes = int(os.environ.get("HARBOR_N_RESUMES", "1").strip() or "1")
            for i in range(n_resumes):
                resume_args = ["job", "resume", "-p", job_path, *resume_filter_args]
                app.logger.info(f"Resume attempt {i + 1}/{n_resumes}: {resume_args}")
                resume_cli = _cli_exec(args=resume_args, cwd=cli_cwd, timeout_sec=float(timeout) + 600.0)
                if _parse_job_metrics(str(resume_cli.get("stdout", "") or "")):
                    cli = resume_cli
                else:
                    app.logger.info(f"Resume {i + 1} produced no parseable metrics, keeping previous result")

        http_status = int(cli.get("_http_status", 500))
        rc = cli.get("returncode", None)
        stdout = str(cli.get("stdout", "") or "")
        stderr = str(cli.get("stderr", "") or "")
        rewards = _parse_rewards(stdout)
        metrics = _parse_job_metrics(stdout)

        remote: Dict[str, Any] = {
            "backend": "cli_service",
            "dataset": dataset_name,
            "cli": {
                "http_status": http_status,
                "returncode": rc,
                "args": cli_args,
                "stdout": stdout[:20000],
                "stderr": stderr[:20000],
            },
        }
        if rewards is not None:
            remote.update(rewards)
            remote["rewards"] = rewards
        if metrics:
            remote["metrics"] = metrics
            # Promote common metric keys to top-level for the verifier script.
            for k in ("resolved_rate", "pass_rate", "accuracy", "score", "mean"):
                if k in metrics and k not in remote:
                    remote[k] = metrics[k]
            # If metric labels were prefixed (e.g. "metric_1_resolved_rate"), still promote.
            for key, value in list(metrics.items()):
                if not isinstance(key, str):
                    continue
                lk = key.lower()
                for k in ("resolved_rate", "pass_rate", "accuracy", "score"):
                    if k in remote:
                        continue
                    if k in lk:
                        remote[k] = value
                        break

        # Best-effort scalar reward for the existing verifier script.
        for key in ("reward", "resolved_rate", "pass_rate", "score", "mean"):
            if key in remote:
                try:
                    remote["reward"] = float(remote[key])
                    break
                except Exception:
                    pass

        ok = (http_status >= 200 and http_status < 300) and (rc == 0)
        return jsonify({"success": bool(ok), "http_status": http_status, "run_id": run_id, "remote": remote}), (
            200 if ok else 502
        )

    except BlockingIOError:
        return jsonify({"success": False, "error": "another eval is running"}), 409
    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "run_id": run_id,
                }
            ),
            500,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TB Meta-Agent Evaluation Proxy API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    def _check_proxy_reachability():
        time.sleep(5)
        host_ip = os.environ.get("HOST_IP", "127.0.0.1")
        proxy_port = os.environ.get("EVALUATION_API_HOST_PORT", "8080")
        proxy_url = f"http://{host_ip}:{proxy_port}/health"
        try:
            r = requests.get(proxy_url, timeout=5)
            if r.status_code == 200:
                app.logger.info(f"[PROXY REACHABILITY] OK - {proxy_url} is reachable from host network")
            else:
                app.logger.error(f"[PROXY REACHABILITY] FAIL - {proxy_url} returned {r.status_code}")
        except Exception as e:
            app.logger.error(f"[PROXY REACHABILITY] FAIL - {proxy_url} unreachable: {e}")
            app.logger.error("[PROXY REACHABILITY] Inner harbor containers will NOT be able to reach the model proxy!")

    threading.Thread(target=_check_proxy_reachability, daemon=True).start()

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
