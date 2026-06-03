"""
Harbor CLI HTTP service (single-file).

This exposes a thin FastAPI wrapper around Harbor's Typer CLI by executing
`python -m harbor.cli.main ...` in a managed subprocess.

Security note: this service can execute Harbor CLI commands. Do not expose it
to untrusted networks without an auth layer / network ACL.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CWD = _REPO_ROOT
_LOG_ROOT = Path(os.environ.get("HARBOR_CLI_SERVICE_LOG_DIR", "/tmp/harbor-cli-service"))


def _base_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)

    # Ensure `import harbor` works when running from a source checkout.
    src_dir = (_REPO_ROOT / "src").as_posix()
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    if src_dir not in parts:
        parts.insert(0, src_dir)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    if extra_env:
        env.update(extra_env)
    return env


def _cli_argv(args: list[str]) -> list[str]:
    # Fixed entrypoint: only Harbor CLI is callable.
    return [sys.executable, "-m", "harbor.cli.main", *args]


def _ensure_log_root() -> None:
    _LOG_ROOT.mkdir(parents=True, exist_ok=True)


def _kill_process_tree(pid: int, sig: signal.Signals) -> None:
    """
    Send a signal to the whole process group when possible.

    We start subprocesses with `start_new_session=True` so on Unix we can signal
    the process group safely. If that fails, fallback to signaling the pid.
    """
    try:
        os.killpg(pid, sig)
    except Exception:
        try:
            os.kill(pid, sig)
        except Exception:
            return


@dataclass
class _ManagedProc:
    id: str
    args: list[str]
    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    started_at_s: float
    log_path: Path
    proc: subprocess.Popen[bytes]
    finished_at_s: float | None = None

    def poll(self) -> int | None:
        rc = self.proc.poll()
        if rc is not None and self.finished_at_s is None:
            self.finished_at_s = time.time()
        return rc


_procs: dict[str, _ManagedProc] = {}
_procs_lock = asyncio.Lock()

_exec_proc: subprocess.Popen[bytes] | None = None
_exec_proc_lock = asyncio.Lock()


class ShellExecRequest(BaseModel):
    cmd: list[str] = Field(..., description="Command + args, e.g. ['docker','network','prune','-f']")
    env: dict[str, str] = Field(default_factory=dict, description="Extra env vars")
    cwd: str | None = Field(default=None, description="Working directory; defaults to repo root")
    timeout_sec: float = Field(default=60.0, ge=1.0, le=300.0, description="Timeout in seconds")


class ExecRequest(BaseModel):
    args: list[str] = Field(..., description="Harbor CLI args, e.g. ['datasets','list']")
    env: dict[str, str] = Field(default_factory=dict, description="Extra env vars")
    cwd: str | None = Field(
        default=None, description="Working directory; defaults to repo root"
    )
    stdin: str | None = Field(
        default=None,
        description="Optional stdin payload (best-effort; useful for simple prompts).",
    )
    timeout_sec: float | None = Field(
        default=600.0,
        description="Timeout for synchronous exec; null means no timeout",
    )


class StartRequest(BaseModel):
    args: list[str] = Field(..., description="Harbor CLI args, e.g. ['run', ...]")
    env: dict[str, str] = Field(default_factory=dict, description="Extra env vars")
    cwd: str | None = Field(
        default=None, description="Working directory; defaults to repo root"
    )
    stdin: str | None = Field(
        default=None,
        description="Optional stdin payload written once then closed (best-effort).",
    )


class ExecResponse(BaseModel):
    command: list[str]
    cwd: str
    returncode: int
    stdout: str
    stderr: str


class StartResponse(BaseModel):
    id: str
    command: list[str]
    cwd: str
    pid: int
    started_at_s: float
    log_path: str


class ProcStatus(BaseModel):
    id: str
    command: list[str]
    cwd: str
    pid: int
    started_at_s: float
    finished_at_s: float | None
    returncode: int | None
    log_path: str


class LogsResponse(BaseModel):
    id: str
    log_path: str
    offset: int
    next_offset: int
    eof: bool
    data: str


def create_app() -> FastAPI:
    app = FastAPI(
        title="Harbor CLI Service",
        description="HTTP wrapper for Harbor CLI commands",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/cli/exec", response_model=ExecResponse)
    async def exec_cli(req: ExecRequest) -> ExecResponse:
        global _exec_proc
        cmd = _cli_argv(req.args)
        cwd = Path(req.cwd).expanduser().resolve() if req.cwd else _DEFAULT_CWD
        env = _base_env(req.env)

        def _run() -> subprocess.CompletedProcess[bytes]:
            global _exec_proc
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if req.stdin is not None else subprocess.DEVNULL,
                start_new_session=True,
            )
            _exec_proc = proc
            try:
                stdin_data = req.stdin.encode("utf-8") if req.stdin is not None else None
                stdout, stderr = proc.communicate(input=stdin_data, timeout=req.timeout_sec)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
                raise
            finally:
                _exec_proc = None
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

        try:
            cp = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="Command timed out")
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))

        return ExecResponse(
            command=cmd,
            cwd=str(cwd),
            returncode=int(cp.returncode),
            stdout=cp.stdout.decode("utf-8", errors="replace"),
            stderr=cp.stderr.decode("utf-8", errors="replace"),
        )

    @app.post("/v1/cli/cancel")
    async def cancel_exec(
        grace_sec: float = Query(default=10.0, ge=0.0, le=60.0),
    ) -> dict[str, Any]:
        """Cancel the currently running /v1/cli/exec subprocess."""
        proc = _exec_proc
        if proc is None:
            return {"status": "no_exec_running"}

        if proc.poll() is not None:
            return {"status": "already_exited", "returncode": proc.returncode}

        _kill_process_tree(proc.pid, signal.SIGTERM)

        def _wait_then_kill() -> int | None:
            try:
                proc.wait(timeout=grace_sec)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            return proc.poll()

        rc = await asyncio.to_thread(_wait_then_kill)

        async def _docker_cleanup() -> dict[str, str]:
            results = {}
            for cmd, label in [
                (["docker", "system", "prune", "-a", "-f"], "system_prune"),
                (["docker", "network", "prune", "-f"], "network_prune"),
            ]:
                try:
                    cp = await asyncio.to_thread(
                        subprocess.run, cmd,
                        capture_output=True, timeout=60, check=False,
                    )
                    results[label] = cp.stdout.decode("utf-8", errors="replace").strip()[:500]
                except Exception as e:
                    results[label] = f"error: {e}"
            return results

        cleanup = await _docker_cleanup()
        return {"status": "cancelled", "returncode": rc, "cleanup": cleanup}

    _SHELL_ALLOWLIST = {
        ("docker", "network", "prune", "-f"),
    }

    @app.post("/v1/shell/exec", response_model=ExecResponse)
    async def shell_exec(req: ShellExecRequest) -> ExecResponse:
        """Run an allowlisted host command (not routed through Harbor CLI)."""
        if tuple(req.cmd) not in _SHELL_ALLOWLIST:
            raise HTTPException(status_code=403, detail=f"Command not in allowlist: {req.cmd}")

        cwd = Path(req.cwd).expanduser().resolve() if req.cwd else _DEFAULT_CWD
        env = _base_env(req.env)

        def _run() -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                req.cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=req.timeout_sec,
                check=False,
            )

        try:
            cp = await asyncio.to_thread(_run)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=408, detail="Command timed out")
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))

        return ExecResponse(
            command=req.cmd,
            cwd=str(cwd),
            returncode=int(cp.returncode),
            stdout=cp.stdout.decode("utf-8", errors="replace"),
            stderr=cp.stderr.decode("utf-8", errors="replace"),
        )

    @app.post("/v1/cli/start", response_model=StartResponse)
    async def start_cli(req: StartRequest) -> StartResponse:
        _ensure_log_root()

        cmd = _cli_argv(req.args)
        cwd = Path(req.cwd).expanduser().resolve() if req.cwd else _DEFAULT_CWD
        env = _base_env(req.env)

        proc_id = uuid4().hex
        log_path = _LOG_ROOT / f"{proc_id}.log"
        started_at_s = time.time()

        try:
            # Note: open in binary mode; we do our own decoding on read.
            log_f = log_path.open("ab", buffering=0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to open log: {e}")

        try:
            popen = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdin=subprocess.PIPE if req.stdin is not None else subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as e:
            try:
                log_f.close()
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to start process: {e}")

        if req.stdin is not None and popen.stdin is not None:
            try:
                popen.stdin.write(req.stdin.encode("utf-8"))
                popen.stdin.flush()
            except Exception:
                pass
            finally:
                try:
                    popen.stdin.close()
                except Exception:
                    pass

        # Close our copy of the FD; child keeps it.
        try:
            log_f.close()
        except Exception:
            pass

        managed = _ManagedProc(
            id=proc_id,
            args=req.args,
            cmd=cmd,
            cwd=cwd,
            env=env,
            started_at_s=started_at_s,
            log_path=log_path,
            proc=popen,
        )

        async with _procs_lock:
            _procs[proc_id] = managed

        return StartResponse(
            id=proc_id,
            command=cmd,
            cwd=str(cwd),
            pid=int(popen.pid),
            started_at_s=started_at_s,
            log_path=str(log_path),
        )

    @app.get("/v1/cli/procs", response_model=list[ProcStatus])
    async def list_procs() -> list[ProcStatus]:
        async with _procs_lock:
            procs = list(_procs.values())

        statuses: list[ProcStatus] = []
        for mp in procs:
            rc = mp.poll()
            statuses.append(
                ProcStatus(
                    id=mp.id,
                    command=mp.cmd,
                    cwd=str(mp.cwd),
                    pid=int(mp.proc.pid),
                    started_at_s=mp.started_at_s,
                    finished_at_s=mp.finished_at_s,
                    returncode=rc,
                    log_path=str(mp.log_path),
                )
            )
        return statuses

    @app.get("/v1/cli/procs/{proc_id}", response_model=ProcStatus)
    async def get_proc(proc_id: str) -> ProcStatus:
        async with _procs_lock:
            mp = _procs.get(proc_id)
        if mp is None:
            raise HTTPException(status_code=404, detail="Unknown proc id")

        rc = mp.poll()
        return ProcStatus(
            id=mp.id,
            command=mp.cmd,
            cwd=str(mp.cwd),
            pid=int(mp.proc.pid),
            started_at_s=mp.started_at_s,
            finished_at_s=mp.finished_at_s,
            returncode=rc,
            log_path=str(mp.log_path),
        )

    @app.post("/v1/cli/procs/{proc_id}/stop")
    async def stop_proc(
        proc_id: str,
        sig: str = Query(
            default="int",
            description="Signal to send first: 'int' (SIGINT) or 'term' (SIGTERM).",
        ),
        grace_sec: float = Query(default=10.0, ge=0.0, le=300.0),
        kill: bool = Query(default=True),
    ) -> dict[str, Any]:
        async with _procs_lock:
            mp = _procs.get(proc_id)
        if mp is None:
            raise HTTPException(status_code=404, detail="Unknown proc id")

        if mp.poll() is not None:
            return {"status": "already_exited", "returncode": mp.proc.returncode}

        first_sig = signal.SIGINT if sig.lower() == "int" else signal.SIGTERM
        _kill_process_tree(mp.proc.pid, first_sig)

        try:
            await asyncio.to_thread(mp.proc.wait, timeout=grace_sec)
        except subprocess.TimeoutExpired:
            if not kill:
                return {"status": "term_sent", "returncode": None}
            _kill_process_tree(mp.proc.pid, signal.SIGKILL)
            try:
                await asyncio.to_thread(mp.proc.wait, timeout=10.0)
            except subprocess.TimeoutExpired:
                return {"status": "kill_sent", "returncode": None}

        mp.poll()
        return {"status": "stopped", "returncode": mp.proc.returncode}

    @app.get("/v1/cli/procs/{proc_id}/logs", response_model=LogsResponse)
    async def read_logs(
        proc_id: str,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=65536, ge=1, le=1024 * 1024),
    ) -> LogsResponse:
        async with _procs_lock:
            mp = _procs.get(proc_id)
        if mp is None:
            raise HTTPException(status_code=404, detail="Unknown proc id")

        if not mp.log_path.exists():
            raise HTTPException(status_code=404, detail="Log file not found")

        def _read() -> tuple[bytes, int, bool]:
            size = mp.log_path.stat().st_size
            if offset >= size:
                return b"", offset, True
            with mp.log_path.open("rb") as f:
                f.seek(offset)
                chunk = f.read(limit)
            next_off = offset + len(chunk)
            eof = next_off >= size
            return chunk, next_off, eof

        chunk, next_off, eof = await asyncio.to_thread(_read)
        return LogsResponse(
            id=mp.id,
            log_path=str(mp.log_path),
            offset=offset,
            next_offset=next_off,
            eof=eof,
            data=chunk.decode("utf-8", errors="replace"),
        )

    return app


app = create_app()


def main() -> None:
    """
    Convenience entrypoint:
      python -m harbor.cli_service
    """
    import uvicorn

    host = os.environ.get("HARBOR_CLI_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("HARBOR_CLI_SERVICE_PORT", "8899"))
    uvicorn.run("harbor.cli_service:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
