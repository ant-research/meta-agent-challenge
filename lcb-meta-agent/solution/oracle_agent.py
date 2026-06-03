#!/usr/bin/env python3
"""Oracle agent for LiveCodeBench test split.

The agent returns precomputed correct code from the hidden test ground truth.
It prefers the already-decrypted full test dataset path injected by evaluation_api
so oracle execution does not decrypt a second multi-GB copy unnecessarily.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import hashlib
import json
import os
import struct
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, "/app/tools")

from base_agent import BaseLCBAgent, Prediction, Problem

_VERIFIER_SECRET = "lcb-verifier-secret-do-not-share"
_CHUNKED_FERNET_MAGIC = b"LCBFERNETv1\n"
_ORACLE_FILE_ENV = "LCB_ORACLE_TEST_FULL_FILE"


def _build_fernet(secret: str):
    from cryptography.fernet import Fernet

    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key)


def _decrypt_file_to_path(encrypted_path: str, secret: str, output_path: str) -> None:
    fernet = _build_fernet(secret)
    with open(encrypted_path, "rb") as src:
        prefix = src.read(len(_CHUNKED_FERNET_MAGIC))
        if prefix == _CHUNKED_FERNET_MAGIC:
            with open(output_path, "wb") as dst:
                while True:
                    length_bytes = src.read(4)
                    if not length_bytes:
                        break
                    if len(length_bytes) != 4:
                        raise ValueError(f"Corrupted encrypted file: {encrypted_path}")
                    token_length = struct.unpack(">I", length_bytes)[0]
                    if token_length <= 0:
                        raise ValueError(
                            f"Invalid encrypted chunk length in: {encrypted_path}"
                        )
                    token = src.read(token_length)
                    if len(token) != token_length:
                        raise ValueError(f"Truncated encrypted chunk in: {encrypted_path}")
                    dst.write(fernet.decrypt(token))
            return

    with open(encrypted_path, "rb") as src, open(output_path, "wb") as dst:
        dst.write(fernet.decrypt(src.read()))


class OraclePredictions(Iterable[Prediction]):
    def __init__(self, ordered_ids: list[str], data_path: Path, offsets: dict[str, int]):
        self.ordered_ids = ordered_ids
        self.data_path = data_path
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.ordered_ids)

    def __iter__(self) -> Iterator[Prediction]:
        with open(self.data_path, "r", encoding="utf-8") as src:
            for idx in self.ordered_ids:
                if idx not in self.offsets:
                    raise ValueError(f"Missing oracle code for idx={idx}")
                src.seek(self.offsets[idx])
                row = json.loads(src.readline())
                code = row.get("gt")
                if not isinstance(code, str) or not code.strip():
                    raise ValueError(f"Invalid oracle code for idx={idx}")
                yield Prediction(idx=idx, pred=code)


class OracleAgent(BaseLCBAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._temp_oracle_file: Path | None = None
        self.oracle_file = self._resolve_oracle_file()
        atexit.register(self._cleanup)

    def _resolve_oracle_file(self) -> Path:
        injected_path = os.environ.get(_ORACLE_FILE_ENV)
        if injected_path:
            candidate = Path(injected_path)
            if candidate.exists():
                print(f"Using injected oracle dataset: {candidate}")
                return candidate

        plaintext_path = Path("/app/data/lcb_test_full.jsonl")
        if plaintext_path.exists():
            print(f"Using plaintext oracle dataset: {plaintext_path}")
            return plaintext_path

        encrypted_path = Path("/app/data/lcb_test_full.jsonl.enc")
        if not encrypted_path.exists():
            raise FileNotFoundError(
                f"Oracle data not found. Checked {plaintext_path} and {encrypted_path}"
            )

        tmp = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".jsonl",
            prefix="lcb_oracle_test_full_",
            dir="/tmp",
            delete=False,
        )
        tmp.close()
        _decrypt_file_to_path(str(encrypted_path), _VERIFIER_SECRET, tmp.name)
        self._temp_oracle_file = Path(tmp.name)
        print(f"Decrypted oracle dataset to {self._temp_oracle_file}")
        return self._temp_oracle_file

    def _cleanup(self) -> None:
        if self._temp_oracle_file is not None:
            self._temp_oracle_file.unlink(missing_ok=True)

    def _build_offsets(self, target_ids: set[str]) -> dict[str, int]:
        offsets: dict[str, int] = {}
        with open(self.oracle_file, "r", encoding="utf-8") as src:
            while True:
                offset = src.tell()
                line = src.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                row_idx = str(row.get("idx", ""))
                if row_idx in target_ids:
                    offsets[row_idx] = offset
                    if len(offsets) == len(target_ids):
                        break
        return offsets

    def solve(
        self,
        problems: list[Problem],
        timeout_sec: int = 21600,
    ) -> OraclePredictions:
        self._start_timer(timeout_sec)
        ordered_ids = [str(problem.idx) for problem in problems]
        offsets = self._build_offsets(set(ordered_ids))
        missing = [idx for idx in ordered_ids if idx not in offsets]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Missing oracle code for {len(missing)} problems. Sample ids: {preview}"
            )

        print(f"Oracle Agent: {len(ordered_ids)} problems, {len(offsets)} oracle codes")
        return OraclePredictions(ordered_ids, self.oracle_file, offsets)

    def __repr__(self) -> str:
        return f"OracleAgent(oracle_file={self.oracle_file})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LCB oracle agent")
    parser.add_argument("--input", required=True, help="Input JSONL problems")
    parser.add_argument("--output", required=True, help="Output JSONL predictions")
    parser.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args()

    problems = BaseLCBAgent.load_problems(args.input)
    agent = OracleAgent()
    predictions = agent.solve(problems, timeout_sec=args.timeout)
    BaseLCBAgent.save_predictions(predictions, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
