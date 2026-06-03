#!/usr/bin/env python3
"""
Oracle Agent - Golden Answer Provider

Supports multiple modes via ORACLE_MODE env var:
  full       - Answer all problems (100% baseline, default)
  first_50   - Answer only first 50% of problems by idx
  last_50    - Answer only last 50% of problems by idx
  random_50  - Answer a random 50% of problems (seeded by ORACLE_SEED)
"""

import json
import hashlib
import base64
import os
import random
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, "/app/tools")

from base_agent import BaseAIMEAgent, Problem, Prediction

_VERIFIER_SECRET = 'aime-verifier-secret-do-not-share'


def _decrypt_file(encrypted_path: str, secret: str) -> bytes:
    from cryptography.fernet import Fernet
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key).decrypt(Path(encrypted_path).read_bytes())


class OracleAgent(BaseAIMEAgent):
    """
    Oracle agent that uses golden answers.

    ORACLE_MODE controls which problems are answered:
      full      - all problems (default)
      first_50  - first half by sorted idx
      last_50   - last half by sorted idx
      random_50 - random half (seed via ORACLE_SEED env var, default 42)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = os.environ.get('ORACLE_MODE', 'full')
        self.seed = int(os.environ.get('ORACLE_SEED', '42'))
        self.golden_answers = {}
        self._load_golden_answers()

    def _load_golden_answers(self):
        enc_path = Path('/app/data/aime_2024_2025.jsonl.enc')
        plain_path = Path('/app/data/aime_2024_2025.jsonl')

        if enc_path.exists():
            raw = _decrypt_file(str(enc_path), _VERIFIER_SECRET).decode('utf-8')
        elif plain_path.exists():
            raw = plain_path.read_text()
        else:
            raise FileNotFoundError(f"Ground truth not found at {enc_path} or {plain_path}")

        for line in raw.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            idx = item.get('id')
            answer = item.get('answer')
            if idx is not None and answer is not None:
                self.golden_answers[idx] = str(answer)

        print(f"OracleAgent: loaded {len(self.golden_answers)} golden answers, mode={self.mode}")

    def _select_problems(self, sorted_problems: List[Problem]) -> List[Problem]:
        n = len(sorted_problems)
        half = n // 2

        if self.mode == 'full':
            return sorted_problems
        elif self.mode == 'first_50':
            return sorted_problems[:half]
        elif self.mode == 'last_50':
            return sorted_problems[half:]
        elif self.mode == 'random_50':
            rng = random.Random(self.seed)
            selected = rng.sample(sorted_problems, half)
            return sorted(selected, key=lambda p: p.idx)
        else:
            raise ValueError(f"Unknown ORACLE_MODE: {self.mode!r}. "
                             f"Valid modes: full, first_50, last_50, random_50")

    def solve(self, problems: List[Problem], timeout_sec: int = 21600) -> List[Prediction]:
        sorted_problems = sorted(problems, key=lambda p: p.idx)
        selected = self._select_problems(sorted_problems)

        predictions = []
        for problem in selected:
            if problem.idx in self.golden_answers:
                predictions.append(Prediction(idx=problem.idx, pred=self.golden_answers[problem.idx]))
            else:
                print(f"Warning: No golden answer for problem idx={problem.idx}")

        print(f"OracleAgent: {len(predictions)}/{len(problems)} predictions submitted")
        return sorted(predictions, key=lambda p: p.idx)

    def __repr__(self):
        return f"OracleAgent(mode={self.mode}, answers={len(self.golden_answers)})"
