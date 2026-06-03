#!/usr/bin/env python3
"""Quick sanity checks for lcb-meta-agent data + evaluation stack."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    task_dir = Path(__file__).resolve().parent.parent
    data_dir = task_dir / "data"

    eval_public = data_dir / "lcb_eval.jsonl"
    test_public = data_dir / "lcb_test.jsonl"
    eval_full = data_dir / "lcb_eval_full.jsonl"
    test_full = data_dir / "lcb_test_full.jsonl"

    required = [eval_public, test_public, eval_full, test_full]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("Missing files:")
        for m in missing:
            print(f"  - {m}")
        print("Run scripts/prepare_datasets.py first.")
        return 1

    for split_name, pub_path, full_path in [
        ("eval", eval_public, eval_full),
        ("test", test_public, test_full),
    ]:
        pub_rows = list(load_jsonl(pub_path))
        full_rows = list(load_jsonl(full_path))

        print(f"[{split_name}] public={len(pub_rows)} full={len(full_rows)}")
        if len(pub_rows) != len(full_rows):
            print(f"❌ Count mismatch in split {split_name}")
            return 1

        fn_counter = Counter(row.get("fn_name") is not None for row in pub_rows)
        print(
            f"[{split_name}] with_fn_name={fn_counter[True]} without_fn_name={fn_counter[False]}"
        )

        for row in pub_rows[:3]:
            assert "idx" in row
            assert "question_content" in row
            assert "starter_code" in row

        for row in full_rows[:3]:
            assert "public_test_cases" in row
            assert "private_test_cases" in row

    print("✅ Basic evaluation flow checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())