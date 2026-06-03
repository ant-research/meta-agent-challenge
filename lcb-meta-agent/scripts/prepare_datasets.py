#!/usr/bin/env python3
"""Prepare LiveCodeBench datasets for lcb-meta-agent.

Outputs in `data/`:
- `lcb_eval.jsonl` / `lcb_test.jsonl`: agent-visible inputs (no private tests)
- `lcb_eval_full.jsonl` / `lcb_test_full.jsonl`: evaluator-only full records
- `split_summary.json`: split metadata
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Record:
    source_file: str
    contest_date: date
    raw: Dict[str, Any]
    fn_name: Optional[str]
    original_idx: str


def parse_original_idx(raw_obj: Dict[str, Any]) -> str:
    """Use upstream question_id as the canonical problem id."""
    question_id = raw_obj.get("question_id")
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError(f"Missing valid question_id in source record: {raw_obj}")
    return question_id


def parse_metadata_fn_name(raw_metadata: Any) -> Optional[str]:
    if isinstance(raw_metadata, dict):
        return raw_metadata.get("func_name")
    if isinstance(raw_metadata, str):
        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                return parsed.get("func_name")
        except Exception:
            return None
    return None


def load_raw_records(source_dir: Path) -> List[Record]:
    files = sorted(source_dir.glob("test*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No test*.jsonl found in: {source_dir}")

    records: List[Record] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                d = datetime.fromisoformat(obj["contest_date"]).date()
                fn_name = parse_metadata_fn_name(obj.get("metadata", "{}"))
                original_idx = parse_original_idx(obj)
                records.append(
                    Record(
                        source_file=path.name,
                        contest_date=d,
                        raw=obj,
                        fn_name=fn_name,
                        original_idx=original_idx,
                    )
                )

    records.sort(key=lambda x: (x.contest_date, x.source_file, x.raw.get("question_id", "")))
    return records


def split_records_by_date_window(
    records: List[Record],
    start_date: date,
    end_date: date,
) -> Tuple[List[Record], List[Record]]:
    # Test uses strict window: start_date < contest_date < end_date
    test_records = [r for r in records if start_date < r.contest_date < end_date]
    # Eval uses all remaining records (including boundaries and outside the window)
    eval_records = [r for r in records if not (start_date < r.contest_date < end_date)]
    return eval_records, test_records


def choose_cutoff_by_ratio(records: List[Record], test_ratio: float) -> date:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")

    by_date = defaultdict(list)
    for rec in records:
        by_date[rec.contest_date].append(rec)

    ordered_dates = sorted(by_date.keys())
    target_eval = int(round(len(records) * (1.0 - test_ratio)))

    if target_eval <= 0:
        return ordered_dates[0]
    if target_eval >= len(records):
        return ordered_dates[-1]

    current = 0
    cutoff = ordered_dates[-1]
    for d in ordered_dates:
        next_count = current + len(by_date[d])
        if next_count >= target_eval:
            cutoff = d
            break
        current = next_count
    return cutoff


def to_agent_visible(record: Record) -> Dict[str, Any]:
    raw = record.raw
    return {
        "idx": record.original_idx,
        "question_title": raw.get("question_title", ""),
        "question_content": raw.get("question_content", ""),
        "platform": raw.get("platform", ""),
        "question_id": raw.get("question_id", ""),
        "contest_id": raw.get("contest_id", ""),
        "contest_date": raw.get("contest_date", ""),
        "starter_code": raw.get("starter_code", ""),
        "difficulty": raw.get("difficulty", ""),
        "fn_name": record.fn_name,
    }


def to_full_record(record: Record) -> Dict[str, Any]:
    raw = dict(record.raw)
    ordered = {
        "idx": record.original_idx,
        "question_title": raw.get("question_title", ""),
        "question_content": raw.get("question_content", ""),
        "platform": raw.get("platform", ""),
        "question_id": raw.get("question_id", ""),
        "contest_id": raw.get("contest_id", ""),
        "contest_date": raw.get("contest_date", ""),
        "starter_code": raw.get("starter_code", ""),
        "difficulty": raw.get("difficulty", ""),
        "public_test_cases": raw.get("public_test_cases", "[]"),
        "private_test_cases": raw.get("private_test_cases", "[]"),
        "metadata": raw.get("metadata", "{}"),
        "fn_name": record.fn_name,
    }
    return ordered


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_split(name: str, records: List[Record]) -> Dict[str, Any]:
    with_fn = sum(1 for r in records if r.fn_name)
    without_fn = len(records) - with_fn
    platforms = defaultdict(int)
    for r in records:
        platforms[r.raw.get("platform", "unknown")] += 1

    return {
        "name": name,
        "count": len(records),
        "with_fn_name": with_fn,
        "without_fn_name": without_fn,
        "min_date": min((r.contest_date for r in records), default=None),
        "max_date": max((r.contest_date for r in records), default=None),
        "platform_counts": dict(sorted(platforms.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare LiveCodeBench split files for lcb-meta-agent")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="/mnt/data2/wangpengbo2025/livecodebench/code_generation_lite",
        help="Directory containing test*.jsonl from code_generation_lite",
    )
    args = parser.parse_args()

    task_dir = Path(__file__).resolve().parent.parent
    data_dir = task_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.source_dir)
    print("=" * 70)
    print("Preparing LiveCodeBench datasets")
    print("=" * 70)
    print(f"Source dir: {source_dir}")

    records = load_raw_records(source_dir)
    print(f"Loaded {len(records)} records")
    unique_idx_count = len({r.original_idx for r in records})
    if unique_idx_count != len(records):
        raise ValueError(
            "Duplicate source idx detected. "
            "Please ensure source question_id values are unique."
        )

    test_start_date = date(2024, 8, 1)
    test_end_date = date(2025, 2, 1)
    eval_records, test_records = split_records_by_date_window(
        records=records,
        start_date=test_start_date,
        end_date=test_end_date,
    )
    if not eval_records or not test_records:
        raise ValueError(
            "Split produced an empty eval or test set. "
            "Check source data and date-window configuration."
        )

    eval_public = [to_agent_visible(rec) for rec in eval_records]
    test_public = [to_agent_visible(rec) for rec in test_records]
    eval_full = [to_full_record(rec) for rec in eval_records]
    test_full = [to_full_record(rec) for rec in test_records]

    write_jsonl(data_dir / "lcb_eval.jsonl", eval_public)
    write_jsonl(data_dir / "lcb_test.jsonl", test_public)
    write_jsonl(data_dir / "lcb_eval_full.jsonl", eval_full)
    write_jsonl(data_dir / "lcb_test_full.jsonl", test_full)

    summary = {
        "source_dir": str(source_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_start_date": test_start_date.isoformat(),
        "test_end_date": test_end_date.isoformat(),
        "split_rule": "test if test_start_date < contest_date < test_end_date; else eval",
        "total": len(records),
        "eval": summarize_split("eval", eval_records),
        "test": summarize_split("test", test_records),
    }
    with open(data_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print()
    print("✓ Generated:")
    print(f"  - {data_dir / 'lcb_eval.jsonl'}")
    print(f"  - {data_dir / 'lcb_test.jsonl'}")
    print(f"  - {data_dir / 'lcb_eval_full.jsonl'}")
    print(f"  - {data_dir / 'lcb_test_full.jsonl'}")
    print(f"  - {data_dir / 'split_summary.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
