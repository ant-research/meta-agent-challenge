#!/usr/bin/env python3
"""
Data preprocessing script for science-meta-agent.

Downloads GPQA Diamond and HLE datasets from HuggingFace and produces
four JSONL files used by the evaluation pipeline:

  gpqa_full.jsonl   - GPQA Diamond (198 questions) with answers   -> evaluation-api only
  gpqa_eval.jsonl   - GPQA Diamond without answers                -> agent visible (eval split)
  hle_mc_full.jsonl - HLE multiple-choice (591 questions) with answers -> evaluation-api only
  hle_mc_test.jsonl - HLE multiple-choice without answers         -> agent visible (test split)

Usage:
    pip install datasets
    python preprocess.py            # writes to current directory
    python preprocess.py /out/dir   # writes to specified directory

Source datasets (HuggingFace):
    - Idavidrein/gpqa  (config: gpqa_diamond, split: train)
    - cais/hle         (split: test, filtered to answer_type == "multipleChoice")
"""

import json
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

SEED = 42


def process_gpqa(out_dir: Path) -> None:
    """
    Download GPQA Diamond and write gpqa_full.jsonl / gpqa_eval.jsonl.

    Transformation:
        - Collect correct + 3 incorrect answers
        - Shuffle with fixed seed (per-question) to avoid answer-position bias
        - Assign A/B/C/D labels; record which letter is correct
    """
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    print(f"GPQA Diamond: {len(ds)} questions loaded")

    full_records = []
    eval_records = []
    labels = "ABCD"

    for idx, row in enumerate(ds):
        question = row["Question"]
        correct = row["Correct Answer"].strip()
        raw_choices = [
            correct,
            row["Incorrect Answer 1"].strip(),
            row["Incorrect Answer 2"].strip(),
            row["Incorrect Answer 3"].strip(),
        ]

        # Deterministic per-question shuffle
        rng = random.Random(SEED + idx)
        rng.shuffle(raw_choices)

        choices = [f"{labels[i]}) {text}" for i, text in enumerate(raw_choices)]
        answer = labels[raw_choices.index(correct)]

        full_records.append({
            "idx": idx,
            "question": question,
            "choices": choices,
            "answer": answer,
        })
        eval_records.append({
            "idx": idx,
            "question": question,
            "choices": choices,
        })

    _write_jsonl(out_dir / "gpqa_full.jsonl", full_records)
    _write_jsonl(out_dir / "gpqa_eval.jsonl", eval_records)
    print(f"  -> gpqa_full.jsonl  ({len(full_records)} records)")
    print(f"  -> gpqa_eval.jsonl  ({len(eval_records)} records)")


def process_hle(out_dir: Path) -> None:
    """
    Download HLE and write hle_mc_full.jsonl / hle_mc_test.jsonl.

    Filters to answer_type == "multipleChoice" only.

    Transformation:
        - Split question text on "Answer Choices:\\n" separator
        - Question stem -> question
        - Parse "A. text" lines -> choices as "A) text"
        - answer stays as-is (single letter)
    """
    ds = load_dataset("cais/hle", split="test")
    mc_rows = [row for row in ds if row["answer_type"] == "multipleChoice"]
    print(f"HLE: {len(ds)} total, {len(mc_rows)} multiple-choice")

    full_records = []
    test_records = []

    for idx, row in enumerate(mc_rows):
        question_text = row["question"]

        # Split on "Answer Choices:\n"
        parts = question_text.split("Answer Choices:\n")
        stem = parts[0].strip()
        choices_text = parts[1].strip() if len(parts) > 1 else ""

        # Parse "A. choice text" lines
        parsed = re.findall(r"^([A-Z])\.\s*(.+?)$", choices_text, re.MULTILINE)
        choices = [f"{letter}) {text}" for letter, text in parsed]

        full_records.append({
            "idx": idx,
            "question": stem,
            "choices": choices,
            "answer": row["answer"],
        })
        test_records.append({
            "idx": idx,
            "question": stem,
            "choices": choices,
        })

    _write_jsonl(out_dir / "hle_mc_full.jsonl", full_records)
    _write_jsonl(out_dir / "hle_mc_test.jsonl", test_records)
    print(f"  -> hle_mc_full.jsonl  ({len(full_records)} records)")
    print(f"  -> hle_mc_test.jsonl  ({len(test_records)} records)")


def _write_jsonl(path: Path, records: list) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir.resolve()}\n")

    process_gpqa(out_dir)
    print()
    process_hle(out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
