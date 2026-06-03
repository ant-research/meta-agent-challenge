#!/usr/bin/env python3
"""
Prepare sanitized AIME datasets (without answers) for agent evaluation.
The evaluation backend will automatically add golden answers when scoring.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def sanitize_dataset(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove all answer-related fields from dataset.

    Keeps only: idx, question
    Removes: solution, answer, gt, url, etc.
    """
    sanitized = []
    for item in data:
        sanitized_item = {}

        # Normalize ID field
        if 'id' in item:
            sanitized_item['idx'] = item['id']
        elif 'idx' in item:
            sanitized_item['idx'] = item['idx']

        # Normalize question field
        if 'problem' in item:
            sanitized_item['question'] = item['problem']
        elif 'question' in item:
            sanitized_item['question'] = item['question']

        sanitized.append(sanitized_item)

    return sanitized


def save_jsonl(data: List[Dict[str, Any]], filepath: Path):
    """Save data to JSONL file."""
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(data)} items to {filepath}")


def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    tests_dir = base_dir / "tests"

    print("=" * 60)
    print("Preparing Sanitized AIME Datasets (No Answers)")
    print("=" * 60)
    print()

    # 1. Process development set
    print("1. Processing development set...")
    dev_file = data_dir / "aime_2022_2023.jsonl"

    if not dev_file.exists():
        print(f"ERROR: {dev_file} not found!")
        return 1

    dev_data = []
    with open(dev_file, 'r') as f:
        for line in f:
            if line.strip():
                dev_data.append(json.loads(line))

    print(f"   Loaded {len(dev_data)} items from aime_2022_2023.jsonl")

    dev_sanitized = sanitize_dataset(dev_data)
    save_jsonl(dev_sanitized, data_dir / "aime_eval.jsonl")

    print()

    # 2. Process test sets
    print("2. Processing test sets...")

    test_files = [
        ("aime24", tests_dir / "aime24" / "test.jsonl"),
        ("aime25", tests_dir / "aime25" / "test.jsonl")
    ]

    all_test_data = []

    for name, test_file in test_files:
        if not test_file.exists():
            print(f"   WARNING: {test_file} not found")
            continue

        test_data = []
        with open(test_file, 'r') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))

        print(f"   Loaded {len(test_data)} items from {name}")
        all_test_data.extend(test_data)

    if all_test_data:
        test_sanitized = sanitize_dataset(all_test_data)
        save_jsonl(test_sanitized, data_dir / "aime_test.jsonl")

    print()
    print("=" * 60)
    print("✓ Dataset Preparation Complete!")
    print("=" * 60)
    print()
    print("Created files (NO ANSWERS - safe for agent):")
    print(f"  - data/aime_eval.jsonl ({len(dev_sanitized)} problems)")
    print(f"  - data/aime_test.jsonl ({len(test_sanitized)} problems)")
    print()
    print("Format: {\"idx\": <int>, \"question\": \"<problem text>\"}")
    print()
    print("⚠️  The evaluation backend will automatically add golden")
    print("    answers from the original files when scoring predictions.")

    return 0


if __name__ == "__main__":
    exit(main())
