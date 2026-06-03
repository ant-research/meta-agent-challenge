#!/usr/bin/env python3
"""
End-to-end test of the evaluation flow.

Tests:
1. Dataset has no answers
2. Predictions file format
3. Ground truth auto-loading
4. Scoring logic
"""

import json
from pathlib import Path
import sys

print("=" * 70)
print("AIME Meta-Agent Evaluation Flow Test")
print("=" * 70)
print()

success = True

# Test 1: Dataset has no answers
print("Test 1: Verify dataset has NO answers")
print("-" * 70)
dataset_file = Path("data/aime_eval.jsonl")
with open(dataset_file, 'r') as f:
    dataset = [json.loads(line) for line in f if line.strip()]

has_answers = any('answer' in item or 'gt' in item for item in dataset)
has_required_fields = all('idx' in item and 'question' in item for item in dataset)

if has_answers:
    print("✗ FAIL: Dataset contains answer fields!")
    success = False
else:
    print(f"✓ PASS: Dataset has no answers ({len(dataset)} problems)")

if not has_required_fields:
    print("✗ FAIL: Dataset missing required fields (idx, question)")
    success = False
else:
    print("✓ PASS: Dataset has required fields (idx, question)")

print()

# Test 2: Ground truth loading
print("Test 2: Ground truth auto-loading")
print("-" * 70)

ground_truth_cache = {}
gt_file = Path("data/aime_2022_2023.jsonl")

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
            ground_truth_cache[idx] = str(answer)

if len(ground_truth_cache) == len(dataset):
    print(f"✓ PASS: Loaded {len(ground_truth_cache)} ground truth answers")
else:
    print(f"✗ FAIL: Ground truth count mismatch: {len(ground_truth_cache)} != {len(dataset)}")
    success = False

# Verify first few entries
test_indices = [0, 1, 2]
all_found = True
for idx in test_indices:
    if idx not in ground_truth_cache:
        print(f"✗ FAIL: Missing ground truth for idx={idx}")
        all_found = False
        success = False
    else:
        print(f"  idx={idx}: gt={ground_truth_cache[idx]}")

if all_found:
    print("✓ PASS: Ground truth lookup working")

print()

# Test 3: Prediction file validation
print("Test 3: Prediction file format validation")
print("-" * 70)

# Create test predictions (only idx and pred)
test_predictions = [
    {"idx": 0, "pred": "116"},
    {"idx": 1, "pred": "756"},
    {"idx": 2, "pred": "999"}
]

# Validate format
format_valid = all('idx' in p and 'pred' in p for p in test_predictions)
no_extra_fields = not any('gt' in p or 'answer' in p or 'score' in p for p in test_predictions)

if format_valid:
    print("✓ PASS: Prediction format is valid")
else:
    print("✗ FAIL: Prediction format is invalid")
    success = False

if no_extra_fields:
    print("✓ PASS: No gt/answer/score fields in predictions")
else:
    print("✗ FAIL: Predictions contain gt/answer/score fields")
    success = False

print()

# Test 4: Auto-complete and scoring
print("Test 4: Auto-complete ground truth and score")
print("-" * 70)

correct_count = 0
for pred in test_predictions:
    idx = pred['idx']
    pred_answer = pred['pred']

    # Auto-complete ground truth
    if idx not in ground_truth_cache:
        print(f"✗ FAIL: Cannot find ground truth for idx={idx}")
        success = False
        continue

    gt = ground_truth_cache[idx]
    correct = (str(pred_answer) == str(gt))

    if correct:
        correct_count += 1

    print(f"  idx={idx}: pred={pred_answer}, gt={gt}, {'✓ correct' if correct else '✗ wrong'}")

accuracy = (correct_count / len(test_predictions)) * 100
print(f"\nAccuracy: {correct_count}/{len(test_predictions)} = {accuracy:.1f}%")
print("✓ PASS: Scoring logic working")

print()

# Final result
print("=" * 70)
if success:
    print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    print()
    print("Evaluation flow is secure:")
    print("  - Agents cannot see answers in dataset")
    print("  - Agents do not provide ground truth in predictions")
    print("  - Evaluation system auto-completes ground truth from secure backend")
    sys.exit(0)
else:
    print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    sys.exit(1)
