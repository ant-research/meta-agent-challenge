"""
Grading logic for science multiple-choice questions.

Much simpler than AIME math grading - just need to match choice letters (A/B/C/D/E).
"""

import re
from typing import Union


def choice_answer_clean(pred: str) -> str:
    """
    Extract choice letter from prediction string.

    Handles various formats:
    - "A" -> "A"
    - "The answer is B" -> "B"
    - "I think it's C." -> "C"
    - "D) Some text" -> "D"

    Args:
        pred: Prediction string

    Returns:
        Cleaned choice letter (A/B/C/D/E/F) or original string if no match
    """
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    # Try to find choice letter (A-F)
    tmp = re.findall(r"\b([A-F])\b", pred.upper(), re.ASCII)
    if tmp:
        # Return the last match (most likely the final answer)
        pred = tmp[-1]
    else:
        # No letter found, return cleaned string
        pred = pred.strip().strip(".")

    # Remove trailing punctuation
    pred = pred.rstrip(".").rstrip("/")

    return pred


def choice_equal(prediction: Union[str, None], reference: Union[str, None]) -> bool:
    """
    Check if multiple choice answer matches reference.

    Args:
        prediction: Predicted answer (e.g., "A", "The answer is B")
        reference: Ground truth answer (e.g., "A", "B")

    Returns:
        True if answers match, False otherwise
    """
    if prediction is None or reference is None:
        return False

    # Clean both prediction and reference
    pred_clean = choice_answer_clean(str(prediction))
    ref_clean = choice_answer_clean(str(reference))

    # Case-insensitive comparison
    return pred_clean.upper() == ref_clean.upper()


def _test_choice_equal():
    """Test cases covering real LLM output patterns"""
    test_cases = [
        # === Basic / clean outputs ===
        ("A", "A", True),
        ("B", "A", False),
        ("a", "A", True),  # lowercase
        ("D", "d", True),

        # === "The answer is X" patterns ===
        ("The answer is A", "A", True),
        ("The answer is B.", "B", True),
        ("The answer is (C)", "C", True),
        ("the answer is: D", "D", True),
        ("So the answer is A.", "A", True),
        ("The answer is C)", "C", True),
        ("BBBBBBB The answer is C)", "C", True),

        # === "correct answer/option" patterns ===
        ("The correct answer is B", "B", True),
        ("The correct option is C.", "C", True),
        ("The correct choice is A because it describes...", "A", True),

        # === Reasoning then answer ===
        ("Let me analyze each option:\nA) Wrong\nB) Also wrong\nC) Correct\nThe answer is C", "C", True),
        ("Option A is about X. Option B is about Y. I think B is correct.", "B", True),
        ("A is wrong, B is also unlikely, C is correct", "C", True),

        # === Parenthesized ===
        ("(A)", "A", True),
        ("(B) because of quantum effects", "B", True),
        ("I would choose (D)", "D", True),

        # === Boxed (LaTeX) ===
        (r"The answer is \boxed{A}", "A", True),
        (r"\boxed{C}", "C", True),

        # === Bold / markdown ===
        ("The answer is **B**", "B", True),
        ("**A**", "A", True),

        # === "Answer:" prefix ===
        ("Answer: A", "A", True),
        ("Answer: C.", "C", True),
        ("Answer - B", "B", True),

        # === Chinese outputs ===
        ("答案是A", "A", True),
        ("答案为B", "B", True),
        ("答案：C", "C", True),
        ("答案是a", "A", True),  # Chinese + lowercase
        ("答案为b", "B", True),

        # === Trailing letter after long reasoning ===
        ("Based on the analysis above, considering all factors...\n\nB", "B", True),
        ("...therefore the mitochondria is the powerhouse.\n\nA", "A", True),

        # === Edge: letter appears in reasoning context (not as answer) ===
        # Model says "C is correct" but then changes mind
        ("C seems right at first, but actually the answer is D", "D", True),

        # === Edge: single letter with punctuation ===
        ("B.", "B", True),
        ("A\n", "A", True),
        (" C ", "C", True),

        # === Edge: no valid answer ===
        ("", "A", False),
        (None, "A", False),
        ("A", None, False),
        ("I'm not sure about this question", "A", False),
        ("The answer could be either option", "B", False),

        # === Edge: letter in non-answer context should NOT match ===
        # "vitamin D" — D is part of a word context, tricky
        # We accept this may match D; it's a known limitation

        # === "option X is correct" ===
        ("option A is correct", "A", True),
        ("Option B is correct because...", "B", True),

        # === "choose/select" ===
        ("I would choose A", "A", True),
        ("I select B as my answer", "B", True),
    ]

    print("Running choice_equal tests...")
    passed = 0
    failed = 0

    for pred, ref, expected in test_cases:
        result = choice_equal(pred, ref)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  {status} choice_equal({repr(pred)}, {repr(ref)}) = {result} (expected {expected})")

    if failed:
        print(f"\n  Failed cases shown above")
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)}")


if __name__ == "__main__":
    _test_choice_equal()
