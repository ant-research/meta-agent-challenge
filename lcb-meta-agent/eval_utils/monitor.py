#!/usr/bin/env python3
"""
API Usage Monitor

Monitors and validates that only the approved API endpoint is being used.
Detects attempts to use unauthorized model APIs or services.
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Set, Tuple


# Known API endpoints that should be blocked
BLOCKED_ENDPOINTS = {
    'api.anthropic.com',
    'api.openai.com',  # Block unless explicitly allowed
    'generativelanguage.googleapis.com',  # Google AI
    'api.cohere.ai',
    'api.ai21.com',
    'api.together.xyz',
    'api.replicate.com',
    'api.huggingface.co',
    'inference.huggingface.co',
    'api-inference.huggingface.co',
    'api.mistral.ai',
    'api.perplexity.ai',
    'api.groq.com',
}

# Known local model indicators
LOCAL_MODEL_INDICATORS = [
    'localhost',
    '127.0.0.1',
    'ollama',
    'llama.cpp',
    'vllm',
    'text-generation-webui',
    'oobabooga',
    'koboldcpp',
]


def get_allowed_endpoint() -> str:
    """Get the approved API endpoint from environment."""
    base_url = os.environ.get('TASK_MODEL_API_BASE', '')
    if not base_url:
        return 'api.openai.com'  # Default, will be blocked if not explicitly set

    # Extract domain from URL
    match = re.search(r'https?://([^/]+)', base_url)
    if match:
        return match.group(1)
    return base_url


def should_skip_file(file_path: str) -> bool:
    """Check if a file should be skipped from scanning."""
    file_str = str(file_path)

    # Skip monitoring tools and task-provided utility files
    skip_patterns = [
        'eval_utils/monitor.py',
        'tools/test_',              # Test files (e.g., test_api_integration.py)
        'tools/example_',           # Example files (e.g., example_usage.py)
        'eval_utils/evaluation_api.py',     # API server code
        'tools/openai_helper.py',      # Helper utilities
        'eval_utils/grader.py',             # Grading utilities
        'eval_utils/parser.py',             # Answer parsing utilities
        'eval_utils/utils.py',              # Shared utilities
        'eval_utils/examples.py',           # Few-shot examples
    ]

    return any(pattern in file_str for pattern in skip_patterns)


def is_code_line(line: str) -> bool:
    """Check if a line contains actual code (not just comments or whitespace)."""
    stripped = line.strip()

    # Empty line
    if not stripped:
        return False

    # Pure comment line
    if stripped.startswith('#'):
        return False

    # Docstring line (simple heuristic)
    if stripped.startswith('"""') or stripped.startswith("'''"):
        return False

    return True


def extract_code_part(line: str) -> str:
    """Extract the code part of a line, removing inline comments."""
    # Remove inline comments (but preserve strings containing #)
    in_string = False
    string_char = None

    for i, char in enumerate(line):
        if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
        elif char == '#' and not in_string:
            return line[:i]

    return line


def scan_code_for_violations(workspace_dir: str) -> List[Tuple[str, str, int]]:
    """
    Scan workspace for code that violates API restrictions.

    Returns:
        List of (file_path, violation_description, line_number) tuples
    """
    violations = []
    violations_set = set()  # Track unique violations to avoid duplicates
    allowed_endpoint = get_allowed_endpoint()

    # Patterns to search for (focusing on actual API usage)
    patterns = {
        'anthropic': r'(?:from|import)\s+anthropic|Anthropic\(|anthropic\.Client',
        'openai_direct': r'openai\.com',
        'google_ai': r'google\.generativeai|from\s+google\.ai|googleapis\.com',
        'cohere': r'(?:from|import)\s+cohere|cohere\.Client',
        'huggingface': r'from\s+transformers|AutoModel|(?:from|import)\s+huggingface|huggingface_hub|InferenceClient\(',
        'replicate': r'(?:from|import)\s+replicate',
        'together': r'together\.ai|together\.xyz',
        'ollama': r'(?:from|import)\s+ollama|OllamaClient\(',
        'vllm': r'from\s+vllm|vllm\.LLM',
        'llama_cpp': r'llama\.cpp|llama_cpp',
    }

    # Scan Python files
    workspace_path = Path(workspace_dir)
    for py_file in workspace_path.rglob('*.py'):
        # Skip task-provided utility files
        if should_skip_file(py_file):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Skip non-code lines (comments, empty lines)
                    if not is_code_line(line):
                        continue

                    # Extract code part (remove inline comments)
                    code_part = extract_code_part(line)

                    # Track if this line already has a violation
                    line_has_violation = False

                    # Check patterns
                    for name, pattern in patterns.items():
                        if re.search(pattern, code_part, re.IGNORECASE):
                            # Check if it's the allowed endpoint
                            if 'openai' in name and allowed_endpoint in code_part:
                                continue  # Allowed

                            violation_key = (str(py_file), line_num, name)
                            if violation_key not in violations_set:
                                violations_set.add(violation_key)
                                violations.append((
                                    str(py_file),
                                    f"Detected {name} usage",
                                    line_num
                                ))
                                line_has_violation = True

                    # Check for blocked endpoints
                    for blocked in BLOCKED_ENDPOINTS:
                        if blocked in code_part and blocked != allowed_endpoint:
                            violation_key = (str(py_file), line_num, f"blocked_{blocked}")
                            if violation_key not in violations_set:
                                violations_set.add(violation_key)
                                violations.append((
                                    str(py_file),
                                    f"Detected blocked endpoint: {blocked}",
                                    line_num
                                ))
                                line_has_violation = True

                    # Check for local model indicators (only localhost/127.0.0.1 in URLs)
                    # Only flag if it looks like an actual API call
                    if not line_has_violation:
                        # Look for localhost/127.0.0.1 in URL context
                        localhost_pattern = r'https?://(?:localhost|127\.0\.0\.1):\d+'
                        if re.search(localhost_pattern, code_part, re.IGNORECASE):
                            violation_key = (str(py_file), line_num, "localhost_url")
                            if violation_key not in violations_set:
                                violations_set.add(violation_key)
                                violations.append((
                                    str(py_file),
                                    "Detected localhost API endpoint usage",
                                    line_num
                                ))
                                line_has_violation = True

                        # Check for other local model frameworks (only if no violation yet)
                        if not line_has_violation:
                            for indicator in ['ollama', 'llama.cpp', 'vllm', 'text-generation-webui', 'oobabooga', 'koboldcpp']:
                                # Only flag if it's in an import or instantiation context
                                if re.search(rf'(?:from|import)\s+\w*{indicator}|{indicator}.*\(', code_part, re.IGNORECASE):
                                    violation_key = (str(py_file), line_num, indicator)
                                    if violation_key not in violations_set:
                                        violations_set.add(violation_key)
                                        violations.append((
                                            str(py_file),
                                            f"Detected local model framework: {indicator}",
                                            line_num
                                        ))
                                        break

        except Exception as e:
            print(f"Warning: Could not scan {py_file}: {e}", file=sys.stderr)

    return violations


def check_environment() -> List[str]:
    """Check environment variables for violations."""
    violations = []
    allowed_endpoint = get_allowed_endpoint()

    # Check for suspicious environment variables
    suspicious_vars = [
        'ANTHROPIC_API_KEY',
        'GOOGLE_API_KEY',
        'COHERE_API_KEY',
        'TOGETHER_API_KEY',
        'REPLICATE_API_TOKEN',
        'HF_TOKEN',
        'HUGGINGFACE_TOKEN',
    ]

    for var in suspicious_vars:
        if os.environ.get(var):
            violations.append(f"Detected unauthorized API key: {var}")

    # Validate TASK_MODEL_API_BASE if set
    base_url = os.environ.get('TASK_MODEL_API_BASE', '')
    if base_url:
        for blocked in BLOCKED_ENDPOINTS:
            if blocked in base_url and blocked != allowed_endpoint:
                violations.append(f"TASK_MODEL_API_BASE points to blocked endpoint: {blocked}")

    return violations


def main():
    """Run API usage monitoring."""
    print("=" * 60)
    print("API Usage Monitor - Checking for Violations")
    print("=" * 60)
    print()

    # Get allowed endpoint
    allowed_endpoint = get_allowed_endpoint()
    print(f"Allowed API endpoint: {allowed_endpoint}")
    print()

    # Check environment
    print("Checking environment variables...")
    env_violations = check_environment()

    if env_violations:
        print("❌ Environment violations detected:")
        for violation in env_violations:
            print(f"  - {violation}")
        print()
    else:
        print("✓ No environment violations detected")
        print()

    # Scan code
    workspace = os.environ.get('WORKSPACE', '/workspace')
    print(f"Scanning code in {workspace}...")
    code_violations = scan_code_for_violations(workspace)

    if code_violations:
        print(f"❌ Code violations detected ({len(code_violations)}):")

        # Group by file
        by_file = {}
        for file_path, desc, line_num in code_violations:
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append((desc, line_num))

        for file_path, issues in by_file.items():
            print(f"\n  File: {file_path}")
            for desc, line_num in issues:
                print(f"    Line {line_num}: {desc}")
        print()
    else:
        print("✓ No code violations detected")
        print()

    # Summary
    total_violations = len(env_violations) + len(code_violations)

    print("=" * 60)
    if total_violations > 0:
        print(f"❌ VIOLATIONS DETECTED: {total_violations}")
        print()
        print("You may ONLY use the provided API endpoint.")
        print("Remove all unauthorized API usage before submission.")
        print("=" * 60)
        sys.exit(1)
    else:
        print("✓ No violations detected - API usage compliant")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
