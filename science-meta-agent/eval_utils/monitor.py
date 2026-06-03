#!/usr/bin/env python3
"""
API Usage Monitor for Science Meta-Agent

Monitors and validates that only approved APIs are being used:
- LLM API (from TASK_MODEL_API_BASE)
- Search API (from SEARCH_API_BASE)

Detects attempts to use unauthorized model APIs or search services.
"""

import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# Known API endpoints that should be blocked
BLOCKED_LLM_ENDPOINTS = {
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

# Blocked search APIs (only the endpoint from SEARCH_API_BASE is allowed)
BLOCKED_SEARCH_APIS = {
    'api.bing.com',
    'api.search.brave.com',
    'serpapi.com',
    'serper.dev',
    'api.tavily.com',
    'duckduckgo',
    'searx',
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


def get_allowed_llm_endpoint() -> str:
    """Get the approved LLM API endpoint from environment."""
    base_url = os.environ.get('TASK_MODEL_API_BASE', '')
    if not base_url:
        return 'api.openai.com'  # Default

    # Extract domain from URL
    match = re.search(r'https?://([^/]+)', base_url)
    if match:
        return match.group(1)
    return base_url


def get_allowed_search_endpoint() -> str:
    """Get the approved search API endpoint from environment."""
    base_url = os.environ.get('SEARCH_API_BASE', '')
    if not base_url:
        return ''

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
        'tools/test_',
        'tools/example_',
        'eval_utils/evaluation_api.py',
        'tools/openai_helper.py',
        'tools/search_helper.py',
        'eval_utils/grader.py',
    ]

    return any(pattern in file_str for pattern in skip_patterns)


def is_code_line(line: str) -> bool:
    """Check if a line contains actual code (not just comments or whitespace)."""
    stripped = line.strip()

    if not stripped:
        return False

    if stripped.startswith('#'):
        return False

    if stripped.startswith('"""') or stripped.startswith("'''"):
        return False

    return True


def extract_code_part(line: str) -> str:
    """Extract the code part of a line, removing inline comments."""
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
    violations_set = set()
    allowed_llm_endpoint = get_allowed_llm_endpoint()
    allowed_search_endpoint = get_allowed_search_endpoint()

    # Patterns for LLM APIs
    llm_patterns = {
        'anthropic': r'(?:from|import)\s+anthropic|Anthropic\(|anthropic\.Client',
        'openai_direct': r'openai\.com',
        'google_ai': r'google\.generativeai|from\s+google\.ai|googleapis\.com/ai',
        'cohere': r'(?:from|import)\s+cohere|cohere\.Client',
        'huggingface': r'from\s+transformers|AutoModel|(?:from|import)\s+huggingface|huggingface_hub|InferenceClient\(',
        'replicate': r'(?:from|import)\s+replicate',
        'together': r'together\.ai|together\.xyz',
        'ollama': r'(?:from|import)\s+ollama|OllamaClient\(',
        'vllm': r'from\s+vllm|vllm\.LLM',
        'llama_cpp': r'llama\.cpp|llama_cpp',
    }

    # Patterns for blocked search APIs
    search_patterns = {
        'bing_search': r'api\.bing\.com|BingSearchAPI',
        'brave_search': r'api\.search\.brave\.com|BraveSearch',
        'serpapi': r'serpapi\.com|SerpApi',
        'serper': r'serper\.dev|SerperDev',
        'tavily': r'tavily\.com|TavilySearch',
        'duckduckgo': r'duckduckgo_search|DDGS\(',
        'searx': r'searx|SearxSearch',
    }

    # Scan Python files
    workspace_path = Path(workspace_dir)
    for py_file in workspace_path.rglob('*.py'):
        if should_skip_file(py_file):
            continue

        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not is_code_line(line):
                        continue

                    code_part = extract_code_part(line)
                    line_has_violation = False

                    # Check LLM patterns
                    for name, pattern in llm_patterns.items():
                        if re.search(pattern, code_part, re.IGNORECASE):
                            if 'openai' in name and allowed_llm_endpoint in code_part:
                                continue  # Allowed

                            violation_key = (str(py_file), line_num, name)
                            if violation_key not in violations_set:
                                violations_set.add(violation_key)
                                violations.append((
                                    str(py_file),
                                    f"Detected unauthorized LLM API: {name}",
                                    line_num
                                ))
                                line_has_violation = True

                    # Check search API patterns
                    for name, pattern in search_patterns.items():
                        if re.search(pattern, code_part, re.IGNORECASE):
                            violation_key = (str(py_file), line_num, f"search_{name}")
                            if violation_key not in violations_set:
                                violations_set.add(violation_key)
                                violations.append((
                                    str(py_file),
                                    f"Detected unauthorized search API: {name}",
                                    line_num
                                ))
                                line_has_violation = True

                    # Check for blocked LLM endpoints
                    for blocked in BLOCKED_LLM_ENDPOINTS:
                        if blocked in code_part and blocked != allowed_llm_endpoint:
                            violation_key = (str(py_file), line_num, f"blocked_{blocked}")
                            if violation_key not in violations_set:
                                violations_set.add(violation_key)
                                violations.append((
                                    str(py_file),
                                    f"Detected blocked LLM endpoint: {blocked}",
                                    line_num
                                ))
                                line_has_violation = True

                    # Check for blocked search endpoints
                    for blocked in BLOCKED_SEARCH_APIS:
                        if blocked in code_part:
                            if allowed_search_endpoint and blocked in allowed_search_endpoint:
                                continue  # Allowed via SEARCH_API_BASE
                            violation_key = (str(py_file), line_num, f"blocked_search_{blocked}")
                            if violation_key not in violations_set:
                                violations_set.add(violation_key)
                                violations.append((
                                    str(py_file),
                                    f"Detected blocked search endpoint: {blocked}",
                                    line_num
                                ))
                                line_has_violation = True

                    # Check for local model indicators
                    if not line_has_violation:
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

                        if not line_has_violation:
                            for indicator in ['ollama', 'llama.cpp', 'vllm', 'text-generation-webui', 'oobabooga', 'koboldcpp']:
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
    allowed_llm_endpoint = get_allowed_llm_endpoint()

    # Check for suspicious LLM API keys
    suspicious_llm_vars = [
        'ANTHROPIC_API_KEY',
        'COHERE_API_KEY',
        'TOGETHER_API_KEY',
        'REPLICATE_API_TOKEN',
        'HF_TOKEN',
        'HUGGINGFACE_TOKEN',
    ]

    for var in suspicious_llm_vars:
        if os.environ.get(var):
            violations.append(f"Detected unauthorized LLM API key: {var}")

    # SEARCH_API_KEY is allowed (for the configured search endpoint)
    # SEARCH_API_BASE is allowed (configures the search endpoint)

    # Validate TASK_MODEL_API_BASE if set
    base_url = os.environ.get('TASK_MODEL_API_BASE', '')
    if base_url:
        for blocked in BLOCKED_LLM_ENDPOINTS:
            if blocked in base_url and blocked != allowed_llm_endpoint:
                violations.append(f"TASK_MODEL_API_BASE points to blocked endpoint: {blocked}")

    return violations


def check_agent_code(agent_file: str) -> List[Dict[str, Any]]:
    """
    Check a specific agent file for violations.

    Args:
        agent_file: Path to agent file

    Returns:
        List of violation dicts with 'file', 'line', 'description'
    """
    violations = []

    # Get workspace directory
    workspace = Path(agent_file).parent
    while workspace.name not in ['workspace', ''] and workspace != workspace.parent:
        workspace = workspace.parent

    if workspace.name != 'workspace':
        workspace = Path('/workspace')

    # Scan code
    code_violations = scan_code_for_violations(str(workspace))

    for file_path, desc, line_num in code_violations:
        violations.append({
            'file': file_path,
            'line': line_num,
            'description': desc
        })

    return violations


def main():
    """Run API usage monitoring."""
    print("=" * 60)
    print("API Usage Monitor - Science Meta-Agent")
    print("=" * 60)
    print()

    # Get allowed endpoints
    allowed_llm = get_allowed_llm_endpoint()
    allowed_search = get_allowed_search_endpoint()
    print(f"Allowed LLM API: {allowed_llm}")
    print(f"Allowed Search API: {allowed_search or '(not set - SEARCH_API_BASE not configured)'}")
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
        print("Allowed APIs:")
        print("  - LLM: Use provided TASK_MODEL_API_BASE")
        print("  - Search: Google Custom Search API only")
        print()
        print("Remove all unauthorized API usage before submission.")
        print("=" * 60)
        sys.exit(1)
    else:
        print("✓ No violations detected - API usage compliant")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
