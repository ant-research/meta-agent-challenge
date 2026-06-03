#!/bin/bash
# LiveCodeBench Meta-Agent Solution Script
# Supports multiple solution modes via SOLUTION_MODE env var:
#   oracle   - Oracle Agent with hidden golden code (default)
#   baseline - Qwen3 thinking baseline agent for verifier-driven evaluation

set -e

SOLUTION_MODE="${SOLUTION_MODE:-oracle}"

echo "============================================================"
echo "LiveCodeBench Meta-Agent Solution - Mode: $SOLUTION_MODE"
echo "============================================================"
echo ""

case "$SOLUTION_MODE" in
    oracle)
        echo "Using Oracle Agent with hidden golden code"
        echo ""

        echo "Setting up Oracle Agent for verifier..."
        cp /solution/oracle_agent.py /workspace/agent.py
        echo "✓ Oracle Agent copied to /workspace/agent.py"
        echo ""

        echo "Calling evaluation API with Oracle Agent..."
        echo ""

        python3 <<'PYTHON_SCRIPT'
import requests

response = requests.post(
    'http://evaluation-api:8080/evaluate/agent',
    headers={
        'X-Verifier-Secret': 'lcb-verifier-secret-do-not-share'
    },
    json={
        'agent_file': '/workspace/agent.py',
        'split': 'test',
        'timeout': 3600,
        'case_timeout': 10,
    },
)

if response.status_code != 200:
    print(f"❌ API call failed with status {response.status_code}")
    print(f"Response: {response.text}")
    raise SystemExit(1)

result = response.json()
if not result.get('success'):
    print(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
    raise SystemExit(1)

print("=" * 60)
print("Oracle Agent Evaluation Results")
print("=" * 60)
print(f"Accuracy: {result['accuracy']:.2f}%")
print(f"Correct: {result['correct']}/{result['total']}")
print(f"Covered: {result.get('covered', result['total'])}/{result['total']} ({result.get('coverage', 100.0):.2f}%)")
print(f"Success: {'✓' if result['success'] else '✗'}")
print("=" * 60)
print("")
print("Note: Evaluation API handles agent execution and scoring internally.")
print("Expected result: 100% accuracy when gt is present for every test problem.")
print("")
PYTHON_SCRIPT
        ;;

    baseline)
        echo "Using Baseline Thinking Agent (Qwen3 LiveCodeBench baseline)"
        echo ""
        echo "This agent will be executed by the verifier (tests/test.sh) via the"
        echo "evaluation API. solve.sh only copies the agent file."
        echo ""

        cp /solution/baseline_agent.py /workspace/agent.py
        echo "✓ Baseline Thinking Agent copied to /workspace/agent.py"
        echo ""
        ;;

    *)
        echo "❌ ERROR: Unknown SOLUTION_MODE: $SOLUTION_MODE"
        echo "Supported modes: oracle, baseline"
        exit 1
        ;;
esac

echo "✓ Solution complete!"
echo "============================================================"
