#!/bin/bash
# AIME Meta-Agent Solution Script
# Supports multiple solution modes via SOLUTION_MODE env var:
#   oracle        - Oracle Agent, all problems (100% baseline, default)
#   oracle_first_50  - Oracle Agent, first 50% by idx (~50% baseline)
#   oracle_last_50   - Oracle Agent, last 50% by idx (~50% baseline)
#   oracle_random_50 - Oracle Agent, random 50% (seed via ORACLE_SEED, default 42)
#   baseline      - Baseline Thinking Agent using Qwen3-8B thinking mode

set -e

SOLUTION_MODE="${SOLUTION_MODE:-oracle}"

echo "============================================================"
echo "AIME Meta-Agent Solution - Mode: $SOLUTION_MODE"
echo "============================================================"
echo ""

# Helper: run oracle agent with a given ORACLE_MODE
run_oracle() {
    local oracle_mode="$1"
    local label="$2"
    local expected="$3"

    echo "Using Oracle Agent - $label"
    echo ""

    cp /solution/oracle_agent.py /workspace/agent.py
    sed -i "s/os.environ.get('ORACLE_MODE', 'full')/'$oracle_mode'/g" /workspace/agent.py
    echo "✓ Oracle Agent copied to /workspace/agent.py (mode=$oracle_mode)"

    echo "Calling evaluation API..."
    echo ""

    ORACLE_MODE="$oracle_mode" ORACLE_SEED="${ORACLE_SEED:-42}" python3 <<PYTHON_SCRIPT
import requests, os

response = requests.post(
    'http://evaluation-api:8080/evaluate/agent',
    headers={'X-Verifier-Secret': 'aime-verifier-secret-do-not-share'},
    json={
        'agent_file': '/workspace/agent.py',
        'split': 'test',
        'timeout': 60
    }
)

if response.status_code != 200:
    print(f"❌ API call failed with status {response.status_code}")
    print(f"Response: {response.text}")
    exit(1)

results = response.json()
if not results.get('success'):
    print(f"❌ Evaluation failed: {results.get('error', 'Unknown error')}")
    exit(1)

print("=" * 60)
print(f"Oracle Agent Evaluation Results ({os.environ.get('ORACLE_MODE', 'full')})")
print("=" * 60)
print(f"Accuracy:            {results['accuracy']:.2f}%")
print(f"Correct:             {results['correct']}/{results['total']}")
print(f"Missing predictions: {results.get('missing_predictions', 'N/A')}")
print(f"Expected:            $expected")
print("=" * 60)
PYTHON_SCRIPT
}

case "$SOLUTION_MODE" in
    oracle)
        run_oracle "full" "all problems (100% baseline)" "100% accuracy"
        ;;

    oracle_first_50)
        run_oracle "first_50" "first 50% by idx" "~50% accuracy"
        ;;

    oracle_last_50)
        run_oracle "last_50" "last 50% by idx" "~50% accuracy"
        ;;

    oracle_random_50)
        run_oracle "random_50" "random 50% (seed=${ORACLE_SEED:-42})" "~50% accuracy"
        ;;

    baseline)
        echo "Using Baseline Thinking Agent (Qwen3-8B thinking mode)"
        echo ""
        echo "This agent will be executed by the verifier (test.sh) via the"
        echo "evaluation API. solve.sh only copies the agent file."
        echo ""

        cp /solution/baseline_thinking_agent.py /workspace/agent.py
        echo "✓ Baseline Thinking Agent copied to /workspace/agent.py"
        echo ""
        ;;

    *)
        echo "❌ ERROR: Unknown SOLUTION_MODE: $SOLUTION_MODE"
        echo "Supported modes: oracle, oracle_first_50, oracle_last_50, oracle_random_50, baseline"
        exit 1
        ;;
esac

echo "✓ Solution complete!"
echo "============================================================"
