#!/bin/bash
# SWE Meta-Agent Solution Script
# Supports multiple solution modes via SOLUTION_MODE env var:
#   baseline              - Terminus-2 baseline (copies agent.py, calls eval API)
#   baseline_openhands    - OpenHands built-in agent (no agent.py, HARBOR_AGENT=openhands)
#   baseline_claude_code  - Claude Code built-in agent (no agent.py, HARBOR_AGENT=claude-code)

set -e

SOLUTION_MODE="${SOLUTION_MODE:-baseline}"

echo "============================================================"
echo "SWE Meta-Agent Solution - Mode: $SOLUTION_MODE"
echo "============================================================"
echo ""

# Evaluation timeout: how long the Harbor CLI run is allowed to take.
# Must fit inside the agent phase (task.toml: agent.timeout_sec = 86400).
# Leave headroom for resume rounds (3x) and overhead.
EVAL_TIMEOUT="${EVAL_TIMEOUT:-21600}"
# Client read timeout: must exceed server total (initial run + resumes + overhead).
# Set generously so the client never disconnects while the server is still working.
CLIENT_TIMEOUT="${CLIENT_TIMEOUT:-86400}"

# Evaluation API URL: docker-compose sets this to http://evaluation-api:8080.
EVAL_API="${EVALUATION_API_URL:-http://evaluation-api:8080}"

# Helper: call evaluation API with an agent file (custom agent via --agent-import-path)
eval_with_agent_file() {
    local agent_file="$1"

    echo "Calling evaluation API with agent_file=$agent_file ..."
    echo "  eval_timeout=${EVAL_TIMEOUT}s, client_timeout=${CLIENT_TIMEOUT}s"
    echo ""

    python3 <<PYTHON_SCRIPT
import requests
import json
import sys

resp = requests.post(
    "${EVAL_API}/evaluate/agent",
    json={
        "agent_file": "$agent_file",
        "timeout": ${EVAL_TIMEOUT},
    },
    timeout=${CLIENT_TIMEOUT},
)

print(f"Status: {resp.status_code}")
try:
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:4000])
except Exception:
    print(resp.text[:2000])

if resp.status_code != 200:
    sys.exit(1)
PYTHON_SCRIPT
}

# Helper: call evaluation API without an agent file (built-in agent via --agent)
eval_builtin_agent() {
    echo "Calling evaluation API (built-in agent mode, HARBOR_AGENT=$HARBOR_AGENT) ..."
    echo "  eval_timeout=${EVAL_TIMEOUT}s, client_timeout=${CLIENT_TIMEOUT}s"
    echo ""

    python3 <<PYTHON_SCRIPT
import requests
import json
import sys

resp = requests.post(
    "${EVAL_API}/evaluate/agent",
    json={
        "timeout": ${EVAL_TIMEOUT},
    },
    timeout=${CLIENT_TIMEOUT},
)

print(f"Status: {resp.status_code}")
try:
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False)[:4000])
except Exception:
    print(resp.text[:2000])

if resp.status_code != 200:
    sys.exit(1)
PYTHON_SCRIPT
}

case "$SOLUTION_MODE" in
    baseline)
        echo "Using Terminus-2 Baseline Agent"
        echo ""
        cp /solution/agent.py /workspace/agent.py
        cp -r /solution/terminus_2 /workspace/terminus_2
        echo "Copied agent.py and terminus_2/ to /workspace/"
        echo "Agent will be executed by the verifier (test.sh) via the evaluation API."
        echo ""
        ;;

    baseline_openhands)
        echo "Using OpenHands Built-in Agent"
        echo ""
        echo "Built-in agent will be executed by the verifier (test.sh) via the"
        echo "evaluation API. solve.sh is a no-op for built-in agents."
        echo ""
        ;;

    baseline_claude_code)
        echo "Using Claude Code Built-in Agent"
        echo ""
        echo "Built-in agent will be executed by the verifier (test.sh) via the"
        echo "evaluation API. solve.sh is a no-op for built-in agents."
        echo ""
        ;;

    *)
        echo "ERROR: Unknown SOLUTION_MODE: $SOLUTION_MODE"
        echo "Supported modes: baseline, baseline_openhands, baseline_claude_code"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Solution complete!"
echo "============================================================"
