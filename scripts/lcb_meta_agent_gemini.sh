#!/bin/bash
# Run LCB Meta-Agent task with Gemini CLI agent.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }

# --- Gemini CLI credentials ---
# Point gemini-cli at a proxy you have already started (e.g. gemini_openai_proxy.py).
export GOOGLE_GEMINI_BASE_URL="${GOOGLE_GEMINI_BASE_URL:?set in .env}"
export GEMINI_API_KEY="${GEMINI_API_KEY:?set in .env}"

# Task model API
export TASK_MODEL_API_BASE="${TASK_MODEL_API_BASE:?set in .env}"
export TASK_MODEL_NAME="${TASK_MODEL_NAME:?set in .env}"
export TASK_MODEL_API_KEY="${TASK_MODEL_API_KEY:?set in .env}"

# Model README
export MODEL_READMES_DIR="${MODEL_READMES_DIR:-$REPO_ROOT/model_readmes}"
export MODEL_README="${MODEL_README:-Qwen3-8B.md}"

# Verifier secret
export VERIFIER_SECRET="${VERIFIER_SECRET:-lcb-verifier-secret-do-not-share}"

TASK_DIR="$REPO_ROOT/lcb-meta-agent"

if [ ! -f "$MODEL_READMES_DIR/$MODEL_README" ]; then
    echo "[ERROR] Missing model readme: $MODEL_READMES_DIR/$MODEL_README" >&2
    exit 1
fi

echo "[INFO] Checking required LCB data files ..."
for f in \
    "$TASK_DIR/data/lcb_eval.jsonl" \
    "$TASK_DIR/data/lcb_eval_full.jsonl" \
    "$TASK_DIR/data/lcb_test.jsonl" \
    "$TASK_DIR/data/lcb_test_full.jsonl" \
    "$TASK_DIR/data/split_summary.json"
do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing data file: $f" >&2
        exit 1
    fi
done

cd "$REPO_ROOT"

# Stop any running meta-agent-related containers
echo "Stopping any running meta-agent-related containers..."
RUNNING_CONTAINERS=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | awk '/hb__(lcb)-meta-agent-evaluation/ || /meta-agent/{print $1}')
if [ -n "$RUNNING_CONTAINERS" ]; then
    echo "Found containers: $(docker ps --format '{{.Names}} {{.Image}}' | awk '/hb__(lcb)-meta-agent-evaluation/ || /meta-agent/{print $1}')"
    docker stop $RUNNING_CONTAINERS
    echo "Containers stopped."
else
    echo "No meta-agent-related containers running."
fi

# Clean up stale Docker volumes
docker volume prune -a -f
docker network prune -f
docker system prune -a -f

harbor run \
    -p "$TASK_DIR" \
    --agent gemini-cli \
    --model google/gemini-3.1-pro-preview \
    --agent-env GEMINI_CLI_TRUST_WORKSPACE=true \
    --agent-kwarg include_directories=/app/tools \
    --agent-kwarg version=0.39.1 \
    --n-concurrent 1 \
    --jobs-dir ./meta-agent-lcb-gemini \
    --force-build \
    --yes
