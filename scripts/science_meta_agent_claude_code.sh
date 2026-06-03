#!/bin/bash
# Test script for science-meta-agent with claude-code agent
# This script runs a full evaluation using claude-code as the meta-agent

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:?set in .env}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?set in .env}"

# Task model API: the model that the agent will use to solve science problems
# These are picked up by docker-compose.yaml for both main and evaluation-api containers
export TASK_MODEL_API_KEY="${TASK_MODEL_API_KEY:?set in .env}"
export TASK_MODEL_API_BASE="${TASK_MODEL_API_BASE:?set in .env}"
export TASK_MODEL_NAME="${TASK_MODEL_NAME:?set in .env}"

# Search API (required for science questions)
export SEARCH_API_KEY="${SEARCH_API_KEY:?set in .env}"
export SEARCH_API_BASE="${SEARCH_API_BASE:?set in .env}"

# Model README: specify the filename in model_readmes/ to load into /workspace/model_readme.md
export MODEL_READMES_DIR="${MODEL_READMES_DIR:-$REPO_ROOT/model_readmes}"
export MODEL_README="${MODEL_README:-Qwen3-8B.md}"

cd "$REPO_ROOT"

# Stop any running meta-agent-related containers before starting
echo "Stopping any running meta-agent-related containers..."
RUNNING_CONTAINERS=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | awk '/hb__(aime|science)-meta-agent-evaluation/ || /meta-agent/{print $1}')
if [ -n "$RUNNING_CONTAINERS" ]; then
    echo "Found containers: $(docker ps --format '{{.Names}} {{.Image}}' | awk '/hb__(aime|science)-meta-agent-evaluation/ || /meta-agent/{print $1}')"
    docker stop $RUNNING_CONTAINERS
    echo "Containers stopped."
else
    echo "No meta-agent-related containers running."
fi

# Clean up stale Docker volumes to ensure fresh workspace
echo "Cleaning up Docker volumes..."
docker volume prune -a -f
docker network prune -f
docker system prune -a -f

harbor run \
    -p ./science-meta-agent \
    --agent claude-code \
    --ak version=2.1.119 \
    --model claude-sonnet-4-6 \
    --n-concurrent 1 \
    --jobs-dir ./meta-agent-science \
    --force-build \
    --yes
