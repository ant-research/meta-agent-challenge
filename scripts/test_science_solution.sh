#!/bin/bash
# Test script for science-meta-agent oracle solution
# This script tests the evaluation pipeline using the oracle agent

export TASK_MODEL_API_BASE="https://api.siliconflow.cn/v1"
export TASK_MODEL_NAME="Qwen/Qwen3-8B"
export TASK_MODEL_API_KEY=""

# Search API (use existing env vars or set dummy values)
export SEARCH_API_KEY="${SEARCH_API_KEY:-dummy}"
export SEARCH_API_BASE="${SEARCH_API_BASE:-http://localhost:8080}"

export SOLUTION_MODE="oracle"

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

# Run harbor with science-meta-agent task
echo "Running science-meta-agent oracle test..."
harbor run \
    -p "/mnt/data1/meta-agent-challenge/science-meta-agent" \
    --jobs-dir ./test-science-meta-agent-oracle \
    --agent oracle \
    --force-build

echo ""
echo "Test complete! Check results in ./test-science-meta-agent-oracle/"
