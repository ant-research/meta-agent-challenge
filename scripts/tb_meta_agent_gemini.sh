#!/bin/bash
# Run TB Meta-Agent task with Gemini CLI agent.
# The meta-agent (gemini-cli) runs inside the container, builds/optimizes
# an agent.py, and calls the evaluation proxy to score it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }

export HOST_IP=$(ip route get 1 2>/dev/null | awk '{print $7; exit}' || hostname -I | awk '{print $1}')

# Point gemini-cli at a proxy you have already started (e.g. gemini_openai_proxy.py).
export GOOGLE_GEMINI_BASE_URL="${GOOGLE_GEMINI_BASE_URL:?set in .env}"
export GEMINI_API_KEY="${GEMINI_API_KEY:?set in .env}"

# Task model
export TASK_MODEL_API_BASE="${TASK_MODEL_API_BASE:?set in .env}"
export TASK_MODEL_NAME="${TASK_MODEL_NAME:?set in .env}"
export TASK_MODEL_API_KEY="${TASK_MODEL_API_KEY:?set in .env}"

# Harbor evaluation config
export HARBOR_DEV_DATASET_NAME="${HARBOR_DEV_DATASET_NAME:-terminal-bench-pro}"
export HARBOR_TEST_DATASET_NAME="${HARBOR_TEST_DATASET_NAME:-terminal-bench@2.0}"
export HARBOR_AGENT_IMPORT_PATH="${HARBOR_AGENT_IMPORT_PATH:-agent:TBAgent}"
export HARBOR_JOBS_DIR="${HARBOR_JOBS_DIR:-evaluation_logs}"
export HARBOR_CLI_CWD="${HARBOR_CLI_CWD:-}"
export HARBOR_N_ATTEMPTS="${HARBOR_N_ATTEMPTS:-1}"
export HARBOR_N_CONCURRENT="${HARBOR_N_CONCURRENT:-8}"
export HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER="${HARBOR_AGENT_SETUP_TIMEOUT_MULTIPLIER:-10}"

# Solution mode: not a baseline — the meta-agent creates its own agent.py
export SOLUTION_MODE="${SOLUTION_MODE:-}"
export VERIFIER_SECRET="${VERIFIER_SECRET:-tb-verifier-secret-do-not-share}"

PORT="${HARBOR_CLI_SERVICE_PORT:-8899}"
export HARBOR_CLI_SERVICE_URL="http://${HOST_IP}:${PORT}"
export HARBOR_CLI_SERVICE_API_KEY="${HARBOR_CLI_SERVICE_API_KEY:-}"
export EVALUATION_API_URL="http://localhost:8080"

cd "$REPO_ROOT"

# Stop ALL running containers to free resources
ALL_CONTAINERS=$(docker ps -q)
if [ -n "$ALL_CONTAINERS" ]; then
    echo "WARNING: Will stop ALL $(echo "$ALL_CONTAINERS" | wc -l) running Docker containers."
    read -t 30 -p "Press Enter to continue or Ctrl+C to abort (auto-continues in 30s)... " || true
    echo "Stopping all running Docker containers..."
    docker stop $ALL_CONTAINERS
    echo "All containers stopped. Waiting 10s for resources to settle..."
    sleep 10
else
    echo "No running containers."
fi

# Kill any leftover cli_service processes on the target port
echo "Killing any existing cli_service on port ${PORT}..."
OLD_PIDS=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
if [ -n "$OLD_PIDS" ]; then
    echo "Found processes on port ${PORT}: $OLD_PIDS"
    kill $OLD_PIDS 2>/dev/null || true
    sleep 1
    STILL_ALIVE=$(lsof -ti tcp:"${PORT}" 2>/dev/null || true)
    if [ -n "$STILL_ALIVE" ]; then
        kill -9 $STILL_ALIVE 2>/dev/null || true
        sleep 0.5
    fi
    echo "Old cli_service killed."
else
    echo "No existing process on port ${PORT}."
fi

docker volume prune -a -f
docker network prune -f
docker system prune -a -f

START_TIME=$(date +%s)
mkdir -p ./logs

cleanup() {
    set +e
    echo ""
    echo "Cleaning up..."
    if [ -n "${CLI_PID:-}" ]; then
        kill "$CLI_PID" 2>/dev/null || true
    fi
    docker ps -a --filter "name=tb-meta-agent" --format "{{.ID}} {{.CreatedAt}}" | while read container_id created_at; do
        container_time=$(date -d "$created_at" +%s 2>/dev/null || echo 0)
        if [ "$container_time" -ge "$START_TIME" ]; then
            echo "Stopping and removing container: $container_id"
            docker stop "$container_id" 2>/dev/null
            docker rm "$container_id" 2>/dev/null
        fi
    done
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

echo "Starting cli_service on :${PORT}..."
python3 -m uvicorn cli_service:create_app --factory --host 0.0.0.0 --port "${PORT}" >./logs/cli_service.log 2>&1 &
CLI_PID="$!"

for _ in $(seq 1 50); do
    curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
    sleep 0.2
done
curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 || {
    echo "cli_service not healthy; see ./logs/cli_service.log"
    exit 1
}
echo "cli_service healthy (HARBOR_CLI_SERVICE_URL=${HARBOR_CLI_SERVICE_URL})"

harbor run \
    -p ./tb-meta-agent \
    --agent gemini-cli \
    --model google/gemini-3.1-pro-preview \
    --agent-env GEMINI_CLI_TRUST_WORKSPACE=true \
    --agent-kwarg include_directories=/app/tools \
    --agent-kwarg version=0.39.1 \
    --n-concurrent 1 \
    --jobs-dir ./meta-tb-agent-gemini \
    --force-build \
    --yes
