SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }

# Codex agent credentials (used by codex CLI via harbor)
export OPENAI_BASE_URL="${OPENAI_BASE_URL:?set in .env}"
export OPENAI_API_KEY="${OPENAI_API_KEY:?set in .env}"

# codex --provider CUSTOM

# Task model API: the model that the agent will use to solve AIME problems
# These are picked up by docker-compose.yaml for both main and evaluation-api containers
export TASK_MODEL_API_KEY="${TASK_MODEL_API_KEY:?set in .env}"
export TASK_MODEL_API_BASE="${TASK_MODEL_API_BASE:?set in .env}"
export TASK_MODEL_NAME="${TASK_MODEL_NAME:?set in .env}"

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
docker volume prune -a -f
docker network prune -f
docker system prune -a -f

harbor run \
    -p ./aime-meta-agent \
    --agent codex \
    --model gpt-5.3-codex \
    --n-concurrent 1 \
    --jobs-dir ./meta-agent-codex \
    --ak version=0.116.0 \
    --force-build
