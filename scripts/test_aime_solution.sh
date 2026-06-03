export ANTHROPIC_BASE_URL="http://121.229.107.80:8127/"
export ANTHROPIC_API_KEY="sk-key"

export TASK_MODEL_API_BASE="https://api.siliconflow.cn/v1"
export TASK_MODEL_NAME="Qwen/Qwen3-8B"
export TASK_MODEL_API_KEY=""

# Select oracle mode (solve.sh uses oracle_agent.py)
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
docker volume prune -a -f
docker network prune -f
docker system prune -a -f

harbor run -p "/mnt/data/luxinyu2021/meta-agent-challenge/aime-meta-agent" \
        --agent oracle \
        --jobs-dir ./test-aime-meta-agent-gt \
        --force-build