export TASK_MODEL_API_BASE="https://api.siliconflow.cn/v1"
export TASK_MODEL_NAME="Qwen/Qwen3-8B"
export TASK_MODEL_API_KEY=""

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

# Run harbor with lcb-meta-agent task
echo "Running lcb-meta-agent oracle test..."
harbor run \
    -p "/mnt/data1/meta-agent-challenge/lcb-meta-agent" \
    --jobs-dir ./test-lcb-meta-agent-oracle \
    --force-build

echo ""
echo "Test complete! Check results in ./test-lcb-meta-agent-oracle/"