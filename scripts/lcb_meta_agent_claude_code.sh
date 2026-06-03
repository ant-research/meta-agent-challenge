#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }

# LiveCodeBench 任务目录，Harbor 会通过该目录运行 lcb-meta-agent。
TASK_DIR="$REPO_ROOT/lcb-meta-agent"

# Harbor 使用的智能体类型，这里保持原值。
HARBOR_AGENT="claude-code"

# Harbor 使用的元智能体模型名称，这里保持原值。
HARBOR_MODEL="claude-opus-4-6"

# Harbor 单次运行的并发任务数，这里保持原值。
HARBOR_N_CONCURRENT="1"

# Harbor 输出作业目录，这里保持原值。
HARBOR_JOBS_DIR="./meta-agent-lcb-opus"

# Claude Code 兼容接口的基础地址
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:?set in .env}"

# Claude Code 兼容接口的认证令牌
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$ANTHROPIC_API_KEY}"

# Anthropic API Key
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?set in .env}"

# 任务求解模型的 API 基础地址
export TASK_MODEL_API_BASE="${TASK_MODEL_API_BASE:?set in .env}"

# 任务求解模型名称
export TASK_MODEL_NAME="${TASK_MODEL_NAME:?set in .env}"

# 任务求解模型的 API Key
export TASK_MODEL_API_KEY="${TASK_MODEL_API_KEY:?set in .env}"

# model_readmes 目录路径
export MODEL_READMES_DIR="${MODEL_READMES_DIR:-$REPO_ROOT/model_readmes}"

# 要加载的模型说明文件名
export MODEL_README="${MODEL_README:-Qwen3-8B.md}"

if [ ! -f "$MODEL_READMES_DIR/$MODEL_README" ]; then
    echo "[ERROR] 缺少模型说明文件：$MODEL_READMES_DIR/$MODEL_README" >&2
    exit 1
fi

echo "[INFO] 正在检查所需的 LCB 数据文件 ..."
for f in \
    "$TASK_DIR/data/lcb_eval.jsonl" \
    "$TASK_DIR/data/lcb_eval_full.jsonl" \
    "$TASK_DIR/data/lcb_test.jsonl" \
    "$TASK_DIR/data/lcb_test_full.jsonl" \
    "$TASK_DIR/data/split_summary.json"
do
    if [ ! -f "$f" ]; then
        echo "[ERROR] 缺少所需数据文件：$f" >&2
        exit 1
    fi
done

cd "$REPO_ROOT"

echo "Stopping any running meta-agent-related containers..."
RUNNING_CONTAINERS=$(docker ps --format '{{.ID}} {{.Image}} {{.Names}}' | awk '/hb__(lcb)-meta-agent-evaluation/ || /meta-agent/{print $1}')
if [ -n "$RUNNING_CONTAINERS" ]; then
    echo "Found containers: $(docker ps --format '{{.Names}} {{.Image}}' | awk '/hb__(lcb)-meta-agent-evaluation/ || /meta-agent/{print $1}')"
    docker stop $RUNNING_CONTAINERS
    echo "Containers stopped."
else
    echo "No meta-agent-related containers running."
fi

# Clean up stale Docker volumes to ensure fresh workspace
docker volume prune -a -f
docker network prune -f
docker system prune -a -f

python3 "$TASK_DIR/scripts/test_evaluation_flow.py"

echo "[INFO] 开始为 lcb-meta-agent 运行 Harbor ..."
harbor run \
    -p "$TASK_DIR" \
    --agent "$HARBOR_AGENT" \
    --model "$HARBOR_MODEL" \
    --n-concurrent "$HARBOR_N_CONCURRENT" \
    --jobs-dir "$HARBOR_JOBS_DIR" \
    --force-build \
    --yes
