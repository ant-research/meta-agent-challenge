#!/bin/bash
#
# 测试脚本：在容器内对已运行的 evaluation-api 执行 AIME proxy + split 测试。
#
# 用法：
#   bash aime-meta-agent/tests/test_aime_proxy.sh
#   bash aime-meta-agent/tests/test_aime_proxy.sh --verifier-secret "$VERIFIER_SECRET"  # 可选：验证 test split 可用
#   bash aime-meta-agent/tests/test_aime_proxy.sh --skip-proxy                          # 可选：跳过 proxy 测试

set -e

if [ -z "${TASK_DIR:-}" ]; then
  TASK_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

export TEST_FILE="${TASK_DIR}/tests/test_proxy.py"

echo "==> Running AIME proxy + split tests against evaluation-api..."
python3 "${TEST_FILE}" \
  --base-url http://evaluation-api:8080 \
  "$@"
