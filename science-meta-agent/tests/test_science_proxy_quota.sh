#!/bin/bash
# 测试脚本：在容器内对已运行的 evaluation-api 执行 proxy + quota 端点测试。
# 需先在宿主机启动 mock server：python3 science-meta-agent/tests/mock_server.py
# 然后在容器内运行此脚本。
#
# 用法：
#   bash science-meta-agent/tests/test_science_proxy_quota.sh              # 全部测试
#   bash science-meta-agent/tests/test_science_proxy_quota.sh --skip-proxy # 跳过 proxy 测试
#   bash science-meta-agent/tests/test_science_proxy_quota.sh --skip-search # 跳过 search query 测试
#   bash science-meta-agent/tests/test_science_proxy_quota.sh --exhaustion  # 附加 quota exhaustion 测试

set -e

TEST_FILE="${TASK_DIR}/tests/test_proxy_quota.py"

echo "==> Running proxy + quota tests against evaluation-api..."
python3 "${TEST_FILE}" \
    --base-url http://evaluation-api:8080 \
    --exhaustion \
    "$@"
