#!/usr/bin/env bash
# Launch the harbor trial viewer. Pass the jobs folder as $1
# (default: ./run_to_inspect/meta-agent-lcb-glm-5/).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

FOLDER="${1}"

HARBOR_BIN="${HARBOR_BIN:-/mnt/data1/envs/harbor_distributed/bin/harbor}"

exec "$HARBOR_BIN" view "$FOLDER" --host 0.0.0.0 --port 8083 --jobs
