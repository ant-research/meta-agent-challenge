#!/usr/bin/env bash
set -euo pipefail

# Audit every "experiment root" under PARENT. An experiment root is any
# directory that contains at least one timestamped subdir (YYYY-MM-DD__HH-MM-SS).
# For each, audit_agent.py --batch produces ONE JSON aggregating all trials.
# Output: outputs/audit_<parent-basename>.json + .log
#
# Concurrency knobs:
#   N_PARENTS     — how many parent dirs audited simultaneously (default 4)
#   N_CONCURRENT  — trials per parent (passed to audit_agent.py, default 5)
#   Total active audits ≈ N_PARENTS * N_CONCURRENT.
#
# Usage:
#   bash scripts/audit_all.sh
#   N_PARENTS=8 N_CONCURRENT=5 bash scripts/audit_all.sh
#   PARENT=/path bash scripts/audit_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$REPO_ROOT/.env" ] && { set -a; . "$REPO_ROOT/.env"; set +a; }

: "${ANTHROPIC_BASE_URL:?set in .env}"
: "${ANTHROPIC_API_KEY:?set in .env}"

PARENT="${PARENT:-$REPO_ROOT}"
N_CONCURRENT="${N_CONCURRENT:-5}"
N_PARENTS="${N_PARENTS:-4}"
MAXDEPTH="${MAXDEPTH:-3}"

# Machine-specific: directory containing the `claude` CLI on the claude user's
# PATH. Override via env if it's installed somewhere else.
AUDIT_CLAUDE_BIN_DIR="${AUDIT_CLAUDE_BIN_DIR:-/mnt/data2/envs/data/bin}"

mkdir -p /tmp/claude && chmod 777 /tmp/claude
mkdir -p "$REPO_ROOT/outputs"
chmod 777 "$REPO_ROOT/outputs"

# Collect parent dirs of timestamped subdirs (deduped, sorted).
EXP_LIST="$(mktemp)"
trap 'rm -f "$EXP_LIST"' EXIT
find "$PARENT" -maxdepth "$MAXDEPTH" -type d \
    -regex '.*/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]__[0-9][0-9]-[0-9][0-9]-[0-9][0-9]$' \
    2>/dev/null \
    | xargs -I{} dirname {} \
    | sort -u > "$EXP_LIST"
chmod 644 "$EXP_LIST"
N="$(wc -l < "$EXP_LIST")"
echo "Found $N experiment root(s) under $PARENT (list: $EXP_LIST)"
echo "N_PARENTS=$N_PARENTS  N_CONCURRENT=$N_CONCURRENT  (≈$((N_PARENTS * N_CONCURRENT)) concurrent audits)"
echo ""

# Do NOT use `sudo -E`: preserves HOME=/root → claude CLI EACCES on
# /root/.claude/debug/. See scripts/test_audit_agent.sh.
sudo -u claude bash <<OUTER_EOF
set -uo pipefail
export PATH="${AUDIT_CLAUDE_BIN_DIR}:\$PATH"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
export AUDIT_N_CONCURRENT="${N_CONCURRENT}"
export AUDIT_LOG_DIR="${REPO_ROOT}/outputs"

cd "${REPO_ROOT}"

audit_one_parent() {
    local ROOT_DIR="\$1"
    local NAME OUT LOG
    NAME="\$(basename "\$ROOT_DIR")"
    OUT="\$AUDIT_LOG_DIR/audit_\${NAME}.json"
    LOG="\$AUDIT_LOG_DIR/audit_\${NAME}.log"

    if [ -s "\$OUT" ]; then
        echo "[skip]  \$NAME (already audited)"
        return 0
    fi

    if ! find "\$ROOT_DIR" -maxdepth 3 -type d -name agent -print -quit 2>/dev/null | grep -q .; then
        echo "[skip]  \$NAME (no trials)"
        return 0
    fi

    echo "[start] \$NAME"
    if python scripts/audit_agent.py \\
            --batch "\$ROOT_DIR" \\
            --base-url "\$ANTHROPIC_BASE_URL" \\
            --api-keys "\$ANTHROPIC_API_KEY" \\
            --model claude-opus-4-6 \\
            --n-concurrent "\$AUDIT_N_CONCURRENT" \\
            -o "\$OUT" >"\$LOG" 2>&1; then
        echo "[done]  \$NAME"
    else
        echo "[FAIL]  \$NAME (see \$LOG)"
    fi
}
export -f audit_one_parent

# xargs -P N_PARENTS runs N parent-audits concurrently. Stdout lines from
# parallel audit_one_parent calls interleave, but each parent's full output
# still lands atomically in its own audit_<name>.log.
xargs -P ${N_PARENTS} -I{} bash -c 'audit_one_parent "\$@"' _ {} < "${EXP_LIST}"

echo ""
echo "=== All audits finished ==="
echo "  results:   \$AUDIT_LOG_DIR"
OUTER_EOF
