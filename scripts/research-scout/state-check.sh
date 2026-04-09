#!/usr/bin/env bash
# State check: verify state files exist and are non-empty after explorer.
#
# Usage: state-check.sh <state_dir> <sp_id>
# Exit 1 if critical files missing (fail_open: false in hooks.yaml).

set -euo pipefail

STATE_DIR="${1:?Usage: state-check.sh <state_dir> <sp_id>}"
SP_ID="${2:?Usage: state-check.sh <state_dir> <sp_id>}"

SP_DIR="$STATE_DIR/sp_$SP_ID"
ERRORS=0

for required in seeds.yaml findings.yaml; do
    path="$SP_DIR/$required"
    if [ ! -f "$path" ]; then
        echo "FAIL: $path does not exist"
        ERRORS=$((ERRORS + 1))
    elif [ ! -s "$path" ]; then
        echo "FAIL: $path is empty"
        ERRORS=$((ERRORS + 1))
    else
        echo "OK: $path"
    fi
done

if [ $ERRORS -gt 0 ]; then
    echo "State check failed: $ERRORS errors"
    exit 1
fi
echo "State check passed for sp_$SP_ID"
