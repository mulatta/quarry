#!/usr/bin/env bash
# Pre-synthesis verification: check seed validity and bridge consistency.
#
# Usage: verification.sh <state_dir>
# Exit 1 if critical issues found.

set -euo pipefail

STATE_DIR="${1:?Usage: verification.sh <state_dir>}"
ERRORS=0
WARNINGS=0

echo "=== Verification: pre-synthesis checks ==="

# 1. Every sub-problem has at least 1 seed
for sp_dir in "$STATE_DIR"/sp_*/; do
    [ -d "$sp_dir" ] || continue
    sp_id=$(basename "$sp_dir" | sed 's/sp_//')
    seeds="$sp_dir/seeds.yaml"
    if [ ! -f "$seeds" ] || [ ! -s "$seeds" ]; then
        echo "ERROR: sp_$sp_id has no seeds"
        ERRORS=$((ERRORS + 1))
    else
        count=$(grep -c "work_id:" "$seeds" 2>/dev/null || echo 0)
        if [ "$count" -eq 0 ]; then
            echo "ERROR: sp_$sp_id seeds.yaml has no work_id entries"
            ERRORS=$((ERRORS + 1))
        else
            echo "OK: sp_$sp_id has $count seed(s)"
        fi
    fi
done

# 2. No duplicate seeds across sub-problems
all_seeds=$(cat "$STATE_DIR"/sp_*/seeds.yaml 2>/dev/null | grep "work_id:" | awk '{print $2}' | sort)
dups=$(echo "$all_seeds" | uniq -d)
if [ -n "$dups" ]; then
    echo "WARNING: duplicate seeds across sub-problems: $dups"
    WARNINGS=$((WARNINGS + 1))
fi

# 3. At least 1 bridge result exists
bridge_count=$(ls -d "$STATE_DIR"/bridge_*/ 2>/dev/null | wc -l)
if [ "$bridge_count" -eq 0 ]; then
    echo "WARNING: no bridge results found"
    WARNINGS=$((WARNINGS + 1))
else
    echo "OK: $bridge_count bridge pair(s)"
fi

# 4. Serendipity log exists (may be empty)
if [ -f "$STATE_DIR/serendipity.yaml" ]; then
    scount=$(grep -c "source:" "$STATE_DIR/serendipity.yaml" 2>/dev/null || echo 0)
    echo "OK: serendipity.yaml with $scount entries"
else
    echo "INFO: no serendipity.yaml (will be noted in report)"
fi

echo "=== Result: $ERRORS errors, $WARNINGS warnings ==="
if [ $ERRORS -gt 0 ]; then
    exit 1
fi
