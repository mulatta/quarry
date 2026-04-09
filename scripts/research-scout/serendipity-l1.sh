#!/usr/bin/env bash
# Serendipity L1: deterministic keyword/MeSH grep across sub-problems.
# Called by orchestrator after each explorer completes.
#
# Usage: serendipity-l1.sh <state_dir> <completed_sp_id>
#
# Reads: state/sub_problems.yaml (all search terms)
#        state/sp_{completed}/seeds.yaml (paper titles + MeSH)
# Writes: state/sp_{completed}/serendipity_flags.yaml

set -euo pipefail

STATE_DIR="${1:?Usage: serendipity-l1.sh <state_dir> <completed_sp_id>}"
COMPLETED_SP="${2:?Usage: serendipity-l1.sh <state_dir> <completed_sp_id>}"

SP_DIR="$STATE_DIR/sp_$COMPLETED_SP"
FLAGS_FILE="$SP_DIR/serendipity_flags.yaml"

# Extract search terms for ALL other sub-problems
OTHER_TERMS=$(python3 -c "
import yaml, sys
with open('$STATE_DIR/sub_problems.yaml') as f:
    data = yaml.safe_load(f)
for sp in data.get('sub_problems', []):
    if str(sp['id']) != '$COMPLETED_SP':
        for term in sp.get('search', []):
            print(f\"{sp['id']}|{term}\")
")

if [ -z "$OTHER_TERMS" ]; then
    echo "serendipity_flags: []" > "$FLAGS_FILE"
    exit 0
fi

# Check seeds and expand_raw for keyword matches
FLAGS="serendipity_flags:"
FOUND=0

for file in "$SP_DIR/seeds.yaml" "$SP_DIR/expand_raw.txt"; do
    [ -f "$file" ] || continue
    while IFS='|' read -r other_sp_id term; do
        # Case-insensitive grep for the term in paper titles
        matches=$(grep -i "$term" "$file" 2>/dev/null | head -3)
        if [ -n "$matches" ]; then
            FLAGS="$FLAGS
  - current_sp: $COMPLETED_SP
    matched_sp: $other_sp_id
    match_type: title_keyword
    matched_term: \"$term\"
    source_file: $(basename "$file")
    match_preview: \"$(echo "$matches" | head -1 | cut -c1-100)\""
            FOUND=$((FOUND + 1))
        fi
    done <<< "$OTHER_TERMS"
done

echo "$FLAGS" > "$FLAGS_FILE"
echo "L1 serendipity: $FOUND flags written to $FLAGS_FILE"
