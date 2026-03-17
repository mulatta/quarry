#!/usr/bin/env bash
# Monthly: download iCite OCC + rebuild CSR + update metrics.
# Add to cron: 0 2 15 * * /path/to/monthly_citation.sh >> /var/log/quarry_monthly.log 2>&1
set -euo pipefail

ICITE_DIR="${QUARRY_ICITE_DIR:-/workspace/seungwon/quarry/icite}"

echo "$(date -Iseconds) === Monthly citation rebuild start ==="

mkdir -p "$ICITE_DIR"
cd "$ICITE_DIR"

echo "--- Download iCite OCC from figshare ---"
# URL changes monthly; update as needed
OCC_URL="https://nih.figshare.com/ndownloader/files/open_citation_collection.zip"
if [ ! -f open_citation_collection.csv ] || [ "$(find open_citation_collection.csv -mtime +35)" ]; then
    curl -L -o occ.zip "$OCC_URL"
    unzip -o occ.zip
    rm -f occ.zip
fi

echo "--- Build CSR mmap graph ---"
python -m quarry.etl.icite citations --csv "$ICITE_DIR/open_citation_collection.csv"

echo "--- Download iCite metadata ---"
METADATA_URL="https://nih.figshare.com/ndownloader/files/icite_metadata.zip"
if [ ! -f icite_metadata.csv ] || [ "$(find icite_metadata.csv -mtime +35)" ]; then
    curl -L -o meta.zip "$METADATA_URL"
    unzip -o meta.zip
    rm -f meta.zip
fi

echo "--- Update papers metrics ---"
python -m quarry.etl.icite metrics --csv "$ICITE_DIR/icite_metadata.csv"

echo "$(date -Iseconds) === Monthly citation rebuild done ==="
