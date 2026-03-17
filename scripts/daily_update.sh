#!/usr/bin/env bash
# Daily PubMed update: FTP sync + parse + embed.
# Add to cron: 0 6 * * * /path/to/daily_update.sh >> /var/log/quarry_daily.log 2>&1
set -euo pipefail

UPDATE_DIR="${QUARRY_PUBMED_UPDATE_DIR:-/workspace/seungwon/quarry/pubmed/updatefiles}"

echo "$(date -Iseconds) === Daily update start ==="

echo "--- Sync update files ---"
mkdir -p "$UPDATE_DIR"
cd "$UPDATE_DIR"
lftp -c "
  open ftp.ncbi.nlm.nih.gov;
  mirror --only-newer --parallel=2 /pubmed/updatefiles/ ./ --include-glob=pubmed*.xml.gz;
"

echo "--- Parse + upsert into DuckDB ---"
python -m quarry.etl.runner update

echo "--- Fetch bioRxiv preprints (last 2 days) ---"
python -m quarry.etl.biorxiv --days 2
python -m quarry.etl.biorxiv --server medrxiv --days 2

echo "$(date -Iseconds) === Daily update done ==="
