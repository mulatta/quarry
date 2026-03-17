#!/usr/bin/env bash
# PubMed FTP baseline download + ETL into DuckDB.
# Run once on initial setup.
set -euo pipefail

BASELINE_DIR="${QUARRY_PUBMED_BASELINE_DIR:-/workspace/seungwon/quarry/pubmed/baseline}"

echo "=== Step 1: Download PubMed baseline ==="
mkdir -p "$BASELINE_DIR"
cd "$BASELINE_DIR"
# Download all baseline .xml.gz files (~657 files, ~11GB total)
# Uses lftp for parallel FTP download
lftp -c "
  open ftp.ncbi.nlm.nih.gov;
  mirror --only-newer --parallel=4 /pubmed/baseline/ ./ --include-glob=pubmed*.xml.gz;
"

echo "=== Step 2: Load into DuckDB ==="
python -m quarry.etl.runner baseline --batch-size 10000

echo "=== Done ==="
