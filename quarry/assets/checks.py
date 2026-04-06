"""Dagster asset checks: post-materialization data quality validation.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import AssetCheckResult, asset_check

from quarry.assets.load import pg_load
from quarry.resources import PGResource

# Minimum expected row counts (order-of-magnitude sanity check).
_MIN_ROWS = {
    "works": 100_000_000,
    "papers": 30_000_000,
    "work_citations": 1_000_000_000,
}


@asset_check(asset=pg_load, description="Verify PG key tables have expected row counts.")
def pg_row_count_check(pg: PGResource) -> AssetCheckResult:
    results = {}
    all_passed = True
    for table, minimum in _MIN_ROWS.items():
        rows = pg.store.query(f"SELECT count(*) AS n FROM {table}")[0]["n"]
        results[f"{table}_rows"] = rows
        results[f"{table}_min"] = minimum
        if rows < minimum:
            all_passed = False
    return AssetCheckResult(passed=all_passed, metadata=results)
