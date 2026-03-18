"""Dagster schedules for periodic ETL jobs."""

from dagster import (
    AssetSelection,
    ScheduleDefinition,
)

# Daily: download updates → stage → load into DuckDB, plus bioRxiv
daily_update_schedule = ScheduleDefinition(
    name="daily_update",
    target=AssetSelection.keys(
        "pubmed_updates_sync",
        "pubmed_updates_stage",
        "biorxiv_stage",
        "duckdb_load",
    ),
    cron_schedule="0 6 * * *",  # 06:00 UTC daily
)

# Monthly: download iCite → stage + CSR graph → load
monthly_citation_schedule = ScheduleDefinition(
    name="monthly_citation",
    target=AssetSelection.keys(
        "icite_occ_sync",
        "csr_graph",
        "icite_metadata_sync",
        "icite_metrics_stage",
        "duckdb_load",
    ),
    cron_schedule="0 2 1 * *",  # 02:00 UTC, 1st of each month
)
