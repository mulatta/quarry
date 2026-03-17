"""Dagster schedules for periodic ETL jobs."""

from dagster import (
    AssetSelection,
    ScheduleDefinition,
)

# Daily: PubMed updates + bioRxiv preprints
daily_update_schedule = ScheduleDefinition(
    name="daily_update",
    target=AssetSelection.keys("pubmed_daily_update", "biorxiv_preprints"),
    cron_schedule="0 6 * * *",  # 06:00 UTC daily
)

# Monthly: iCite citation graph rebuild + metrics update
monthly_citation_schedule = ScheduleDefinition(
    name="monthly_citation",
    target=AssetSelection.keys("csr_graph", "icite_metrics"),
    cron_schedule="0 2 1 * *",  # 02:00 UTC, 1st of each month
)
