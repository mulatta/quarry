"""Dagster schedules for periodic ETL jobs.

Schedules only trigger download assets. Downstream stage/load assets
run automatically via AutomationCondition.eager() when upstream
DataVersion changes.
"""

from dagster import (
    AssetSelection,
    ScheduleDefinition,
)

# Daily: download PubMed updates + bioRxiv fetch
# Stage/load triggered by AutomationCondition on DataVersion change
daily_update_schedule = ScheduleDefinition(
    name="daily_update",
    target=AssetSelection.assets(
        "pubmed_updates_sync",
    ),
    cron_schedule="0 6 * * *",  # 06:00 UTC daily
)

# Monthly: download iCite snapshot
# Downstream load/enrich/CSR triggered by AutomationCondition on DataVersion change
monthly_citation_schedule = ScheduleDefinition(
    name="monthly_citation",
    target=AssetSelection.assets(
        "icite_metadata_sync",
    ),
    cron_schedule="0 2 1 * *",  # 02:00 UTC, 1st of each month
)

# Weekly: kick off distributed pipeline (sensors chain r2_download → serve)
weekly_distributed_schedule = ScheduleDefinition(
    name="weekly_distributed",
    target=AssetSelection.assets("etl_on_batch"),
    cron_schedule="0 2 * * 0",  # 02:00 UTC, Sundays
)
