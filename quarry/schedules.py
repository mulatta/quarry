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
    target=AssetSelection.keys(
        "pubmed_updates_sync",
        "biorxiv_stage",
    ),
    cron_schedule="0 6 * * *",  # 06:00 UTC daily
)

# Monthly: download iCite + CSR graph rebuild
# Stage/load triggered by AutomationCondition on DataVersion change
monthly_citation_schedule = ScheduleDefinition(
    name="monthly_citation",
    target=AssetSelection.keys(
        "icite_occ_sync",
        "icite_metadata_sync",
        "csr_graph",
    ),
    cron_schedule="0 2 1 * *",  # 02:00 UTC, 1st of each month
)
