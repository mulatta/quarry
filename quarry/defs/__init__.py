"""Dagster Definitions: combine all assets, resources, and schedules."""

from dagster import Definitions, in_process_executor, load_assets_from_modules

from quarry.assets import citations, pubmed, search, supplementary
from quarry.resources import DuckDBResource, LanceDBResource
from quarry.schedules import daily_update_schedule, monthly_citation_schedule

defs = Definitions(
    assets=load_assets_from_modules(
        [pubmed, citations, search, supplementary],
    ),
    resources={
        "duckdb": DuckDBResource(),
        "lancedb": LanceDBResource(),
    },
    schedules=[
        daily_update_schedule,
        monthly_citation_schedule,
    ],
    executor=in_process_executor,
)
