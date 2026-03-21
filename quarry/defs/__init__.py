"""Dagster Definitions: combine all assets, resources, and schedules."""

from dagster import Definitions, load_assets_from_modules, multiprocess_executor

from quarry.assets import citations, download, load, search, stage
from quarry.resources import PGResource
from quarry.schedules import daily_update_schedule, monthly_citation_schedule

defs = Definitions(
    assets=load_assets_from_modules(
        [download, stage, load, citations, search],
    ),
    resources={
        "pg": PGResource(),
    },
    schedules=[
        daily_update_schedule,
        monthly_citation_schedule,
    ],
    executor=multiprocess_executor,
)
