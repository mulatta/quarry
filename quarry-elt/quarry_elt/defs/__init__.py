"""Dagster Definitions: combine all assets, resources, jobs, and schedules."""

from dagster import Definitions, load_assets_from_modules, multiprocess_executor

from quarry_elt.assets import (
    citations,
    download,
    export,
    import_,
    load,
    search,
    stage,
)
from quarry_elt.assets.checks import pg_row_count_check
from quarry_elt.jobs import (
    build_job,
    embeddings_job,
    full_job,
    load_job,
    r2_download_job,
    r2_upload_job,
)
from quarry_elt.resources import PGResource
from quarry_elt.schedules import (
    daily_update_schedule,
    monthly_citation_schedule,
)
from quarry_elt.sensors import distributed_r2_sync, distributed_serve, r2_upload_sensor

defs = Definitions(
    assets=load_assets_from_modules(
        [download, stage, load, export, import_, citations, search],
    ),
    asset_checks=[pg_row_count_check],
    jobs=[
        build_job,
        load_job,
        embeddings_job,
        full_job,
        r2_upload_job,
        r2_download_job,
    ],
    resources={
        "pg": PGResource(),
    },
    schedules=[
        daily_update_schedule,
        monthly_citation_schedule,
    ],
    sensors=[
        r2_upload_sensor,
        distributed_r2_sync,
        distributed_serve,
    ],
    executor=multiprocess_executor,
)
