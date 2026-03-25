"""Dagster Definitions: combine all assets, resources, jobs, and schedules."""

try:
    from dagster import Definitions, load_assets_from_modules, multiprocess_executor
except ImportError:
    raise ImportError("pip install quarry[elt]") from None

from quarry.assets import (
    batch,
    citations,
    download,
    export,
    import_,
    load,
    search,
    stage,
)
from quarry.jobs import (
    embeddings_job,
    etl_job,
    full_job,
    r2_download_job,
    serve_job,
)
from quarry.resources import PGResource
from quarry.schedules import (
    daily_update_schedule,
    monthly_citation_schedule,
    weekly_distributed_schedule,
)
from quarry.sensors import distributed_r2_sync, distributed_serve

defs = Definitions(
    assets=load_assets_from_modules(
        [download, stage, load, export, import_, batch, citations, search],
    ),
    jobs=[
        etl_job,
        serve_job,
        embeddings_job,
        full_job,
        r2_download_job,
    ],
    resources={
        "pg": PGResource(),
    },
    schedules=[
        daily_update_schedule,
        monthly_citation_schedule,
        weekly_distributed_schedule,
    ],
    sensors=[
        distributed_r2_sync,
        distributed_serve,
    ],
    executor=multiprocess_executor,
)
