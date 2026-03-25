"""Dagster sensors for distributed pipeline chaining.

Distributed workflow (sensor-driven):
  weekly schedule → etl_on_batch
                        ↓ distributed_r2_sync sensor
                   r2_download
                        ↓ distributed_serve sensor
                   serve job (pg_load + csr_graph + embeddings)
"""

from dagster import (
    AssetKey,
    RunRequest,
    asset_sensor,
)

from quarry.jobs import r2_download_job, serve_job


@asset_sensor(asset_key=AssetKey("etl_on_batch"), job=r2_download_job)
def distributed_r2_sync(context):
    """Trigger R2 download when Batch ETL completes."""
    yield RunRequest()


@asset_sensor(asset_key=AssetKey("r2_download"), job=serve_job)
def distributed_serve(context):
    """Trigger serve pipeline when R2 download completes."""
    yield RunRequest()
