"""Dagster sensors for pipeline chaining.

Local workflow (sensor-driven R2 sync after build):
  build job done → r2_upload_sensor → r2_upload

Distributed workflow (sensor-driven cross-machine):
  r2_upload done → distributed_r2_sync → r2_download (remote)
  r2_download done → distributed_serve → load job (remote)
"""

from dagster import (
    AssetKey,
    RunRequest,
    asset_sensor,
)

from quarry.jobs import load_job, r2_download_job, r2_upload_job


@asset_sensor(asset_key=AssetKey("parquet_export"), job=r2_upload_job)
def r2_upload_sensor(context):
    """Trigger R2 upload when parquet export completes."""
    yield RunRequest()


@asset_sensor(asset_key=AssetKey("r2_upload"), job=r2_download_job)
def distributed_r2_sync(context):
    """Trigger R2 download when upload completes (remote instance)."""
    yield RunRequest()


@asset_sensor(asset_key=AssetKey("r2_download"), job=load_job)
def distributed_serve(context):
    """Trigger serve pipeline when R2 download completes (remote instance)."""
    yield RunRequest()
