"""R2 upload asset: sync local Parquet files to Cloudflare R2.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.assets.helpers import run
from quarry.assets.load import parquet_export
from quarry.config import settings


@asset(
    group_name="export",
    deps=[parquet_export],
    description="Sync Parquet files → Cloudflare R2 via aws s3 sync.",
    kinds={"s3"},
)
def r2_upload(context: AssetExecutionContext) -> MaterializeResult:
    dest = f"s3://{settings.r2_bucket}/latest/parquet/"
    cmd = [
        "aws",
        "s3",
        "sync",
        str(settings.parquet_dir),
        dest,
        "--endpoint-url",
        settings.r2_endpoint,
    ]
    run(cmd, context, label=f"[R2] sync {settings.parquet_dir} → {dest}")
    return MaterializeResult(
        metadata={"destination": MetadataValue.text(dest)},
    )
