"""R2 download asset: sync Parquet files from Cloudflare R2 to local.

DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry_elt.assets.helpers import run
from quarry.config import settings


@asset(
    group_name="import",
    description="Sync Parquet files from Cloudflare R2 → local via aws s3 sync.",
    kinds={"s3"},
)
def r2_download(context: AssetExecutionContext) -> MaterializeResult:
    src = f"s3://{settings.r2_bucket}/latest/parquet/"
    cmd = [
        "aws",
        "s3",
        "sync",
        src,
        str(settings.parquet_dir),
        "--endpoint-url",
        settings.r2_endpoint,
    ]
    run(cmd, context, label=f"[R2] sync {src} → {settings.parquet_dir}")
    return MaterializeResult(
        metadata={"source": MetadataValue.text(src)},
    )
