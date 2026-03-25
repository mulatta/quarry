"""AWS Batch ETL wrapper asset.

Submits the quarry-etl job to AWS Batch (Spot) and polls until completion.
DO NOT use `from __future__ import annotations` here — Dagster inspects types at runtime.
"""

import time
from datetime import date

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry.config import settings


@asset(
    group_name="batch",
    description="Submit quarry-etl job to AWS Batch, poll until SUCCEEDED/FAILED.",
    kinds={"aws"},
)
def etl_on_batch(context: AssetExecutionContext) -> MaterializeResult:
    import boto3

    batch = boto3.client("batch")
    job_name = f"quarry-etl-{date.today().isoformat()}"

    context.log.info(f"[Batch] submitting {job_name}")
    resp = batch.submit_job(
        jobName=job_name,
        jobQueue=settings.batch_job_queue,
        jobDefinition=settings.batch_job_definition,
    )
    job_id = resp["jobId"]
    context.log.info(f"[Batch] submitted: {job_id}")

    # Poll with exponential backoff (cap at 5 min)
    delay = 30
    while True:
        desc = batch.describe_jobs(jobs=[job_id])
        status = desc["jobs"][0]["status"]
        context.log.info(f"[Batch] {job_id}: {status}")
        if status == "SUCCEEDED":
            break
        if status == "FAILED":
            reason = desc["jobs"][0].get("statusReason", "unknown")
            raise RuntimeError(f"Batch job failed: {reason}")
        time.sleep(delay)
        delay = min(delay * 2, 300)

    return MaterializeResult(
        metadata={
            "job_id": MetadataValue.text(job_id),
            "job_name": MetadataValue.text(job_name),
        }
    )
