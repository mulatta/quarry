"""papeline — Academic paper data pipeline (OpenAlex)."""

from papeline._native import (
    Filter,
    HttpPool,
    OAShard,
    RunSummary,
    TransferSummary,
    fetch_manifest,
    is_shard_complete,
    pull,
    push,
    run,
    run_hive,
    tables,
)

__all__ = [
    "Filter",
    "HttpPool",
    "OAShard",
    "RunSummary",
    "TransferSummary",
    "fetch_manifest",
    "is_shard_complete",
    "pull",
    "push",
    "run",
    "run_hive",
    "tables",
]
