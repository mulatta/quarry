"""Dagster assets: CH-native ingest from OpenAlex gzip JSONL via file() function."""

import subprocess
from importlib import resources as pkg_resources

import dagster as dg

from quarry.config import settings


def _run_ch_sql(
    sql_file: str, path_glob: str, context: dg.AssetExecutionContext
) -> int:
    """Read a SQL template from quarry/sql/, substitute {path}, execute via clickhouse-client."""
    sql_text = pkg_resources.files("quarry.sql").joinpath(sql_file).read_text()
    sql = sql_text.replace("{path}", path_glob)

    cmd = [
        "clickhouse-client",
        f"--host={settings.clickhouse_host}",
        "--port=9000",
        f"--database={settings.clickhouse_database}",
        "--max_execution_time=7200",
        "--max_memory_usage=32000000000",
    ]
    context.log.info(f"Executing {sql_file} with path={path_glob}")
    result = subprocess.run(cmd, input=sql, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"clickhouse-client failed: {result.stderr}")
    return 0


@dg.asset(
    description="Ingest OpenAlex papers into ClickHouse via native file() function.",
    kinds={"clickhouse"},
)
def ch_papers(context: dg.AssetExecutionContext) -> dg.Output[dict]:
    glob = "openalex_works/**/*.gz"
    _run_ch_sql("ingest_papers.sql", glob, context)

    # Report count
    result = subprocess.run(
        [
            "clickhouse-client",
            f"--host={settings.clickhouse_host}",
            "--query=SELECT count() FROM quarry.papers",
        ],
        capture_output=True,
        text=True,
    )
    count = int(result.stdout.strip()) if result.stdout.strip() else 0
    context.log.info(f"Papers: {count:,} rows")
    return dg.Output({"count": count}, metadata={"count": count})


@dg.asset(
    deps=["ch_papers"],
    description="Ingest citation edges into ClickHouse via native file() function.",
    kinds={"clickhouse"},
)
def ch_edges(context: dg.AssetExecutionContext) -> dg.Output[dict]:
    glob = "openalex_works/**/*.gz"
    _run_ch_sql("ingest_edges.sql", glob, context)

    result = subprocess.run(
        [
            "clickhouse-client",
            f"--host={settings.clickhouse_host}",
            "--query=SELECT count() FROM quarry.edges",
        ],
        capture_output=True,
        text=True,
    )
    count = int(result.stdout.strip()) if result.stdout.strip() else 0
    context.log.info(f"Edges: {count:,} rows")
    return dg.Output({"count": count}, metadata={"count": count})


@dg.asset(
    deps=["ch_papers"],
    description="Ingest authors into ClickHouse via native file() function.",
    kinds={"clickhouse"},
)
def ch_authors(context: dg.AssetExecutionContext) -> dg.Output[dict]:
    glob = "openalex_works/**/*.gz"
    _run_ch_sql("ingest_authors.sql", glob, context)

    result = subprocess.run(
        [
            "clickhouse-client",
            f"--host={settings.clickhouse_host}",
            "--query=SELECT count() FROM quarry.authors",
        ],
        capture_output=True,
        text=True,
    )
    count = int(result.stdout.strip()) if result.stdout.strip() else 0
    context.log.info(f"Authors: {count:,} rows")
    return dg.Output({"count": count}, metadata={"count": count})
