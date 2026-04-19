"""CH schema initialization asset.

Reads ch_schema.sql and executes DDL statements to create the CH database
and staging tables. Separated from load.py (DML) — different change frequency
and failure modes.
"""

import re
from pathlib import Path

from dagster import (
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    asset,
)

from quarry_elt.assets.helpers import run
from quarry.config import settings

_SQL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "sql"
_CH_SCHEMA_SQL = _SQL_DIR / "ch_schema.sql"


def _ch_query_no_db(
    query: str, context: AssetExecutionContext, label: str | None = None
) -> None:
    """Run a CH query without --database (for CREATE DATABASE)."""
    cmd = [
        "clickhouse-client",
        "--host",
        settings.ch_host,
        "--port",
        str(settings.ch_port),
        "--query",
        query,
    ]
    run(cmd, context, label=label or f"[CH] {query}")


def _ch_query_init(
    query: str, context: AssetExecutionContext, label: str | None = None
) -> None:
    """Run a CH query with --database for DDL statements."""
    cmd = [
        "clickhouse-client",
        "--host",
        settings.ch_host,
        "--port",
        str(settings.ch_port),
        "--database",
        settings.ch_database,
        "--query",
        query,
    ]
    run(cmd, context, label=label or f"[CH] {query}")


@asset(
    group_name="init",
    description="Create CH database + tables from ch_schema.sql.",
    kinds={"clickhouse"},
)
def ch_init(context: AssetExecutionContext) -> MaterializeResult:
    content = _CH_SCHEMA_SQL.read_text()
    # Strip block comments (/* ... */) before splitting
    cleaned = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    stmts = [s.strip() for s in cleaned.split(";") if s.strip()]

    for stmt in stmts:
        if stmt.startswith("--"):
            continue
        # CREATE DATABASE runs without --database flag
        if stmt.upper().startswith("CREATE DATABASE"):
            _ch_query_no_db(stmt, context, label=f"[CH] {stmt.split(chr(10))[0]}")
        elif stmt.upper().startswith("USE"):
            continue  # --database flag handles this
        else:
            # DROP + CREATE to ensure schema matches ch_schema.sql.
            # CH tables are staging-only; data is rebuilt every pipeline run.
            m = re.match(
                r"CREATE TABLE IF NOT EXISTS\s+(\S+)",
                stmt,
                re.IGNORECASE,
            )
            if m:
                table = m.group(1)
                _ch_query_init(f"DROP TABLE IF EXISTS {table}", context)
                stmt = stmt.replace("IF NOT EXISTS ", "", 1)
            _ch_query_init(stmt, context)

    return MaterializeResult(
        metadata={"status": MetadataValue.text("ok")},
    )
