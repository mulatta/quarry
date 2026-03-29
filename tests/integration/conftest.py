"""Pytest conftest: set QUARRY_* env vars before any quarry module is imported.

This must run before quarry.config.settings singleton is created.

Test isolation:
  - CH: quarry_test database (port 9000)
  - PG: quarry_test database (socket /tmp/quarry-pg)
  - Paths: tests/data/ and tests/data/parsed/
  - Databases are created at session start, dropped at session end.
"""

import logging
import os
import subprocess
from pathlib import Path

import pytest

logger = logging.getLogger("quarry.test")

TEST_DIR = Path(__file__).parent
TESTS_ROOT = TEST_DIR.parent
DATA_DIR = TEST_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
SQL_DIR = TESTS_ROOT.parent / "sql"

# Test database names
TEST_CH_DATABASE = "quarry_test"
TEST_PG_DBNAME = "quarry_test"
TEST_PG_CONNINFO = f"host=/tmp/quarry-pg dbname={TEST_PG_DBNAME}"

# Override all quarry-data paths to point at tests/data/
_OVERRIDES = {
    "QUARRY_OA_LOCAL_DIR": str(DATA_DIR / "oa" / "works"),
    "QUARRY_PUBMED_BASELINE_DIR": str(DATA_DIR / "pubmed" / "baseline"),
    "QUARRY_PUBMED_UPDATE_DIR": str(DATA_DIR / "pubmed" / "updatefiles"),
    "QUARRY_PUBMED_MESH_DIR": str(DATA_DIR / "pubmed" / "mesh"),
    "QUARRY_ICITE_DIR": str(DATA_DIR / "icite"),
    "QUARRY_OA_PARQUET_DIR": str(PARSED_DIR / "oa"),
    "QUARRY_PM_PARQUET_DIR": str(PARSED_DIR / "pubmed"),
    "QUARRY_MESH_PARQUET_DIR": str(PARSED_DIR / "mesh"),
    "QUARRY_PARQUET_DIR": str(DATA_DIR / "parquet"),
    # Test-isolated databases
    "QUARRY_CH_DATABASE": TEST_CH_DATABASE,
    "QUARRY_PG_CONNINFO": TEST_PG_CONNINFO,
}

for key, val in _OVERRIDES.items():
    os.environ[key] = val

# Add quarry-parse binary to PATH (debug build from crates/)
_PARSE_BIN_DIR = str(
    Path(__file__).parent.parent / "crates" / "quarry-parse" / "target" / "debug"
)
if Path(_PARSE_BIN_DIR).exists():
    os.environ["PATH"] = _PARSE_BIN_DIR + os.pathsep + os.environ.get("PATH", "")


# ── Database helpers ──


def _run_cmd(
    cmd: list[str], label: str, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command with logging."""
    logger.info(f"[{label}] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout.strip():
        for line in r.stdout.strip().split("\n"):
            logger.debug(f"  stdout: {line}")
    if r.stderr.strip():
        for line in r.stderr.strip().split("\n"):
            logger.info(f"  stderr: {line}")
    if check and r.returncode != 0:
        raise RuntimeError(f"[{label}] exit {r.returncode}: {r.stderr}")
    return r


def _ch_cmd(database: str = "default") -> list[str]:
    return [
        "clickhouse-client",
        "--host",
        "localhost",
        "--port",
        "9000",
        "--database",
        database,
    ]


def _ch_query(query: str, database: str = "default", label: str = "CH") -> None:
    _run_cmd(_ch_cmd(database) + ["--query", query], label)


def _pg_conninfo(dbname: str = "quarry") -> str:
    return f"host=/tmp/quarry-pg dbname={dbname}"


def _psql(
    sql: str, dbname: str = "quarry", label: str = "PG"
) -> subprocess.CompletedProcess:
    return _run_cmd(["psql", _pg_conninfo(dbname), "-c", sql], label, check=False)


# ── Session-scoped database fixtures ──


@pytest.fixture(scope="session", autouse=True)
def test_databases():
    """Create quarry_test databases in CH and PG, drop on teardown."""
    # ── CH: create quarry_test ──
    logger.info("=== Creating CH quarry_test database ===")
    _ch_query(f"DROP DATABASE IF EXISTS {TEST_CH_DATABASE}", label="CH setup")
    _ch_query(f"CREATE DATABASE {TEST_CH_DATABASE}", label="CH setup")

    # Apply CH schema (replace USE quarry → USE quarry_test)
    ch_schema = (SQL_DIR / "ch_schema.sql").read_text()
    ch_schema = ch_schema.replace("USE quarry;", f"USE {TEST_CH_DATABASE};")
    _run_cmd(
        _ch_cmd(TEST_CH_DATABASE) + ["--multiquery", "--query", ch_schema],
        "CH schema",
    )
    logger.info("CH quarry_test schema applied")

    # ── PG: create quarry_test ──
    logger.info("=== Creating PG quarry_test database ===")
    # Terminate existing connections
    _psql(
        f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{TEST_PG_DBNAME}' AND pid <> pg_backend_pid()",
        dbname="quarry",
        label="PG setup",
    )
    _psql(
        f"DROP DATABASE IF EXISTS {TEST_PG_DBNAME}", dbname="quarry", label="PG setup"
    )
    r = _psql(f"CREATE DATABASE {TEST_PG_DBNAME}", dbname="quarry", label="PG setup")
    if r.returncode != 0:
        pytest.skip(f"Cannot create PG test database: {r.stderr}")

    # Apply PG schema
    _run_cmd(
        ["psql", _pg_conninfo(TEST_PG_DBNAME), "-f", str(SQL_DIR / "schema.sql")],
        "PG schema",
    )
    logger.info("PG quarry_test schema applied")

    yield

    # ── Teardown ──
    logger.info("=== Dropping test databases ===")

    # Kill active CH queries on quarry_test to prevent orphaned connections
    _run_cmd(
        _ch_cmd("system")
        + [
            "--query",
            f"KILL QUERY WHERE current_database = '{TEST_CH_DATABASE}' SYNC",
        ],
        "CH teardown",
        check=False,
    )
    # Kill any lingering clickhouse-client processes targeting quarry_test
    subprocess.run(
        ["pkill", "-f", f"clickhouse-client.*{TEST_CH_DATABASE}"],
        capture_output=True,
    )

    _ch_query(f"DROP DATABASE IF EXISTS {TEST_CH_DATABASE}", label="CH teardown")
    _psql(
        f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{TEST_PG_DBNAME}' AND pid <> pg_backend_pid()",
        dbname="quarry",
        label="PG teardown",
    )
    _psql(
        f"DROP DATABASE IF EXISTS {TEST_PG_DBNAME}",
        dbname="quarry",
        label="PG teardown",
    )
    logger.info("Test databases dropped")
