"""Arrow Feather staging: intermediate storage between parse and DB load.

Parse assets write Feather files here; the single duckdb_load asset reads them.
This decouples CPU-bound parsing from the DuckDB single-writer constraint.
"""

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather


def write_tables(staging_dir: Path, tables: dict[str, pa.Table | pa.RecordBatch]):
    """Write a dict of Arrow tables/batches as Feather files."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    for name, data in tables.items():
        if data is None:
            continue
        if isinstance(data, pa.RecordBatch):
            data = pa.Table.from_batches([data])
        if data.num_rows == 0:
            continue
        feather.write_feather(data, staging_dir / f"{name}.feather")


def read_tables(staging_dir: Path) -> dict[str, pa.Table]:
    """Read all Feather files from a staging directory."""
    result = {}
    if not staging_dir.exists():
        return result
    for path in sorted(staging_dir.glob("*.feather")):
        result[path.stem] = feather.read_table(path)
    return result


def iter_chunks(staging_dir: Path):
    """Iterate over numbered chunk subdirectories, yielding tables from each."""
    if not staging_dir.exists():
        return
    for chunk_dir in sorted(d for d in staging_dir.iterdir() if d.is_dir()):
        yield read_tables(chunk_dir)


def clear(staging_dir: Path):
    """Remove all staging files in a subdirectory."""
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
