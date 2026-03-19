"""MeSH descriptor XML → DuckDB mesh_tree table.

Uses Rust quarry_parse.parse_mesh_xml() for fast XML parsing → Arrow RecordBatch.
"""

import time
from pathlib import Path

import pyarrow as pa
import quarry_parse

from quarry.store.duckdb import DuckDBStore


def parse_mesh_descriptors(xml_path: Path) -> pa.Table:
    """Parse MeSH descriptor XML → Arrow Table (descriptor_ui, descriptor_name, tree_number)."""
    batch = quarry_parse.parse_mesh_xml(str(xml_path))
    return pa.Table.from_batches([batch])


def load_mesh_tree(xml_path: Path, db: DuckDBStore | None = None):
    """Parse MeSH XML and replace mesh_tree table in DuckDB."""
    close_db = False
    if db is None:
        db = DuckDBStore()
        db.init_schema()
        close_db = True

    print(f"Parsing MeSH descriptors from {xml_path} ...")
    t0 = time.time()
    table = parse_mesh_descriptors(xml_path)
    print(f"  Parsed {table.num_rows:,} tree entries in {time.time() - t0:.1f}s")

    db.conn.execute("DELETE FROM mesh_tree")
    db.conn.register("_tmp_mesh", table)
    db.conn.execute(
        "INSERT INTO mesh_tree (descriptor_ui, descriptor_name, tree_number) "
        "SELECT descriptor_ui, descriptor_name, tree_number FROM _tmp_mesh"
    )
    db.conn.unregister("_tmp_mesh")
    print(f"  Loaded {table.num_rows:,} rows into mesh_tree")

    if close_db:
        db.close()
