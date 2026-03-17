"""MeSH descriptor XML → DuckDB mesh_tree table.

Parses the NLM MeSH descriptor XML (desc20XX.xml) to extract
descriptor_ui, descriptor_name, and tree_number(s).

Usage:
    python -m quarry.etl.mesh --xml ~/data/mesh/desc2025.xml
"""

import argparse
import time
from pathlib import Path

import pyarrow as pa
from lxml import etree

from quarry.store.duckdb import DuckDBStore


def parse_mesh_descriptors(xml_path: Path) -> list[dict]:
    """Parse MeSH descriptor XML → list of (descriptor_ui, descriptor_name, tree_number)."""
    rows = []

    context = etree.iterparse(str(xml_path), events=("end",), tag="DescriptorRecord")

    for _, elem in context:
        ui = elem.findtext("DescriptorUI", "").strip()
        name_el = elem.find(".//DescriptorName/String")
        name = name_el.text.strip() if name_el is not None and name_el.text else ""

        if not ui or not name:
            elem.clear()
            continue

        for tree_num in elem.findall(".//TreeNumber"):
            tn = (tree_num.text or "").strip()
            if tn:
                rows.append(
                    {
                        "descriptor_ui": ui,
                        "descriptor_name": name,
                        "tree_number": tn,
                    }
                )

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return rows


def load_mesh_tree(xml_path: Path, db: DuckDBStore | None = None):
    """Parse MeSH XML and replace mesh_tree table in DuckDB."""
    close_db = False
    if db is None:
        db = DuckDBStore()
        db.init_schema()
        close_db = True

    print(f"Parsing MeSH descriptors from {xml_path} ...")
    t0 = time.time()
    rows = parse_mesh_descriptors(xml_path)
    print(f"  Parsed {len(rows):,} tree entries in {time.time() - t0:.1f}s")

    # Replace table contents via pyarrow bulk load
    db.conn.execute("DELETE FROM mesh_tree")
    arrow_table = pa.table(
        {
            "descriptor_ui": pa.array(
                [r["descriptor_ui"] for r in rows], type=pa.string()
            ),
            "descriptor_name": pa.array(
                [r["descriptor_name"] for r in rows], type=pa.string()
            ),
            "tree_number": pa.array([r["tree_number"] for r in rows], type=pa.string()),
        }
    )
    db.conn.register("_tmp_mesh", arrow_table)
    db.conn.execute(
        "INSERT INTO mesh_tree (descriptor_ui, descriptor_name, tree_number) "
        "SELECT descriptor_ui, descriptor_name, tree_number FROM _tmp_mesh"
    )
    db.conn.unregister("_tmp_mesh")
    print(f"  Loaded {len(rows):,} rows into mesh_tree")

    if close_db:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="MeSH descriptor XML → DuckDB mesh_tree"
    )
    parser.add_argument("--xml", type=Path, required=True, help="Path to descXXXX.xml")
    args = parser.parse_args()
    load_mesh_tree(args.xml)


if __name__ == "__main__":
    main()
