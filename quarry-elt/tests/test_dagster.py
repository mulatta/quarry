"""Dagster ELT pipeline tests.

Requires: quarry-parse parquet files exist (run parse tests first),
ClickHouse on port 9000, PostgreSQL on /tmp/quarry-pg.
Uses quarry_test databases (created by conftest.py fixtures).
"""

import logging
import subprocess
from pathlib import Path

# Share test data with tests/integration/
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "tests" / "integration" / "data"


class TestDefsLoad:
    def test_definitions_resolve(self):
        """Verify Dagster Definitions load without import/validation errors."""
        from quarry_elt.defs import defs

        assert defs.resolve_implicit_global_asset_job_def() is not None


class TestDagsterPipeline:
    """Full Dagster E2E: ch_load → ch_transform → parquet_export → pg_load."""

    def test_e2e(self):
        from dagster import AssetKey, Definitions, load_assets_from_modules

        from quarry_elt.assets import download, load, stage
        from quarry_elt.resources import PGResource

        defs = Definitions(
            assets=load_assets_from_modules([download, stage, load]),
            resources={"pg": PGResource()},
        )

        logging.getLogger("dagster").setLevel(logging.INFO)

        job = defs.resolve_implicit_global_asset_job_def()
        result = job.execute_in_process(
            asset_selection=[
                AssetKey("oa_parse"),
                AssetKey("pm_parse"),
                AssetKey("mesh_stage"),
                AssetKey("ch_load_oa"),
                AssetKey("ch_load_pm"),
                AssetKey("ch_load_mesh"),
                AssetKey("ch_load_icite"),
                AssetKey("ch_transform"),
                AssetKey("parquet_export"),
                AssetKey("pg_load"),
            ],
        )

        for event in result.all_events:
            if hasattr(event, "event_type_value"):
                if event.event_type_value == "STEP_FAILURE":
                    print(f"FAILED: {event.step_key}")
                elif event.event_type_value == "STEP_SUCCESS":
                    print(f"OK: {event.step_key}")

        assert result.success, "Pipeline failed — see step failures above"

        # Verify Parquet export created files
        pq_dir = DATA_DIR / "parquet"
        flat_files = list(pq_dir.glob("*.parquet")) if pq_dir.exists() else []
        works_files = (
            list((pq_dir / "works").rglob("*.parquet"))
            if (pq_dir / "works").exists()
            else []
        )
        tier_dirs = (
            [d for d in (pq_dir / "works").iterdir() if d.is_dir()]
            if (pq_dir / "works").exists()
            else []
        )
        print(
            f"\n=== Parquet: {len(flat_files)} flat + {len(works_files)} works "
            f"in {len(tier_dirs)} tier dirs ==="
        )
        for f in sorted(flat_files):
            print(f"  {f.name}: {f.stat().st_size:,} bytes")
        for d in sorted(tier_dirs):
            n = len(list(d.glob("*.parquet")))
            print(f"  {d.name}/: {n} files")
        assert len(flat_files) == 14, (
            f"Expected 14 flat parquet files, got {len(flat_files)}"
        )
        assert len(tier_dirs) > 0, "Expected hive tier directories"
        assert len(works_files) > 0, "Expected works bucket parquet files"

        # Verify PG has data
        from quarry.config import settings

        r = subprocess.run(
            ["psql", settings.pg_conninfo, "-t", "-c", "SELECT count(*) FROM works"],
            capture_output=True,
            text=True,
        )
        count = int(r.stdout.strip())
        assert count > 0, f"Expected works in PG, got {count}"
        print(f"\n=== PG works: {count} ===")

        # Verify join worked (works_export has PM/iCite enrichment)
        r = subprocess.run(
            [
                "psql",
                settings.pg_conninfo,
                "-t",
                "-c",
                "SELECT count(*) FROM works WHERE rcr IS NOT NULL",
            ],
            capture_output=True,
            text=True,
        )
        enriched = int(r.stdout.strip())
        print(f"=== Works with iCite RCR: {enriched} ===")
        assert enriched > 0, "No iCite enrichment — JOIN failed"

        # Summary: table row counts
        tables = [
            "works",
            "papers",
            "work_authors",
            "work_topics",
            "work_citations",
            "mesh_headings",
            "mesh_tree",
            "id_crosswalk",
            "cited_by_clin",
        ]
        print("\n=== PG table summary ===")
        for t in tables:
            r = subprocess.run(
                ["psql", settings.pg_conninfo, "-t", "-c", f"SELECT count(*) FROM {t}"],
                capture_output=True,
                text=True,
            )
            cnt = r.stdout.strip() if r.returncode == 0 else "ERROR"
            print(f"  {t}: {cnt}")

        # Verify hive parquet → PyArrow scan (embedding pipeline input)
        import pyarrow.dataset as pa_ds

        works_dir = pq_dir / "works"
        dataset = pa_ds.dataset(works_dir, format="parquet", partitioning="hive")
        scan = dataset.scanner(
            columns=["work_id", "title", "abstract", "tier", "type"],
            filter=pa_ds.field("tier").isin(["t1", "t2"]),
        ).to_table()
        assert len(scan) > 0, "No rows from hive scan with tier filter"
        assert "tier" in scan.column_names, "tier column missing from hive partition"
        print(
            f"\n=== Hive scan (t1+t2): {len(scan)} rows, cols={scan.column_names} ==="
        )

        # Verify LanceStore round-trip with dummy vectors (no GPU needed)
        from quarry.store.lance import LanceStore

        lance_uri = str(DATA_DIR / "lance_test")
        lance = LanceStore(lance_uri)
        lance.create_table()
        row = {
            "work_id": scan.column("work_id")[0].as_py(),
            "content_hash": b"\x00" * 32,
            "title": scan.column("title")[0].as_py(),
            "abstract": scan.column("abstract")[0].as_py(),
            "vec_retrieval": [0.0] * 256,
            "vec_cluster": [0.0] * 256,
        }
        lance.upsert([row])
        count = lance.table.count_rows()
        assert count == 1, f"Expected 1 row in LanceDB, got {count}"
        result = lance.table.search().limit(1).to_list()
        assert result[0]["work_id"] == row["work_id"]
        print(f"=== LanceDB round-trip OK: {row['work_id']} ===")
