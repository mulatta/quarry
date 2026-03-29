"""E2E pipeline test: API → raw files → quarry-parse → CH → Parquet → PG.

Fetches 100 T1 works from OpenAlex/PubMed/iCite APIs, writes raw input
files, then runs the full Dagster pipeline (parse → ch_load → ch_transform
→ parquet_export → pg_load) with paths overridden to tests/data/.

QUARRY_* env vars are set in conftest.py before any quarry module is imported.

Usage:
    pytest tests/test_pipeline.py -v -s
    pytest tests/test_pipeline.py -v -s -k test_e2e
"""

import csv
import gzip
import json
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from quarry.config import settings

# ── Paths ──

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"

TEST_COUNT = 100

# ── API helpers ──

OA_API = "https://api.openalex.org/works"
ICITE_API = "https://icite.od.nih.gov/api/pubs"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

ICITE_CSV_COLUMNS = [
    "pmid",
    "doi",
    "title",
    "authors",
    "year",
    "journal",
    "is_research_article",
    "citation_count",
    "field_citation_rate",
    "expected_citations_per_year",
    "citations_per_year",
    "relative_citation_ratio",
    "nih_percentile",
    "human",
    "animal",
    "molecular_cellular",
    "x_coord",
    "y_coord",
    "apt",
    "is_clinical",
    "cited_by_clin",
    "cited_by",
    "references",
    "provisional",
    "last_modified",
]


def _api_get(url: str) -> bytes:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "quarry-test/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  retry {attempt + 1}: {e}")
            time.sleep(2)
    raise RuntimeError("unreachable")


# ── Data generation ──


def _fetch_oa_works(count: int) -> list[dict]:
    print(f"[OA] fetching {count} works (pmid + abstract) ...")
    works: list[dict] = []
    cursor = "*"
    per_page = min(count, 200)

    while len(works) < count:
        url = (
            f"{OA_API}?filter=has_pmid:true,has_abstract:true"
            f"&per_page={per_page}&cursor={cursor}"
            f"&select=id,ids,title,abstract_inverted_index,publication_year,"
            f"publication_date,type,cited_by_count,primary_location,"
            f"open_access,is_retracted,updated_date,authorships,topics,"
            f"referenced_works"
        )
        data = json.loads(_api_get(url))
        results = data.get("results", [])
        if not results:
            break
        works.extend(results)
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        print(f"  {len(works)}/{count}")
        time.sleep(0.1)

    return works[:count]


def _extract_pmids(works: list[dict]) -> list[int]:
    pmids = []
    for w in works:
        pmid_url = (w.get("ids") or {}).get("pmid", "")
        if pmid_url:
            try:
                pmids.append(int(pmid_url.rstrip("/").split("/")[-1]))
            except ValueError:
                pass
    return pmids


def _write_oa_jsonl_gz(works: list[dict]) -> None:
    part_dir = DATA_DIR / "oa" / "works" / "updated_date=2025-01-01"
    part_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(part_dir / "test.jsonl.gz", "wt", encoding="utf-8") as f:
        for w in works:
            json.dump(w, f, ensure_ascii=False)
            f.write("\n")
    print(f"[OA] wrote {len(works)} lines")


def _fetch_and_write_pubmed(pmids: list[int]) -> bytes:
    print(f"[PM] fetching {len(pmids)} articles ...")
    pmid_str = ",".join(str(p) for p in pmids)
    xml_data = _api_get(f"{PUBMED_EFETCH}?db=pubmed&id={pmid_str}&retmode=xml")

    pm_dir = DATA_DIR / "pubmed" / "baseline"
    pm_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(pm_dir / "pubmed_test.xml.gz", "wb") as f:
        f.write(xml_data)

    (DATA_DIR / "pubmed" / "updatefiles").mkdir(parents=True, exist_ok=True)
    print(f"[PM] wrote {len(xml_data)} bytes")
    return xml_data


def _icite_field(pub: dict, col: str) -> str:
    val = pub.get(col)
    if val is None:
        return ""
    if col in ("cited_by_clin", "cited_by", "references"):
        if isinstance(val, list):
            return " ".join(str(v) for v in val)
        return str(val)
    if isinstance(val, bool):
        return "Yes" if val else "No"
    return str(val)


def _fetch_and_write_icite(pmids: list[int]) -> None:
    print(f"[iCite] fetching {len(pmids)} pmids ...")
    all_pubs: list[dict] = []
    for i in range(0, len(pmids), 1000):
        batch = pmids[i : i + 1000]
        url = f"{ICITE_API}?pmids={','.join(str(p) for p in batch)}"
        data = json.loads(_api_get(url))
        all_pubs.extend(data.get("data", []))
        time.sleep(0.1)

    icite_dir = DATA_DIR / "icite"
    icite_dir.mkdir(parents=True, exist_ok=True)
    with open(icite_dir / "icite_metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(ICITE_CSV_COLUMNS)
        for pub in all_pubs:
            writer.writerow([_icite_field(pub, col) for col in ICITE_CSV_COLUMNS])
    print(f"[iCite] wrote {len(all_pubs)} rows")


def _write_mesh_xml(xml_data: bytes) -> None:
    uis: set[str] = set()
    root = ET.fromstring(xml_data)
    for desc in root.iter("DescriptorName"):
        ui = desc.get("UI")
        if ui:
            uis.add(ui)

    mesh_dir = DATA_DIR / "pubmed" / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh_root = ET.Element("DescriptorRecordSet")
    for i, ui in enumerate(sorted(uis)):
        rec = ET.SubElement(mesh_root, "DescriptorRecord")
        ui_el = ET.SubElement(rec, "DescriptorUI")
        ui_el.text = ui
        name_el = ET.SubElement(rec, "DescriptorName")
        str_el = ET.SubElement(name_el, "String")
        str_el.text = f"Descriptor_{ui}"
        tree_list = ET.SubElement(rec, "TreeNumberList")
        tree_el = ET.SubElement(tree_list, "TreeNumber")
        tree_el.text = f"Z99.{i:04d}"

    tree = ET.ElementTree(mesh_root)
    ET.indent(tree, space="  ")
    tree.write(mesh_dir / "desc_test.xml", encoding="unicode", xml_declaration=True)
    print(f"[MeSH] wrote {len(uis)} descriptors")


def _data_exists() -> bool:
    return (
        (
            DATA_DIR / "oa" / "works" / "updated_date=2025-01-01" / "test.jsonl.gz"
        ).exists()
        and (DATA_DIR / "pubmed" / "baseline" / "pubmed_test.xml.gz").exists()
        and (DATA_DIR / "icite" / "icite_metadata.csv").exists()
        and (DATA_DIR / "pubmed" / "mesh" / "desc_test.xml").exists()
    )


# ── Fixtures ──


@pytest.fixture(scope="session")
def test_data(test_databases):
    """Fetch test data from APIs (cached: skips if files already exist).

    Depends on test_databases to ensure CH/PG quarry_test is ready.
    """
    if _data_exists():
        print("Test data cached, skipping fetch")
        return

    works = _fetch_oa_works(TEST_COUNT)
    pmids = _extract_pmids(works)
    assert len(pmids) > 0, "No pmids extracted from OA works"

    _write_oa_jsonl_gz(works)
    xml_data = _fetch_and_write_pubmed(pmids)
    _fetch_and_write_icite(pmids)
    _write_mesh_xml(xml_data)

    print(f"\n=== Test data: {len(works)} works, {len(pmids)} pmids ===")


# ── Tests ──


class TestDefsLoad:
    def test_definitions_resolve(self):
        """Verify Dagster Definitions load without import/validation errors."""
        from quarry.defs import defs

        assert defs.resolve_implicit_global_asset_job_def() is not None


class TestDataCreation:
    def test_files_exist(self, test_data):
        assert (
            DATA_DIR / "oa" / "works" / "updated_date=2025-01-01" / "test.jsonl.gz"
        ).exists()
        assert (DATA_DIR / "pubmed" / "baseline" / "pubmed_test.xml.gz").exists()
        assert (DATA_DIR / "icite" / "icite_metadata.csv").exists()
        assert (DATA_DIR / "pubmed" / "mesh" / "desc_test.xml").exists()


class TestParse:
    def test_oa(self, test_data):
        out_dir = PARSED_DIR / "oa"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                "quarry-parse",
                "oa",
                "--input-dir",
                str(settings.oa_local_dir),
                "--output-dir",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
        )
        print(r.stderr)
        assert r.returncode == 0, f"quarry-parse oa failed: {r.stderr}"
        assert any(out_dir.rglob("*.parquet"))

    def test_pubmed(self, test_data):
        out_dir = PARSED_DIR / "pubmed"
        out_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                "quarry-parse",
                "pubmed",
                "--input-dir",
                str(settings.pubmed_baseline_dir),
                "--output-dir",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
        )
        print(r.stderr)
        assert r.returncode == 0, f"quarry-parse pubmed failed: {r.stderr}"
        assert any(out_dir.rglob("*.parquet"))

    def test_mesh(self, test_data):
        out_dir = PARSED_DIR / "mesh"
        out_dir.mkdir(parents=True, exist_ok=True)
        xml_path = DATA_DIR / "pubmed" / "mesh" / "desc_test.xml"
        r = subprocess.run(
            [
                "quarry-parse",
                "mesh",
                "--xml-path",
                str(xml_path),
                "--output-dir",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
        )
        print(r.stderr)
        assert r.returncode == 0, f"quarry-parse mesh failed: {r.stderr}"
        assert any(out_dir.rglob("*.parquet"))


class TestDagsterPipeline:
    """Full Dagster E2E: ch_load → ch_transform → parquet_export → pg_load.

    Requires: quarry-parse tests passed (parquet files exist),
    ClickHouse running on port 9001, PostgreSQL running on /tmp/quarry-pg.
    Uses quarry_test databases (created by test_databases fixture).
    """

    def test_e2e(self, test_data):
        import logging

        from dagster import AssetKey, Definitions, load_assets_from_modules

        from quarry.assets import download, load, stage

        # Load only ELT assets (skip search/citations that need quarry_core)
        defs = Definitions(
            assets=load_assets_from_modules([download, stage, load]),
        )

        # Ensure Dagster logs are visible in pytest output
        logging.getLogger("dagster").setLevel(logging.INFO)

        job = defs.resolve_implicit_global_asset_job_def()
        result = job.execute_in_process(
            asset_selection=[
                AssetKey("oa_parse"),
                AssetKey("pm_parse"),
                AssetKey("mesh_stage"),
                AssetKey("ch_load"),
                AssetKey("ch_transform"),
                AssetKey("parquet_export"),
                AssetKey("pg_load"),
            ],
        )

        # Print step-level results for debugging
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
        assert len(flat_files) == 12, (
            f"Expected 12 flat parquet files, got {len(flat_files)}"
        )
        assert len(tier_dirs) > 0, "Expected hive tier directories"
        assert len(works_files) > 0, "Expected works bucket parquet files"

        # Verify PG has data
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
            "content_hash": b"\x00" * 32,  # dummy hash for round-trip test
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
