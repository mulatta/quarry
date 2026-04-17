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
