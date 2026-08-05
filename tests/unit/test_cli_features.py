"""Unit tests for CLI features added in dogfood sessions.

Tests run against the live PG database (same as test_expand.py
runs against live CSR). Requires quarry PG to be running.

Covers: info, mesh, shrink, mesh_lookup, host_venue enrichment.
"""

import json
import subprocess

import pytest


def _quarry(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run quarry CLI command and return result."""
    return subprocess.run(
        ["quarry", *args], capture_output=True, text=True, timeout=timeout, check=False
    )


def _quarry_json(*args: str, timeout: int = 120) -> dict:
    """Run quarry CLI with --format json and parse output."""
    r = _quarry(*args, "--format", "json", timeout=timeout)
    assert r.returncode == 0, f"quarry {' '.join(args)} failed: {r.stderr}"
    return json.loads(r.stdout)


# ── info ──


class TestInfo:
    def test_single_paper(self):
        r = _quarry("info", "W2766599608")
        assert r.returncode == 0
        assert "Programmable base editing" in r.stdout

    def test_multiple_papers(self):
        r = _quarry("info", "W2766599608", "W2336828812")
        assert r.returncode == 0
        assert r.stdout.count("W2") >= 2

    def test_not_found(self):
        r = _quarry("info", "W9999999999")
        assert r.returncode == 1
        assert "Not found" in r.stderr

    def test_json_format(self):
        d = _quarry_json("info", "W2766599608")
        assert d["work_id"] == "W2766599608"
        assert d["pub_year"] is not None
        assert d["host_venue"] is not None

    def test_mesh_flag(self):
        r = _quarry("info", "W2766599608", "--mesh")
        assert r.returncode == 0
        assert "mesh:" in r.stdout
        # ABE paper should have Gene Editing MeSH
        assert "Gene Editing" in r.stdout


# ── mesh ──


class TestMesh:
    def test_name_search_single(self):
        # "CRISPR-Cas" is specific enough to match one descriptor
        r = _quarry("mesh", "CRISPR-Cas Systems")
        assert r.returncode == 0
        assert "D064113" in r.stdout

    def test_name_search_multiple(self):
        r = _quarry("mesh", "phosphorylation")
        assert r.returncode == 0
        assert "D010766" in r.stdout  # Phosphorylation descriptor

    def test_token_matching(self):
        """Token-AND matching: order shouldn't matter."""
        r = _quarry("mesh", "fluorescence hybridization")
        assert r.returncode == 0
        assert "D017404" in r.stdout

    def test_descriptor_ui_lookup(self):
        r = _quarry("mesh", "D017404")
        assert r.returncode == 0
        assert "In Situ Hybridization, Fluorescence" in r.stdout

    def test_tree_option(self):
        r = _quarry("mesh", "D017404", "--tree")
        assert r.returncode == 0
        assert "↑" in r.stdout or "↓" in r.stdout  # hierarchy arrows

    def test_not_found(self):
        r = _quarry("mesh", "xyznonexistent12345")
        assert r.returncode == 1

    def test_historical_descriptor(self):
        """D011499 is not in mesh_tree but exists in mesh_lookup."""
        r = _quarry("mesh", "post-translational")
        assert r.returncode == 0
        assert "D011499" in r.stdout

    def test_entry_term_search(self):
        """Entry terms from desc*.xml should be searchable."""
        r = _quarry("mesh", "post-translational modification")
        assert r.returncode == 0
        # Should find something (Histone Code or similar via entry term)
        assert "D0" in r.stdout


# ── sql --schema ──


class TestSqlSchema:
    def test_schema_works(self):
        r = _quarry("sql", "--schema", "works")
        assert r.returncode == 0
        assert "pub_year" in r.stdout
        assert "host_venue" in r.stdout

    def test_schema_mesh_lookup(self):
        r = _quarry("sql", "--schema", "mesh_lookup")
        assert r.returncode == 0
        assert "term" in r.stdout
        assert "entry_term" in r.stdout

    def test_schema_unknown(self):
        r = _quarry("sql", "--schema", "nonexistent")
        assert r.returncode == 1

    def test_no_args_shows_help(self):
        r = _quarry("sql")
        assert r.returncode == 1
        assert "Available tables" in r.stdout or "Available tables" in r.stderr


# ── expand output enrichment ──


class TestExpandEnrichment:
    @pytest.fixture(scope="class")
    def expand_result(self):
        return _quarry_json("expand", "W2766599608", "--limit", "5", timeout=120)

    def test_has_host_venue(self, expand_result):
        for p in expand_result["papers"]:
            assert "host_venue" in p

    def test_has_cited_by(self, expand_result):
        for p in expand_result["papers"]:
            assert "cited_by" in p["quality"]

    def test_mesh_summary(self):
        r = _quarry(
            "expand", "W2766599608", "--limit", "10", "--mesh-summary", timeout=120
        )
        assert r.returncode == 0
        assert "MeSH summary" in r.stdout


# ── shrink ──


class TestShrink:
    @pytest.fixture(scope="class")
    def shrink_result(self):
        return _quarry_json("shrink", "W2042789810", "--top", "3", timeout=180)

    def test_returns_selected(self, shrink_result):
        assert len(shrink_result["selected"]) == 3

    def test_has_coverage(self, shrink_result):
        assert shrink_result["stats"]["coverage"] > 0

    def test_cumulative_coverage_increases(self, shrink_result):
        coverages = [s["cumulative_coverage"] for s in shrink_result["selected"]]
        assert coverages == sorted(coverages)

    def test_deterministic(self):
        """Same seed → same result."""
        r1 = _quarry_json("shrink", "W2042789810", "--top", "3", timeout=180)
        r2 = _quarry_json("shrink", "W2042789810", "--top", "3", timeout=180)
        ids1 = [s["work_id"] for s in r1["selected"]]
        ids2 = [s["work_id"] for s in r2["selected"]]
        assert ids1 == ids2

    def test_no_foundation(self):
        r = _quarry_json(
            "shrink", "W2766599608", "--top", "3", "--no-foundation", timeout=180
        )
        for s in r["selected"]:
            assert s["relation"] != "foundation"

    def test_centralization_warning(self):
        """ABE seed should trigger centralization warning."""
        r = _quarry("shrink", "W2766599608", "--top", "5", timeout=180)
        assert r.returncode == 0
        assert "centralized" in r.stdout.lower() or r.stdout.count("+") > 0

    def test_custom_venue(self):
        r = _quarry_json(
            "shrink",
            "W2042789810",
            "--top",
            "3",
            "--venue",
            "Nature,Science",
            timeout=180,
        )
        for s in r["selected"]:
            assert s["host_venue"] in ("Nature", "Science")
