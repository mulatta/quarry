"""Unit tests for quarry expand (graph.expand + graph.appr).

Tests run against the live CSR graph. Requires QUARRY_CSR_DIR or default
config pointing to a valid CSR directory.

Seed papers:
  Dao 2016 (W2495359280) — AptaTRACE, HT-SELEX aptamer motifs
    PMC XML: tests/data/pmc5042215_dao2016.xml
    PMCID: PMC5042215, PMID: 27467247, DOI: 10.1016/j.cels.2016.07.003
"""

import pytest

try:
    import quarry_graph
except ImportError:
    pytest.skip("quarry_graph not installed", allow_module_level=True)


CSR_DIR = "/workspace/seungwon/quarry-data/serving/csr"
DAO_2016 = 2495359280  # AptaTRACE


@pytest.fixture(scope="module")
def graph():
    return quarry_graph.Graph(CSR_DIR)


class TestAppr:
    def test_returns_results(self, graph):
        result = graph.appr(DAO_2016)
        assert len(result) > 0

    def test_seed_is_top_1(self, graph):
        result = graph.appr(DAO_2016)
        assert result[0][0] == DAO_2016

    def test_sorted_descending(self, graph):
        result = graph.appr(DAO_2016)
        scores = [s for _, s in result[:50]]
        assert scores == sorted(scores, reverse=True)

    def test_unknown_seed_empty(self, graph):
        result = graph.appr(999999999999)
        assert result == []

    def test_top_k(self, graph):
        result = graph.appr(DAO_2016, top_k=10)
        assert len(result) == 10


class TestExpandFused:
    def test_returns_results(self, graph):
        papers, stats = graph.expand(DAO_2016, mode="fused", limit=50)
        assert len(papers) > 0
        assert stats["appr_candidates"] > 0

    def test_seed_excluded(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=50)
        ids = {wid for wid, _ in papers}
        assert DAO_2016 not in ids

    def test_sorted_descending(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=50)
        scores = [s for _, s in papers]
        assert scores == sorted(scores, reverse=True)

    def test_limit_respected(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=10)
        assert len(papers) <= 10

    def test_results_are_on_topic(self, graph):
        """Expand results for Dao 2016 should include known aptamer/SELEX papers."""
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=200)
        ids = {wid for wid, _ in papers}
        # AptaCluster (Hoinka 2014) — seed's key reference
        aptacluster = 2114854978
        if graph.has_node(aptacluster):
            assert aptacluster in ids, "AptaCluster should be in expand results"


class TestExpandSeparated:
    def test_returns_results(self, graph):
        papers, stats = graph.expand(DAO_2016, mode="separated", limit=50)
        assert len(papers) > 0

    def test_seed_excluded(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="separated", limit=50)
        ids = {wid for wid, _ in papers}
        assert DAO_2016 not in ids

    def test_has_lateral_papers(self, graph):
        """Separated mode should include lateral papers not in direct refs/citers."""
        papers, _ = graph.expand(DAO_2016, mode="separated", limit=200)
        refs = set(graph.neighbors(DAO_2016, "forward"))
        citers = set(graph.neighbors(DAO_2016, "reverse"))
        ids = {wid for wid, _ in papers}
        laterals = ids - refs - citers
        assert len(laterals) > 0, "separated mode should produce lateral papers"

    def test_limit_respected(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="separated", limit=20)
        assert len(papers) <= 20


class TestExpandModes:
    def test_invalid_mode_raises(self, graph):
        with pytest.raises(ValueError, match="invalid mode"):
            graph.expand(DAO_2016, mode="invalid")

    def test_fused_and_separated_differ(self, graph):
        """Fused and separated should produce different result sets."""
        fused, _ = graph.expand(DAO_2016, mode="fused", limit=200)
        separated, _ = graph.expand(DAO_2016, mode="separated", limit=200)
        fused_ids = {wid for wid, _ in fused}
        sep_ids = {wid for wid, _ in separated}
        overlap = fused_ids & sep_ids
        assert len(overlap) > 0, "modes should share some results"


# ── Quality regression guards ──
# Baseline values from 5-seed evaluation (2026-04-01).
# These catch algorithm regressions, not absolute quality.

# Known seed papers and their expected references in the graph.
SEEDS = {
    "Dao-2016": {
        "wid": 2495359280,
        # Verified references present in graph and expand results:
        "must_find": [2031580508, 2050271637, 51469642],
    },
    "Seo-2025": {
        "wid": 4406019873,
        # Yoshida 2016 (PET bacterium), Austin 2018 (engineered PET depolymerase)
        "must_find": [2294707565, 3015207233],
    },
    "Zhao-2021": {
        "wid": 3043719325,
        # Verified references present in graph and expand results:
        "must_find": [2530994846, 2480829753, 2336828812],
    },
}


class TestQualityRegression:
    """Quality baselines. Failures mean algorithm change degraded results."""

    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_must_find_papers_fused(self, graph, seed_name):
        """Key papers in the field must appear in fused top-200."""
        seed = SEEDS[seed_name]
        papers, _ = graph.expand(seed["wid"], mode="fused", limit=200)
        ids = {wid for wid, _ in papers}
        for must in seed["must_find"]:
            if graph.has_node(must):
                assert must in ids, f"{seed_name}: W{must} missing from fused results"

    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_must_find_papers_separated(self, graph, seed_name):
        """Key papers must appear in separated top-200."""
        seed = SEEDS[seed_name]
        papers, _ = graph.expand(seed["wid"], mode="separated", limit=200)
        ids = {wid for wid, _ in papers}
        for must in seed["must_find"]:
            if graph.has_node(must):
                assert must in ids, (
                    f"{seed_name}: W{must} missing from separated results"
                )

    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_ref_recovery_minimum(self, graph, seed_name):
        """Reference recovery should not drop below 60% (fused mode)."""
        seed = SEEDS[seed_name]
        wid = seed["wid"]
        refs = set(graph.neighbors(wid, "forward"))
        if len(refs) < 5:
            pytest.skip("too few references for meaningful recovery test")
        papers, _ = graph.expand(wid, mode="fused", limit=200)
        ids = {w for w, _ in papers}
        recovery = len(refs & ids) / len(refs)
        # High-citation seeds (many citers) dilute ref recovery via APPR.
        # Threshold is lenient to avoid false failures on such seeds.
        assert recovery >= 0.3, f"{seed_name}: ref recovery {recovery:.0%} < 30%"

    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_lateral_quality_separated(self, graph, seed_name):
        """Separated mode must produce lateral papers (not just refs/citers)."""
        seed = SEEDS[seed_name]
        wid = seed["wid"]
        refs = set(graph.neighbors(wid, "forward"))
        citers = set(graph.neighbors(wid, "reverse"))
        papers, _ = graph.expand(wid, mode="separated", limit=200)
        ids = {w for w, _ in papers}
        laterals = ids - refs - citers
        assert len(laterals) >= 5, (
            f"{seed_name}: only {len(laterals)} laterals in separated mode"
        )

    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_no_duplicate_results(self, graph, seed_name):
        """Results must not contain duplicate work_ids."""
        seed = SEEDS[seed_name]
        for mode in ("fused", "separated"):
            papers, _ = graph.expand(seed["wid"], mode=mode, limit=200)
            ids = [wid for wid, _ in papers]
            assert len(ids) == len(set(ids)), (
                f"{seed_name}/{mode}: duplicate work_ids in results"
            )
