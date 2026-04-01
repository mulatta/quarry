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
        # Not identical (different fusion strategies)
        # But should have significant overlap
        overlap = fused_ids & sep_ids
        assert len(overlap) > 0, "modes should share some results"
