"""Unit tests for quarry expand (graph.expand + graph.appr).

Tests run against the live CSR graph.

Seed papers:
  Dao 2016 (W2495359280) — AptaTRACE, HT-SELEX aptamer motifs
    PMC XML: tests/data/pmc5042215_dao2016.xml
"""

import pytest

try:
    import quarry_graph
except ImportError:
    pytest.skip("quarry_graph not installed", allow_module_level=True)


CSR_DIR = "/workspace/seungwon/quarry-data/serving/csr"
DAO_2016 = 2495359280


@pytest.fixture(scope="module")
def graph():
    return quarry_graph.Graph(CSR_DIR)


def _ids(papers):
    return {p["work_id"] for p in papers}


def _scores(papers):
    return [p["fused_score"] for p in papers]


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
        assert graph.appr(999999999999) == []

    def test_top_k(self, graph):
        assert len(graph.appr(DAO_2016, top_k=10)) == 10


class TestExpandFused:
    def test_returns_results(self, graph):
        papers, stats = graph.expand(DAO_2016, mode="fused", limit=50)
        assert len(papers) > 0
        assert stats["appr_candidates"] > 0

    def test_seed_excluded(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=50)
        assert DAO_2016 not in _ids(papers)

    def test_sorted_descending(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=50)
        scores = _scores(papers)
        assert scores == sorted(scores, reverse=True)

    def test_limit_respected(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=10)
        assert len(papers) <= 10

    def test_has_appr_score(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=10)
        # At least some papers should have appr_score
        has_appr = [p for p in papers if p["appr_score"] is not None]
        assert len(has_appr) > 0

    def test_results_are_on_topic(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=200)
        ids = _ids(papers)
        aptacluster = 51469642
        if graph.has_node(aptacluster):
            assert aptacluster in ids


class TestExpandSeparated:
    def test_returns_results(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="separated", limit=50)
        assert len(papers) > 0

    def test_seed_excluded(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="separated", limit=50)
        assert DAO_2016 not in _ids(papers)

    def test_has_lateral_papers(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="separated", limit=200)
        refs = set(graph.neighbors(DAO_2016, "forward"))
        citers = set(graph.neighbors(DAO_2016, "reverse"))
        laterals = _ids(papers) - refs - citers
        assert len(laterals) > 0

    def test_limit_respected(self, graph):
        papers, _ = graph.expand(DAO_2016, mode="separated", limit=20)
        assert len(papers) <= 20


class TestBridges:
    def test_lateral_has_bridges(self, graph):
        """Lateral papers should have non-empty bridges."""
        papers, _ = graph.expand(DAO_2016, mode="separated", limit=200)
        refs = set(graph.neighbors(DAO_2016, "forward"))
        citers = set(graph.neighbors(DAO_2016, "reverse"))
        laterals = [
            p for p in papers if p["work_id"] not in refs and p["work_id"] not in citers
        ]
        bridged = [p for p in laterals if p["bridges"]]
        assert len(bridged) > 0, "some lateral papers should have bridges"

    def test_bridge_structure(self, graph):
        """Bridge dicts should have required fields."""
        papers, _ = graph.expand(DAO_2016, mode="separated", limit=200)
        for p in papers:
            if p["bridges"]:
                for b in p["bridges"]:
                    assert "work_id" in b
                    assert "type" in b
                    assert b["type"] in ("shared_ref", "shared_citer")
                    assert "weight" in b
                    assert b["weight"] > 0
                break  # one paper enough

    def test_foundation_no_bridges(self, graph):
        """Foundation/follow-up papers should not have bridges."""
        papers, _ = graph.expand(DAO_2016, mode="fused", limit=50)
        refs = set(graph.neighbors(DAO_2016, "forward"))
        for p in papers:
            if p["work_id"] in refs:
                assert p["bridges"] == [], (
                    f"foundation W{p['work_id']} should have no bridges"
                )
                break


class TestExpandModes:
    def test_invalid_mode_raises(self, graph):
        with pytest.raises(ValueError, match="invalid mode"):
            graph.expand(DAO_2016, mode="invalid")

    def test_fused_and_separated_differ(self, graph):
        fused, _ = graph.expand(DAO_2016, mode="fused", limit=200)
        separated, _ = graph.expand(DAO_2016, mode="separated", limit=200)
        assert len(_ids(fused) & _ids(separated)) > 0


# ── Quality regression guards ──

SEEDS = {
    "Dao-2016": {
        "wid": 2495359280,
        "must_find": [51469642, 2050271637, 2252393116],
    },
    "Seo-2025": {
        "wid": 4406019873,
        "must_find": [2294707565, 3015207233],
    },
    "Zhao-2021": {
        "wid": 3043719325,
        "must_find": [2336828812, 2766599608],
    },
}


class TestQualityRegression:
    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_must_find_papers_fused(self, graph, seed_name):
        seed = SEEDS[seed_name]
        papers, _ = graph.expand(seed["wid"], mode="fused", limit=200)
        ids = _ids(papers)
        for must in seed["must_find"]:
            if graph.has_node(must):
                assert must in ids, f"{seed_name}: W{must} missing from fused"

    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_must_find_papers_separated(self, graph, seed_name):
        seed = SEEDS[seed_name]
        papers, _ = graph.expand(seed["wid"], mode="separated", limit=200)
        ids = _ids(papers)
        for must in seed["must_find"]:
            if graph.has_node(must):
                assert must in ids, f"{seed_name}: W{must} missing from separated"

    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_ref_recovery_minimum(self, graph, seed_name):
        seed = SEEDS[seed_name]
        wid = seed["wid"]
        refs = set(graph.neighbors(wid, "forward"))
        if len(refs) < 5:
            pytest.skip("too few references")
        papers, _ = graph.expand(wid, mode="fused", limit=200)
        recovery = len(refs & _ids(papers)) / len(refs)
        assert recovery >= 0.3, f"{seed_name}: ref recovery {recovery:.0%} < 30%"

    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_lateral_quality_separated(self, graph, seed_name):
        seed = SEEDS[seed_name]
        wid = seed["wid"]
        refs = set(graph.neighbors(wid, "forward"))
        citers = set(graph.neighbors(wid, "reverse"))
        papers, _ = graph.expand(wid, mode="separated", limit=200)
        laterals = _ids(papers) - refs - citers
        assert len(laterals) >= 5, f"{seed_name}: only {len(laterals)} laterals"

    @pytest.mark.parametrize("seed_name", SEEDS.keys())
    def test_no_duplicate_results(self, graph, seed_name):
        seed = SEEDS[seed_name]
        for mode in ("fused", "separated"):
            papers, _ = graph.expand(seed["wid"], mode=mode, limit=200)
            ids = [p["work_id"] for p in papers]
            assert len(ids) == len(set(ids)), f"{seed_name}/{mode}: duplicates"
