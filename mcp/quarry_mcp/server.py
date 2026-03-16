"""FastMCP server exposing 8 tools for academic paper exploration.

Tools:
  search_papers   — hybrid search (BM25 + ANN + reranker) + MeSH expansion
  get_paper       — single paper detail by PMID or DOI
  expand_citations — citation/cited-by N-hop expansion
  find_path       — citation chain between two papers
  similar_papers  — embedding similarity search
  get_subgraph    — session papers graph structure
  query_metadata  — DuckDB SQL (SELECT only)
  mesh_explore    — MeSH hierarchy navigation
"""

from mcp.server.fastmcp import FastMCP

from quarry.config import settings
from quarry.store.csr import CSRGraph
from quarry.store.duckdb import DuckDBStore
from quarry.store.subgraph import SessionSubgraph

mcp = FastMCP("quarry")

# Lazy-initialized singletons
_db: DuckDBStore | None = None
_csr: CSRGraph | None = None
_subgraph: SessionSubgraph | None = None


def _get_db() -> DuckDBStore:
    global _db
    if _db is None:
        _db = DuckDBStore()
    return _db


def _get_csr() -> CSRGraph:
    global _csr
    if _csr is None:
        _csr = CSRGraph(settings.csr_dir)
    return _csr


def _get_subgraph() -> SessionSubgraph:
    global _subgraph
    if _subgraph is None:
        _subgraph = SessionSubgraph()
    return _subgraph


@mcp.tool()
def search_papers(
    query: str,
    limit: int = 20,
    mode: str = "hybrid",
    enrich: bool = True,
) -> list[dict]:
    """Hybrid search for papers using BM25 + vector ANN + RRF fusion.

    Args:
        query: Natural language search query
        limit: Max results (default 20)
        mode: "hybrid", "vector", or "text"
        enrich: If true, include full DuckDB metadata
    """
    from quarry.search.hybrid import HybridSearcher

    searcher = HybridSearcher(db=_get_db())
    return searcher.search(query, limit=limit, mode=mode, enrich=enrich)


@mcp.tool()
def get_paper(pmid: int | None = None, doi: str | None = None) -> dict | None:
    """Get detailed paper metadata by PMID or DOI.

    Args:
        pmid: PubMed ID (integer)
        doi: DOI string (alternative to pmid)
    """
    db = _get_db()
    if pmid is not None:
        return db.get_paper(pmid)
    if doi is not None:
        results = db.query(
            f"SELECT * FROM papers WHERE doi = '{doi}' AND NOT is_deleted LIMIT 1"
        )
        return results[0] if results else None
    return None


@mcp.tool()
def expand_citations(
    pmid: int,
    direction: str = "both",
    hops: int = 1,
    max_nodes: int = 500,
    enrich: bool = True,
) -> dict:
    """Expand citation graph around a paper.

    Args:
        pmid: Center paper PMID
        direction: "forward" (cites), "reverse" (cited by), or "both"
        hops: Number of hops (default 1)
        max_nodes: Maximum nodes to return
        enrich: Include paper metadata
    """
    csr = _get_csr()
    neighbors = csr.k_hop(str(pmid), k=hops, direction=direction, max_nodes=max_nodes)

    result = {
        "center": pmid,
        "direction": direction,
        "hops": hops,
        "count": len(neighbors),
    }

    if enrich:
        pmids = [int(n) for n in neighbors]
        db = _get_db()
        papers = db.get_papers(pmids[:100])  # limit enrichment
        result["papers"] = papers
    else:
        result["pmids"] = [int(n) for n in neighbors]

    return result


@mcp.tool()
def find_path(
    source_pmid: int,
    target_pmid: int,
    max_depth: int = 6,
    enrich: bool = True,
) -> dict:
    """Find citation chain path between two papers.

    Args:
        source_pmid: Starting paper PMID
        target_pmid: Target paper PMID
        max_depth: Maximum path length to search
        enrich: Include paper metadata for path nodes
    """
    csr = _get_csr()
    path = csr.shortest_path(str(source_pmid), str(target_pmid), max_depth=max_depth)

    if path is None:
        return {"found": False, "source": source_pmid, "target": target_pmid}

    result = {
        "found": True,
        "path_length": len(path),
        "path_pmids": [int(p) for p in path],
    }

    if enrich:
        db = _get_db()
        papers = db.get_papers([int(p) for p in path])
        paper_map = {p["pmid"]: p for p in papers}
        result["path_papers"] = [paper_map.get(int(p), {"pmid": int(p)}) for p in path]

    return result


@mcp.tool()
def similar_papers(
    pmid: int,
    limit: int = 20,
) -> list[dict]:
    """Find papers similar to a given paper using embedding similarity.

    Args:
        pmid: Source paper PMID
        limit: Max results
    """
    from quarry.search.hybrid import HybridSearcher

    searcher = HybridSearcher(db=_get_db())
    return searcher.similar(pmid, limit=limit)


@mcp.tool()
def get_subgraph(
    pmids: list[int],
    include_metrics: bool = True,
) -> dict:
    """Build session subgraph from a set of papers and compute graph metrics.

    Args:
        pmids: List of PMIDs to include in the subgraph
        include_metrics: If true, compute PageRank and betweenness centrality
    """
    sg = _get_subgraph()
    csr = _get_csr()

    str_ids = [str(p) for p in pmids]
    sg.add_edges_from_csr(csr, str_ids)

    result = sg.to_json()
    result["num_nodes"] = sg.num_nodes
    result["num_edges"] = sg.num_edges

    if include_metrics and sg.num_nodes > 0:
        result["pagerank"] = sg.pagerank()
        if sg.num_nodes < 10_000:
            result["betweenness"] = sg.betweenness_centrality()
        result["components"] = [list(c) for c in sg.connected_components()]

    return result


@mcp.tool()
def query_metadata(sql: str) -> list[dict]:
    """Execute a read-only SQL query against DuckDB metadata.

    Only SELECT/WITH/EXPLAIN queries are allowed.

    Args:
        sql: SQL query string
    """
    db = _get_db()
    return db.query(sql)


@mcp.tool()
def mesh_explore(
    descriptor_ui: str | None = None,
    descriptor_name: str | None = None,
    direction: str = "descendants",
) -> dict:
    """Explore MeSH hierarchy: find descendants or ancestors of a descriptor.

    Args:
        descriptor_ui: MeSH descriptor UI (e.g., "D009369" for Neoplasms)
        descriptor_name: Alternative: search by name (partial match)
        direction: "descendants" (default) or "info"
    """
    db = _get_db()

    # Resolve descriptor
    if descriptor_name and not descriptor_ui:
        results = db.query(
            f"SELECT DISTINCT descriptor_ui, descriptor_name "
            f"FROM mesh_tree WHERE descriptor_name ILIKE '%{descriptor_name}%' LIMIT 10"
        )
        if not results:
            return {"error": f"No MeSH descriptor found matching '{descriptor_name}'"}
        if len(results) > 1:
            return {
                "matches": results,
                "hint": "Multiple matches. Specify descriptor_ui.",
            }
        descriptor_ui = results[0]["descriptor_ui"]

    if not descriptor_ui:
        return {"error": "Provide descriptor_ui or descriptor_name"}

    # Get tree numbers for this descriptor
    tree_entries = db.query(
        f"SELECT * FROM mesh_tree WHERE descriptor_ui = '{descriptor_ui}'"
    )
    if not tree_entries:
        return {"error": f"Descriptor {descriptor_ui} not found in mesh_tree"}

    result = {
        "descriptor_ui": descriptor_ui,
        "descriptor_name": tree_entries[0]["descriptor_name"],
        "tree_numbers": [e["tree_number"] for e in tree_entries],
    }

    if direction == "descendants":
        all_descendants: list[dict] = []
        for entry in tree_entries:
            descendants = db.mesh_descendants(entry["tree_number"])
            all_descendants.extend(descendants)
        # Deduplicate
        seen = set()
        unique = []
        for d in all_descendants:
            key = (d["descriptor_ui"], d["tree_number"])
            if key not in seen:
                seen.add(key)
                unique.append(d)
        result["descendants"] = unique
        result["descendant_count"] = len(unique)

    return result


def main():
    mcp.run()


if __name__ == "__main__":
    main()
