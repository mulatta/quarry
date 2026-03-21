"""FastMCP server exposing 8 tools for academic paper exploration.

Tools:
  search_papers   — hybrid search (BM25 + ANN + reranker) + MeSH expansion
  get_paper       — single paper detail by work_id, PMID, or DOI
  expand_citations — citation/cited-by N-hop expansion
  find_path       — citation chain between two papers
  similar_papers  — embedding similarity search
  get_subgraph    — session papers graph structure
  query_metadata  — PG SQL (SELECT only)
  mesh_explore    — MeSH hierarchy navigation
"""

from mcp.server.fastmcp import FastMCP

import quarry_graph
from quarry.config import settings
from quarry.store.pg import PGStore

mcp = FastMCP("quarry")

# Lazy-initialized singletons
_db: PGStore | None = None
_graph: quarry_graph.Graph | None = None


def _get_db() -> PGStore:
    global _db
    if _db is None:
        _db = PGStore(settings.pg_conninfo)
    return _db


def _get_graph() -> quarry_graph.Graph:
    global _graph
    if _graph is None:
        _graph = quarry_graph.Graph(str(settings.csr_dir))
    return _graph


def _resolve_graph_id(
    db: PGStore,
    work_id: str | None = None,
    pmid: int | None = None,
) -> int | None:
    """Resolve work_id or pmid → work_id_int (CSR graph node ID)."""
    with db.conn.cursor() as cur:
        if work_id:
            cur.execute("SELECT work_id_int FROM works WHERE work_id = %s", (work_id,))
        elif pmid:
            cur.execute("SELECT work_id_int FROM works WHERE pmid = %s", (pmid,))
        else:
            return None
        row = cur.fetchone()
        return row[0] if row else None


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
        enrich: If true, include full metadata
    """
    from quarry.search.hybrid import HybridSearcher

    searcher = HybridSearcher(db=_get_db())
    return searcher.search(query, limit=limit, mode=mode, enrich=enrich)


@mcp.tool()
def get_paper(
    work_id: str | None = None,
    pmid: int | None = None,
    doi: str | None = None,
) -> dict | None:
    """Get detailed paper metadata by work_id, PMID, or DOI.

    Args:
        work_id: OpenAlex work ID (e.g., "W2741809807")
        pmid: PubMed ID (integer)
        doi: DOI string
    """
    db = _get_db()
    if work_id is not None:
        return db.get_work(work_id)
    if pmid is not None:
        return db.get_work_by_pmid(pmid)
    if doi is not None:
        return db.get_work_by_doi(doi)
    return None


@mcp.tool()
def expand_citations(
    work_id: str | None = None,
    pmid: int | None = None,
    direction: str = "both",
    hops: int = 1,
    max_nodes: int = 500,
    enrich: bool = True,
) -> dict:
    """Expand citation graph around a paper.

    Args:
        work_id: OpenAlex work ID
        pmid: PubMed ID (alternative)
        direction: "forward" (cites), "reverse" (cited by), or "both"
        hops: Number of hops (default 1)
        max_nodes: Maximum nodes to return
        enrich: Include paper metadata
    """
    db = _get_db()
    graph = _get_graph()

    node_id = _resolve_graph_id(db, work_id=work_id, pmid=pmid)
    if node_id is None:
        return {"error": "Paper not found in graph"}

    neighbors = graph.k_hop(node_id, k=hops, direction=direction, max_nodes=max_nodes)

    result = {
        "center": work_id or f"pmid:{pmid}",
        "direction": direction,
        "hops": hops,
        "count": len(neighbors),
    }

    if enrich:
        work_id_int_list = neighbors[:100]
        with db.conn.cursor() as cur:
            cur.execute(
                "SELECT work_id FROM works WHERE work_id_int = ANY(%s)",
                (work_id_int_list,),
            )
            found_work_ids = [r[0] for r in cur.fetchall()]
        result["papers"] = db.get_works(found_work_ids)
    else:
        result["node_ids"] = neighbors

    return result


@mcp.tool()
def find_path(
    source_work_id: str | None = None,
    source_pmid: int | None = None,
    target_work_id: str | None = None,
    target_pmid: int | None = None,
    max_depth: int = 6,
    enrich: bool = True,
) -> dict:
    """Find citation chain path between two papers.

    Args:
        source_work_id: Starting paper work_id
        source_pmid: Starting paper PMID (alternative)
        target_work_id: Target paper work_id
        target_pmid: Target paper PMID (alternative)
        max_depth: Maximum path length to search
        enrich: Include paper metadata for path nodes
    """
    db = _get_db()
    graph = _get_graph()

    src_id = _resolve_graph_id(db, work_id=source_work_id, pmid=source_pmid)
    dst_id = _resolve_graph_id(db, work_id=target_work_id, pmid=target_pmid)

    if src_id is None or dst_id is None:
        return {"found": False, "error": "Source or target not found in graph"}

    path = graph.shortest_path(src_id, dst_id, max_depth=max_depth)

    if path is None:
        return {"found": False}

    result = {
        "found": True,
        "path_length": len(path),
        "path_node_ids": path,
    }

    if enrich:
        with db.conn.cursor() as cur:
            cur.execute(
                "SELECT work_id_int, work_id FROM works WHERE work_id_int = ANY(%s)",
                (path,),
            )
            int_to_wid = {r[0]: r[1] for r in cur.fetchall()}
        path_work_ids = [int_to_wid[nid] for nid in path if nid in int_to_wid]
        papers = db.get_works(path_work_ids)
        paper_map = {p["work_id"]: p for p in papers}
        result["path_papers"] = [
            paper_map.get(int_to_wid.get(nid), {"work_id_int": nid}) for nid in path
        ]

    return result


@mcp.tool()
def similar_papers(
    work_id: str | None = None,
    pmid: int | None = None,
    limit: int = 20,
) -> list[dict]:
    """Find papers similar to a given paper using embedding similarity.

    Args:
        work_id: OpenAlex work ID
        pmid: PubMed ID (alternative — resolved to work_id)
        limit: Max results
    """
    from quarry.search.hybrid import HybridSearcher

    db = _get_db()
    if pmid is not None and work_id is None:
        work_id = db.resolve_pmid_to_work_id(pmid)
        if work_id is None:
            return []

    searcher = HybridSearcher(db=db)
    return searcher.similar(work_id, limit=limit)


@mcp.tool()
def get_subgraph(
    work_ids: list[str] | None = None,
    pmids: list[int] | None = None,
    include_metrics: bool = True,
) -> dict:
    """Build session subgraph from a set of papers and compute graph metrics.

    Args:
        work_ids: List of OpenAlex work IDs
        pmids: List of PMIDs (alternative — resolved to work_id_ints)
        include_metrics: If true, compute PageRank and betweenness centrality
    """
    db = _get_db()
    graph = _get_graph()

    node_ids = []
    if work_ids:
        for wid in work_ids:
            nid = _resolve_graph_id(db, work_id=wid)
            if nid is not None:
                node_ids.append(nid)
    if pmids:
        for p in pmids:
            nid = _resolve_graph_id(db, pmid=p)
            if nid is not None:
                node_ids.append(nid)

    edges = graph.subgraph_edges(node_ids)
    result = {
        "nodes": [{"id": n} for n in node_ids],
        "edges": [{"source": s, "target": t} for s, t in edges],
        "num_nodes": len(node_ids),
        "num_edges": len(edges),
    }

    if include_metrics and len(node_ids) > 0:
        result["pagerank"] = dict(
            graph.subgraph_pagerank(node_ids, alpha=0.85, max_iter=100, tol=1e-6)
        )
        if len(node_ids) < 10_000:
            result["betweenness"] = dict(
                graph.subgraph_betweenness(node_ids, normalized=True)
            )
        result["components"] = graph.subgraph_components(node_ids)

    return result


@mcp.tool()
def query_metadata(sql: str) -> list[dict]:
    """Execute a read-only SQL query against PostgreSQL metadata.

    Only SELECT/WITH/EXPLAIN queries are allowed.
    Both v1 (papers) and v2 (works) tables are accessible.

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

    if descriptor_name and not descriptor_ui:
        results = db.mesh_search_by_name(descriptor_name)
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

    tree_entries = db.mesh_by_ui(descriptor_ui)
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
