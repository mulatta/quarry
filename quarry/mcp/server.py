"""FastMCP server exposing 12 tools for academic paper exploration.

CLI-equivalent tools (1:1 mapping):
  expand          — quarry expand (APPR + wRRF citation neighborhood)
  bridge          — quarry bridge (7 bridge types across seeds)
  shrink          — quarry shrink (greedy min-cover from top venues)
  search_papers   — quarry search / vsearch (hybrid BM25 + vector)
  get_paper       — quarry info [+include_mesh for --mesh]
  mesh_explore    — quarry mesh [+papers, +tree directions]

Lower-level / utility tools:
  expand_citations — k-hop neighborhood (kept for backward compatibility)
  find_path        — shortest citation chain between two papers
  similar_papers   — embedding similarity search
  get_subgraph     — session subgraph structure + metrics
  query_metadata   — PG SQL escape hatch (SELECT only)
"""

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError("pip install quarry[server]") from None

try:
    import quarry_graph
except ImportError:
    raise ImportError("pip install quarry[server]") from None

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
    include_mesh: bool = False,
    include_abstract: bool = False,
) -> dict | None:
    """Get detailed paper metadata by work_id, PMID, or DOI.

    Args:
        work_id: OpenAlex work ID (e.g., "W2741809807")
        pmid: PubMed ID (integer)
        doi: DOI string
        include_mesh: If true, include MeSH descriptor tags (equiv. quarry info --mesh)
        include_abstract: If true, include abstract text (equiv. quarry info --full)
    """
    db = _get_db()
    if work_id is not None:
        result = db.get_work(work_id)
    elif pmid is not None:
        result = db.get_work_by_pmid(pmid)
    elif doi is not None:
        result = db.get_work_by_doi(doi)
    else:
        return None
    if result is not None and not include_abstract:
        result.pop("abstract", None)
    if result and include_mesh:
        result["mesh"] = db.get_work_mesh(result["work_id"])
    return result


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
    limit: int = 15,
) -> dict:
    """Explore MeSH hierarchy or discover papers tagged with a descriptor.

    Args:
        descriptor_ui: MeSH descriptor UI (e.g., "D009369" for Neoplasms)
        descriptor_name: Alternative: search by name (partial match)
        direction: "descendants" (default), "info", "papers", or "tree"
            - "descendants": all descendant descriptors in hierarchy
            - "info": descriptor metadata only
            - "papers": top cited papers tagged with this descriptor (quarry mesh default)
            - "tree": parent descriptor + direct children in hierarchy (quarry mesh --tree)
        limit: Max papers returned when direction="papers" (default 15)
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

    tree_numbers = [e["tree_number"] for e in tree_entries]
    result: dict = {
        "descriptor_ui": descriptor_ui,
        "descriptor_name": tree_entries[0]["descriptor_name"],
        "tree_numbers": tree_numbers,
    }

    if direction == "descendants":
        all_descendants: list[dict] = []
        for entry in tree_entries:
            descendants = db.mesh_descendants(entry["tree_number"])
            all_descendants.extend(descendants)
        seen: set = set()
        unique = []
        for d in all_descendants:
            key = (d["descriptor_ui"], d["tree_number"])
            if key not in seen:
                seen.add(key)
                unique.append(d)
        result["descendants"] = unique
        result["descendant_count"] = len(unique)

    elif direction == "papers":
        works, total = db.get_top_works_by_mesh(descriptor_ui, limit=limit)
        result["papers"] = works
        result["total_count"] = total
        if total > 50_000:
            result["breadth_warning"] = (
                f"{total:,} papers — too broad for seed discovery. "
                "Use direction='tree' to find sub-descriptors."
            )

    elif direction == "tree":
        parents = [db.mesh_parent(tn) for tn in tree_numbers]
        result["parents"] = [p for p in parents if p is not None]
        children: list[dict] = []
        seen_children: set = set()
        for tn in tree_numbers:
            for c in db.mesh_descendants(tn):
                # Only direct children: one level deeper, not self
                if (
                    c["tree_number"] != tn
                    and c["tree_number"].startswith(tn + ".")
                    and c["tree_number"].count(".") == tn.count(".") + 1
                    and c["descriptor_ui"] not in seen_children
                ):
                    seen_children.add(c["descriptor_ui"])
                    children.append(c)
        result["children"] = children

    return result


@mcp.tool()
def expand(
    seed: str,
    mode: str = "fused",
    limit: int = 200,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    min_citations: int = 0,
    include_abstract: bool = False,
    mesh_summary: bool = False,
) -> dict:
    """APPR-based citation graph expansion around a seed paper (quarry expand).

    Uses Approximate Personalized PageRank + wRRF to surface the most
    structurally relevant papers in the citation neighborhood.

    Args:
        seed: Seed paper — W<id>, DOI, or integer work_id_int
        mode: "fused" (APPR+wRRF, default) or "separated" (APPR only)
        limit: Max papers returned (default 200)
        alpha: PPR teleport probability (default 0.15)
        epsilon: PPR convergence threshold (default 1e-6)
        min_citations: Filter out papers with fewer citations than this
        include_abstract: Include abstract text in each paper entry
        mesh_summary: Append top-10 MeSH descriptor distribution across results
    """
    from quarry.core.expand import run_expand

    result = run_expand(
        seed=seed,
        csr_dir=str(settings.csr_dir),
        pg_conninfo=settings.pg_conninfo,
        mode=mode,
        alpha=alpha,
        epsilon=epsilon,
        limit=limit,
        min_citations=min_citations,
        include_abstract=include_abstract,
    )

    if mesh_summary:
        db = _get_db()
        work_ids = [f"W{p['work_id']}" for p in result["papers"]]
        result["mesh_summary"] = db.top_mesh_by_work_ids(work_ids, limit=10)

    return result


@mcp.tool()
def bridge(
    seeds: list[str],
    types: list[str] | None = None,
    limit: int = 100,
    max_neighbor_degree: int = 10_000,
    max_path_depth: int = 5,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    include_abstract: bool = False,
    mesh_summary: bool = False,
) -> dict:
    """Find papers that connect multiple seed papers (quarry bridge).

    Runs 7 bridge algorithms (common_refs, common_citers, coupling,
    cocitation, path, ppr, steiner) to surface cross-domain connectors.

    Args:
        seeds: 2+ seed papers — W<id>, DOI, or integer work_id_int
        types: Bridge types to run; None = all applicable types
        limit: Max results per bridge type (default 100)
        max_neighbor_degree: Skip hub nodes above this degree (default 10000)
        max_path_depth: Max hops for path-based bridges (default 5)
        alpha: PPR teleport probability (default 0.15)
        epsilon: PPR convergence threshold (default 1e-6)
        include_abstract: Include abstract text in each paper entry
        mesh_summary: Append top-10 MeSH descriptor distribution across all bridge results
    """
    from quarry.core.bridge import run_bridge

    result = run_bridge(
        seeds=seeds,
        csr_dir=str(settings.csr_dir),
        pg_conninfo=settings.pg_conninfo,
        types=types,
        limit=limit,
        max_neighbor_degree=max_neighbor_degree,
        max_path_depth=max_path_depth,
        alpha=alpha,
        epsilon=epsilon,
        include_abstract=include_abstract,
    )

    if mesh_summary:
        db = _get_db()
        bridge_type_keys = [
            "common_refs",
            "common_citers",
            "coupling_bridges",
            "cocitation_bridges",
            "path_bridges",
            "ppr_bridges",
            "steiner_bridges",
        ]
        seen: set[str] = set()
        all_work_ids: list[str] = []
        for key in bridge_type_keys:
            for p in result.get(key, []):
                wid = f"W{p['work_id']}"
                if wid not in seen:
                    seen.add(wid)
                    all_work_ids.append(wid)
        result["mesh_summary"] = db.top_mesh_by_work_ids(all_work_ids, limit=10)

    return result


@mcp.tool()
def shrink(
    seed: str,
    top_n: int = 5,
    venues: list[str] | None = None,
    expand_limit: int = 200,
    no_foundation: bool = False,
    exclude: list[str] | None = None,
) -> dict:
    """Greedy minimum covering set from top-venue papers (quarry shrink).

    Expands seed, filters to high-impact venues, then selects the smallest
    set of papers that collectively cite/are-cited-by the most candidates.

    Args:
        seed: Seed paper — W<id>, DOI, or integer work_id_int
        top_n: Number of papers to select (default 5)
        venues: Venue names to include; None = NCS_PLUS defaults
        expand_limit: Candidate pool size from expand (default 200)
        no_foundation: Exclude relation=foundation papers from pool
        exclude: Work IDs to exclude — W<id> strings (e.g., ["W12345"])
    """
    from quarry.core.shrink import NCS_PLUS, run_shrink

    venue_set = set(venues) if venues is not None else NCS_PLUS
    exclude_set = {
        int(w.lstrip("W")) for w in (exclude or []) if w.lstrip("W").isdigit()
    }

    return run_shrink(
        seed=seed,
        csr_dir=str(settings.csr_dir),
        pg_conninfo=settings.pg_conninfo,
        top_n=top_n,
        venues=venue_set,
        expand_limit=expand_limit,
        no_foundation=no_foundation,
        exclude=exclude_set,
    )


@mcp.tool()
async def get_full_text(
    work_id: str | None = None,
    pmid: int | None = None,
    doi: str | None = None,
) -> dict:
    """Fetch full text for a paper using a three-source fallback chain.

    Source priority:
      1. PMC full-text  — if pmc_id available (PDF then HTML)
      2. Unpaywall      — resolves best OA URL via Unpaywall API
      3. oa_url         — OpenAlex OA URL (oa_url column in works table)

    Each source runs through the fetch waterfall:
      L1 httpx → L2 Playwright+stealth → L3 HTML text

    Args:
        work_id: OpenAlex work ID (e.g., "W2741809807")
        pmid: PubMed ID (integer)
        doi: DOI string
    """
    from quarry.fetch import fetch, fetch_pmc, fetch_unpaywall

    db = _get_db()
    if work_id is not None:
        row = db.get_work(work_id)
    elif pmid is not None:
        row = db.get_work_by_pmid(pmid)
    elif doi is not None:
        row = db.get_work_by_doi(doi)
    else:
        return {"error": "Provide work_id, pmid, or doi"}

    if row is None:
        return {"error": "Paper not found"}

    pmc_id: str | None = row.get("pmc_id")
    paper_doi: str | None = row.get("doi")
    oa_url: str | None = row.get("oa_url")
    source_tried: list[str] = []

    result = None

    # 1. PMC
    if pmc_id:
        source_tried.append(f"PMC:{pmc_id}")
        result = await fetch_pmc(pmc_id)
        if result.success:
            source = f"PMC:{pmc_id}"
            return _full_text_response(row, result, source)

    # 2. Unpaywall
    if paper_doi:
        source_tried.append(f"unpaywall:{paper_doi}")
        result = await fetch_unpaywall(paper_doi)
        if result.success:
            return _full_text_response(row, result, f"unpaywall:{paper_doi}")

    # 3. oa_url
    if oa_url:
        source_tried.append(f"oa_url:{oa_url[:60]}")
        result = await fetch(oa_url)
        if result.success:
            return _full_text_response(row, result, "oa_url")

    # All sources exhausted
    last_notes = result.notes if result else []
    return {
        "work_id": row.get("work_id"),
        "pmid": row.get("pmid"),
        "doi": paper_doi,
        "success": False,
        "sources_tried": source_tried,
        "notes": last_notes,
    }


def _full_text_response(row: dict, result, source: str) -> dict:
    return {
        "work_id": row.get("work_id"),
        "pmid": row.get("pmid"),
        "doi": row.get("doi"),
        "source": source,
        "layer": result.layer,
        "success": True,
        "pdf_bytes": result.pdf_bytes,
        "text_chars": len(result.text),
        "text": result.text,
        "notes": result.notes,
    }


def main():
    # host/port must be set on the FastMCP instance before run(), not passed to run()
    mcp.settings.host = settings.mcp_host
    mcp.settings.port = settings.mcp_port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
