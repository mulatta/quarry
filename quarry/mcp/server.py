"""FastMCP server — academic paper exploration via citation graph, MeSH, and semantic search.

## Tool overview

| Tool              | When to use                                                  |
|-------------------|--------------------------------------------------------------|
| mesh_explore      | Discover MeSH descriptors; find field entry-point papers     |
| search_papers     | Keyword/semantic search when you have a natural-language topic|
| get_paper         | Look up a specific paper by work_id / PMID / DOI             |
| expand            | Map citation neighborhood of a seed (APPR + wRRF)           |
| bridge            | Find cross-domain connectors between 2+ seeds                |
| shrink            | Build a minimal reading list from top-venue papers           |
| get_full_text     | Fetch full text (PMC → Unpaywall → OA URL)                   |
| similar_papers    | Embedding-based similarity search                            |
| expand_citations  | Raw k-hop graph walk (simpler than expand)                   |
| find_path         | Shortest citation chain between two papers                   |
| get_subgraph      | Structural metrics for a set of papers                       |
| query_metadata    | SQL escape hatch for ad-hoc metadata queries                 |

## Typical exploration sequences

Discovery (unknown field):
  mesh_explore(descriptor_name) → mesh_explore(direction="papers") → expand(mesh_summary=True)

From a known paper:
  get_paper(include_mesh=True) → expand → bridge([seed1, seed2])

Reading list:
  expand → shrink

Full-text:
  get_paper → get_full_text(work_id/pmid/doi)

Fallback when mesh_explore misses:
  query_metadata("SELECT ... FROM mesh_lookup WHERE descriptor_name ILIKE '%term%'")
"""

from __future__ import annotations

from typing import Annotated

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
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


@mcp.tool(annotations=_RO)
def search_papers(
    query: Annotated[
        str, "Natural language query — topic, title fragment, or author+year"
    ],
    limit: Annotated[int, "Max results (default 20, max ~200)"] = 20,
    mode: Annotated[
        str,
        '"hybrid" (BM25+vector, best quality), "vector" (semantic only), "text" (BM25 only)',
    ] = "hybrid",
    enrich: Annotated[
        bool,
        "Include full metadata (title, authors, year, venue, cited_by). Set False for IDs only.",
    ] = True,
) -> list[dict]:
    """Search papers by natural-language query using BM25 + vector ANN + RRF fusion.

    Use this when you have a topic in plain English but no known work_id.
    Returns papers ranked by relevance; each has work_id, title, year, cited_by_count.

    Prefer mesh_explore when the topic maps cleanly to a MeSH descriptor (more precise).
    Prefer similar_papers when you already have a seed paper and want neighbors by content.

    Fallback if results are poor: mesh_explore(descriptor_name=<topic keyword>)
    """
    from quarry.search.hybrid import HybridSearcher

    searcher = HybridSearcher(db=_get_db())
    return searcher.search(query, limit=limit, mode=mode, enrich=enrich)


@mcp.tool(annotations=_RO)
def get_paper(
    work_id: Annotated[
        str | None, "OpenAlex work ID — format W<digits>, e.g. 'W2741809807'"
    ] = None,
    pmid: Annotated[
        int | None,
        "PubMed ID (integer). Use when you have a PMID from a search result.",
    ] = None,
    doi: Annotated[
        str | None, "DOI string, with or without https://doi.org/ prefix"
    ] = None,
    include_mesh: Annotated[
        bool,
        "Add MeSH descriptor tags — useful for seed validation and field characterisation",
    ] = False,
    include_abstract: Annotated[
        bool, "Add abstract text — required before calling get_full_text"
    ] = False,
) -> dict | None:
    """Get detailed metadata for a single paper by work_id, PMID, or DOI.

    Use this to:
    - Verify a paper exists before passing its work_id to expand/bridge/shrink
    - Get identifiers (work_id, pmid, pmc_id, doi) needed by other tools
    - Check MeSH tags to assess whether a paper is a good seed (include_mesh=True)
    - Preview abstract to judge relevance before fetching full text (include_abstract=True)

    Returns None if the paper is not in the database.

    Typical flow:
      search_papers or mesh_explore → get_paper(include_mesh=True) → expand
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


@mcp.tool(annotations=_RO)
def expand_citations(
    work_id: Annotated[str | None, "OpenAlex work ID (W<digits>)"] = None,
    pmid: Annotated[int | None, "PubMed ID (alternative to work_id)"] = None,
    direction: Annotated[
        str,
        '"forward" (papers this paper cites), "reverse" (papers that cite this), "both"',
    ] = "both",
    hops: Annotated[
        int,
        "Graph walk depth. 1 = direct neighbors, 2 = neighbors-of-neighbors. Keep ≤2 to avoid explosion.",
    ] = 1,
    max_nodes: Annotated[int, "Cap on returned nodes (default 500)"] = 500,
    enrich: Annotated[
        bool, "Include full paper metadata. Set False to get raw node IDs only."
    ] = True,
) -> dict:
    """Raw k-hop citation graph walk around a paper.

    Simpler and faster than expand — does a breadth-first walk rather than APPR.
    Use expand instead when you want relevance-ranked results.

    Use expand_citations when:
    - You want the complete direct-citation neighborhood (direction="forward"/"reverse")
    - You are tracing intellectual lineage hop-by-hop
    - expand is overkill for a simple neighbor lookup
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


@mcp.tool(annotations=_RO)
def find_path(
    source_work_id: Annotated[str | None, "Starting paper work_id (W<digits>)"] = None,
    source_pmid: Annotated[
        int | None, "Starting paper PMID (alternative to source_work_id)"
    ] = None,
    target_work_id: Annotated[str | None, "Target paper work_id (W<digits>)"] = None,
    target_pmid: Annotated[
        int | None, "Target paper PMID (alternative to target_work_id)"
    ] = None,
    max_depth: Annotated[
        int, "Maximum citation hops to search (default 6). Increase for distant papers."
    ] = 6,
    enrich: Annotated[bool, "Include full metadata for each paper on the path"] = True,
) -> dict:
    """Find the shortest citation chain connecting two papers.

    Returns the sequence of papers where each paper cites the next.
    Useful for tracing intellectual lineage or showing how field A influenced field B.

    Returns {"found": False} if no path exists within max_depth hops.

    Use bridge instead when you want multiple connectors rather than a single shortest path.
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


@mcp.tool(annotations=_RO)
def similar_papers(
    work_id: Annotated[str | None, "OpenAlex work ID (W<digits>)"] = None,
    pmid: Annotated[int | None, "PubMed ID (alternative to work_id)"] = None,
    limit: Annotated[int, "Max results (default 20)"] = 20,
) -> list[dict]:
    """Find papers with similar content using embedding (vector) similarity.

    Searches LanceDB vector store — results reflect semantic/topic similarity,
    not citation relationships. Complements expand (which is citation-based).

    Use this when:
    - You want papers on the same topic as a seed, across different citation communities
    - expand returns too few results (paper is new or isolated in the citation graph)
    - You want to cross-check citation-based findings with content-based ones
    """
    from quarry.search.hybrid import HybridSearcher

    db = _get_db()
    if pmid is not None and work_id is None:
        work_id = db.resolve_pmid_to_work_id(pmid)
        if work_id is None:
            return []

    if work_id is None:
        return []
    searcher = HybridSearcher(db=db)
    return searcher.similar(work_id, limit=limit)


@mcp.tool(annotations=_RO)
def get_subgraph(
    work_ids: Annotated[
        list[str] | None, "List of OpenAlex work IDs (W<digits>)"
    ] = None,
    pmids: Annotated[
        list[int] | None, "List of PMIDs (alternative to work_ids)"
    ] = None,
    include_metrics: Annotated[
        bool,
        "Compute betweenness centrality and connected components (skip for >10k nodes)",
    ] = True,
) -> dict:
    """Build a citation subgraph from a set of papers and compute structural metrics.

    Use this to analyse the internal structure of a paper set — e.g. after expand or bridge.
    Betweenness centrality identifies papers that act as structural bridges within the set.
    Connected components reveal whether the set is a single coherent cluster or disjoint.

    Typical use: get a set of work_ids from expand → pass to get_subgraph to rank by centrality.
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


@mcp.tool(annotations=_RO)
def query_metadata(
    sql: Annotated[
        str,
        "SELECT/WITH/EXPLAIN query against PostgreSQL. Tables: works, work_mesh, mesh_lookup, mesh_tree, venues.",
    ],
) -> list[dict]:
    """Execute a read-only SQL query against the PostgreSQL metadata database.

    Use as a fallback when other tools don't cover the query:
    - mesh_explore misses a descriptor → query mesh_lookup directly:
        SELECT descriptor_ui, descriptor_name FROM mesh_lookup WHERE descriptor_name ILIKE '%term%'
    - Need papers in a specific year/venue range:
        SELECT work_id, title, year FROM works WHERE venue_id = X AND year BETWEEN 2018 AND 2023
    - Count papers per MeSH tag:
        SELECT descriptor_name, COUNT(*) FROM work_mesh JOIN mesh_lookup USING(descriptor_ui) GROUP BY 1 ORDER BY 2 DESC LIMIT 20

    Only SELECT/WITH/EXPLAIN are permitted. No writes.
    Key tables: works (445M rows), work_mesh (379M), mesh_lookup (272K), mesh_tree (65K).
    """
    db = _get_db()
    return db.query(sql)


@mcp.tool(annotations=_RO)
def mesh_explore(
    descriptor_ui: Annotated[
        str | None,
        "MeSH descriptor UI, e.g. 'D009369'. Provide this OR descriptor_name.",
    ] = None,
    descriptor_name: Annotated[
        str | None,
        "Partial name search, e.g. 'neural plasticity'. Returns error if ambiguous — use descriptor_ui then.",
    ] = None,
    direction: Annotated[
        str,
        '"descendants": all sub-descriptors in hierarchy (default — use to find specific sub-topics). '
        '"papers": top cited papers tagged with this descriptor (good first-pass seed discovery). '
        '"tree": parent + direct children only (navigate hierarchy one level). '
        '"info": descriptor metadata only.',
    ] = "descendants",
    limit: Annotated[
        int, "Max papers returned when direction='papers' (default 15)"
    ] = 15,
) -> dict:
    """Explore the MeSH ontology and discover field entry-point papers.

    MeSH (Medical Subject Headings) is a controlled vocabulary used to index biomedical papers.
    Start here when you have a biomedical topic and need to find the right descriptor.

    Recommended discovery workflow:
      1. mesh_explore(descriptor_name="your topic") — find the right descriptor_ui
      2. If result has breadth_warning → direction="tree" to find sub-descriptors
      3. mesh_explore(descriptor_ui=X, direction="papers") — get top cited seed papers
      4. get_paper(work_id=..., include_mesh=True) — verify seed before expand

    If descriptor_name returns "multiple matches", pick one descriptor_ui and retry.
    If descriptor is not found at all, fall back to:
      query_metadata("SELECT descriptor_ui, descriptor_name FROM mesh_lookup WHERE descriptor_name ILIKE '%term%'")
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


@mcp.tool(annotations=_RO)
def expand(
    seed: Annotated[
        str,
        "Seed paper: W<id> (OpenAlex), DOI, or integer work_id_int. Verify with get_paper first.",
    ],
    mode: Annotated[
        str,
        '"fused" (APPR+wRRF, best quality — default) or "separated" (APPR only, faster)',
    ] = "fused",
    limit: Annotated[
        int,
        "Max papers returned (default 200). Increase to 500 for broad field mapping.",
    ] = 200,
    alpha: Annotated[
        float, "PPR teleport probability (default 0.15). Lower = broader neighborhood."
    ] = 0.15,
    epsilon: Annotated[
        float, "PPR convergence threshold (default 1e-6). Rarely needs changing."
    ] = 1e-6,
    min_citations: Annotated[
        int,
        "Exclude papers with fewer citations. Use 5-10 to filter noise; 0 = include all.",
    ] = 0,
    include_abstract: Annotated[
        bool, "Include abstract text — needed before calling get_full_text on results"
    ] = False,
    mesh_summary: Annotated[
        bool,
        "Append top-10 MeSH topic distribution — useful for field characterisation",
    ] = False,
) -> dict:
    """Map the citation neighborhood of a seed paper using APPR + wRRF ranking.

    The core exploration tool. Returns papers ranked by structural relevance
    to the seed in the citation graph. Covers both citing and cited papers.

    Use this to:
    - Map the full citation landscape around a known paper
    - Discover the key papers in a field starting from one entry point
    - Get a MeSH topic breakdown of a field (mesh_summary=True)

    Typical sequence:
      mesh_explore → get_paper(include_mesh=True) → expand(mesh_summary=True) → bridge

    If seed not found in graph (error "not in CSR"):
      Try a highly-cited paper in the same field from mesh_explore(direction="papers").
      New papers (< 2 years) may not be indexed yet — use similar_papers instead.
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


@mcp.tool(annotations=_RO)
def bridge(
    seeds: Annotated[
        list[str],
        "2+ seed papers from different domains. W<id>, DOI, or integer work_id_int. Use seeds from different sub-problems for cross-domain discovery.",
    ],
    types: Annotated[
        list[str] | None,
        (
            "Bridge types to run. None = all. Options: "
            "common_refs (share references — bibliographic coupling), "
            "common_citers (co-cited by same papers), "
            "coupling (strong bibliographic coupling), "
            "cocitation (frequently co-cited), "
            "path (shortest citation path), "
            "ppr (PPR score convergence across seeds), "
            "steiner (Steiner tree connecting seeds). "
            "Start with all types; narrow to coupling/cocitation for most reliable connectors."
        ),
    ] = None,
    limit: Annotated[int, "Max results per bridge type (default 100)"] = 100,
    max_neighbor_degree: Annotated[
        int,
        "Skip hub nodes above this degree — removes review journals and textbooks that connect everything (default 10000)",
    ] = 10_000,
    max_path_depth: Annotated[int, "Max hops for path-based bridges (default 5)"] = 5,
    alpha: Annotated[float, "PPR teleport probability (default 0.15)"] = 0.15,
    epsilon: Annotated[float, "PPR convergence threshold (default 1e-6)"] = 1e-6,
    include_abstract: Annotated[
        bool, "Include abstract text — set True when assessing bridge paper quality"
    ] = False,
    mesh_summary: Annotated[
        bool, "Append MeSH topic distribution across all bridge results"
    ] = False,
) -> dict:
    """Find papers that structurally connect two or more seed papers across domains.

    Runs up to 7 bridge algorithms and returns a dict keyed by type.
    The best connectors appear in multiple types (coupling + cocitation overlap is strong signal).

    Use this to:
    - Discover cross-domain connections between two research areas
    - Find the "bridge paper" that first connected two fields
    - Surface unexpected serendipitous connections

    Typical sequence:
      expand(seed1) + expand(seed2) → pick best seeds → bridge([seed1, seed2], mesh_summary=True)
      → get_paper(work_id=top_result, include_abstract=True) to assess each connector

    Quality assessment: prefer coupling/cocitation results over path/steiner.
    High-citation classics often produce trivial results — check mid-citation recent papers too.
    If all types return empty: seeds may be too closely related (same sub-field) or too distant.
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


@mcp.tool(annotations=_RO)
def shrink(
    seed: Annotated[str, "Seed paper: W<id>, DOI, or integer work_id_int"],
    top_n: Annotated[
        int,
        "Number of papers to select (default 5). Use 3-5 for a reading list, 10+ for a literature review.",
    ] = 5,
    venues: Annotated[
        list[str] | None,
        "Venue names to include. None = NCS_PLUS (Nature/Cell/Science + high-impact journals). Pass custom list for domain-specific venues.",
    ] = None,
    expand_limit: Annotated[
        int, "Candidate pool size from upstream expand (default 200)"
    ] = 200,
    no_foundation: Annotated[
        bool,
        "Exclude foundational/seminal papers (relation=foundation) — set True to focus on recent work",
    ] = False,
    exclude: Annotated[
        list[str] | None,
        "Work IDs to exclude from selection, e.g. ['W12345'] — use to remove known papers",
    ] = None,
) -> dict:
    """Build a minimal reading list: the fewest papers that cover the most ground.

    Runs expand internally, filters to top-venue papers, then greedily selects
    the set that collectively covers the most of the citation neighborhood.

    Use this to:
    - Build a 3-5 paper starting point for an unfamiliar field
    - Generate a reading list for a grant proposal or review paper
    - Find the canonical papers in a subfield

    NCS_PLUS venues include: Nature, Science, Cell, NEJM, Lancet, JAMA, PNAS,
    Nature Medicine, Nature Biotechnology, and ~20 other high-impact journals.

    If results feel too narrow: increase top_n or use no_foundation=True to surface recent work.
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


@mcp.tool(annotations=_RO_OPEN)
async def get_full_text(
    work_id: Annotated[
        str | None,
        "OpenAlex work ID (W<digits>). Get from get_paper or expand results.",
    ] = None,
    pmid: Annotated[
        int | None, "PubMed ID (integer). Best identifier for PMC papers."
    ] = None,
    doi: Annotated[str | None, "DOI string. Used for Unpaywall OA lookup."] = None,
) -> dict:
    """Fetch the full text of a paper as Markdown (PDF parsed via docling).

    Tries three sources in priority order:
      1. PMC full-text — highest quality; available if paper has pmc_id
      2. Unpaywall     — finds best open-access URL via Unpaywall API
      3. oa_url        — OpenAlex OA URL as last resort

    Each source tries: direct httpx download → Playwright browser (for JS-gated sites) → HTML extraction.

    Returns: {success, text, text_chars, source, layer, work_id, pmid, doi, notes}
    - success=False means all sources failed (paywalled or no OA version)
    - layer: "L1" (httpx), "L2" (playwright), "L3" (html) — indicates fetch method used
    - text: full Markdown text from PDF or HTML

    Recommended: call get_paper first to confirm the paper exists and has pmc_id/doi.
    PMID is the most reliable identifier for PMC papers.
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


@mcp.resource("quarry://docs/tools")
def resource_tool_overview() -> str:
    """Tool overview table and exploration sequences."""
    return __doc__ or ""


@mcp.resource("quarry://docs/bridge-types")
def resource_bridge_types() -> str:
    """Bridge algorithm reference: when to use each type and how to assess results."""
    return """\
# Bridge Types

| Type          | Mechanism                                    | Signal strength |
|---------------|----------------------------------------------|-----------------|
| coupling      | Share many references (bibliographic coupling)| Strong — shared intellectual foundation |
| cocitation    | Frequently cited together                     | Strong — treated as related by community |
| common_refs   | Share ≥1 reference                            | Broad — use for initial discovery |
| common_citers | Cited by the same papers                      | Moderate — similar impact area |
| ppr           | High PPR score across all seeds               | Good for multi-seed (3+) scenarios |
| path          | Lies on shortest citation path                | Directional lineage — A influenced B |
| steiner       | Minimum Steiner tree connecting seeds         | Structural — may be indirect |

## Quality assessment
- Prefer coupling + cocitation overlap: paper appears in both = strong cross-domain connector
- Avoid relying on very high-citation papers (>50k citations) — they connect everything trivially
- Recent papers (≤3 years) are underweighted by citation count; check abstracts directly
- If all types empty: seeds are co-located (same sub-field) or completely disconnected

## Serendipity signals
A bridge paper is serendipitous when:
1. From a completely different field than both seeds
2. Provides a mechanistic link (not just co-occurrence)
3. Has mid-range citations (10–500) — not a textbook, not a preprint
4. Abstract mentions BOTH domains or a shared mechanism
"""


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
