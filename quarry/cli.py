"""Quarry CLI: paper search, citation graph, SQL queries.

Requires: pip install quarry[cli]
"""

try:
    import typer
except ImportError:
    raise ImportError("pip install quarry[cli]") from None

app = typer.Typer(name="quarry", help="Academic paper exploration CLI.")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="hybrid|vector|text"),
):
    """Hybrid search for papers."""
    from quarry.search.hybrid import HybridSearcher
    from quarry.store.pg import PGStore
    from quarry.config import settings

    db = PGStore(settings.pg_conninfo)
    searcher = HybridSearcher(db=db)
    results = searcher.search(query, limit=limit, mode=mode)
    for r in results:
        wid = r.get("work_id", "?")
        title = r.get("title", "")[:80]
        year = r.get("pub_year", "")
        typer.echo(f"  {wid}  {year}  {title}")


@app.command()
def similar(
    work_id: str = typer.Argument(..., help="OpenAlex work ID"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """Find similar papers by embedding."""
    from quarry.search.hybrid import HybridSearcher
    from quarry.store.pg import PGStore
    from quarry.config import settings

    db = PGStore(settings.pg_conninfo)
    searcher = HybridSearcher(db=db)
    results = searcher.similar(work_id, limit=limit)
    for r in results:
        wid = r.get("work_id", "?")
        title = r.get("title", "")[:80]
        typer.echo(f"  {wid}  {title}")


@app.command()
def sql(
    query: str = typer.Argument(..., help="SQL SELECT query"),
):
    """Execute read-only SQL against PG."""
    from quarry.store.pg import PGStore
    from quarry.config import settings

    db = PGStore(settings.pg_conninfo)
    rows = db.query(query)
    for row in rows[:50]:
        typer.echo(row)


@app.command()
def expand(
    seed: str = typer.Argument(
        ..., help="Seed paper: work_id_int, DOI (--doi), or PMID (--pmid)"
    ),
    limit: int = typer.Option(200, "--limit", "-n", help="Max results"),
    fmt: str = typer.Option("table", "--format", "-f", help="table|json"),
    alpha: float = typer.Option(0.15, "--alpha", help="APPR teleport probability"),
    epsilon: float = typer.Option(1e-6, "--epsilon", help="APPR precision threshold"),
):
    """Expand seed paper into a ranked subgraph of related papers."""
    import json
    import time

    import quarry_graph

    from quarry.config import settings

    # Load graph
    g = quarry_graph.Graph(str(settings.csr_dir))

    # Resolve seed to work_id_int
    seed_id = _resolve_seed(seed)

    if not g.has_node(seed_id):
        typer.echo(f"Error: seed {seed_id} not found in graph", err=True)
        raise typer.Exit(1)

    # Run expand
    t0 = time.perf_counter()
    papers, stats = g.expand(seed_id, alpha=alpha, epsilon=epsilon, limit=limit)
    elapsed = time.perf_counter() - t0

    # Enrich with metadata from PG
    metadata = _enrich_metadata([wid for wid, _ in papers])

    # Relation tagging
    seed_fwd = set(g.neighbors(seed_id, "forward"))
    seed_rev = set(g.neighbors(seed_id, "reverse"))

    results = []
    for wid, score in papers:
        meta = metadata.get(wid, {})
        cites_seed = wid in seed_rev  # paper → seed
        cited_by_seed = wid in seed_fwd  # seed → paper
        if cited_by_seed and cites_seed:
            relation = "mutual"
        elif cited_by_seed:
            relation = "foundation"
        elif cites_seed:
            relation = "follow-up"
        else:
            relation = "lateral"
        results.append(
            {
                "work_id": wid,
                "title": meta.get("title", ""),
                "year": meta.get("pub_year"),
                "cited_by": meta.get("cited_by_count"),
                "relation": relation,
                "fused_score": round(score, 6),
            }
        )

    if fmt == "json":
        output = {
            "seed": seed_id,
            "params": {"alpha": alpha, "epsilon": epsilon, "limit": limit},
            "papers": results,
            "stats": {
                **stats,
                "returned": len(results),
                "elapsed_s": round(elapsed, 3),
            },
        }
        typer.echo(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        typer.echo(f"Seed: W{seed_id}  ({elapsed:.2f}s, {len(results)} results)")
        typer.echo(
            f"Stats: appr={stats['appr_candidates']} coupling={stats['coupling_candidates']} cocite={stats['cocitation_candidates']}"
        )
        typer.echo()
        for i, r in enumerate(results[:limit]):
            rel = r["relation"][:8].ljust(8)
            year = r["year"] or "?"
            title = (r["title"] or "")[:70]
            typer.echo(
                f"  {i + 1:3d}. [{rel}] {year}  {r['fused_score']:.6f}  W{r['work_id']}  {title}"
            )


def _resolve_seed(seed: str) -> int:
    """Resolve seed string to work_id_int."""
    # Direct integer
    try:
        return int(seed)
    except ValueError:
        pass

    # DOI or PMID — resolve via PG
    from quarry.store.pg import PGStore
    from quarry.config import settings

    db = PGStore(settings.pg_conninfo)

    if seed.startswith("10.") or seed.startswith("https://doi.org/"):
        doi = seed.removeprefix("https://doi.org/")
        rows = db.query(
            "SELECT work_id_int FROM works WHERE doi = %s OR doi = %s LIMIT 1",
            (doi, f"https://doi.org/{doi}"),
        )
        if rows:
            return rows[0][0]
        raise typer.Exit(f"DOI not found: {doi}")

    # Try PMID
    try:
        pmid = int(seed)
        rows = db.query(
            "SELECT work_id_int FROM id_crosswalk WHERE pmid = %s LIMIT 1",
            (pmid,),
        )
        if rows:
            return rows[0][0]
    except ValueError:
        pass

    raise typer.Exit(f"Cannot resolve seed: {seed}")


def _enrich_metadata(work_ids: list[int]) -> dict:
    """Fetch title, year, cited_by from PG for a batch of work_ids."""
    if not work_ids:
        return {}

    try:
        from quarry.store.pg import PGStore
        from quarry.config import settings

        db = PGStore(settings.pg_conninfo)
        placeholders = ",".join(["%s"] * len(work_ids))
        rows = db.query(
            f"SELECT work_id_int, title, pub_year, cited_by_count "
            f"FROM works WHERE work_id_int IN ({placeholders})",
            tuple(work_ids),
        )
        return {
            row[0]: {"title": row[1], "pub_year": row[2], "cited_by_count": row[3]}
            for row in rows
        }
    except Exception:
        # PG unavailable — return empty metadata
        return {}


@app.command()
def mcp_server():
    """Start MCP server."""
    from quarry.mcp.server import main

    main()


if __name__ == "__main__":
    app()
