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
    mode: str = typer.Option(
        "fused", "--mode", "-m", help="fused (focused) | separated (broad)"
    ),
    fmt: str = typer.Option("table", "--format", "-f", help="table|json|detail"),
    alpha: float = typer.Option(0.15, "--alpha", help="APPR teleport probability"),
    epsilon: float = typer.Option(1e-6, "--epsilon", help="APPR precision threshold"),
):
    """Expand seed paper into a ranked subgraph of related papers.

    Modes:
      fused     — wRRF fusion, all candidates compete (focused exploration)
      separated — APPR + guaranteed lateral slots (broad survey)
    """
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
    papers, stats = g.expand(
        seed_id, alpha=alpha, epsilon=epsilon, mode=mode, limit=limit
    )
    elapsed = time.perf_counter() - t0

    # Enrich with metadata from PG
    include_abstract = fmt == "detail"
    metadata = _enrich_metadata(
        [wid for wid, _ in papers], include_abstract=include_abstract
    )

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
        entry = {
            "work_id": wid,
            "title": meta.get("title", ""),
            "year": meta.get("pub_year"),
            "cited_by": meta.get("cited_by_count"),
            "relation": relation,
            "fused_score": round(score, 6),
        }
        if include_abstract:
            entry["abstract"] = meta.get("abstract", "")
        results.append(entry)

    if fmt in ("json", "detail"):
        output = {
            "seed": seed_id,
            "params": {
                "alpha": alpha,
                "epsilon": epsilon,
                "mode": mode,
                "limit": limit,
            },
            "papers": results,
            "stats": {
                **stats,
                "returned": len(results),
                "elapsed_s": round(elapsed, 3),
            },
        }
        typer.echo(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        from rich.console import Console
        from rich.table import Table

        from rich.text import Text

        console = Console()
        console.print(
            f"Seed: W{seed_id}  ({elapsed:.2f}s, {len(results)} results)  "
            f"[dim]appr={stats['appr_candidates']} coup={stats['coupling_candidates']} cocite={stats['cocitation_candidates']}[/dim]"
        )

        table = Table(show_edge=False, pad_edge=False, box=None, expand=True)
        table.add_column("#", justify="right", style="dim", width=4, no_wrap=True)
        table.add_column("rel", width=10, no_wrap=True)
        table.add_column("year", justify="right", width=4, no_wrap=True)
        table.add_column("score", justify="right", width=8, no_wrap=True)
        table.add_column("work_id", width=13, no_wrap=True)
        table.add_column("title", ratio=1, overflow="ellipsis", no_wrap=True)

        for i, r in enumerate(results[:limit]):
            rel = r["relation"]
            year = str(r["year"]) if r["year"] else "-"
            title = r["title"] or "-"
            table.add_row(
                str(i + 1),
                rel,
                year,
                f"{r['fused_score']:.6f}",
                f"W{r['work_id']}",
                Text(title),
            )

        console.print(table)


def _resolve_seed(seed: str) -> int:
    """Resolve seed string to work_id_int.

    Accepts: work_id_int, W<id>, DOI, https://doi.org/DOI, --pmid <pmid>.
    """
    # Strip W prefix (OpenAlex format)
    if seed.upper().startswith("W") and seed[1:].isdigit():
        return int(seed[1:])

    # Direct integer → work_id_int
    try:
        return int(seed)
    except ValueError:
        pass

    # DOI — normalize to OA format (https://doi.org/ + lowercase) for exact match
    if seed.startswith("10.") or seed.startswith("https://doi.org/"):
        doi = "https://doi.org/" + seed.removeprefix("https://doi.org/").lower()
        return _pg_lookup(
            "SELECT work_id_int FROM works WHERE doi = %s LIMIT 1",
            (doi,),
            f"DOI not found: {doi}",
        )

    raise typer.Exit(f"Cannot resolve seed: {seed}")


def _pg_lookup(sql: str, params: tuple, err_msg: str) -> int:
    """Execute parameterized PG query, return first column of first row."""
    import psycopg
    from quarry.config import settings

    with psycopg.connect(settings.pg_conninfo) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
        if row:
            return row[0]
    raise typer.Exit(err_msg)


def _enrich_metadata(work_ids: list[int], *, include_abstract: bool = False) -> dict:
    """Fetch metadata from PG for a batch of work_ids."""
    if not work_ids:
        return {}

    try:
        import psycopg
        from quarry.config import settings

        cols = "work_id_int, title, pub_year, cited_by_count"
        if include_abstract:
            cols += ", abstract"
        with psycopg.connect(settings.pg_conninfo) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols} FROM works WHERE work_id_int = ANY(%s)",
                (list(work_ids),),
            )
            result = {}
            for row in cur.fetchall():
                entry = {"title": row[1], "pub_year": row[2], "cited_by_count": row[3]}
                if include_abstract:
                    entry["abstract"] = row[4] or ""
                result[row[0]] = entry
            return result
    except Exception:
        return {}


@app.command()
def mcp_server():
    """Start MCP server."""
    from quarry.mcp.server import main

    main()


if __name__ == "__main__":
    app()
