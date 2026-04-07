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
        ..., help="Seed paper: work_id_int, W<id>, DOI, or https://doi.org/..."
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

    Output: complete JSON with metadata. Use jq/SQL for filtering/sorting.
    """
    import json

    from quarry.config import settings
    from quarry.core.expand import run_expand

    try:
        result = run_expand(
            seed=seed,
            csr_dir=str(settings.csr_dir),
            pg_conninfo=settings.pg_conninfo,
            mode=mode,
            alpha=alpha,
            epsilon=epsilon,
            limit=limit,
            include_abstract=(fmt == "detail"),
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if fmt in ("json", "detail"):
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_table(result)


def _print_table(result: dict) -> None:
    """Print expand result as a rich table."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    stats = result["stats"]
    seed_id = result["seed"]["work_id"]
    papers = result["papers"]

    console = Console()
    console.print(
        f"Seed: W{seed_id}  ({stats['elapsed_s']}s, {stats['returned']} results)  "
        f"[dim]appr={stats['appr_candidates']} coup={stats['coupling_candidates']} cocite={stats['cocitation_candidates']}[/dim]"
    )

    table = Table(show_edge=False, pad_edge=False, box=None, expand=True)
    table.add_column("#", justify="right", style="dim", width=4, no_wrap=True)
    table.add_column("rel", width=10, no_wrap=True)
    table.add_column("year", justify="right", width=4, no_wrap=True)
    table.add_column("score", justify="right", width=8, no_wrap=True)
    table.add_column("work_id", width=13, no_wrap=True)
    table.add_column("title", ratio=1, overflow="ellipsis", no_wrap=True)

    for p in papers:
        year = str(p["year"]) if p["year"] else "-"
        table.add_row(
            str(p["rank"]),
            p["relation"],
            year,
            f"{p['scores']['fused']:.6f}",
            f"W{p['work_id']}",
            Text(p["title"] or "-"),
        )

    console.print(table)


@app.command()
def bridge(
    seeds: list[str] = typer.Argument(
        ...,
        help="Two or more seed papers: work_id_int, W<id>, DOI, or https://doi.org/...",
    ),
    types: list[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Bridge types to compute (default: all). "
        "Options: common_refs, common_citers, coupling, cocitation, path",
    ),
    limit: int = typer.Option(100, "--limit", "-n", help="Max results per type"),
    max_neighbor_degree: int = typer.Option(
        10_000, "--max-degree", help="Prune neighbors above this degree (Type 3-4)"
    ),
    max_path_depth: int = typer.Option(
        5, "--max-path-depth", help="Max BFS depth for path bridges (Type 5)"
    ),
    fmt: str = typer.Option("table", "--format", "-f", help="table|json|detail"),
):
    """Discover bridge papers connecting two or more seeds.

    Each bridge type answers a different structural question:
      common_refs    — shared intellectual foundation
      common_citers  — who already synthesized both
      coupling       — who combined both methods
      cocitation     — what both communities read
      path           — stepping-stone reading path between seeds

    Output: complete JSON with metadata. Use jq/SQL for filtering/sorting.
    """
    import json

    from quarry.config import settings
    from quarry.core.bridge import run_bridge

    if len(seeds) < 2:
        typer.echo("Error: bridge requires at least 2 seeds", err=True)
        raise typer.Exit(1)

    try:
        result = run_bridge(
            seeds=seeds,
            csr_dir=str(settings.csr_dir),
            pg_conninfo=settings.pg_conninfo,
            types=types or None,
            limit=limit,
            max_neighbor_degree=max_neighbor_degree,
            max_path_depth=max_path_depth,
            include_abstract=(fmt == "detail"),
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if fmt in ("json", "detail"):
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_bridge_table(result)


def _print_bridge_table(result: dict) -> None:
    """Print bridge result as rich tables, one per type."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    stats = result["stats"]

    # Header
    seed_strs = [f"W{s['work_id']}" for s in result["seeds"]]
    sp_len = stats.get("shortest_path_length")
    sp_info = f" sp={sp_len}" if sp_len is not None else ""
    console.print(
        f"Bridge: {' ↔ '.join(seed_strs)}  "
        f"({stats['elapsed_s']}s)  "
        f"[dim]refs overlap={stats['overlap_refs']} "
        f"citers overlap={stats['overlap_citers']}{sp_info}[/dim]"
    )

    _SECTIONS = [
        ("common_refs", "Common Refs (shared foundation)"),
        ("common_citers", "Common Citers (synthesis papers)"),
        ("coupling_bridges", "Coupling Bridges (combined methods)"),
        ("cocitation_bridges", "Cocitation Bridges (community reads)"),
        ("path_bridges", "Path Bridges (stepping stones)"),
    ]

    for key, label in _SECTIONS:
        entries = result.get(key, [])
        if not entries:
            continue

        console.print(f"\n[bold]{label}[/bold]  ({len(entries)})")

        is_path = "hop_from" in entries[0]
        is_scored = "score" in entries[0]

        table = Table(show_edge=False, pad_edge=False, box=None, expand=True)
        table.add_column("#", justify="right", style="dim", width=4, no_wrap=True)
        table.add_column("year", justify="right", width=4, no_wrap=True)

        if is_path:
            table.add_column("hops", justify="right", width=7, no_wrap=True)
            table.add_column("paths", justify="right", width=6, no_wrap=True)
        elif is_scored:
            table.add_column("score", justify="right", width=10, no_wrap=True)
        else:
            table.add_column("aa", justify="right", width=8, no_wrap=True)

        table.add_column("work_id", width=13, no_wrap=True)
        table.add_column("title", ratio=1, overflow="ellipsis", no_wrap=True)

        for i, e in enumerate(entries):
            year = str(e["year"]) if e.get("year") else "-"
            if is_path:
                hops = "+".join(str(h) for h in e["hop_from"])
                table.add_row(
                    str(i + 1),
                    year,
                    hops,
                    str(e["path_count"]),
                    f"W{e['work_id']}",
                    Text(e.get("title") or "-"),
                )
            else:
                score_col = (
                    f"{e['score']:.6f}" if is_scored else f"{e['aa_weight']:.4f}"
                )
                table.add_row(
                    str(i + 1),
                    year,
                    score_col,
                    f"W{e['work_id']}",
                    Text(e.get("title") or "-"),
                )

        console.print(table)


@app.command()
def mcp_server():
    """Start MCP server."""
    from quarry.mcp.server import main

    main()


if __name__ == "__main__":
    app()
