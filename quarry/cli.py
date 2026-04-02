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
def mcp_server():
    """Start MCP server."""
    from quarry.mcp.server import main

    main()


if __name__ == "__main__":
    app()
