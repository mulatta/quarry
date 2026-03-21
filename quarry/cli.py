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
def mcp_server():
    """Start MCP server."""
    from quarry.mcp.server import main

    main()


if __name__ == "__main__":
    app()
