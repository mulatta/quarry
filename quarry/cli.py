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
def vsearch(
    work_id: str = typer.Argument(..., help="OpenAlex work ID"),
    limit: int = typer.Option(20, "--limit", "-n"),
):
    """Vector search: find similar papers by embedding."""
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
def info(
    work_ids: list[str] = typer.Argument(
        ..., help="One or more work IDs: W<id>, DOI, PMID"
    ),
    full: bool = typer.Option(False, "--full", help="Show full abstract"),
    fmt: str = typer.Option("table", "--format", "-f", help="table|json"),
):
    """Lookup metadata for one or more papers."""
    import json as json_mod

    from rich.console import Console

    from quarry.config import settings
    from quarry.store.pg import PGStore

    db = PGStore(settings.pg_conninfo)
    results = []
    for wid_str in work_ids:
        row = _resolve_and_get(wid_str, db)
        if row is None:
            typer.echo(f"Not found: {wid_str}", err=True)
            continue
        results.append(row)

    if not results:
        raise typer.Exit(1)

    if fmt == "json":
        out = []
        for r in results:
            entry = {
                "work_id": r.get("work_id"),
                "title": r.get("title"),
                "pub_year": r.get("pub_year"),
                "doi": r.get("doi"),
                "host_venue": r.get("host_venue"),
                "cited_by_count": r.get("cited_by_count"),
                "rcr": r.get("rcr"),
                "oa_status": r.get("oa_status"),
            }
            if full:
                entry["abstract"] = r.get("abstract", "")
            out.append(entry)
        typer.echo(
            json_mod.dumps(
                out if len(out) > 1 else out[0], indent=2, ensure_ascii=False
            )
        )
    else:
        console = Console()
        for r in results:
            console.print(
                f"[bold]{r.get('work_id', '?')}[/bold]  {r.get('title', '-')}"
            )
            console.print(
                f"  year={r.get('pub_year', '-')}  cited_by={r.get('cited_by_count', '-')}  "
                f"rcr={r.get('rcr', '-')}  oa={r.get('oa_status', '-')}"
            )
            console.print(f"  venue={r.get('host_venue', '-')}")
            console.print(f"  doi={r.get('doi', '-')}")
            if full and r.get("abstract"):
                console.print(f"  [dim]{r['abstract']}[/dim]")
            console.print()


def _resolve_and_get(identifier: str, db) -> dict | None:
    """Resolve identifier to a work dict. Accepts W<id>, DOI, PMID."""
    # W-prefix
    if identifier.upper().startswith("W") and identifier[1:].isdigit():
        return db.get_work(identifier.upper())

    # PMID (plain integer)
    try:
        pmid = int(identifier)
        return db.get_work_by_pmid(pmid)
    except ValueError:
        pass

    # DOI
    if identifier.startswith("10.") or identifier.startswith("https://doi.org/"):
        doi = "https://doi.org/" + identifier.removeprefix("https://doi.org/").lower()
        return db.get_work_by_doi(doi)

    return None


_SCHEMA_HELP: dict[str, list[tuple[str, str]]] = {
    "works": [
        ("work_id", "text — OpenAlex ID (e.g. W1234567890)"),
        ("work_id_int", "bigint — numeric part of work_id"),
        ("title", "text"),
        ("abstract", "text"),
        ("pub_year", "smallint"),
        ("pub_date", "date"),
        ("doi", "text — full URL (https://doi.org/...)"),
        ("pmid", "integer"),
        ("host_venue", "text — journal/venue name"),
        ("cited_by_count", "integer"),
        ("rcr", "real — relative citation ratio (iCite)"),
        ("fwci", "real — field-weighted citation impact"),
        ("citation_normalized_percentile", "real — 0-1 range (cnp)"),
        ("oa_status", "text — gold/green/hybrid/bronze/closed"),
        ("type", "text — article/review/preprint/..."),
        ("language", "text"),
        ("is_retracted", "boolean"),
    ],
    "work_citations": [
        ("citing_work_id_int", "bigint"),
        ("cited_work_id_int", "bigint"),
    ],
    "work_authors": [
        ("work_id", "text"),
        ("author_position", "text — first/middle/last"),
        ("author_id", "text — OpenAlex author ID"),
        ("author_name", "text"),
        ("raw_affiliation", "text"),
        ("institution_id", "text"),
        ("institution_name", "text"),
    ],
    "work_topics": [
        ("work_id", "text"),
        ("topic_id", "text"),
        ("topic_name", "text"),
        ("subfield_name", "text"),
        ("field_name", "text"),
        ("domain_name", "text"),
        ("score", "real"),
    ],
    "work_mesh": [
        ("work_id", "text"),
        ("descriptor_ui", "text"),
        ("descriptor_name", "text"),
        ("qualifier_ui", "text"),
        ("qualifier_name", "text"),
        ("is_major_topic", "boolean"),
    ],
    "mesh_tree": [
        ("descriptor_ui", "text"),
        ("descriptor_name", "text"),
        ("tree_number", "text"),
    ],
    "id_crosswalk": [
        ("work_id", "text"),
        ("pmid", "integer"),
        ("doi", "text"),
    ],
}


@app.command()
def sql(
    query: str | None = typer.Argument(None, help="SQL SELECT query"),
    schema: str | None = typer.Option(
        None, "--schema", "-s", help="Show schema for a table"
    ),
):
    """Execute read-only SQL against PG (development only)."""
    from quarry.store.pg import PGStore
    from quarry.config import settings

    if schema is not None:
        tbl = schema.lower()
        if tbl not in _SCHEMA_HELP:
            typer.echo(f"Unknown table: {tbl}", err=True)
            typer.echo(f"Available: {', '.join(sorted(_SCHEMA_HELP))}")
            raise typer.Exit(1)
        typer.echo(f"\n  {tbl}")
        typer.echo(f"  {'─' * 50}")
        for col, desc in _SCHEMA_HELP[tbl]:
            typer.echo(f"  {col:<35} {desc}")
        typer.echo()
        return

    if query is None:
        typer.echo("Provide a query or use --schema <table>", err=True)
        typer.echo(f"Available tables: {', '.join(sorted(_SCHEMA_HELP))}")
        raise typer.Exit(1)

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
    table.add_column("cited", justify="right", width=6, no_wrap=True)
    table.add_column("score", justify="right", width=8, no_wrap=True)
    table.add_column("work_id", width=13, no_wrap=True)
    table.add_column("title", ratio=1, overflow="ellipsis", no_wrap=True)

    for p in papers:
        year = str(p["year"]) if p["year"] else "-"
        cited = (
            str(p["quality"]["cited_by"])
            if p.get("quality", {}).get("cited_by") is not None
            else "-"
        )
        table.add_row(
            str(p["rank"]),
            p["relation"],
            year,
            cited,
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
        "Options: common_refs, common_citers, coupling, cocitation, path, ppr",
    ),
    limit: int = typer.Option(100, "--limit", "-n", help="Max results per type"),
    max_neighbor_degree: int = typer.Option(
        10_000, "--max-degree", help="Prune neighbors above this degree (Type 3-4)"
    ),
    max_path_depth: int = typer.Option(
        5, "--max-path-depth", help="Max BFS depth for path bridges (Type 5)"
    ),
    alpha: float = typer.Option(
        0.15, "--alpha", help="APPR teleport probability (Type 6)"
    ),
    epsilon: float = typer.Option(
        1e-6,
        "--epsilon",
        help="APPR precision threshold (Type 6, lower = wider search)",
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
      ppr            — structurally central to both networks (global)

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
            alpha=alpha,
            epsilon=epsilon,
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
    if sp_len is not None and sp_len >= 6:
        console.print(
            f"[yellow]⚠ sp={sp_len}: long citation chain — "
            f"path_bridges may include off-topic papers[/yellow]"
        )

    _SECTIONS = [
        ("common_refs", "Common Refs (shared foundation)"),
        ("common_citers", "Common Citers (synthesis papers)"),
        ("coupling_bridges", "Coupling Bridges (combined methods)"),
        ("cocitation_bridges", "Cocitation Bridges (community reads)"),
        ("path_bridges", "Path Bridges (stepping stones)"),
        ("ppr_bridges", "PPR Bridges (structural centrality)"),
        ("steiner_bridges", "Steiner Bridges (minimal connecting set, k≥3)"),
    ]

    for key, label in _SECTIONS:
        entries = result.get(key, [])
        if not entries:
            continue

        console.print(f"\n[bold]{label}[/bold]  ({len(entries)})")

        is_path = "hop_from" in entries[0]
        is_scored = "score" in entries[0]
        is_simple = "aa_weight" not in entries[0] and not is_scored and not is_path

        table = Table(show_edge=False, pad_edge=False, box=None, expand=True)
        table.add_column("#", justify="right", style="dim", width=4, no_wrap=True)
        table.add_column("year", justify="right", width=4, no_wrap=True)

        if is_path:
            table.add_column("hops", justify="right", width=7, no_wrap=True)
            table.add_column("paths", justify="right", width=6, no_wrap=True)
        elif is_scored:
            table.add_column("score", justify="right", width=10, no_wrap=True)
        elif not is_simple:
            table.add_column("aa", justify="right", width=8, no_wrap=True)

        table.add_column("cited", justify="right", width=6, no_wrap=True)
        table.add_column("work_id", width=13, no_wrap=True)
        table.add_column("title", ratio=1, overflow="ellipsis", no_wrap=True)

        for i, e in enumerate(entries):
            year = str(e["year"]) if e.get("year") else "-"
            cited = (
                str(e.get("quality", {}).get("cited_by", "-"))
                if e.get("quality", {}).get("cited_by") is not None
                else "-"
            )
            if is_path:
                hops = "+".join(str(h) for h in e["hop_from"])
                table.add_row(
                    str(i + 1),
                    year,
                    hops,
                    str(e["path_count"]),
                    cited,
                    f"W{e['work_id']}",
                    Text(e.get("title") or "-"),
                )
            elif is_simple:
                table.add_row(
                    str(i + 1),
                    year,
                    cited,
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
                    cited,
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
