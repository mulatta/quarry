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
    show_mesh: bool = typer.Option(False, "--mesh", help="Show MeSH descriptors"),
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

    # Fetch MeSH tags if requested
    mesh_map: dict[str, list[dict]] = {}
    if show_mesh:
        for r in results:
            wid = r.get("work_id", "")
            mesh_map[wid] = db.get_work_mesh(wid)

    if fmt == "json":
        out = []
        for r in results:
            entry = {
                "work_id": r.get("work_id"),
                "title": r.get("title"),
                "pub_year": r.get("pub_year"),
                "doi": r.get("doi"),
                "pmid": r.get("pmid"),
                "pmc_id": r.get("pmc_id"),
                "host_venue": r.get("host_venue"),
                "cited_by_count": r.get("cited_by_count"),
                "rcr": r.get("rcr"),
                "oa_status": r.get("oa_status"),
            }
            if full:
                entry["abstract"] = r.get("abstract", "")
            if show_mesh:
                entry["mesh"] = mesh_map.get(r.get("work_id", ""), [])
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
            pmid_str = r.get("pmid") or "-"
            pmc_str = r.get("pmc_id") or "-"
            console.print(f"  venue={r.get('host_venue', '-')}")
            console.print(f"  doi={r.get('doi', '-')}")
            console.print(f"  pmid={pmid_str}  pmc={pmc_str}")
            if full and r.get("abstract"):
                console.print(f"  [dim]{r['abstract']}[/dim]")
            if show_mesh:
                tags = mesh_map.get(r.get("work_id", ""), [])
                if tags:
                    console.print("  mesh:")
                    for t in tags:
                        marker = "★" if t.get("is_major_topic") else " "
                        qual = (
                            f" / {t['qualifier_name']}"
                            if t.get("qualifier_name")
                            else ""
                        )
                        console.print(
                            f"    {marker} {t['descriptor_name']} ({t['descriptor_ui']}){qual}"
                        )
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
        return db.get_work_by_doi(identifier)

    return None


@app.command()
def mesh(
    query: str = typer.Argument(
        ..., help="MeSH descriptor name (partial) or UI (e.g. D018626)"
    ),
    tree: bool = typer.Option(
        False, "--tree", "-t", help="Show hierarchy (parents/children)"
    ),
    limit: int = typer.Option(15, "--limit", "-n", help="Max papers to show"),
):
    """MeSH-based paper discovery and hierarchy browsing.

    Find papers by curated MeSH vocabulary — precise topic filtering
    without keyword noise. Also browse the MeSH hierarchy.

    Examples:
      quarry mesh "retroelement"        # search by name
      quarry mesh D018626               # by descriptor UI
      quarry mesh D018626 --tree        # show hierarchy
    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    from quarry.config import settings
    from quarry.store.pg import PGStore

    console = Console()
    db = PGStore(settings.pg_conninfo)

    # Resolve query to descriptor_ui
    if query.startswith("D") and query[1:].isdigit():
        descriptor_ui = query
        entries = db.mesh_by_ui(descriptor_ui)
        if entries:
            descriptor_name = entries[0]["descriptor_name"]
        else:
            # Not in mesh_tree — check mesh_lookup (entry terms + historical)
            from psycopg.rows import dict_row

            with db.conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT DISTINCT descriptor_name FROM mesh_lookup "
                    "WHERE descriptor_ui = %s LIMIT 1",
                    (descriptor_ui,),
                )
                row = cur.fetchone()
            if not row:
                typer.echo(f"Not found: {descriptor_ui}", err=True)
                raise typer.Exit(1)
            descriptor_name = row["descriptor_name"]
            entries = []
    else:
        matches = db.mesh_search_by_name(query, limit=10)
        if not matches:
            typer.echo(f"No MeSH descriptor matching: {query}", err=True)
            raise typer.Exit(1)
        if len(matches) > 1:
            console.print(f"[bold]Multiple matches for '{query}':[/bold]")
            for m in matches:
                console.print(f"  {m['descriptor_ui']}  {m['descriptor_name']}")
            console.print("\nSpecify descriptor_ui to select one.")
            return
        descriptor_ui = matches[0]["descriptor_ui"]
        descriptor_name = matches[0]["descriptor_name"]
        entries = db.mesh_by_ui(descriptor_ui)

    tree_numbers = [e["tree_number"] for e in entries] if entries else []

    if tree:
        # Show hierarchy
        console.print(f"\n[bold]{descriptor_ui}  {descriptor_name}[/bold]")
        for tn in tree_numbers:
            console.print(f"  tree: {tn}")
            # Parent
            parent = db.mesh_parent(tn)
            if parent:
                console.print(
                    f"    ↑ {parent['descriptor_name']} ({parent['descriptor_ui']})"
                )
            # Children (direct, not all descendants)
            children = db.mesh_descendants(tn)
            direct = [
                c
                for c in children
                if c["tree_number"] != tn
                and c["tree_number"].startswith(tn + ".")
                and c["tree_number"].count(".") == tn.count(".") + 1
            ]
            for c in direct:
                console.print(f"    ↓ {c['descriptor_name']} ({c['descriptor_ui']})")
        console.print()
        return

    # Paper discovery
    _BROAD_THRESHOLD = 50_000
    works, total = db.get_top_works_by_mesh(descriptor_ui, limit=limit)

    console.print(
        f"\n[bold]{descriptor_ui}  {descriptor_name}[/bold]  "
        f"papers={total} (major topic)"
    )

    if total > _BROAD_THRESHOLD:
        console.print(
            f"[yellow]⚠ {total:,} papers — too broad for seed discovery. "
            f"Use --tree to find sub-descriptors.[/yellow]"
        )

    if not works:
        console.print("  No papers found.")
        return

    table = Table(show_edge=False, pad_edge=False, box=None, expand=True)
    table.add_column("#", justify="right", style="dim", width=4, no_wrap=True)
    table.add_column("year", justify="right", width=4, no_wrap=True)
    table.add_column("cited", justify="right", width=6, no_wrap=True)
    table.add_column("work_id", width=13, no_wrap=True)
    table.add_column("title", ratio=1, overflow="ellipsis", no_wrap=True)

    for i, w in enumerate(works):
        year = str(w["pub_year"]) if w.get("pub_year") else "-"
        cited = str(w["cited_by_count"]) if w.get("cited_by_count") is not None else "-"
        table.add_row(
            str(i + 1),
            year,
            cited,
            w["work_id"],
            Text(w.get("title") or "-"),
        )

    console.print(table)
    console.print()


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
    "mesh_lookup": [
        ("descriptor_ui", "text"),
        ("descriptor_name", "text — official MeSH name"),
        ("term", "text — searchable name (synonym, entry term, abbreviation)"),
        (
            "source",
            "text — 'entry_term' (from desc*.xml) | 'historical' (from work_mesh)",
        ),
        ("has_tree", "boolean — true if descriptor has tree_number in current MeSH"),
    ],
    "id_crosswalk": [
        ("work_id", "text"),
        ("pmid", "integer"),
        ("pmc_id", "text — PMC ID (e.g. PMC12749249)"),
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
    mesh_summary: bool = typer.Option(
        False, "--mesh-summary", help="Show MeSH topic distribution of results"
    ),
    alpha: float = typer.Option(0.15, "--alpha", help="APPR teleport probability"),
    epsilon: float = typer.Option(1e-6, "--epsilon", help="APPR precision threshold"),
    min_citations: int = typer.Option(
        0, "--min-citations", help="Minimum cited_by_count filter (0=disabled)"
    ),
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
            min_citations=min_citations,
            include_abstract=(fmt == "detail"),
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if fmt in ("json", "detail"):
        if mesh_summary:
            result["mesh_summary"] = _get_mesh_summary(result)
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_table(result, mesh_summary=mesh_summary)


def _get_mesh_summary(result: dict) -> list[dict]:
    """Compute MeSH topic distribution for expand/bridge results."""
    from quarry.config import settings
    from quarry.store.pg import PGStore

    work_ids = [f"W{p['work_id']}" for p in result.get("papers", [])]
    if not work_ids:
        return []
    db = PGStore(settings.pg_conninfo)
    return db.top_mesh_by_work_ids(work_ids, limit=10)


def _print_table(result: dict, *, mesh_summary: bool = False) -> None:
    """Print expand result as a rich table."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    stats = result["stats"]
    seed_id = result["seed"]["work_id"]
    papers = result["papers"]
    total = len(papers)

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
    table.add_column("venue", width=15, no_wrap=True, overflow="ellipsis")
    table.add_column("title", ratio=1, overflow="ellipsis", no_wrap=True)

    for p in papers:
        year = str(p["year"]) if p["year"] else "-"
        cited = (
            str(p["quality"]["cited_by"])
            if p.get("quality", {}).get("cited_by") is not None
            else "-"
        )
        venue = p.get("host_venue") or "-"
        table.add_row(
            str(p["rank"]),
            p["relation"],
            year,
            cited,
            f"{p['scores']['fused']:.6f}",
            f"W{p['work_id']}",
            venue,
            Text(p["title"] or "-"),
        )

    console.print(table)

    if mesh_summary:
        summary = _get_mesh_summary(result)
        if summary:
            console.print(
                f"\n[bold]MeSH summary[/bold] (major topics, top {len(summary)})"
            )
            max_cnt = summary[0]["cnt"] if summary else 1
            for s in summary:
                bar = "█" * int(20 * s["cnt"] / max_cnt)
                console.print(
                    f"  {s['descriptor_name']:<40} {s['cnt']:>3}/{total}  {bar}"
                )


@app.command()
def shrink(
    seed: str = typer.Argument(
        ..., help="Seed paper: work_id_int, W<id>, DOI, or https://doi.org/..."
    ),
    top_n: int = typer.Option(5, "--top", "-n", help="Number of papers to select"),
    venue: str = typer.Option(
        "NCS+",
        "--venue",
        "-v",
        help="Venue preset (NCS+) or comma-separated list",
    ),
    limit: int = typer.Option(200, "--limit", help="Internal expand limit"),
    no_foundation: bool = typer.Option(
        False,
        "--no-foundation",
        help="Exclude foundation papers from pool (surfaces recent trends)",
    ),
    exclude_ids: str = typer.Option(
        None,
        "--exclude",
        help="Comma-separated work_ids to exclude (e.g. W2336828812,W2766599608)",
    ),
    fmt: str = typer.Option("table", "--format", "-f", help="table|json"),
):
    """Find minimum set of top-venue papers covering an expand landscape.

    Runs expand internally, filters to specified venues, then selects
    papers by greedy weighted citation coverage. Useful for building
    a concise reading list from top journals.

    In centralized fields where one foundational paper dominates coverage,
    use --no-foundation to surface recent follow-up trends instead.

    Examples:
      quarry shrink W2042789810                    # top 5 from NCS+
      quarry shrink W2042789810 --top 10           # more papers
      quarry shrink W2042789810 --no-foundation    # skip foundations
      quarry shrink W2042789810 --venue "Nature,Science,Cell"
    """
    import json

    from quarry.config import settings
    from quarry.core.shrink import NCS_PLUS, run_shrink

    # Parse venue
    if venue.upper() == "NCS+":
        venues = NCS_PLUS
    else:
        venues = {v.strip() for v in venue.split(",")}

    # Parse exclude list
    exclude: set[int] = set()
    if exclude_ids:
        for wid_str in exclude_ids.split(","):
            wid_str = wid_str.strip().upper()
            if wid_str.startswith("W") and wid_str[1:].isdigit():
                exclude.add(int(wid_str[1:]))
            elif wid_str.isdigit():
                exclude.add(int(wid_str))

    try:
        result = run_shrink(
            seed=seed,
            csr_dir=str(settings.csr_dir),
            pg_conninfo=settings.pg_conninfo,
            top_n=top_n,
            venues=venues,
            expand_limit=limit,
            no_foundation=no_foundation,
            exclude=exclude,
        )
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1) from None

    if fmt == "json":
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        _print_shrink_table(result)


def _print_shrink_table(result: dict) -> None:
    """Print shrink result as a rich table."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    console = Console()
    stats = result["stats"]
    seed_id = result["seed"]["work_id"]
    selected = result["selected"]
    coverage_pct = stats["coverage"] * 100

    console.print(
        f"Shrink: W{seed_id} → {len(selected)} papers "
        f"(coverage: {coverage_pct:.0f}%, {stats['covered']}/{stats['total_candidates']})  "
        f"[dim]pool={stats['venue_pool_size']} expand={stats['expand_elapsed_s']}s[/dim]"
    )

    if stats.get("centralization_warning"):
        console.print(f"[yellow]⚠ {stats['centralization_warning']}[/yellow]")

    if stats["venue_pool_size"] == 0:
        console.print("[yellow]⚠ No papers found in specified venues[/yellow]")
        return

    table = Table(show_edge=False, pad_edge=False, box=None, expand=True)
    table.add_column("#", justify="right", style="dim", width=3, no_wrap=True)
    table.add_column("ex#", justify="right", style="dim", width=4, no_wrap=True)
    table.add_column("year", justify="right", width=4, no_wrap=True)
    table.add_column("cited", justify="right", width=6, no_wrap=True)
    table.add_column("+cov", justify="right", width=5, no_wrap=True)
    table.add_column("cum%", justify="right", width=5, no_wrap=True)
    table.add_column("venue", width=15, no_wrap=True, overflow="ellipsis")
    table.add_column("title", ratio=1, overflow="ellipsis", no_wrap=True)

    for s in selected:
        year = str(s["year"]) if s.get("year") else "-"
        cited = (
            str(s["quality"]["cited_by"])
            if s.get("quality", {}).get("cited_by") is not None
            else "-"
        )
        cum_pct = f"{s['cumulative_coverage'] * 100:.0f}%"
        table.add_row(
            str(s["rank_in_shrink"]),
            str(s["rank_in_expand"]),
            year,
            cited,
            f"+{s['marginal_count']}",
            cum_pct,
            s.get("host_venue") or "-",
            Text(s.get("title") or "-"),
        )

    console.print(table)

    uncovered = stats["uncovered"]
    if uncovered > 0:
        console.print(
            f"\n  [dim]Uncovered: {uncovered} papers ({100 - coverage_pct:.0f}%)[/dim]"
        )


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


def _empty_bridge_reason(key: str, result: dict) -> str | None:
    """Return a human-readable reason why a bridge type is empty, or None."""
    k = len(result.get("seeds", []))
    sp = result.get("stats", {}).get("shortest_path_length")

    if key == "steiner_bridges":
        if k < 3:
            return "requires 3+ seeds"
        pairwise = result.get("stats", {}).get("pairwise_sp", [])
        unreachable = sum(1 for _, _, s in pairwise if s is None)
        all_sp1 = all(s == 1 for _, _, s in pairwise if s is not None)
        if unreachable:
            return (
                f"no steiner tree ({unreachable}/{len(pairwise)} pairs unreachable)"
                " → try closer seeds or use pairwise bridge"
            )
        if all_sp1:
            return (
                "no intermediate nodes (all pairs sp=1)"
                " → seeds directly connected, use expand instead"
            )
        return "no intermediate nodes found"
    if key == "path_bridges":
        if sp == 1:
            return "seeds directly connected (sp=1), no stepping stones needed"
        return "no path found within max depth → try increasing --max-path-depth"
    return None


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
    # Only show overlap when common_refs/common_citers were computed
    has_common = bool(result.get("common_refs")) or bool(result.get("common_citers"))
    overlap_info = (
        f"refs overlap={stats['overlap_refs']} citers overlap={stats['overlap_citers']}"
        if has_common or (stats["overlap_refs"] > 0 or stats["overlap_citers"] > 0)
        else ""
    )
    # Pairwise sp matrix
    pairwise_sp = stats.get("pairwise_sp", [])
    if pairwise_sp:
        sp_parts = []
        for i, j, sp in pairwise_sp:
            si = seed_strs[i] if i < len(seed_strs) else f"#{i}"
            sj = seed_strs[j] if j < len(seed_strs) else f"#{j}"
            sp_parts.append(f"{si}↔{sj}={sp if sp is not None else '∞'}")
        sp_info = " sp: " + "  ".join(sp_parts)
    elif sp_len is not None:
        sp_info = f" sp={sp_len}"
    else:
        sp_info = ""
    console.print(
        f"Bridge: {' ↔ '.join(seed_strs)}  "
        f"({stats['elapsed_s']}s)  "
        f"[dim]{overlap_info}{sp_info}[/dim]"
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

    # Map result keys to param type names for filtering
    _KEY_TO_TYPE = {
        "common_refs": "common_refs",
        "common_citers": "common_citers",
        "coupling_bridges": "coupling",
        "cocitation_bridges": "cocitation",
        "path_bridges": "path",
        "ppr_bridges": "ppr",
        "steiner_bridges": "steiner",
    }
    requested_types = set(result.get("params", {}).get("types", []))

    for key, label in _SECTIONS:
        entries = result.get(key, [])
        type_name = _KEY_TO_TYPE.get(key, "")
        was_requested = type_name in requested_types
        if not entries:
            if was_requested:
                reason = _empty_bridge_reason(key, result)
                if reason:
                    console.print(f"\n[dim]{label}  — {reason}[/dim]")
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
