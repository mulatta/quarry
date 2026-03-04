"""papeline CLI — Academic paper data pipeline (OpenAlex)."""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich import print as rprint

import papeline

app = typer.Typer(
    name="papeline",
    help="Academic paper data pipeline (OpenAlex)",
    no_args_is_help=True,
)

# ============================================================
# Config loading
# ============================================================

CONFIG_SEARCH = ("papeline.toml", ".papeline.toml")


@dataclass
class Config:
    """Parsed TOML configuration with defaults."""

    output: str | None = None
    zstd_level: int | None = None
    concurrency: int | None = None
    max_retries: int | None = None
    read_timeout: int | None = None
    outer_retries: int | None = None
    retry_delay: int | None = None
    filter: FilterCfg = field(default_factory=lambda: FilterCfg())
    hive: HiveCfg = field(default_factory=lambda: HiveCfg())
    upload: UploadCfg = field(default_factory=lambda: UploadCfg())


@dataclass
class FilterCfg:
    domains: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    work_types: list[str] = field(default_factory=list)
    require_abstract: bool = False


@dataclass
class HiveCfg:
    enable: bool | None = None
    clean_raw: bool | None = None
    zstd_level: int | None = None
    row_group_size: int | None = None
    num_shards: int | None = None
    threads: int | None = None
    memory_limit: str | None = None


@dataclass
class UploadCfg:
    bucket: str | None = None
    endpoint: str | None = None
    region: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    prefix: str | None = None


def load_config(path: str | None) -> Config:
    """Load TOML config from explicit path or auto-discovered location."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            _abort(f"Config file not found: {p}")
        return _parse_config(p)

    for name in CONFIG_SEARCH:
        p = Path(name)
        if p.exists():
            return _parse_config(p)

    return Config()


def _parse_config(p: Path) -> Config:
    with open(p, "rb") as f:
        raw = tomllib.load(f)

    cfg = Config(
        output=raw.get("output"),
        zstd_level=raw.get("zstd_level"),
        concurrency=raw.get("concurrency"),
        max_retries=raw.get("max_retries"),
        read_timeout=raw.get("read_timeout"),
        outer_retries=raw.get("outer_retries"),
        retry_delay=raw.get("retry_delay"),
    )

    if "filter" in raw:
        f = raw["filter"]
        cfg.filter = FilterCfg(
            domains=f.get("domains", []),
            topics=f.get("topics", []),
            languages=f.get("languages", []),
            work_types=f.get("work_types", []),
            require_abstract=f.get("require_abstract", False),
        )

    if "hive" in raw:
        h = raw["hive"]
        cfg.hive = HiveCfg(
            enable=h.get("enable"),
            clean_raw=h.get("clean_raw"),
            zstd_level=h.get("zstd_level"),
            row_group_size=h.get("row_group_size"),
            num_shards=h.get("num_shards"),
            threads=h.get("threads"),
            memory_limit=h.get("memory_limit"),
        )

    if "upload" in raw:
        u = raw["upload"]
        cfg.upload = UploadCfg(
            bucket=u.get("bucket"),
            endpoint=u.get("endpoint"),
            region=u.get("region"),
            access_key=u.get("access_key"),
            secret_key=u.get("secret_key"),
            prefix=u.get("prefix"),
        )

    return cfg


def _resolve_upload(cfg: UploadCfg) -> dict[str, str]:
    """Resolve upload config with env var fallback. Raises on missing required fields."""
    bucket = cfg.bucket
    if not bucket:
        _abort("Upload requires [upload].bucket in config")
    endpoint = cfg.endpoint
    if not endpoint:
        _abort("Upload requires [upload].endpoint in config")

    access_key = (
        cfg.access_key
        or os.environ.get("R2_ACCESS_KEY_ID")
        or os.environ.get("AWS_ACCESS_KEY_ID")
    )
    secret_key = (
        cfg.secret_key
        or os.environ.get("R2_SECRET_ACCESS_KEY")
        or os.environ.get("AWS_SECRET_ACCESS_KEY")
    )
    if not access_key or not secret_key:
        _abort(
            "Upload requires access_key/secret_key "
            "(config or R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY env vars)"
        )

    return {
        "bucket": bucket,
        "endpoint": endpoint,
        "region": cfg.region or "auto",
        "access_key": access_key,
        "secret_key": secret_key,
        "prefix": cfg.prefix or "",
    }


def _abort(msg: str) -> None:
    rprint(f"[red]Error:[/red] {msg}", file=sys.stderr)
    raise typer.Exit(code=2)


def _format_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if n != int(n) else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


# ============================================================
# Commands
# ============================================================


@app.command()
def run(
    output_dir: str = typer.Option("./outputs", "-o", "--output-dir", help="Output directory"),
    config: str | None = typer.Option(None, "-c", "--config", help="Config file (TOML)"),
    # Download
    concurrency: int | None = typer.Option(None, help="Concurrent downloads [default: 8]"),
    read_timeout: int | None = typer.Option(None, help="HTTP read timeout in seconds [default: 30]"),
    max_retries: int | None = typer.Option(None, help="Per-shard retry attempts [default: 3]"),
    outer_retries: int | None = typer.Option(None, help="Outer retry passes [default: 3]"),
    retry_delay: int | None = typer.Option(None, help="Delay between outer retries in seconds [default: 30]"),
    # Output
    zstd_level: int | None = typer.Option(None, help="ZSTD compression level [default: 3]"),
    # Filter
    domain: list[str] | None = typer.Option(None, help="Filter by OA domain (repeatable)"),
    topic: list[str] | None = typer.Option(None, help="Filter by OA topic ID (repeatable)"),
    language: list[str] | None = typer.Option(None, help="Filter by language code (repeatable)"),
    work_type: list[str] | None = typer.Option(None, help="Filter by work type (repeatable)"),
    require_abstract: bool = typer.Option(False, help="Only keep works with an abstract"),
    # Run modes
    force: bool = typer.Option(False, help="Force re-download all shards"),
    since: str | None = typer.Option(None, help="Only process shards with updated_date >= DATE"),
    max_shards: int | None = typer.Option(None, help="Limit total shards (for testing)"),
    dry_run: bool = typer.Option(False, help="Show what would be done"),
    hive: bool = typer.Option(False, help="Auto-run hive after download"),
    clean_raw: bool = typer.Option(False, help="Remove raw files after hive"),
) -> None:
    """Download and process OpenAlex works snapshot to Parquet."""
    import time

    cfg = load_config(config)
    root = cfg.output or output_dir

    # Resolve parameters: CLI > config > default
    r_concurrency = concurrency or cfg.concurrency or 8
    r_read_timeout = read_timeout or cfg.read_timeout or 30
    r_max_retries = max_retries or cfg.max_retries or 3
    r_outer_retries = outer_retries or cfg.outer_retries or 3
    r_retry_delay = retry_delay or cfg.retry_delay or 30
    r_zstd_level = zstd_level or cfg.zstd_level or 3

    # Build filter: CLI > config
    domains = domain or cfg.filter.domains
    topics = topic or cfg.filter.topics
    languages = language or cfg.filter.languages
    work_types = work_type or cfg.filter.work_types
    r_require_abstract = require_abstract or cfg.filter.require_abstract

    filt = papeline.Filter(
        domains=domains,
        topics=topics,
        languages=languages,
        work_types=work_types,
        require_abstract=r_require_abstract,
    )

    raw_dir = str(Path(root) / "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Fetch manifest
    pool = papeline.HttpPool(
        concurrency=r_concurrency,
        read_timeout_secs=r_read_timeout,
        max_retries=r_max_retries,
    )
    rprint("[bold]Fetching manifest...[/bold]")
    shards = papeline.fetch_manifest(pool, "works")
    rprint(f"  {len(shards)} shards, ~{sum(s.record_count for s in shards):,} records")

    # Filter by --since
    if since:
        shards = [s for s in shards if s.updated_date and s.updated_date >= since]
        rprint(f"  After --since {since}: {len(shards)} shards")

    # Filter completed (unless --force)
    if not force:
        shards = [s for s in shards if not papeline.is_shard_complete(raw_dir, s.shard_idx)]
        rprint(f"  Pending: {len(shards)} shards")

    if max_shards:
        shards = shards[:max_shards]

    if not shards:
        rprint("[green]All shards up to date. Nothing to do.[/green]")
        raise typer.Exit()

    if dry_run:
        rprint(f"\n[yellow]Dry run:[/yellow] would process {len(shards)} shards")
        for s in shards[:10]:
            rprint(f"  shard {s.shard_idx}: {s.record_count:,} records (date={s.updated_date})")
        if len(shards) > 10:
            rprint(f"  ... and {len(shards) - 10} more")
        raise typer.Exit()

    # Process with outer retry loop
    pending = shards
    total_completed = 0
    total_rows = 0

    for pass_num in range(r_outer_retries + 1):
        if not pending:
            break

        if pass_num > 0:
            rprint(f"\n[yellow]Retry pass {pass_num}/{r_outer_retries} ({len(pending)} shards)[/yellow]")
            time.sleep(r_retry_delay)

        result = papeline.run(
            shards=pending,
            output_dir=raw_dir,
            pool=pool,
            filter=filt,
            zstd_level=r_zstd_level,
            concurrency=r_concurrency,
        )
        total_completed += result.completed
        total_rows += result.total_rows

        if result.failed == 0:
            pending = []
            break

        pending = [pending[i] for i in result.failed_indices]

    # Summary
    failed = len(pending)
    rprint(f"\n[bold]Done:[/bold] {total_completed} completed, {failed} failed, {total_rows:,} rows")

    if failed:
        rprint(f"[red]Failed shards:[/red] {[s.shard_idx for s in pending]}")

    # Auto-hive
    do_hive = hive or cfg.hive.enable
    if do_hive and failed == 0:
        rprint("\n[bold]Running hive partitioning...[/bold]")
        _run_hive_with_cfg(root, cfg, clean_raw=clean_raw or cfg.hive.clean_raw)

    if failed:
        raise typer.Exit(code=1)


@app.command()
def status(
    output_dir: str = typer.Option("./outputs", "-o", "--output-dir", help="Output directory"),
    config: str | None = typer.Option(None, "-c", "--config", help="Config file (TOML)"),
) -> None:
    """Show remote manifest info and local progress."""
    cfg = load_config(config)
    root = cfg.output or output_dir
    raw_dir = str(Path(root) / "raw")

    pool = papeline.HttpPool(concurrency=1, read_timeout_secs=30, max_retries=3)

    try:
        shards = papeline.fetch_manifest(pool, "works")
    except RuntimeError as e:
        _abort(f"Failed to fetch manifest: {e}")

    total_records = sum(s.record_count for s in shards)
    total_bytes = sum(s.content_length or 0 for s in shards)
    dates = sorted({s.updated_date for s in shards if s.updated_date})
    date_range = f"{dates[0]} .. {dates[-1]}" if dates else "unknown"

    rprint("[bold]Remote (OpenAlex manifest):[/bold]")
    rprint(f"  Shards:     {len(shards)}")
    rprint(f"  Records:    ~{total_records:,}")
    rprint(f"  Size:       ~{total_bytes // (1024**3)} GiB (compressed)")
    rprint(f"  Partitions: updated_date={date_range}")
    rprint()

    completed = sum(1 for s in shards if papeline.is_shard_complete(raw_dir, s.shard_idx))
    rprint(f"[bold]Local ({root}):[/bold]")
    rprint(f"  Completed:  {completed}/{len(shards)} shards")

    # Check hive dir
    hive_dir = Path(root) / "hive"
    if hive_dir.exists():
        hive_tables = [d.name for d in hive_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        rprint(f"  Hive:       {len(hive_tables)} tables")
    else:
        rprint("  Hive:       not built")


@app.command()
def hive(
    output_dir: str = typer.Option("./outputs", "-o", "--output-dir", help="Output directory"),
    config: str | None = typer.Option(None, "-c", "--config", help="Config file (TOML)"),
    force: bool = typer.Option(False, help="Force rebuild"),
    dry_run: bool = typer.Option(False, help="Show what would be done"),
    zstd_level: int | None = typer.Option(None, help="ZSTD compression level"),
    row_group_size: int | None = typer.Option(None, help="Parquet row group size [default: 500000]"),
    num_shards: int | None = typer.Option(None, help="Shards per partition [default: 4]"),
    threads: int | None = typer.Option(None, help="Worker threads [default: 3/4 CPUs]"),
    memory_limit: str | None = typer.Option(None, help='Memory limit (e.g. "32GB") [default: 65% RAM]'),
) -> None:
    """Repartition raw shards into year-partitioned Hive parquet."""
    cfg = load_config(config)
    root = cfg.output or output_dir

    try:
        papeline.run_hive(
            output_dir=root,
            zstd_level=zstd_level or cfg.hive.zstd_level or cfg.zstd_level or 8,
            row_group_size=row_group_size or cfg.hive.row_group_size or 500_000,
            num_shards=num_shards or cfg.hive.num_shards or 4,
            threads=threads or cfg.hive.threads or 0,
            memory_limit=memory_limit or cfg.hive.memory_limit or "32GB",
            force=force,
            dry_run=dry_run,
        )
    except RuntimeError as e:
        _abort(str(e))

    rprint("[green]Hive partitioning complete.[/green]")


@app.command()
def push(
    output_dir: str = typer.Option("./outputs", "-o", "--output-dir", help="Output directory"),
    config: str | None = typer.Option(None, "-c", "--config", help="Config file (TOML)"),
    raw_only: bool = typer.Option(False, help="Push raw/ only"),
    hive_only: bool = typer.Option(False, help="Push hive/ only"),
    dry_run: bool = typer.Option(False, help="Show what would be pushed"),
    concurrency: int = typer.Option(8, help="Max concurrent uploads"),
) -> None:
    """Sync local files to S3-compatible storage."""
    if raw_only and hive_only:
        _abort("Cannot use --raw-only and --hive-only together")

    cfg = load_config(config)
    root = cfg.output or output_dir
    upload = _resolve_upload(cfg.upload)

    try:
        summary = papeline.push(
            output_dir=root,
            raw=not hive_only,
            hive=not raw_only,
            dry_run=dry_run,
            concurrency=concurrency,
            **upload,
        )
    except RuntimeError as e:
        _abort(str(e))

    rprint(
        f"Push: {summary.files_transferred} transferred "
        f"({_format_bytes(summary.bytes_transferred)}), "
        f"{summary.files_skipped} skipped"
    )


@app.command()
def pull(
    output_dir: str = typer.Option("./outputs", "-o", "--output-dir", help="Output directory"),
    config: str | None = typer.Option(None, "-c", "--config", help="Config file (TOML)"),
    include_hive: bool = typer.Option(False, help="Also pull hive/"),
    dry_run: bool = typer.Option(False, help="Show what would be pulled"),
    concurrency: int = typer.Option(8, help="Max concurrent downloads"),
) -> None:
    """Sync S3-compatible storage to local."""
    cfg = load_config(config)
    root = cfg.output or output_dir
    upload = _resolve_upload(cfg.upload)

    try:
        summary = papeline.pull(
            output_dir=root,
            raw=True,
            hive=include_hive,
            dry_run=dry_run,
            concurrency=concurrency,
            **upload,
        )
    except RuntimeError as e:
        _abort(str(e))

    rprint(
        f"Pull: {summary.files_transferred} transferred "
        f"({_format_bytes(summary.bytes_transferred)}), "
        f"{summary.files_skipped} skipped"
    )


@app.command()
def clean(
    output_dir: str = typer.Option("./outputs", "-o", "--output-dir", help="Output directory"),
) -> None:
    """Remove temporary (.tmp) files from output directory."""
    raw_dir = Path(output_dir) / "raw"
    if not raw_dir.exists():
        rprint(f"Raw directory does not exist: {raw_dir}")
        raise typer.Exit()

    total = 0
    for table in papeline.tables():
        table_dir = raw_dir / table
        if table_dir.exists():
            for f in table_dir.glob("*.tmp"):
                f.unlink()
                total += 1

    if total:
        rprint(f"Removed {total} .tmp files")
    else:
        rprint("No .tmp files found")


# ============================================================
# Helpers
# ============================================================


def _run_hive_with_cfg(
    root: str,
    cfg: Config,
    *,
    clean_raw: bool | None = None,
) -> None:
    """Run hive with config defaults (used by auto-hive in `run`)."""
    try:
        papeline.run_hive(
            output_dir=root,
            zstd_level=cfg.hive.zstd_level or cfg.zstd_level or 8,
            row_group_size=cfg.hive.row_group_size or 500_000,
            num_shards=cfg.hive.num_shards or 4,
            threads=cfg.hive.threads or 0,
            memory_limit=cfg.hive.memory_limit or "32GB",
        )
    except RuntimeError as e:
        rprint(f"[red]Hive error:[/red] {e}", file=sys.stderr)
        return

    if clean_raw:
        raw_dir = Path(root) / "raw"
        if raw_dir.exists():
            import shutil

            shutil.rmtree(raw_dir)
            rprint("Cleaned raw directory")
