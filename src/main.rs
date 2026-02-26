//! papeline -- Academic paper data pipeline (OpenAlex).

mod accumulator;
mod api;
mod config;
mod error;
mod hive;
mod id;
mod manifest;
mod oa;
mod progress;
mod provider;
mod retry;
mod schema;
mod sink;
mod stream;
mod transform;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use rustc_hash::FxHashSet;

use config::{ResolvedConfig, State, SyncLogEntry, build_filter, load_config};
use manifest::{ManifestDiff, ManifestSnapshot, compute_manifest_diff, delete_shard_files};
use oa::{OAProvider, OAShard, TABLES, is_shard_complete};
use progress::{IndicatifMakeWriter, ProgressContext, SharedProgress};
use provider::{RunContext, run_provider};
use stream::{HttpConfig as StreamHttpConfig, init_http, set_http_config};

// ============================================================
// CLI definition
// ============================================================

#[derive(Parser)]
#[command(
    name = "papeline",
    about = "Academic paper data pipeline (OpenAlex)",
    version
)]
struct Cli {
    /// Output directory (raw/ and hive/ are created inside)
    #[arg(short, long, global = true, default_value = "./outputs")]
    output_dir: PathBuf,

    /// Config file (TOML). CLI options override config values
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Enable debug logging
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download and process OpenAlex works snapshot to Parquet
    Run(RunArgs),
    /// Show remote manifest info and local progress
    Status,
    /// Remove temporary (.tmp) files from output directory
    Clean,
    /// Repartition raw shards into year-partitioned Hive parquet
    Hive(HiveArgs),
}

#[derive(clap::Args)]
struct HiveArgs {
    /// Force rebuild even if source manifest unchanged
    #[arg(long)]
    force: bool,
    /// Show what would be done without executing
    #[arg(long)]
    dry_run: bool,

    // -- Compression / output --
    /// ZSTD compression level for hive parquet [default: inherit top-level or 8]
    #[arg(long)]
    zstd_level: Option<i32>,
    /// Parquet row group size [default: 500000]
    #[arg(long)]
    row_group_size: Option<usize>,
    /// Shards per year partition (deterministic via work_id %% N) [default: 4]
    #[arg(long)]
    num_shards: Option<usize>,

    // -- Resource limits --
    /// Number of DataFusion worker threads [default: 3/4 of CPUs]
    #[arg(long)]
    threads: Option<usize>,
    /// Memory limit for DataFusion (e.g. "32GB", "512MB") [default: 65% of system RAM]
    #[arg(long)]
    memory_limit: Option<String>,
}

#[derive(clap::Args)]
struct RunArgs {
    // -- Download options --
    /// Concurrent downloads [default: 8]
    #[arg(long)]
    concurrency: Option<usize>,

    /// HTTP read timeout for stall detection in seconds [default: 30]
    #[arg(long)]
    read_timeout: Option<u64>,

    /// Per-shard retry attempts [default: 3]
    #[arg(long)]
    max_retries: Option<u32>,

    /// Outer retry passes for all failed shards [default: 3]
    #[arg(long)]
    outer_retries: Option<u32>,

    /// Delay between outer retry passes in seconds [default: 30]
    #[arg(long)]
    retry_delay: Option<u64>,

    // -- Output options --
    /// ZSTD compression level for Parquet [default: 3]
    #[arg(long)]
    zstd_level: Option<i32>,

    // -- Filter options --
    /// Filter works by OA domain (repeatable)
    #[arg(long)]
    domain: Vec<String>,

    /// Filter works by OA topic ID, e.g. T11162 (repeatable)
    #[arg(long)]
    topic: Vec<String>,

    // -- Run modes --
    /// Force re-download of all shards (ignore completed)
    #[arg(long)]
    force: bool,

    /// Only process shards with updated_date >= DATE (YYYY-MM-DD)
    #[arg(long)]
    since: Option<String>,

    /// Limit total shards to process (for testing)
    #[arg(long)]
    max_shards: Option<usize>,

    /// Show what would be processed without executing
    #[arg(long)]
    dry_run: bool,
}

// ============================================================
// Entry point
// ============================================================

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Create MultiProgress early so tracing output is routed through it,
    // preventing log lines from corrupting progress bar rendering.
    let multi = indicatif::MultiProgress::new();

    let level = if cli.verbose { "debug" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level)),
        )
        .with_target(false)
        .with_writer(IndicatifMakeWriter::new(multi.clone()))
        .init();

    match &cli.command {
        Commands::Run(args) => cmd_run(&cli.output_dir, cli.config.as_deref(), args, multi),
        Commands::Status => cmd_status(&cli.output_dir),
        Commands::Clean => cmd_clean(&cli.output_dir),
        Commands::Hive(args) => cmd_hive(&cli.output_dir, cli.config.as_deref(), args, &multi),
    }
}

// ============================================================
// `papeline run`
// ============================================================

fn cmd_run(
    output_dir: &Path,
    config_path: Option<&Path>,
    args: &RunArgs,
    multi: indicatif::MultiProgress,
) -> ExitCode {
    let plan = match prepare_run(output_dir, config_path, args) {
        Ok(plan) => plan,
        Err(code) => return code,
    };

    let Some(result) = execute_run(&plan, multi) else {
        save_snapshot(
            &plan.all_shards,
            &FxHashSet::default(),
            &plan.resolved.output_dir,
        );
        return ExitCode::SUCCESS;
    };

    finalize_run(plan, &result)
}

// ---- Phase types ----

/// Everything computed during preparation, needed for execution and finalization.
struct RunPlan {
    resolved: ResolvedConfig,
    state: State,
    all_shards: Vec<OAShard>,
    shards_to_process: Vec<OAShard>,
    removed: Vec<manifest::RemovedShard>,
}

/// Outcome of pipeline execution (only produced when shards were processed).
struct RunResult {
    total_completed: usize,
    total_rows: usize,
    /// Shards that failed after all retry passes.
    pending: Vec<OAShard>,
    started_at: String,
    finished_at: String,
}

// ---- Phase 1: Prepare ----

/// Load config, fetch manifest, compute diff, filter shards.
///
/// Returns `Err(ExitCode)` on fatal errors or after printing dry-run output.
fn prepare_run(
    output_dir: &Path,
    config_path: Option<&Path>,
    args: &RunArgs,
) -> Result<RunPlan, ExitCode> {
    // 1. Load config (file + CLI overrides)
    let file_cfg = load_config(config_path).map_err(|e| {
        tracing::error!("Config error: {e}");
        ExitCode::from(2)
    })?;
    let resolved = resolve_config(output_dir, &file_cfg, args);

    // 2. Initialize HTTP client
    init_http(resolved.concurrency).map_err(|e| {
        tracing::error!("Failed to initialize HTTP: {e}");
        ExitCode::from(2)
    })?;

    // 3. Configure HTTP timeouts
    set_http_config(StreamHttpConfig {
        read_timeout: std::time::Duration::from_secs(resolved.read_timeout),
        max_retries: resolved.max_retries,
    });

    // 4. Fetch manifest
    let all_shards = api::fetch_manifest("works").map_err(|e| {
        tracing::error!("Failed to fetch manifest: {e}");
        ExitCode::from(2)
    })?;

    // 5. Create output dir
    std::fs::create_dir_all(&resolved.output_dir).map_err(|e| {
        tracing::error!("Cannot create output dir: {e}");
        ExitCode::from(2)
    })?;

    // 6. Validate existing parquet schemas (skip with --force)
    if !resolved.force {
        let mismatches = sink::validate_existing_schemas(&resolved.output_dir, TABLES);
        if !mismatches.is_empty() {
            eprintln!("Schema mismatch in existing parquet files:");
            for m in &mismatches {
                let added: Vec<_> = m
                    .expected_fields
                    .iter()
                    .filter(|f| !m.actual_fields.contains(f))
                    .collect();
                let removed: Vec<_> = m
                    .actual_fields
                    .iter()
                    .filter(|f| !m.expected_fields.contains(f))
                    .collect();
                eprintln!(
                    "  {}: expected {} cols, found {} (file: {})",
                    m.table,
                    m.expected_fields.len(),
                    m.actual_fields.len(),
                    m.file.display(),
                );
                if !added.is_empty() {
                    eprintln!(
                        "    new columns:     {}",
                        added
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                if !removed.is_empty() {
                    eprintln!(
                        "    removed columns: {}",
                        removed
                            .iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }
            eprintln!();
            eprintln!("Run with --force to re-download all shards with the current schema.");
            return Err(ExitCode::from(2));
        }
    }

    // 7. Load state + manifest snapshot
    let state = State::load(&resolved.output_dir);
    let old_snapshot = ManifestSnapshot::load(&resolved.output_dir);
    warn_on_filter_change(&state, &resolved);

    // 8. Compute manifest diff (stabilize indices + diff in one step)
    let manifest::ManifestDiffResult {
        shards: all_shards,
        diff,
    } = if resolved.force {
        manifest::ManifestDiffResult {
            shards: all_shards,
            diff: ManifestDiff {
                changed: Vec::new(),
                removed: Vec::new(),
                unchanged_ok: 0,
                unchanged_missing: Vec::new(),
            },
        }
    } else {
        compute_manifest_diff(all_shards, old_snapshot.as_ref(), &resolved.output_dir)
    };

    // Build processing set
    let mut shards_to_process: Vec<OAShard> = if resolved.force {
        all_shards.clone()
    } else {
        let mut set = diff.changed;
        set.extend(diff.unchanged_missing);
        set
    };

    let n_changed = shards_to_process.len();
    let n_unchanged = diff.unchanged_ok;
    let n_removed = diff.removed.len();

    // --since: additional date filter on top of diff results
    if let Some(since) = &resolved.since {
        shards_to_process.retain(|s| {
            s.updated_date
                .as_ref()
                .is_some_and(|d| d.as_str() >= since.as_str())
        });
    }

    if let Some(max) = resolved.max_shards {
        shards_to_process.truncate(max);
    }

    tracing::info!(
        "Manifest diff: {} to process, {} unchanged, {} removed",
        n_changed,
        n_unchanged,
        n_removed
    );

    // Dry-run output (before any side effects)
    if resolved.dry_run {
        print_dry_run(
            &resolved,
            &all_shards,
            &shards_to_process,
            n_changed,
            n_unchanged,
            &diff.removed,
        );
        return Err(ExitCode::SUCCESS);
    }

    Ok(RunPlan {
        resolved,
        state,
        all_shards,
        shards_to_process,
        removed: diff.removed,
    })
}

// ---- Phase 2: Execute ----

/// Delete removed shard files and run the processing pipeline with retries.
///
/// Returns `None` when there are no shards to process.
fn execute_run(plan: &RunPlan, multi: indicatif::MultiProgress) -> Option<RunResult> {
    // Delete files for removed shards (always, even with --since)
    if !plan.removed.is_empty() {
        delete_shard_files(&plan.resolved.output_dir, &plan.removed);
    }

    if plan.shards_to_process.is_empty() {
        println!("All shards up to date. Nothing to do.");
        return None;
    }

    let started_at = chrono::Utc::now().to_rfc3339();
    let progress: SharedProgress = Arc::new(ProgressContext::with_multi(multi));

    let ctx = RunContext {
        output_dir: plan.resolved.output_dir.clone(),
        zstd_level: plan.resolved.zstd_level,
        concurrency: plan.resolved.concurrency,
    };

    let provider = OAProvider {
        filter: plan.resolved.filter.clone(),
    };

    // Outer retry loop
    let mut pending = plan.shards_to_process.clone();
    let mut total_completed: usize = 0;
    let mut total_rows: usize = 0;
    let outer_retries = plan.resolved.outer_retries;

    for pass in 0..=outer_retries {
        if pending.is_empty() {
            break;
        }
        if pass > 0 {
            let delay = plan.resolved.retry_delay * (pass as u64);
            tracing::info!("Outer retry pass {pass}/{outer_retries}, waiting {delay}s...");
            std::thread::sleep(std::time::Duration::from_secs(delay));
        }

        tracing::info!(
            "Processing {} shards (pass {}/{})",
            pending.len(),
            pass,
            outer_retries
        );

        let summary = run_provider(&provider, &pending, &ctx, &progress);
        total_completed += summary.completed;
        total_rows += summary.total_rows;

        tracing::info!(
            "Pass {pass}: {} completed, {} failed, {} rows, {:.1}s",
            summary.completed,
            summary.failed,
            summary.total_rows,
            summary.elapsed.as_secs_f64()
        );

        if summary.failed_indices.is_empty() {
            pending = Vec::new();
            break;
        }

        let failed_shards: Vec<OAShard> = summary
            .failed_indices
            .iter()
            .map(|&i| pending[i].clone())
            .collect();
        pending = failed_shards;
    }

    Some(RunResult {
        total_completed,
        total_rows,
        pending,
        started_at,
        finished_at: chrono::Utc::now().to_rfc3339(),
    })
}

// ---- Phase 3: Finalize ----

/// Persist state, manifest snapshot, sync log, and print summary.
fn finalize_run(plan: RunPlan, result: &RunResult) -> ExitCode {
    let failed_count = result.pending.len();
    let removed_indices: Vec<usize> = plan.removed.iter().map(|r| r.shard_idx).collect();

    // Update state
    let (completed_indices, updated_dates) =
        compute_completed(&plan.shards_to_process, &result.pending);
    let new_state = build_new_state(
        plan.state,
        &completed_indices,
        &updated_dates,
        failed_count,
        plan.all_shards.len(),
        &plan.resolved.filter,
        (&result.started_at, &result.finished_at),
    );
    if let Err(e) = new_state.save(&plan.resolved.output_dir) {
        tracing::warn!("Failed to save state: {e}");
    }

    // Save manifest snapshot (excluding failed shards)
    let failed_urls: FxHashSet<&str> = result.pending.iter().map(|s| s.url.as_str()).collect();
    save_snapshot(&plan.all_shards, &failed_urls, &plan.resolved.output_dir);

    // Summary
    println!(
        "Done: {} completed, {} failed, {} removed, {} rows",
        result.total_completed,
        failed_count,
        removed_indices.len(),
        result.total_rows
    );

    // Sync log
    let log_entry = SyncLogEntry {
        timestamp: chrono::Utc::now().to_rfc3339(),
        shards_processed: completed_indices,
        shards_removed: removed_indices,
        updated_dates,
        rows_written: result.total_rows,
        failed: failed_count,
    };
    if let Err(e) = SyncLogEntry::append(&plan.resolved.output_dir, &log_entry) {
        tracing::warn!("Failed to write sync log: {e}");
    }

    if !result.pending.is_empty() {
        let failed_ids: Vec<usize> = result.pending.iter().map(|s| s.shard_idx).collect();
        eprintln!("Failed shards: {failed_ids:?}");
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

// ============================================================
// cmd_run helpers
// ============================================================

fn resolve_config(
    output_dir: &Path,
    file_cfg: &config::FileConfig,
    args: &RunArgs,
) -> ResolvedConfig {
    let root = file_cfg
        .output
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| output_dir.to_path_buf());
    let output_dir = root.join("raw");

    let filter = build_filter(&args.domain, &args.topic, &file_cfg.run.filter);

    ResolvedConfig {
        output_dir,
        zstd_level: args
            .zstd_level
            .or(file_cfg.run.zstd_level)
            .or(file_cfg.zstd_level)
            .unwrap_or(3),
        concurrency: args.concurrency.or(file_cfg.run.concurrency).unwrap_or(8),
        max_retries: args.max_retries.or(file_cfg.run.max_retries).unwrap_or(3),
        read_timeout: args
            .read_timeout
            .or(file_cfg.run.read_timeout)
            .unwrap_or(30),
        outer_retries: args
            .outer_retries
            .or(file_cfg.run.outer_retries)
            .unwrap_or(3),
        retry_delay: args.retry_delay.or(file_cfg.run.retry_delay).unwrap_or(30),
        filter,
        force: args.force,
        since: args.since.clone(),
        max_shards: args.max_shards,
        dry_run: args.dry_run,
    }
}

fn warn_on_filter_change(state: &State, resolved: &ResolvedConfig) {
    if state.completed_shards.is_empty() || resolved.filter.is_empty() {
        return;
    }
    let state_domains: std::collections::BTreeSet<&str> =
        state.filter.domains.iter().map(|s| s.as_str()).collect();
    let state_topics: std::collections::BTreeSet<&str> =
        state.filter.topics.iter().map(|s| s.as_str()).collect();
    let cur_domains: std::collections::BTreeSet<&str> =
        resolved.filter.domains.iter().map(|s| s.as_str()).collect();
    let cur_topics: std::collections::BTreeSet<&str> = resolved
        .filter
        .topic_ids
        .iter()
        .map(|s| s.as_str())
        .collect();
    if state_domains != cur_domains || state_topics != cur_topics {
        tracing::warn!(
            "Filter changed since last run! Previous: domains={state_domains:?} topics={state_topics:?}"
        );
    }
}

/// Compute completed shard indices and unique updated_dates from the processing set,
/// excluding shards that remain in the `pending` (failed) set.
fn compute_completed(
    shards_to_process: &[OAShard],
    pending: &[OAShard],
) -> (Vec<usize>, Vec<String>) {
    let failed_set: FxHashSet<usize> = pending.iter().map(|s| s.shard_idx).collect();

    let completed_indices: Vec<usize> = shards_to_process
        .iter()
        .filter(|s| !failed_set.contains(&s.shard_idx))
        .map(|s| s.shard_idx)
        .collect();

    let updated_dates: Vec<String> = shards_to_process
        .iter()
        .filter_map(|s| s.updated_date.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    (completed_indices, updated_dates)
}

/// Build the new State after a run, preserving backward-compat fields.
fn build_new_state(
    mut state: State,
    completed_indices: &[usize],
    updated_dates: &[String],
    failed_count: usize,
    manifest_shard_count: usize,
    filter: &transform::Filter,
    timestamps: (&str, &str),
) -> State {
    // Keep completed_shards for backward compat (no longer used for filtering)
    for &idx in completed_indices {
        if !state.completed_shards.contains(&idx) {
            state.completed_shards.push(idx);
        }
    }
    state.completed_shards.sort_unstable();
    state.manifest_shard_count = manifest_shard_count;

    // last_sync_date: informational only
    if failed_count == 0
        && let Some(max_date) = updated_dates.last()
        && state.last_sync_date.as_ref().is_none_or(|d| max_date > d)
    {
        state.last_sync_date = Some(max_date.clone());
    }

    state.filter = config::StateFilter {
        domains: filter.domains.iter().cloned().collect(),
        topics: filter.topic_ids.iter().cloned().collect(),
    };
    let (started_at, finished_at) = timestamps;
    state.started_at = Some(started_at.to_owned());
    state.finished_at = Some(finished_at.to_owned());
    state
}

/// Save manifest snapshot, excluding failed shard URLs.
fn save_snapshot(all_shards: &[OAShard], failed_urls: &FxHashSet<&str>, output_dir: &Path) {
    let snapshot = ManifestSnapshot::from_shards_excluding(all_shards, failed_urls);
    if let Err(e) = snapshot.save(output_dir) {
        tracing::error!("Failed to save manifest snapshot: {e}");
    }
}

fn print_dry_run(
    cfg: &ResolvedConfig,
    all_shards: &[OAShard],
    to_process: &[OAShard],
    n_changed: usize,
    n_unchanged: usize,
    removed: &[manifest::RemovedShard],
) {
    println!(
        "Filter: domains={:?} topics={:?}",
        cfg.filter.domains.iter().collect::<Vec<_>>(),
        cfg.filter.topic_ids.iter().collect::<Vec<_>>()
    );
    if let Some(since) = &cfg.since {
        println!("Since: {since} (cli)");
    }
    println!("Remote manifest: {} shards", all_shards.len());
    println!();
    println!("Manifest diff:");
    println!("  Changed/new:       {n_changed}");
    println!("  Unchanged (ok):    {n_unchanged}");
    println!("  Removed:           {}", removed.len());
    println!();
    println!("To process: {} shards", to_process.len());
    for s in to_process.iter().take(20) {
        let date = s.updated_date.as_deref().unwrap_or("unknown");
        let size = s
            .content_length
            .map(|b| format!("~{} MiB", b / 1_048_576))
            .unwrap_or_else(|| "unknown".to_string());
        println!("  shard_{:04}  updated_date={date}  {size}", s.shard_idx);
    }
    if to_process.len() > 20 {
        println!("  ... and {} more", to_process.len() - 20);
    }
    if !removed.is_empty() {
        println!();
        println!("Removed shards (files will be deleted):");
        for r in removed.iter().take(20) {
            println!("  shard_{:04}", r.shard_idx);
        }
        if removed.len() > 20 {
            println!("  ... and {} more", removed.len() - 20);
        }
    }
    println!();
    println!("Output: {} tables per shard", TABLES.len());
}

// ============================================================
// `papeline status`
// ============================================================

fn cmd_status(output_dir: &Path) -> ExitCode {
    let raw_dir = output_dir.join("raw");
    if let Err(e) = init_http(1) {
        eprintln!("Failed to initialize HTTP: {e}");
        return ExitCode::from(2);
    }
    match api::fetch_manifest("works") {
        Ok(shards) => {
            let total_records: u64 = shards.iter().map(|s| s.record_count).sum();
            let total_bytes: u64 = shards.iter().filter_map(|s| s.content_length).sum();
            let dates: std::collections::BTreeSet<&str> = shards
                .iter()
                .filter_map(|s| s.updated_date.as_deref())
                .collect();
            let date_range = if dates.is_empty() {
                "unknown".to_string()
            } else {
                format!(
                    "{} .. {}",
                    dates.iter().next().unwrap(),
                    dates.iter().last().unwrap()
                )
            };

            println!("Remote (OpenAlex manifest):");
            println!("  Shards:     {}", shards.len());
            println!("  Records:    ~{total_records}");
            println!(
                "  Size:       ~{} GiB (compressed)",
                total_bytes / 1_073_741_824
            );
            println!("  Partitions: updated_date={date_range}");
            println!();

            // Local info
            let state = State::load(&raw_dir);
            let old_snapshot = ManifestSnapshot::load(&raw_dir);
            let completed_count = shards
                .iter()
                .filter(|s| is_shard_complete(&raw_dir, s.shard_idx))
                .count();

            println!("Local ({}):", output_dir.display());
            println!("  Completed:  {completed_count}/{} shards", shards.len());
            println!(
                "  Last sync:  {}",
                state.last_sync_date.as_deref().unwrap_or("never")
            );
            if let Some(snap) = &old_snapshot {
                println!(
                    "  Snapshot:   {} entries (saved {})",
                    snap.entries.len(),
                    snap.saved_at
                );
            } else {
                println!("  Snapshot:   none (first run)");
            }
            if !state.filter.domains.is_empty() || !state.filter.topics.is_empty() {
                println!(
                    "  Filter:     domains={:?} topics={:?}",
                    state.filter.domains, state.filter.topics
                );
            }

            // Show manifest diff preview
            if old_snapshot.is_some() {
                let result = compute_manifest_diff(shards, old_snapshot.as_ref(), &raw_dir);
                let d = &result.diff;
                if d.changed.len() + d.unchanged_missing.len() + d.removed.len() > 0 {
                    println!();
                    println!("  Pending changes:");
                    println!("    Changed/new:       {}", d.changed.len());
                    println!("    Missing locally:   {}", d.unchanged_missing.len());
                    println!("    Removed upstream:  {}", d.removed.len());
                }
            }
        }
        Err(e) => {
            tracing::error!("Failed to fetch manifest: {e}");
            let state = State::load(&raw_dir);
            println!("Local ({}):", output_dir.display());
            println!(
                "  Completed:  {} shards (manifest unavailable)",
                state.completed_shards.len()
            );
            println!(
                "  Last sync:  {}",
                state.last_sync_date.as_deref().unwrap_or("never")
            );
        }
    }

    ExitCode::SUCCESS
}

// ============================================================
// `papeline hive`
// ============================================================

fn cmd_hive(
    output_dir: &Path,
    config_path: Option<&Path>,
    args: &HiveArgs,
    multi: &indicatif::MultiProgress,
) -> ExitCode {
    let file_cfg = match load_config(config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            tracing::error!("Config error: {e}");
            return ExitCode::from(2);
        }
    };

    let output_dir = file_cfg
        .output
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| output_dir.to_path_buf());

    // Merge: CLI > [hive] section
    let hive_cfg = config::HiveConfig {
        zstd_level: args.zstd_level.or(file_cfg.hive.zstd_level),
        row_group_size: args.row_group_size.or(file_cfg.hive.row_group_size),
        num_shards: args.num_shards.or(file_cfg.hive.num_shards),
        threads: args.threads.or(file_cfg.hive.threads),
        memory_limit: args.memory_limit.clone().or(file_cfg.hive.memory_limit),
    };

    let resolved = match config::ResolvedHiveConfig::from_config(
        &output_dir,
        &hive_cfg,
        file_cfg.zstd_level, // top-level fallback for zstd
    ) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Config error: {e}");
            return ExitCode::from(2);
        }
    };

    match hive::run_hive(&resolved, args.force, args.dry_run, multi) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:#}");
            ExitCode::from(1)
        }
    }
}

// ============================================================
// `papeline clean`
// ============================================================

fn cmd_clean(output_dir: &Path) -> ExitCode {
    let raw_dir = output_dir.join("raw");
    if !raw_dir.exists() {
        println!("Raw directory does not exist: {}", raw_dir.display());
        return ExitCode::SUCCESS;
    }

    let mut total = 0usize;
    for table in TABLES {
        let dir = raw_dir.join(table);
        if dir.exists() {
            match sink::cleanup_tmp_files(&dir) {
                Ok(count) => total += count,
                Err(e) => tracing::error!("Error cleaning {}: {e}", dir.display()),
            }
        }
    }

    if total > 0 {
        println!("Removed {total} .tmp files");
    } else {
        println!("No .tmp files found");
    }

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_run_args() -> RunArgs {
        RunArgs {
            concurrency: None,
            read_timeout: None,
            max_retries: None,
            outer_retries: None,
            retry_delay: None,
            zstd_level: None,
            domain: vec![],
            topic: vec![],
            force: false,
            since: None,
            max_shards: None,
            dry_run: false,
        }
    }

    // ---- resolve_config ----

    #[test]
    fn resolve_config_defaults() {
        let cfg = config::FileConfig::default();
        let args = empty_run_args();
        let r = resolve_config(Path::new("./out"), &cfg, &args);
        assert_eq!(r.zstd_level, 3);
        assert_eq!(r.concurrency, 8);
        assert_eq!(r.max_retries, 3);
        assert_eq!(r.read_timeout, 30);
        assert_eq!(r.outer_retries, 3);
        assert_eq!(r.retry_delay, 30);
        assert!(r.filter.is_empty());
    }

    #[test]
    fn resolve_config_cli_override() {
        let cfg = config::FileConfig::default();
        let mut args = empty_run_args();
        args.zstd_level = Some(5);
        args.concurrency = Some(16);
        let r = resolve_config(Path::new("./out"), &cfg, &args);
        assert_eq!(r.zstd_level, 5);
        assert_eq!(r.concurrency, 16);
    }

    #[test]
    fn resolve_config_file_override() {
        let cfg = config::FileConfig {
            zstd_level: Some(7),
            ..Default::default()
        };
        let args = empty_run_args();
        let r = resolve_config(Path::new("./out"), &cfg, &args);
        assert_eq!(r.zstd_level, 7);
    }

    #[test]
    fn resolve_config_cli_wins() {
        let cfg = config::FileConfig {
            zstd_level: Some(7),
            ..Default::default()
        };
        let mut args = empty_run_args();
        args.zstd_level = Some(5);
        let r = resolve_config(Path::new("./out"), &cfg, &args);
        assert_eq!(r.zstd_level, 5);
    }

    #[test]
    fn resolve_config_output_from_file() {
        let cfg = config::FileConfig {
            output: Some("/custom/path".to_string()),
            ..Default::default()
        };
        let args = empty_run_args();
        let r = resolve_config(Path::new("./out"), &cfg, &args);
        assert_eq!(r.output_dir, PathBuf::from("/custom/path/raw"));
    }

    #[test]
    fn resolve_config_output_adds_raw() {
        let cfg = config::FileConfig::default();
        let args = empty_run_args();
        let r = resolve_config(Path::new("./outputs"), &cfg, &args);
        assert_eq!(r.output_dir, PathBuf::from("./outputs/raw"));
    }

    // ---- save_snapshot ----

    fn make_shards(n: usize) -> Vec<OAShard> {
        (0..n)
            .map(|i| OAShard {
                shard_idx: i,
                url: format!(
                    "https://example.com/updated_date=2025-06-{:02}/part.gz",
                    i + 1
                ),
                content_length: Some(1000),
                record_count: 100,
                updated_date: Some(format!("2025-06-{:02}", i + 1)),
            })
            .collect()
    }

    #[test]
    fn save_snapshot_excludes_failed() {
        let dir = tempfile::TempDir::new().unwrap();
        let shards = make_shards(3);
        let mut failed: FxHashSet<&str> = FxHashSet::default();
        failed.insert(shards[1].url.as_str());

        save_snapshot(&shards, &failed, dir.path());

        let loaded = ManifestSnapshot::load(dir.path()).unwrap();
        assert_eq!(loaded.entries.len(), 2);
        let urls: Vec<&str> = loaded.entries.iter().map(|e| e.url.as_str()).collect();
        assert!(!urls.contains(&shards[1].url.as_str()));
        assert!(urls.contains(&shards[0].url.as_str()));
        assert!(urls.contains(&shards[2].url.as_str()));
    }

    // ---- compute_completed ----

    #[test]
    fn compute_completed_excludes_failed() {
        let shards = make_shards(5);
        let pending = vec![shards[1].clone(), shards[3].clone()];
        let (completed, dates) = compute_completed(&shards, &pending);
        assert_eq!(completed, vec![0, 2, 4]);
        assert_eq!(dates.len(), 5); // all unique dates
    }

    #[test]
    fn compute_completed_all_success() {
        let shards = make_shards(3);
        let (completed, _) = compute_completed(&shards, &[]);
        assert_eq!(completed, vec![0, 1, 2]);
    }
}
