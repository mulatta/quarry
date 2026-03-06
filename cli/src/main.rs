//! quarry-etl CLI — Academic paper data pipeline (OpenAlex).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::MultiProgress;
use rayon::prelude::*;

use quarry_etl_core::config::{self, FileConfig, ResolvedHiveConfig, ResolvedUploadConfig};
use quarry_etl_core::embed;
use quarry_etl_core::oa::{self, OAProvider, OAShard};
use quarry_etl_core::progress::{IndicatifMakeWriter, ProgressContext};
use quarry_etl_core::provider::{RunContext, run_provider};
use quarry_etl_core::remote::{RemoteTargets, TransferOpts};
use quarry_etl_core::stream::HttpPool;
use quarry_etl_core::{api, hive, remote};

#[derive(Parser)]
#[command(name = "quarry-etl", about = "Academic paper data pipeline (OpenAlex)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download and process OpenAlex works snapshot to Parquet
    Run(RunArgs),
    /// Repartition raw shards into year-partitioned Hive parquet
    Hive(HiveArgs),
    /// Sync local files to S3-compatible storage
    Push(TransferArgs),
    /// Sync S3-compatible storage to local
    Pull(TransferArgs),
    /// Show remote manifest info and local progress
    Status(StatusArgs),
    /// Remove temporary (.tmp) files from output directory
    Clean(CleanArgs),
    /// Generate embeddings for hive-partitioned works
    Embed(EmbedArgs),
}

// ============================================================
// Shared options
// ============================================================

/// Common options for output and config.
#[derive(Parser)]
struct CommonOpts {
    /// Output directory
    #[arg(short, long, default_value = "./outputs")]
    output_dir: String,
    /// Config file (TOML)
    #[arg(short, long)]
    config: Option<PathBuf>,
}

// ============================================================
// Run
// ============================================================

#[derive(Parser)]
struct RunArgs {
    #[command(flatten)]
    common: CommonOpts,
    /// Concurrent downloads
    #[arg(long)]
    concurrency: Option<usize>,
    /// HTTP read timeout in seconds
    #[arg(long)]
    read_timeout: Option<u64>,
    /// Per-shard retry attempts
    #[arg(long)]
    max_retries: Option<u32>,
    /// Outer retry passes
    #[arg(long)]
    outer_retries: Option<u32>,
    /// Delay between outer retries in seconds
    #[arg(long)]
    retry_delay: Option<u64>,
    /// ZSTD compression level
    #[arg(long)]
    zstd_level: Option<i32>,
    /// Filter by OA domain (repeatable)
    #[arg(long = "domain")]
    domains: Vec<String>,
    /// Filter by OA topic ID (repeatable)
    #[arg(long = "topic")]
    topics: Vec<String>,
    /// Filter by language code (repeatable)
    #[arg(long = "language")]
    languages: Vec<String>,
    /// Filter by work type (repeatable)
    #[arg(long = "work-type")]
    work_types: Vec<String>,
    /// Only keep works with an abstract
    #[arg(long)]
    require_abstract: bool,
    /// Force re-download all shards
    #[arg(long)]
    force: bool,
    /// Only process shards with updated_date >= DATE
    #[arg(long)]
    since: Option<String>,
    /// Limit total shards (for testing)
    #[arg(long)]
    max_shards: Option<usize>,
    /// Show what would be done
    #[arg(long)]
    dry_run: bool,
    /// Auto-run hive after download
    #[arg(long)]
    hive: bool,
    /// Remove raw files after hive
    #[arg(long)]
    clean_raw: bool,
}

// ============================================================
// Hive
// ============================================================

#[derive(Parser)]
struct HiveArgs {
    #[command(flatten)]
    common: CommonOpts,
    /// Force rebuild
    #[arg(long)]
    force: bool,
    /// Show what would be done
    #[arg(long)]
    dry_run: bool,
    /// ZSTD compression level
    #[arg(long)]
    zstd_level: Option<i32>,
    /// Parquet row group size
    #[arg(long)]
    row_group_size: Option<usize>,
    /// Shards per partition
    #[arg(long)]
    num_shards: Option<usize>,
    /// Worker threads
    #[arg(long)]
    threads: Option<usize>,
    /// Memory limit (e.g. "32GB")
    #[arg(long)]
    memory_limit: Option<String>,
}

// ============================================================
// Push / Pull
// ============================================================

#[derive(Parser)]
struct TransferArgs {
    #[command(flatten)]
    common: CommonOpts,
    /// Exclude raw/
    #[arg(long)]
    no_raw: bool,
    /// Exclude hive/
    #[arg(long)]
    no_hive: bool,
    /// Show what would be transferred
    #[arg(long)]
    dry_run: bool,
    /// Force transfer all files (skip diff)
    #[arg(long)]
    force: bool,
    /// Max concurrent transfers
    #[arg(long)]
    concurrency: Option<usize>,
}

// ============================================================
// Status / Clean
// ============================================================

#[derive(Parser)]
struct StatusArgs {
    #[command(flatten)]
    common: CommonOpts,
}

#[derive(Parser)]
struct CleanArgs {
    /// Output directory
    #[arg(short, long, default_value = "./outputs")]
    output_dir: String,
}

// ============================================================
// Embed
// ============================================================

#[derive(Clone, clap::ValueEnum)]
enum EmbedBackend {
    /// OpenAI-compatible HTTP API
    Http,
    /// Local ONNX Runtime inference
    Local,
}

#[derive(Parser)]
struct EmbedArgs {
    #[command(flatten)]
    common: CommonOpts,

    /// Embedding backend
    #[arg(long, value_enum, default_value_t = EmbedBackend::Http)]
    backend: EmbedBackend,

    /// Batch size for embedding calls
    #[arg(long)]
    batch_size: Option<usize>,

    /// Max rows to process (for testing)
    #[arg(long)]
    max_rows: Option<usize>,

    /// Overwrite existing output file
    #[arg(long)]
    force: bool,

    /// Output parquet path (default: {output_dir}/embeddings.parquet)
    #[arg(long)]
    embed_output: Option<PathBuf>,

    // --- HTTP backend options ---
    /// Embedding API endpoint (http backend)
    #[arg(long, env = "EMBED_ENDPOINT")]
    endpoint: Option<String>,

    /// Model name for the API (http backend)
    #[arg(long)]
    model: Option<String>,

    // --- Local backend options ---
    /// Path to ONNX model directory containing model.onnx + tokenizer.json
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Execution device: cpu, cuda, coreml (local backend)
    #[arg(long)]
    device: Option<String>,

    /// Pooling strategy: mean, cls, last_token (local backend)
    #[arg(long)]
    pooling: Option<embed::PoolingStrategy>,

    /// Prompt prefix prepended to each text (local backend)
    #[arg(long)]
    prompt: Option<String>,

    /// Max token length for tokenizer (local backend)
    #[arg(long)]
    max_length: Option<usize>,
}

// ============================================================
// Main
// ============================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Command::Run(args) => cmd_run(args),
        Command::Hive(args) => cmd_hive(args),
        Command::Push(args) => cmd_push(args),
        Command::Pull(args) => cmd_pull(args),
        Command::Status(args) => cmd_status(args),
        Command::Clean(args) => cmd_clean(args),
        Command::Embed(args) => cmd_embed(args),
    }
}

// ============================================================
// Config helpers
// ============================================================

/// Load config: --config > ./etl.toml > ~/quarry/etl.toml > defaults.
fn load_cfg(explicit: Option<&std::path::Path>) -> Result<FileConfig> {
    config::load_config(explicit)
}

fn resolve_root(common: &CommonOpts, cfg: &FileConfig) -> String {
    cfg.output
        .as_deref()
        .unwrap_or(&common.output_dir)
        .to_string()
}

fn resolve_upload(cfg: &FileConfig) -> Result<Option<ResolvedUploadConfig>> {
    if cfg.upload.bucket.is_none() {
        return Ok(None);
    }
    ResolvedUploadConfig::from_config(&cfg.upload).map(Some)
}

/// Install tracing with indicatif-aware writer.
fn init_tracing(multi: &MultiProgress) {
    init_tracing_with_level(multi, "info");
}

/// Install tracing with a custom default level filter.
///
/// `default_level` is used when `RUST_LOG` is not set (e.g. `"warn,ort=error"`).
fn init_tracing_with_level(multi: &MultiProgress, default_level: &str) {
    use tracing_subscriber::EnvFilter;
    let writer = IndicatifMakeWriter::new(multi.clone());
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level)),
        )
        .with_writer(writer)
        .with_target(false)
        .init();
}

/// Install SIGINT handler that sets the cancelled flag.
/// Returns (cancelled_flag, signal_id).
fn install_sigint() -> Result<(Arc<AtomicBool>, signal_hook::SigId)> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&cancelled);
    let sig_id = unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGINT, move || {
            flag.store(true, Ordering::Relaxed);
        })
    }
    .context("Failed to register SIGINT handler")?;
    Ok((cancelled, sig_id))
}

fn format_bytes(n: u64) -> String {
    config::format_bytes(n as usize)
}

// ============================================================
// Commands
// ============================================================

fn cmd_run(args: &RunArgs) -> Result<()> {
    let cfg = load_cfg(args.common.config.as_deref())?;
    let root = resolve_root(&args.common, &cfg);

    // Resolve: CLI > config > default
    let r_concurrency = args.concurrency.or(cfg.concurrency).unwrap_or(8);
    let r_read_timeout = args.read_timeout.or(cfg.read_timeout).unwrap_or(30);
    let r_max_retries = args.max_retries.or(cfg.max_retries).unwrap_or(3);
    let r_outer_retries = args.outer_retries.or(cfg.outer_retries).unwrap_or(3);
    let r_retry_delay = args.retry_delay.or(cfg.retry_delay).unwrap_or(30);
    let r_zstd_level = args.zstd_level.or(cfg.zstd_level).unwrap_or(3);

    let filter = config::build_filter(
        &args.domains,
        &args.topics,
        &args.languages,
        &args.work_types,
        args.require_abstract,
        &cfg.filter,
    );

    let raw_dir = PathBuf::from(&root).join("raw");
    std::fs::create_dir_all(&raw_dir)?;

    let progress: Arc<ProgressContext> = Arc::new(ProgressContext::new());
    init_tracing(progress.multi());

    // Fetch manifest
    let pool = Arc::new(
        HttpPool::new(
            r_concurrency,
            Duration::from_secs(r_read_timeout),
            r_max_retries,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?,
    );
    eprintln!("Fetching manifest...");
    let mut shards = api::fetch_manifest(&pool, "works")?;
    let total_records: u64 = shards.iter().map(|s| s.record_count).sum();
    eprintln!("  {} shards, ~{total_records} records", shards.len());

    // Filter by --since
    if let Some(ref since) = args.since {
        shards.retain(|s| {
            s.updated_date
                .as_ref()
                .is_some_and(|d| d.as_str() >= since.as_str())
        });
        eprintln!("  After --since {since}: {} shards", shards.len());
    }

    // Filter completed (unless --force)
    if !args.force {
        let all_indices: Vec<usize> = shards.iter().map(|s| s.shard_idx).collect();
        let done: std::collections::HashSet<usize> = all_indices
            .into_par_iter()
            .filter(|&idx| oa::is_shard_complete(&raw_dir, idx))
            .collect();
        shards.retain(|s| !done.contains(&s.shard_idx));
        eprintln!("  Pending: {} shards", shards.len());
    }

    if let Some(max) = args.max_shards {
        shards.truncate(max);
    }

    if shards.is_empty() {
        eprintln!("All shards up to date. Nothing to do.");
        return Ok(());
    }

    if args.dry_run {
        eprintln!("Dry run: would process {} shards", shards.len());
        for s in shards.iter().take(10) {
            eprintln!(
                "  shard {}: {} records (date={:?})",
                s.shard_idx, s.record_count, s.updated_date
            );
        }
        if shards.len() > 10 {
            eprintln!("  ... and {} more", shards.len() - 10);
        }
        return Ok(());
    }

    let upload_config = resolve_upload(&cfg)?;
    let (cancelled, sig_id) = install_sigint()?;

    // Outer retry loop
    let mut pending = shards;
    let mut total_completed = 0usize;
    let mut total_rows = 0usize;

    for pass_num in 0..=r_outer_retries {
        if pending.is_empty() {
            break;
        }
        if pass_num > 0 {
            eprintln!(
                "\nRetry pass {pass_num}/{r_outer_retries} ({} shards)",
                pending.len()
            );
            std::thread::sleep(Duration::from_secs(r_retry_delay));
        }

        let ctx = RunContext {
            output_dir: raw_dir.clone(),
            zstd_level: r_zstd_level,
            concurrency: r_concurrency,
            cancelled: Arc::clone(&cancelled),
        };
        let provider = OAProvider {
            filter: filter.clone(),
            pool: Arc::clone(&pool),
        };

        let base_dir = raw_dir.parent().unwrap_or(&raw_dir).to_path_buf();

        let summary = if let Some(ref ucfg) = upload_config {
            let (tx, handle) = remote::spawn_upload_worker(ucfg.clone(), base_dir, 64)?;
            let on_complete = |shard: &OAShard| {
                for table in oa::TABLES {
                    let path =
                        raw_dir.join(format!("{table}/shard_{:04}.parquet", shard.shard_idx));
                    if path.exists() {
                        let _ = tx.send(path);
                    }
                }
            };
            let s = run_provider(&provider, &pending, &ctx, &progress, Some(&on_complete));
            drop(tx);
            match handle.join() {
                Ok(Ok(us)) => {
                    eprintln!("Auto-push: {} uploaded, {} failed", us.uploaded, us.failed)
                }
                Ok(Err(e)) => eprintln!("Auto-push worker error: {e:#}"),
                Err(_) => eprintln!("Auto-push worker panicked"),
            }
            s
        } else {
            run_provider(&provider, &pending, &ctx, &progress, None)
        };

        total_completed += summary.completed;
        total_rows += summary.total_rows;

        if summary.failed == 0 {
            pending = Vec::new();
            break;
        }
        pending = summary
            .failed_indices
            .iter()
            .map(|&i| pending[i].clone())
            .collect();

        if cancelled.load(Ordering::Relaxed) {
            break;
        }
    }

    signal_hook::low_level::unregister(sig_id);

    let failed = pending.len();
    eprintln!("\nDone: {total_completed} completed, {failed} failed, {total_rows} rows");
    if failed > 0 {
        let idxs: Vec<usize> = pending.iter().map(|s| s.shard_idx).collect();
        eprintln!("Failed shards: {idxs:?}");
    }

    // Auto-hive
    let do_hive = args.hive || cfg.hive.enable.unwrap_or(false);
    if do_hive && failed == 0 {
        eprintln!("\nRunning hive partitioning...");
        run_hive_with_cfg(
            &root,
            &cfg,
            args.clean_raw || cfg.hive.clean_raw.unwrap_or(false),
        )?;
    }

    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn cmd_hive(args: &HiveArgs) -> Result<()> {
    let cfg = load_cfg(args.common.config.as_deref())?;
    let root = resolve_root(&args.common, &cfg);

    let multi = MultiProgress::new();
    init_tracing(&multi);

    let hive_config = resolve_hive_config(&root, &cfg, args)?;
    let upload_config = resolve_upload(&cfg)?;

    if let Some(ucfg) = upload_config {
        let output_dir = PathBuf::from(&root);
        let (tx, handle) = remote::spawn_upload_worker(ucfg, output_dir, 64)?;
        hive::run_hive(&hive_config, args.force, args.dry_run, &multi, Some(&tx))?;
        drop(tx);
        match handle.join() {
            Ok(Ok(s)) => eprintln!("Auto-push: {} uploaded, {} failed", s.uploaded, s.failed),
            Ok(Err(e)) => eprintln!("Auto-push worker error: {e:#}"),
            Err(_) => eprintln!("Auto-push worker panicked"),
        }
    } else {
        hive::run_hive(&hive_config, args.force, args.dry_run, &multi, None)?;
    }

    eprintln!("Hive partitioning complete.");
    Ok(())
}

fn cmd_push(args: &TransferArgs) -> Result<()> {
    let cfg = load_cfg(args.common.config.as_deref())?;
    let root = resolve_root(&args.common, &cfg);

    let multi = MultiProgress::new();
    init_tracing(&multi);

    let ucfg = ResolvedUploadConfig::from_config(&cfg.upload)?;
    let r_force = args.force || cfg.upload.force.unwrap_or(false);
    let r_concurrency = args.concurrency.or(cfg.upload.concurrency).unwrap_or(8);

    let opts = TransferOpts {
        targets: RemoteTargets {
            raw: !args.no_raw,
            hive: !args.no_hive,
        },
        dry_run: args.dry_run,
        force: r_force,
        concurrency: r_concurrency,
    };

    let (cancelled, sig_id) = install_sigint()?;
    let progress: Arc<ProgressContext> = Arc::new(ProgressContext::with_multi(multi));
    let summary = remote::run_push(&ucfg, &PathBuf::from(&root), &opts, &progress, &cancelled)?;
    signal_hook::low_level::unregister(sig_id);

    print_transfer_summary("Push", &summary);
    if summary.files_failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn cmd_pull(args: &TransferArgs) -> Result<()> {
    let cfg = load_cfg(args.common.config.as_deref())?;
    let root = resolve_root(&args.common, &cfg);

    let multi = MultiProgress::new();
    init_tracing(&multi);

    let ucfg = ResolvedUploadConfig::from_config(&cfg.upload)?;
    let r_force = args.force || cfg.upload.force.unwrap_or(false);
    let r_concurrency = args.concurrency.or(cfg.upload.concurrency).unwrap_or(8);

    let opts = TransferOpts {
        targets: RemoteTargets {
            raw: !args.no_raw,
            hive: !args.no_hive,
        },
        dry_run: args.dry_run,
        force: r_force,
        concurrency: r_concurrency,
    };

    let (cancelled, sig_id) = install_sigint()?;
    let progress: Arc<ProgressContext> = Arc::new(ProgressContext::with_multi(multi));
    let summary = remote::run_pull(&ucfg, &PathBuf::from(&root), &opts, &progress, &cancelled)?;
    signal_hook::low_level::unregister(sig_id);

    print_transfer_summary("Pull", &summary);
    if summary.files_failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn cmd_status(args: &StatusArgs) -> Result<()> {
    let cfg = load_cfg(args.common.config.as_deref())?;
    let root = resolve_root(&args.common, &cfg);
    let raw_dir = PathBuf::from(&root).join("raw");

    let pool = HttpPool::new(1, Duration::from_secs(30), 3).map_err(|e| anyhow::anyhow!("{e}"))?;
    let shards = api::fetch_manifest(&pool, "works")?;

    let total_records: u64 = shards.iter().map(|s| s.record_count).sum();
    let total_bytes: u64 = shards.iter().filter_map(|s| s.content_length).sum();
    let mut dates: Vec<&str> = shards
        .iter()
        .filter_map(|s| s.updated_date.as_deref())
        .collect();
    dates.sort_unstable();
    dates.dedup();
    let date_range = if dates.is_empty() {
        "unknown".to_string()
    } else {
        format!("{} .. {}", dates[0], dates[dates.len() - 1])
    };

    eprintln!("Remote (OpenAlex manifest):");
    eprintln!("  Shards:     {}", shards.len());
    eprintln!("  Records:    ~{total_records}");
    eprintln!("  Size:       ~{}", format_bytes(total_bytes));
    eprintln!("  Partitions: updated_date={date_range}");
    eprintln!();

    let all_indices: Vec<usize> = shards.iter().map(|s| s.shard_idx).collect();
    let completed: usize = all_indices
        .into_par_iter()
        .filter(|&idx| oa::is_shard_complete(&raw_dir, idx))
        .count();
    eprintln!("Local ({root}):");
    eprintln!("  Completed:  {completed}/{} shards", shards.len());

    let hive_dir = PathBuf::from(&root).join("hive");
    if hive_dir.exists() {
        let tables: Vec<String> = std::fs::read_dir(&hive_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir() && !e.file_name().to_string_lossy().starts_with('.'))
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        eprintln!("  Hive:       {} tables", tables.len());
    } else {
        eprintln!("  Hive:       not built");
    }

    Ok(())
}

fn cmd_clean(args: &CleanArgs) -> Result<()> {
    let raw_dir = PathBuf::from(&args.output_dir).join("raw");
    if !raw_dir.exists() {
        eprintln!("Raw directory does not exist: {}", raw_dir.display());
        return Ok(());
    }

    let mut total = 0usize;
    for table in oa::TABLES {
        let table_dir = raw_dir.join(table);
        if !table_dir.exists() {
            continue;
        }
        for entry in std::fs::read_dir(&table_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "tmp") {
                std::fs::remove_file(&path)?;
                total += 1;
            }
        }
    }

    if total > 0 {
        eprintln!("Removed {total} .tmp files");
    } else {
        eprintln!("No .tmp files found");
    }
    Ok(())
}

fn cmd_embed(args: &EmbedArgs) -> Result<()> {
    use std::io::IsTerminal;

    let cfg = load_cfg(args.common.config.as_deref())?;
    let root = resolve_root(&args.common, &cfg);
    let ecfg = &cfg.embed;
    let is_tty = std::io::stderr().is_terminal();

    // TTY: progress bar only (suppress INFO), non-TTY: structured logs
    // ort=error: suppress ORT C++ WARN/INFO (Memcpy, unassigned node warnings)
    let multi = MultiProgress::new();
    let default_level = if is_tty {
        "warn,ort=error"
    } else {
        "info,ort=error"
    };
    init_tracing_with_level(&multi, default_level);

    // Resolve: CLI > config > default
    let batch_size = args.batch_size.or(ecfg.batch_size).unwrap_or(64);
    let chunk_size = 50_000usize;
    let max_rows = args.max_rows.or(ecfg.max_rows);
    let model_name = args
        .model
        .clone()
        .or_else(|| ecfg.model.clone())
        .unwrap_or_else(|| "jina-embeddings-v3".to_string());

    let hive_dir = PathBuf::from(&root).join("hive/works");
    let out_path = args
        .embed_output
        .clone()
        .or_else(|| ecfg.output.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(&root).join("embeddings.parquet"));

    // Overwrite protection
    if out_path.exists() && !args.force {
        anyhow::bail!(
            "{} already exists (use --force to overwrite)",
            out_path.display()
        );
    }

    // Backend selection
    let backend_str = match args.backend {
        EmbedBackend::Http => "http",
        EmbedBackend::Local => "local",
    };
    let backend_str = ecfg.backend.as_deref().unwrap_or(backend_str);

    // Count total rows (fast metadata-only scan)
    let (total_rows, num_files) = embed::io::count_hive_rows(&hive_dir)?;
    let effective_total = max_rows.map(|m| m.min(total_rows)).unwrap_or(total_rows);

    if effective_total == 0 {
        eprintln!("No works to embed");
        return Ok(());
    }

    let summary = format!(
        "{num_files} hive files, ~{} rows, backend={backend_str}, batch_size={batch_size}",
        quarry_etl_core::progress::fmt_num(effective_total),
    );
    if is_tty {
        eprintln!("{summary}");
    } else {
        tracing::info!("{summary}");
    }

    // Build embedder
    let embedder: Box<dyn embed::Embedder> = match backend_str {
        "http" => {
            let endpoint = args
                .endpoint
                .clone()
                .or_else(|| ecfg.endpoint.clone())
                .or_else(|| std::env::var("EMBED_ENDPOINT").ok())
                .ok_or_else(|| {
                    anyhow::anyhow!("--endpoint or [embed].endpoint required for http backend")
                })?;
            Box::new(embed::HttpEmbedder::new(endpoint, model_name))
        }
        "local" => {
            #[cfg(feature = "local")]
            {
                let model_dir = args
                    .model_dir
                    .clone()
                    .or_else(|| ecfg.model_dir.as_ref().map(PathBuf::from))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "--model-dir or [embed].model_dir required for local backend"
                        )
                    })?;
                let pooling_override = args.pooling.or_else(|| {
                    ecfg.pooling
                        .as_deref()
                        .and_then(|s| s.parse::<embed::PoolingStrategy>().ok())
                });
                let prompt_override = args.prompt.clone().or_else(|| ecfg.prompt.clone());
                let max_length_override = args.max_length.or(ecfg.max_length);
                let device = args
                    .device
                    .clone()
                    .or_else(|| ecfg.device.clone())
                    .unwrap_or_else(|| "cpu".to_string());
                Box::new(embed::ort_backend::OrtEmbedder::new(
                    &model_dir,
                    embed::ort_backend::OrtEmbedderOpts {
                        pooling: pooling_override,
                        max_length: max_length_override,
                        prompt: prompt_override,
                        device,
                    },
                )?)
            }
            #[cfg(not(feature = "local"))]
            anyhow::bail!("local backend not compiled — rebuild with --features local")
        }
        other => anyhow::bail!("unknown embed backend: {other} (expected http|local)"),
    };

    // Progress bar: TTY shows bar with speed, non-TTY hidden
    let bar = if is_tty {
        let b = multi.add(indicatif::ProgressBar::new(effective_total as u64));
        b.set_style(quarry_etl_core::progress::embed_style());
        b.set_prefix("Embedding");
        b
    } else {
        indicatif::ProgressBar::hidden()
    };

    // Stream: read chunks → encode → write (bounded memory)
    let mut writer: Option<embed::io::EmbedWriter> = None;
    let start = std::time::Instant::now();

    let processed =
        embed::io::process_hive_chunks(&hive_dir, chunk_size, max_rows, |work_ids, texts| {
            let (embeddings, dim) = embedder.encode(texts, batch_size, &bar)?;
            if writer.is_none() {
                writer = Some(embed::io::EmbedWriter::new(&out_path, dim)?);
            }
            writer.as_mut().unwrap().write_chunk(work_ids, embeddings)?;
            Ok(())
        })?;

    bar.finish_and_clear();

    if let Some(w) = writer {
        w.close()?;
    }

    // Final summary
    let elapsed = start.elapsed();
    let rps = processed as f64 / elapsed.as_secs_f64();
    let size = out_path.metadata().map(|m| m.len()).unwrap_or(0);
    let done_msg = format!(
        "{} rows → {} ({:.1} MiB, {:.0} rows/s, {:.1}s)",
        quarry_etl_core::progress::fmt_num(processed),
        out_path.display(),
        size as f64 / 1024.0 / 1024.0,
        rps,
        elapsed.as_secs_f64(),
    );
    if is_tty {
        eprintln!("{done_msg}");
    } else {
        tracing::info!("{done_msg}");
    }

    Ok(())
}

// ============================================================
// Helpers
// ============================================================

fn resolve_hive_config(
    root: &str,
    cfg: &FileConfig,
    args: &HiveArgs,
) -> Result<ResolvedHiveConfig> {
    let output_dir = PathBuf::from(root);
    let hive_cfg = config::HiveConfig {
        enable: cfg.hive.enable,
        clean_raw: cfg.hive.clean_raw,
        zstd_level: args.zstd_level.or(cfg.hive.zstd_level),
        row_group_size: args.row_group_size.or(cfg.hive.row_group_size),
        num_shards: args.num_shards.or(cfg.hive.num_shards),
        threads: args.threads.or(cfg.hive.threads),
        memory_limit: args
            .memory_limit
            .clone()
            .or_else(|| cfg.hive.memory_limit.clone()),
    };
    ResolvedHiveConfig::from_config(&output_dir, &hive_cfg, cfg.zstd_level)
}

fn run_hive_with_cfg(root: &str, cfg: &FileConfig, clean_raw: bool) -> Result<()> {
    let output_dir = PathBuf::from(root);
    let hive_config = ResolvedHiveConfig::from_config(&output_dir, &cfg.hive, cfg.zstd_level)?;
    let multi = MultiProgress::new();
    let upload_config = resolve_upload(cfg)?;

    if let Some(ucfg) = upload_config {
        let (tx, handle) = remote::spawn_upload_worker(ucfg, output_dir.clone(), 64)?;
        hive::run_hive(&hive_config, false, false, &multi, Some(&tx))?;
        drop(tx);
        match handle.join() {
            Ok(Ok(s)) => eprintln!("Auto-push: {} uploaded, {} failed", s.uploaded, s.failed),
            Ok(Err(e)) => eprintln!("Auto-push worker error: {e:#}"),
            Err(_) => eprintln!("Auto-push worker panicked"),
        }
    } else {
        hive::run_hive(&hive_config, false, false, &multi, None)?;
    }

    if clean_raw {
        let raw_dir = output_dir.join("raw");
        if raw_dir.exists() {
            std::fs::remove_dir_all(&raw_dir)?;
            eprintln!("Cleaned raw directory");
        }
    }
    Ok(())
}

fn print_transfer_summary(label: &str, summary: &remote::TransferSummary) {
    eprintln!(
        "{label}: {} transferred ({}), {} skipped",
        summary.files_transferred,
        format_bytes(summary.bytes_transferred),
        summary.files_skipped,
    );
    if summary.files_failed > 0 {
        eprintln!("Failed: {} files", summary.files_failed);
        for key in summary.failed_keys.iter().take(20) {
            eprintln!("  {key}");
        }
        if summary.failed_keys.len() > 20 {
            eprintln!("  ... and {} more", summary.failed_keys.len() - 20);
        }
    }
}
