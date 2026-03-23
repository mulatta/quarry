//! quarry-ingest CLI — standalone binary for all data ingestion pipelines.
//!
//! Output protocol:
//! - stdout: JSON stats on success (one line)
//! - stderr: progress logs
//! - exit code: 0 = success, 1 = error
//!
//! Config priority: CLI args > env vars > config.toml > defaults.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use quarry_ingest::build::{
    config::BuildConfig,
    icite,
    oa,
    pg_sink::PgSink,
    pubmed,
};
use quarry_ingest::parse::mesh;

#[derive(Parser)]
#[command(name = "quarry-ingest", about = "Quarry data ingestion CLI")]
struct Cli {
    /// PG connection string
    #[arg(long, env = "QUARRY_PG_CONNINFO", default_value = "host=/tmp/quarry-pg dbname=quarry")]
    pg_conninfo: String,

    /// Path to TOML config file (CLI args override config values)
    #[arg(long, env = "QUARRY_CONFIG")]
    config: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Load data into PostgreSQL
    Load(LoadCmd),
    /// Database management
    Db(DbCmd),
    /// Enrich works with PubMed fields + generate work_mesh
    Enrich,
}

#[derive(Parser)]
struct LoadCmd {
    #[command(subcommand)]
    source: LoadSource,
}

#[derive(Subcommand)]
enum LoadSource {
    /// Load PubMed XML baseline + updates
    Pubmed {
        /// Directory with baseline XML files
        #[arg(long, env = "QUARRY_PUBMED_BASELINE_DIR")]
        xml_dir: PathBuf,

        /// Directory with update XML files
        #[arg(long, env = "QUARRY_PUBMED_UPDATES_DIR")]
        updates_dir: Option<PathBuf>,

        /// Parse thread count (0 = auto from available memory)
        #[arg(long, env = "QUARRY_PUBMED_PARSE_THREADS")]
        threads: Option<usize>,

        /// Parallel PG writer threads
        #[arg(long, env = "QUARRY_PUBMED_PG_WRITERS")]
        pg_writers: Option<usize>,

        /// Bounded channel buffer between parse and PG writers (0 = auto)
        #[arg(long, env = "QUARRY_PUBMED_CHANNEL_BUFFER")]
        channel_buffer: Option<usize>,
    },
    /// Load OpenAlex from S3
    Oa {
        /// S3 URL (s3://bucket/prefix)
        #[arg(long, env = "QUARRY_OA_S3_PREFIX")]
        s3_prefix: String,

        /// S3 download concurrency
        #[arg(long, env = "QUARRY_OA_S3_CONCURRENCY")]
        s3_concurrency: Option<usize>,

        /// Prefetch buffer: downloaded-but-not-yet-parsed file slots (0 = auto)
        #[arg(long, env = "QUARRY_OA_PREFETCH_BUFFER")]
        prefetch_buffer: Option<usize>,

        /// Parallel PG writer threads
        #[arg(long, env = "QUARRY_OA_PG_WRITERS")]
        pg_writers: Option<usize>,

        /// Bounded channel buffer between download and PG writers (0 = auto)
        #[arg(long, env = "QUARRY_OA_CHANNEL_BUFFER")]
        channel_buffer: Option<usize>,

        /// Max retry attempts for S3 fetch (0 = no retry)
        #[arg(long, env = "QUARRY_OA_FETCH_MAX_RETRIES")]
        fetch_max_retries: Option<u32>,

        /// Initial backoff in ms (doubles each retry)
        #[arg(long, env = "QUARRY_OA_FETCH_INITIAL_BACKOFF_MS")]
        fetch_initial_backoff_ms: Option<u64>,

        /// Max backoff in ms (cap for exponential growth)
        #[arg(long, env = "QUARRY_OA_FETCH_MAX_BACKOFF_MS")]
        fetch_max_backoff_ms: Option<u64>,
    },
    /// Load iCite CSV metrics
    Icite {
        /// Path to iCite CSV file
        #[arg(long, env = "QUARRY_ICITE_CSV_PATH")]
        csv_path: PathBuf,
    },
    /// Load MeSH descriptor XML
    Mesh {
        /// Path to MeSH descriptor XML (e.g. desc2025.xml)
        #[arg(long, env = "QUARRY_MESH_XML_PATH")]
        xml_path: PathBuf,
    },
}

#[derive(Parser)]
struct DbCmd {
    #[command(subcommand)]
    action: DbAction,
}

#[derive(Subcommand)]
enum DbAction {
    /// Create tables and indexes (idempotent)
    Init,
    /// Drop non-PK indexes before bulk load
    DropIndexes,
    /// Create indexes after bulk load
    CreateIndexes,
    /// VACUUM ANALYZE all tables
    Vacuum,
    /// TRUNCATE all data tables
    Reset,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    if let Err(e) = run(&cli) {
        eprintln!("error: {}", quarry_ingest::err_chain(&*e));
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

/// Load base config: TOML file if --config provided, otherwise defaults.
fn base_config(cli: &Cli) -> Result<BuildConfig, Box<dyn std::error::Error>> {
    match &cli.config {
        Some(path) => {
            eprintln!("config: loading {}", path.display());
            BuildConfig::from_toml(path)
        }
        None => Ok(BuildConfig::default()),
    }
}

fn run(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    match &cli.cmd {
        Cmd::Load(load) => run_load(cli, load),
        Cmd::Db(db) => run_db(cli, db),
        Cmd::Enrich => run_enrich(cli),
    }
}

fn run_load(cli: &Cli, load: &LoadCmd) -> Result<(), Box<dyn std::error::Error>> {
    // Fast-fail: verify DB schema matches expected COPY columns
    let mut checker = PgSink::connect(&cli.pg_conninfo)?;
    checker.verify_schema()?;
    drop(checker);

    match &load.source {
        LoadSource::Pubmed {
            xml_dir,
            updates_dir,
            threads,
            pg_writers,
            channel_buffer,
        } => {
            let mut config = base_config(cli)?;
            if let Some(v) = threads {
                config.parse_threads = if *v == 0 { None } else { Some(*v) };
            }
            if let Some(v) = pg_writers { config.pg_writer_threads = *v; }
            if let Some(v) = channel_buffer { config.channel_buffer = *v; }

            let stats = pubmed::build_pubmed(
                &config,
                xml_dir,
                updates_dir.as_deref(),
                &cli.pg_conninfo,
            )?;
            let json = serde_json::json!({
                "num_papers": stats.num_papers,
                "num_authors": stats.num_authors,
                "num_mesh": stats.num_mesh,
                "num_grants": stats.num_grants,
                "num_chemicals": stats.num_chemicals,
                "num_files_processed": stats.num_files_processed,
                "num_failed_files": stats.num_failed_files,
                "elapsed_secs": stats.elapsed_secs,
            });
            println!("{json}");
        }

        LoadSource::Oa {
            s3_prefix,
            s3_concurrency,
            prefetch_buffer,
            pg_writers,
            channel_buffer,
            fetch_max_retries,
            fetch_initial_backoff_ms,
            fetch_max_backoff_ms,
        } => {
            let mut config = base_config(cli)?;
            if let Some(v) = s3_concurrency { config.s3_download_concurrency = *v; }
            if let Some(v) = prefetch_buffer { config.prefetch_buffer = *v; }
            if let Some(v) = pg_writers { config.pg_writer_threads = *v; }
            if let Some(v) = channel_buffer { config.channel_buffer = *v; }
            if let Some(v) = fetch_max_retries { config.fetch_max_retries = *v; }
            if let Some(v) = fetch_initial_backoff_ms { config.fetch_initial_backoff_ms = *v; }
            if let Some(v) = fetch_max_backoff_ms { config.fetch_max_backoff_ms = *v; }

            let stats = oa::build_oa_s3(&config, s3_prefix, &cli.pg_conninfo)?;
            let json = serde_json::json!({
                "num_works": stats.num_works,
                "num_authors": stats.num_authors,
                "num_topics": stats.num_topics,
                "num_citations": stats.num_citations,
                "num_crosswalk": stats.num_crosswalk,
                "num_files_processed": stats.num_files_processed,
                "num_failed_lines": stats.num_failed_lines,
                "elapsed_secs": stats.elapsed_secs,
            });
            println!("{json}");
        }

        LoadSource::Icite { csv_path } => {
            let stats = icite::load_icite(&cli.pg_conninfo, csv_path)?;
            let json = serde_json::json!({
                "papers_updated": stats.papers_updated,
                "works_updated": stats.works_updated,
                "cited_by_clin_rows": stats.cited_by_clin_rows,
                "elapsed_secs": stats.elapsed_secs,
            });
            println!("{json}");
        }

        LoadSource::Mesh { xml_path } => {
            let entries = mesh::parse_mesh_xml(xml_path)?;
            let n = entries.len();
            eprintln!("mesh: parsed {n} entries from {}", xml_path.display());

            let mut sink = PgSink::connect(&cli.pg_conninfo)?;
            let rows = sink.write_mesh_tree(&entries)?;
            let json = serde_json::json!({
                "rows_written": rows,
            });
            println!("{json}");
        }
    }
    Ok(())
}

fn run_db(cli: &Cli, db: &DbCmd) -> Result<(), Box<dyn std::error::Error>> {
    let mut sink = PgSink::connect(&cli.pg_conninfo)?;
    match db.action {
        DbAction::Init => {
            sink.init_schema()?;
            eprintln!("db: schema initialized");
        }
        DbAction::DropIndexes => {
            sink.drop_indexes()?;
            eprintln!("db: indexes dropped");
        }
        DbAction::CreateIndexes => {
            sink.create_indexes()?;
            eprintln!("db: indexes created");
        }
        DbAction::Vacuum => {
            sink.vacuum()?;
            eprintln!("db: vacuum complete");
        }
        DbAction::Reset => {
            sink.reset_all()?;
            eprintln!("db: all tables truncated");
        }
    }
    Ok(())
}

fn run_enrich(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let mut sink = PgSink::connect(&cli.pg_conninfo)?;
    sink.verify_schema()?;

    let n_enriched = sink.enrich_works_from_papers()?;
    eprintln!("enrich: {n_enriched} works updated from papers");

    // Atomic: DELETE + regenerate work_mesh
    sink.begin()?;
    let mesh_result = (|| -> Result<u64, Box<dyn std::error::Error>> {
        sink.execute_static("DELETE FROM work_mesh")?;
        sink.generate_work_mesh()
    })();
    let n_mesh = match mesh_result {
        Ok(n) => {
            sink.commit()?;
            n
        }
        Err(e) => {
            let _ = sink.rollback();
            return Err(e);
        }
    };

    let json = serde_json::json!({
        "works_enriched": n_enriched,
        "work_mesh_rows": n_mesh,
    });
    println!("{json}");
    Ok(())
}
