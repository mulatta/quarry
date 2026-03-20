//! quarry-build CLI — Rust ETL build pipeline (PG direct load).

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use quarry_rs::build::{oa, pubmed};
use quarry_rs::build::pg_sink::PgSink;

#[derive(Parser)]
#[command(name = "quarry-build", about = "Quarry ETL build pipeline")]
struct Cli {
    /// PostgreSQL connection string.
    #[arg(long, default_value = "postgresql:///quarry")]
    pg_conninfo: String,

    /// Max parallel parse threads. Auto-detects from available memory if unset.
    /// Each thread holds ~400MB during PubMed XML parsing.
    #[arg(long)]
    threads: Option<usize>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Initialize PostgreSQL schema (create tables + indexes).
    InitSchema,
    /// Build PubMed data from XML files → PG (baseline + optional updates).
    BuildPubmed {
        /// Directory containing PubMed baseline XML (.xml.gz) files.
        #[arg(long)]
        xml_dir: PathBuf,

        /// Directory containing PubMed update XML (.xml.gz) files.
        #[arg(long)]
        updates_dir: Option<PathBuf>,
    },
    /// Build OpenAlex data from JSONL files → PG (local or S3).
    BuildOa {
        /// S3 prefix (e.g. s3://openalex/data/works).
        #[arg(long)]
        s3_prefix: Option<String>,

        /// Local directory containing .gz JSONL files (alternative to S3).
        #[arg(long)]
        local_dir: Option<PathBuf>,

        /// Number of concurrent S3 downloads (max 64).
        #[arg(long, default_value_t = 8)]
        s3_concurrency: usize,
    },
    /// Enrich: UPDATE works SET pm_* FROM papers; generate work_mesh.
    Enrich,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let pg = &cli.pg_conninfo;

    let config = quarry_rs::build::config::BuildConfig {
        parse_threads: cli.threads,
        ..Default::default()
    };

    match cli.command {
        Command::InitSchema => {
            let mut sink = PgSink::connect(pg)?;
            sink.init_schema()?;
            eprintln!("schema: all tables and indexes created");
        }
        Command::BuildPubmed { xml_dir, updates_dir } => {
            let stats = pubmed::build_pubmed(
                &config,
                &xml_dir,
                updates_dir.as_deref(),
                pg,
            )?;
            eprintln!(
                "done: {} papers from {} files in {:.1}s",
                stats.num_papers, stats.num_files_processed, stats.elapsed_secs,
            );
        }
        Command::BuildOa {
            s3_prefix,
            local_dir,
            s3_concurrency,
        } => {
            let mut config = config;
            config.s3_download_concurrency = s3_concurrency;

            let stats = match (local_dir, s3_prefix) {
                (Some(dir), _) => oa::build_oa_local(&config, &dir, pg)?,
                (None, Some(prefix)) => oa::build_oa_s3(&config, &prefix, pg)?,
                (None, None) => {
                    return Err("build-oa requires --local-dir or --s3-prefix".into());
                }
            };
            eprintln!(
                "done: {} works, {} citations from {} files in {:.1}s",
                stats.num_works, stats.num_citations,
                stats.num_files_processed, stats.elapsed_secs,
            );
        }
        Command::Enrich => {
            let mut sink = PgSink::connect(pg)?;
            eprintln!("enrich: UPDATE works SET pm_* FROM papers...");
            let n = sink.enrich_works_from_papers()?;
            eprintln!("enrich: {n} works updated with pm_ fields");

            eprintln!("enrich: generating work_mesh (mesh_headings JOIN id_crosswalk)...");
            sink.execute_static("DELETE FROM work_mesh")?;
            let n = sink.generate_work_mesh()?;
            eprintln!("enrich: {n} work_mesh rows inserted");
        }
    }

    Ok(())
}
