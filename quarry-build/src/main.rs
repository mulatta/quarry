//! quarry-build CLI — Rust ETL build pipeline.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use quarry_build::config::BuildConfig;
use quarry_build::{enrich, oa, pubmed};

#[derive(Parser)]
#[command(name = "quarry-build", about = "Quarry ETL build pipeline")]
struct Cli {
    /// Output directory for Parquet files.
    #[arg(short, long, default_value = "build_output")]
    output: PathBuf,

    /// Parquet row group size.
    #[arg(long, default_value_t = 128 * 1024)]
    row_group_size: usize,

    /// Max rows per output Parquet file.
    #[arg(long, default_value_t = 500_000)]
    max_rows_per_file: usize,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build PubMed Parquet from XML files.
    BuildPubmed {
        /// Directory containing PubMed XML (.xml.gz) files.
        #[arg(long)]
        xml_dir: PathBuf,
    },
    /// Build OpenAlex Parquet from JSONL files (local or S3).
    BuildOa {
        /// S3 prefix (e.g. s3://openalex/data/works).
        #[arg(long)]
        s3_prefix: Option<String>,

        /// Local directory containing .gz JSONL files (alternative to S3).
        #[arg(long)]
        local_dir: Option<PathBuf>,

        /// OA batch size — flush to Parquet every N works.
        #[arg(long, default_value_t = 50_000)]
        batch_size: usize,
    },
    /// Enrich: JOIN works + papers → enriched works with pm_ fields,
    /// JOIN mesh_headings + id_crosswalk → work_mesh.
    Enrich,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let mut config = BuildConfig {
        output_dir: cli.output,
        row_group_size: cli.row_group_size,
        max_rows_per_file: cli.max_rows_per_file,
        ..BuildConfig::default()
    };
    config.ensure_dirs()?;

    match cli.command {
        Command::BuildPubmed { xml_dir } => {
            let stats = pubmed::build_pubmed(&config, &xml_dir)?;
            eprintln!(
                "done: {} papers from {} files in {:.1}s",
                stats.num_papers, stats.num_files_processed, stats.elapsed_secs,
            );
        }
        Command::BuildOa {
            s3_prefix,
            local_dir,
            batch_size,
        } => {
            config.oa_batch_size = batch_size;

            let stats = match (local_dir, s3_prefix) {
                (Some(dir), _) => oa::build_oa_local(&config, &dir)?,
                (None, Some(prefix)) => oa::build_oa_s3(&config, &prefix)?,
                (None, None) => {
                    return Err("build-oa requires --local-dir or --s3-prefix".into());
                }
            };
            eprintln!(
                "done: {} works, {} citations from {} files in {:.1}s",
                stats.num_works,
                stats.num_citations,
                stats.num_files_processed,
                stats.elapsed_secs,
            );
        }
        Command::Enrich => {
            let stats = enrich::enrich(&config)?;
            eprintln!(
                "done: enriched_works ({} parts), work_mesh ({} parts) in {:.1}s",
                stats.enriched_works_files,
                stats.work_mesh_files,
                stats.elapsed_secs,
            );
        }
    }

    Ok(())
}
