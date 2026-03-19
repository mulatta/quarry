//! quarry-build CLI — Rust ETL build pipeline.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use quarry_build::config::BuildConfig;
use quarry_build::pubmed;

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
    // Future: BuildOa, Enrich, BuildAll
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let config = BuildConfig {
        output_dir: cli.output,
        row_group_size: cli.row_group_size,
        max_rows_per_file: cli.max_rows_per_file,
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
    }

    Ok(())
}
