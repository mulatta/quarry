//! quarry-parse CLI — local files → Parquet output.
//!
//! Output protocol:
//! - stdout: JSON stats on success (one line)
//! - stderr: progress logs
//! - exit code: 0 = success, 1 = error

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

use quarry_parse::build::{config::ParseConfig, oa, pubmed};
use quarry_parse::parse::mesh;
use quarry_parse::build::parquet_writer;

#[derive(Parser)]
#[command(name = "quarry-parse", about = "Parse raw data files into Parquet")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Parse OpenAlex .gz JSONL → Parquet
    Oa {
        /// Directory with OA .gz JSONL files
        #[arg(long, env = "QUARRY_OA_INPUT_DIR")]
        input_dir: PathBuf,

        /// Parquet output directory
        #[arg(long, env = "QUARRY_OA_OUTPUT_DIR")]
        output_dir: PathBuf,

        /// Filter to a single Hive partition (e.g. updated_date=2025-01-01)
        #[arg(long)]
        partition: Option<String>,

        /// Parse thread count (0 = auto from available memory)
        #[arg(long, env = "QUARRY_PARSE_THREADS")]
        threads: Option<usize>,
    },
    /// Parse PubMed XML → Parquet
    Pubmed {
        /// Directory with baseline XML files
        #[arg(long, env = "QUARRY_PUBMED_INPUT_DIR")]
        input_dir: PathBuf,

        /// Directory with update XML files
        #[arg(long, env = "QUARRY_PUBMED_UPDATES_DIR")]
        updates_dir: Option<PathBuf>,

        /// Parquet output directory
        #[arg(long, env = "QUARRY_PUBMED_OUTPUT_DIR")]
        output_dir: PathBuf,

        /// Parse thread count (0 = auto from available memory)
        #[arg(long, env = "QUARRY_PARSE_THREADS")]
        threads: Option<usize>,
    },
    /// Parse MeSH descriptor XML → Parquet
    Mesh {
        /// Path to MeSH descriptor XML (e.g. desc2025.xml)
        #[arg(long, env = "QUARRY_MESH_XML_PATH")]
        xml_path: PathBuf,

        /// Parquet output directory
        #[arg(long, env = "QUARRY_MESH_OUTPUT_DIR")]
        output_dir: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    if let Err(e) = run(&cli) {
        eprintln!("error: {}", quarry_parse::err_chain(&*e));
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    match &cli.cmd {
        Cmd::Oa {
            input_dir,
            output_dir,
            partition,
            threads,
        } => {
            let config = make_config(threads);
            let stats = oa::parse_oa(&config, input_dir, output_dir, partition.as_deref())?;
            println!("{}", serde_json::to_string(&stats)?);
        }

        Cmd::Pubmed {
            input_dir,
            updates_dir,
            output_dir,
            threads,
        } => {
            let config = make_config(threads);
            let stats = pubmed::parse_pubmed(
                &config,
                input_dir,
                updates_dir.as_deref(),
                output_dir,
            )?;
            println!("{}", serde_json::to_string(&stats)?);
        }

        Cmd::Mesh {
            xml_path,
            output_dir,
        } => {
            let entries = mesh::parse_mesh_xml(xml_path)?;
            let n = entries.len();
            eprintln!("mesh: parsed {n} entries from {}", xml_path.display());

            let out_path = output_dir.join("mesh_tree.parquet");
            let rows = parquet_writer::write_mesh_tree(&entries, &out_path)?;
            println!("{}", serde_json::json!({"rows_written": rows}));
        }
    }
    Ok(())
}

fn make_config(threads: &Option<usize>) -> ParseConfig {
    let mut config = ParseConfig::default();
    if let Some(v) = threads {
        config.parse_threads = if *v == 0 { None } else { Some(*v) };
    }
    config
}
