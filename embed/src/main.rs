//! quarry-embed — Generate embeddings for hive-partitioned academic papers.
//!
//! Reads `work_id`, `title`, `abstract_text` from hive/works parquet,
//! encodes via embedding backend, writes `{work_id, embedding}` parquet.

mod embedder;
mod http_backend;
mod io;
#[cfg(feature = "local")]
mod ort_backend;
#[cfg(feature = "local")]
mod tokenize;

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use crate::embedder::Embedder;

// ============================================================
// CLI
// ============================================================

#[derive(Clone, clap::ValueEnum)]
enum Backend {
    /// OpenAI-compatible HTTP API
    Http,
    /// Local ONNX Runtime inference
    Local,
}

#[derive(Parser)]
#[command(about = "Generate embeddings for academic paper hive data")]
struct Cli {
    /// Path to hive works directory (e.g. outputs/hive/works)
    hive_dir: PathBuf,

    /// Output parquet path
    #[arg(short, long)]
    output: PathBuf,

    /// Embedding backend
    #[arg(long, value_enum, default_value_t = Backend::Http)]
    backend: Backend,

    /// Batch size for embedding calls
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Max rows to process (for testing)
    #[arg(long)]
    max_rows: Option<usize>,

    // --- HTTP backend options ---
    /// Embedding API endpoint (http backend)
    #[arg(long, env = "EMBED_ENDPOINT")]
    endpoint: Option<String>,

    /// Model name passed to the API (http backend)
    #[arg(long, default_value = "jina-embeddings-v3")]
    model: String,

    // --- Local backend options ---
    /// Path to ONNX model directory containing model.onnx + tokenizer.json
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Execution device (local backend)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Pooling strategy (local backend)
    #[arg(long, value_enum, default_value_t = crate::embedder::PoolingStrategy::Mean)]
    pooling: crate::embedder::PoolingStrategy,

    /// Prompt prefix prepended to each text (local backend, e.g. "Document: ")
    #[arg(long, default_value = "")]
    prompt: String,

    /// Max token length for tokenizer (local backend)
    #[arg(long, default_value_t = 512)]
    max_length: usize,
}

fn encode_bar(n: u64) -> ProgressBar {
    let bar = ProgressBar::new(n);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{prefix:.bold} {bar:30.cyan/black.dim} {pos}/{len} rows [{elapsed}]")
            .expect("valid template")
            .progress_chars("=>-"),
    );
    bar.set_prefix("Encoding");
    bar
}

// ============================================================
// Main
// ============================================================

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Phase 1: Read
    let (work_ids, texts) = io::read_inputs(&cli.hive_dir, cli.max_rows)?;
    if work_ids.is_empty() {
        println!("No works to embed");
        return Ok(());
    }

    // Phase 2: Encode
    let embedder: Box<dyn Embedder> = match cli.backend {
        Backend::Http => {
            let endpoint = cli
                .endpoint
                .ok_or_else(|| anyhow::anyhow!("--endpoint is required for http backend"))?;
            Box::new(http_backend::HttpEmbedder::new(endpoint, cli.model))
        }
        Backend::Local => {
            #[cfg(feature = "local")]
            {
                let model_dir = cli
                    .model_dir
                    .ok_or_else(|| anyhow::anyhow!("--model-dir is required for local backend"))?;
                Box::new(ort_backend::OrtEmbedder::new(
                    &model_dir,
                    cli.pooling,
                    cli.max_length,
                    cli.prompt,
                    &cli.device,
                )?)
            }
            #[cfg(not(feature = "local"))]
            anyhow::bail!("local backend not compiled — rebuild with --features local")
        }
    };

    let bar = encode_bar(texts.len() as u64);
    let (embeddings, dim) = embedder.encode(&texts, cli.batch_size, &bar)?;
    bar.finish_and_clear();

    // Phase 3: Write
    io::write_output(&cli.output, &work_ids, embeddings, dim)?;

    println!("Done: {} works embedded (dim={dim})", work_ids.len());
    Ok(())
}
