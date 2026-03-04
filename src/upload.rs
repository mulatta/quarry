//! Streaming upload of parquet files to S3-compatible storage (e.g. Cloudflare R2).
//!
//! Architecture: processing threads (rayon, CPU-bound) send completed file paths through
//! a bounded `mpsc::SyncSender`. A dedicated upload thread runs a tokio runtime that
//! consumes paths and streams them to the bucket, overlapping compression with network I/O.

use std::path::PathBuf;
use std::sync::mpsc;

use anyhow::{Context, Result};
use s3::creds::Credentials;
use s3::{Bucket, Region};

use crate::config::ResolvedUploadConfig;

/// Spawn a background upload worker thread.
///
/// Returns `(sender, join_handle)`. Send `PathBuf`s of completed files
/// through the sender. Drop the sender to signal completion; the worker will
/// drain remaining items and return.
///
/// `base_dir` is the local root — used to compute the S3 object key
/// by stripping this prefix from each file path.
/// e.g. base_dir = `/output`, file = `/output/raw/works/shard_0000.parquet`
///      → key = `{prefix}/raw/works/shard_0000.parquet`
pub fn spawn_upload_worker(
    config: ResolvedUploadConfig,
    base_dir: PathBuf,
    channel_bound: usize,
) -> Result<(
    mpsc::SyncSender<PathBuf>,
    std::thread::JoinHandle<Result<()>>,
)> {
    let (tx, rx) = mpsc::sync_channel::<PathBuf>(channel_bound);

    let handle = std::thread::Builder::new()
        .name("upload-worker".to_string())
        .spawn(move || upload_thread(config, base_dir, rx))?;

    Ok((tx, handle))
}

/// The upload thread entry point — creates a tokio runtime and processes files.
///
/// Takes ownership because this runs in a `thread::spawn(move || ...)` closure.
#[allow(clippy::needless_pass_by_value)]
fn upload_thread(
    config: ResolvedUploadConfig,
    base_dir: PathBuf,
    rx: mpsc::Receiver<PathBuf>,
) -> Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create upload tokio runtime")?;

    rt.block_on(async {
        let bucket = create_bucket(&config)?;
        let mut uploaded = 0usize;
        let mut bytes = 0u64;

        while let Ok(path) = rx.recv() {
            match upload_file(&bucket, &config.prefix, &base_dir, &path).await {
                Ok(size) => {
                    uploaded += 1;
                    bytes += size;
                    tracing::info!(
                        "uploaded {}: {} bytes (total: {} files, {} bytes)",
                        path.display(),
                        size,
                        uploaded,
                        bytes,
                    );
                }
                Err(e) => {
                    tracing::error!("upload failed for {}: {e:#}", path.display());
                    return Err(e);
                }
            }
        }

        tracing::info!(
            "Upload complete: {} files, {} bytes",
            uploaded,
            crate::config::format_bytes(bytes as usize),
        );
        Ok(())
    })
}

/// Create an S3 Bucket handle from resolved config.
pub fn create_bucket(config: &ResolvedUploadConfig) -> Result<Box<Bucket>> {
    let credentials = Credentials::new(
        Some(&config.access_key),
        Some(&config.secret_key),
        None,
        None,
        None,
    )
    .map_err(|e| anyhow::anyhow!("S3 credentials error: {e}"))?;

    let region = Region::Custom {
        region: config.region.clone(),
        endpoint: config.endpoint.clone(),
    };

    let bucket = Bucket::new(&config.bucket, region, credentials)
        .map_err(|e| anyhow::anyhow!("S3 bucket error: {e}"))?
        .with_path_style();

    Ok(bucket)
}

/// Upload a single file to the bucket via streaming.
///
/// Object key = `{prefix}/{relative_path}` where relative_path is
/// the file path relative to base_dir (e.g. `raw/works/shard_0000.parquet`).
async fn upload_file(
    bucket: &Bucket,
    prefix: &str,
    base_dir: &std::path::Path,
    local_path: &std::path::Path,
) -> Result<u64> {
    let relative = local_path
        .strip_prefix(base_dir)
        .with_context(|| {
            format!(
                "{} is not under {}",
                local_path.display(),
                base_dir.display()
            )
        })?
        .to_string_lossy();

    let key = if prefix.is_empty() {
        relative.to_string()
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), relative)
    };

    let file_len = tokio::fs::metadata(local_path)
        .await
        .with_context(|| format!("Cannot stat {}", local_path.display()))?
        .len();

    let mut file = tokio::fs::File::open(local_path)
        .await
        .with_context(|| format!("Cannot open {}", local_path.display()))?;

    let ct = if key.ends_with(".parquet") {
        "application/vnd.apache.parquet"
    } else {
        "application/json"
    };

    bucket
        .put_object_stream_with_content_type(&mut file, &key, ct)
        .await
        .map_err(|e| anyhow::anyhow!("S3 put_object_stream failed for {key}: {e}"))?;

    Ok(file_len)
}
