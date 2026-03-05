//! S3-compatible remote storage operations (Cloudflare R2).
//!
//! Three modes of remote access:
//! - **Streaming upload**: background worker fed via `mpsc` channel during fetch/hive
//! - **Push**: diff-based local → remote sync (`papeline push`)
//! - **Pull**: diff-based remote → local sync (`papeline pull`)
//!
//! Sync strategy: filename + file_size comparison (no checksums needed because
//! parquet files are immutable in our append-only pipeline).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::task::{Context as TaskContext, Poll};

use anyhow::{Context, Result};
use futures_util::StreamExt;
use s3::creds::Credentials;
use s3::{Bucket, Region};
use tokio::io::{AsyncRead, ReadBuf};

use crate::config::{ResolvedUploadConfig, format_bytes};
use crate::oa::TABLES;
use crate::progress::ProgressReporter;

// ============================================================
// AsyncRead progress adapter
// ============================================================

/// Wraps an `AsyncRead` to report bytes read through a `ProgressReporter`.
struct ProgressReader<R> {
    inner: R,
    reporter: Arc<dyn ProgressReporter>,
}

impl<R: AsyncRead + Unpin> AsyncRead for ProgressReader<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut TaskContext<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let before = buf.filled().len();
        let result = Pin::new(&mut self.inner).poll_read(cx, buf);
        if let Poll::Ready(Ok(())) = &result {
            let delta = buf.filled().len() - before;
            if delta > 0 {
                self.reporter.inc(delta as u64);
            }
        }
        result
    }
}

// ============================================================
// Bucket creation
// ============================================================

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

// ============================================================
// Streaming upload worker (used by fetch --hive and hive --upload)
// ============================================================

/// Summary from the background upload worker.
pub struct UploadWorkerSummary {
    pub uploaded: usize,
    pub bytes: u64,
    pub failed: usize,
}

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
    std::thread::JoinHandle<Result<UploadWorkerSummary>>,
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
) -> Result<UploadWorkerSummary> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create upload tokio runtime")?;

    rt.block_on(async {
        let bucket = create_bucket(&config)?;
        let mut uploaded = 0usize;
        let mut bytes = 0u64;
        let mut failed = 0usize;

        while let Ok(path) = rx.recv() {
            match stream_upload_file(&bucket, &config.prefix, &base_dir, &path).await {
                Ok(size) => {
                    uploaded += 1;
                    bytes += size;
                    tracing::info!(
                        "auto-push: {} ({}) [{} files, {}]",
                        path.display(),
                        format_bytes(size as usize),
                        uploaded,
                        format_bytes(bytes as usize),
                    );
                }
                Err(e) => {
                    failed += 1;
                    tracing::error!("auto-push failed for {}: {e:#}", path.display());
                }
            }
        }

        tracing::info!(
            "Auto-push complete: {} uploaded, {} failed, {}",
            uploaded,
            failed,
            format_bytes(bytes as usize),
        );
        Ok(UploadWorkerSummary {
            uploaded,
            bytes,
            failed,
        })
    })
}

/// Upload a single file to the bucket via streaming (used by the background worker).
async fn stream_upload_file(
    bucket: &Bucket,
    prefix: &str,
    base_dir: &Path,
    local_path: &Path,
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

    let key = make_key(prefix, &relative);

    let file_len = tokio::fs::metadata(local_path)
        .await
        .with_context(|| format!("Cannot stat {}", local_path.display()))?
        .len();

    let mut file = tokio::fs::File::open(local_path)
        .await
        .with_context(|| format!("Cannot open {}", local_path.display()))?;

    let ct = content_type(&relative);

    bucket
        .put_object_stream_with_content_type(&mut file, &key, ct)
        .await
        .map_err(|e| anyhow::anyhow!("S3 put_object_stream failed for {key}: {e}"))?;

    Ok(file_len)
}

// ============================================================
// Diff-based push / pull
// ============================================================

/// Which directories to sync.
pub struct RemoteTargets {
    pub raw: bool,
    pub hive: bool,
}

/// Options for push/pull operations.
pub struct TransferOpts {
    pub targets: RemoteTargets,
    pub dry_run: bool,
    pub force: bool,
    pub concurrency: usize,
}

/// Result summary after push/pull completes.
pub struct TransferSummary {
    pub files_transferred: usize,
    pub bytes_transferred: u64,
    pub files_skipped: usize,
    pub files_failed: usize,
    pub failed_keys: Vec<String>,
}

/// State files that are always overwritten (small, mutable).
const STATE_FILES: &[&str] = &[
    ".manifest.json",
    ".state.json",
    ".sync_log.jsonl",
    ".hive_state.json",
];

// ---- File listing types ----

struct RemoteObject {
    /// Path relative to config prefix (e.g. "raw/works/shard_0000.parquet")
    relative: String,
    size: u64,
}

struct LocalFile {
    /// Path relative to output_dir (e.g. "raw/works/shard_0000.parquet")
    relative: String,
    absolute: PathBuf,
    size: u64,
}

// ---- List remote objects ----

/// List all objects under `{config_prefix}/{sub}/` in the bucket.
async fn list_remote(bucket: &Bucket, config_prefix: &str, sub: &str) -> Result<Vec<RemoteObject>> {
    let search_prefix = make_key(config_prefix, sub);
    // Ensure trailing slash for directory-like listing
    let search_prefix = if search_prefix.ends_with('/') || search_prefix.is_empty() {
        search_prefix
    } else {
        format!("{search_prefix}/")
    };

    let results = bucket
        .list(search_prefix.clone(), None)
        .await
        .map_err(|e| anyhow::anyhow!("S3 list failed for {search_prefix}: {e}"))?;

    let config_prefix_stripped = if config_prefix.is_empty() {
        String::new()
    } else {
        let mut p = config_prefix.trim_end_matches('/').to_string();
        p.push('/');
        p
    };

    let mut objects = Vec::new();
    for result in &results {
        for obj in &result.contents {
            let relative = obj
                .key
                .strip_prefix(&config_prefix_stripped)
                .unwrap_or(&obj.key);
            objects.push(RemoteObject {
                relative: relative.to_string(),
                size: obj.size,
            });
        }
    }

    Ok(objects)
}

// ---- List local files ----

/// Walk local directory and collect all sync-eligible files.
fn list_local(output_dir: &Path, sub: &str) -> Result<Vec<LocalFile>> {
    let base = output_dir.join(sub);
    if !base.exists() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();

    // State files at sub-directory root
    for entry in std::fs::read_dir(&base)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            // Include dotfiles (state) and .jsonl files
            if name.starts_with('.') || name.ends_with(".jsonl") {
                let meta = std::fs::metadata(&path)?;
                files.push(LocalFile {
                    relative: format!("{sub}/{name}"),
                    absolute: path,
                    size: meta.len(),
                });
            }
        }
    }

    // Table directories
    for table in TABLES {
        let table_dir = base.join(table);
        if !table_dir.exists() {
            continue;
        }
        walk_parquet_dir(&table_dir, &format!("{sub}/{table}"), &mut files)?;
    }

    Ok(files)
}

/// Recursively collect .parquet files from a directory.
fn walk_parquet_dir(dir: &Path, prefix: &str, files: &mut Vec<LocalFile>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        if path.is_dir() {
            // e.g. pub_year=2020/
            walk_parquet_dir(&path, &format!("{prefix}/{name}"), files)?;
        } else if name.ends_with(".parquet") {
            let meta = std::fs::metadata(&path)?;
            files.push(LocalFile {
                relative: format!("{prefix}/{name}"),
                absolute: path,
                size: meta.len(),
            });
        }
    }
    Ok(())
}

// ---- Diff computation ----

fn is_state_file(relative: &str) -> bool {
    STATE_FILES.iter().any(|sf| relative.ends_with(sf))
}

/// Files that need to be pushed (local → remote).
fn compute_push_diff<'a>(
    local: &'a [LocalFile],
    remote: &[RemoteObject],
    force: bool,
) -> Vec<&'a LocalFile> {
    if force {
        return local.iter().collect();
    }

    let remote_index: HashMap<&str, u64> = remote
        .iter()
        .map(|r| (r.relative.as_str(), r.size))
        .collect();

    local
        .iter()
        .filter(|lf| {
            if is_state_file(&lf.relative) {
                return true; // always push state files
            }
            match remote_index.get(lf.relative.as_str()) {
                None => true,               // missing remotely
                Some(&sz) => sz != lf.size, // size mismatch
            }
        })
        .collect()
}

/// Files that need to be pulled (remote → local).
fn compute_pull_diff<'a>(
    remote: &'a [RemoteObject],
    local: &[LocalFile],
    force: bool,
) -> Vec<&'a RemoteObject> {
    if force {
        return remote.iter().collect();
    }

    let local_index: HashMap<&str, u64> = local
        .iter()
        .map(|lf| (lf.relative.as_str(), lf.size))
        .collect();

    remote
        .iter()
        .filter(|ro| {
            if is_state_file(&ro.relative) {
                return true; // always pull state files
            }
            match local_index.get(ro.relative.as_str()) {
                None => true,               // missing locally
                Some(&sz) => sz != ro.size, // size mismatch
            }
        })
        .collect()
}

// ---- Transfer operations ----

fn content_type(relative: &str) -> &'static str {
    if relative.ends_with(".parquet") {
        "application/vnd.apache.parquet"
    } else {
        "application/json"
    }
}

fn make_key(prefix: &str, relative: &str) -> String {
    if prefix.is_empty() {
        relative.to_string()
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), relative)
    }
}

/// Upload a single file to the bucket (used by push).
async fn push_file(
    bucket: &Bucket,
    prefix: &str,
    lf: &LocalFile,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<u64> {
    let key = make_key(prefix, &lf.relative);

    let file = tokio::fs::File::open(&lf.absolute)
        .await
        .with_context(|| format!("Cannot open {}", lf.absolute.display()))?;

    reporter.upgrade_to_determinate(lf.size);
    let mut reader = ProgressReader {
        inner: file,
        reporter: Arc::clone(reporter),
    };

    bucket
        .put_object_stream_with_content_type(&mut reader, &key, content_type(&lf.relative))
        .await
        .map_err(|e| anyhow::anyhow!("S3 upload failed for {key}: {e}"))?;

    reporter.finish_and_clear();
    Ok(lf.size)
}

/// Download a single file from the bucket via streaming. Atomic: write to .tmp then rename.
async fn pull_file(
    bucket: &Bucket,
    prefix: &str,
    output_dir: &Path,
    ro: &RemoteObject,
    reporter: &Arc<dyn ProgressReporter>,
) -> Result<u64> {
    let local_path = output_dir.join(&ro.relative);
    let tmp_path = local_path.with_extension("tmp");

    if let Some(parent) = local_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("Cannot create dir {}", parent.display()))?;
    }

    let key = make_key(prefix, &ro.relative);
    reporter.upgrade_to_determinate(ro.size);

    let response = bucket
        .get_object_stream(&key)
        .await
        .map_err(|e| anyhow::anyhow!("S3 download failed for {key}: {e}"))?;

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("Cannot create {}", tmp_path.display()))?;

    let mut total = 0u64;
    let mut stream = response.bytes;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("Stream read error for {key}"))?;
        let len = chunk.len() as u64;
        total += len;
        reporter.inc(len);
        tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
            .await
            .with_context(|| format!("Write error for {}", tmp_path.display()))?;
    }

    tokio::fs::rename(&tmp_path, &local_path)
        .await
        .with_context(|| {
            format!(
                "Cannot rename {} → {}",
                tmp_path.display(),
                local_path.display()
            )
        })?;

    reporter.finish_and_clear();
    Ok(total)
}

// ============================================================
// Public API
// ============================================================

/// Push local files to R2 (upload new/changed).
pub fn run_push(
    config: &ResolvedUploadConfig,
    output_dir: &Path,
    opts: &TransferOpts,
    progress: &crate::progress::SharedProgress,
    cancelled: &Arc<AtomicBool>,
) -> Result<TransferSummary> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create tokio runtime")?;

    rt.block_on(async {
        let bucket = create_bucket(config)?;
        let subs = target_subs(&opts.targets);

        // Dry-run: collect across all subs, print, return early
        if opts.dry_run {
            let mut total = 0usize;
            let mut diff_count = 0usize;
            for sub in &subs {
                let local = list_local(output_dir, sub)?;
                let remote = list_remote(&bucket, &config.prefix, sub).await?;
                let diff = compute_push_diff(&local, &remote, opts.force);
                total += local.len();
                diff_count += diff.len();
                if !diff.is_empty() {
                    eprintln!("{sub}/: {} to transfer", diff.len());
                    for f in &diff {
                        eprintln!("  + {} ({})", f.relative, format_bytes(f.size as usize));
                    }
                }
            }
            eprintln!(
                "Push: {diff_count} to transfer, {} up to date",
                total - diff_count
            );
            return Ok(TransferSummary {
                files_transferred: 0,
                bytes_transferred: 0,
                files_skipped: total,
                files_failed: 0,
                failed_keys: Vec::new(),
            });
        }

        // Transfer: per-sub loop with dir_bar + file_bar
        let mut total_bytes = 0u64;
        let mut total_transferred = 0usize;
        let mut total_skipped = 0usize;
        let mut failed_keys_out = Vec::new();

        for sub in &subs {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }

            let local = list_local(output_dir, sub)?;
            let remote = list_remote(&bucket, &config.prefix, sub).await?;
            let sub_total = local.len();
            let to_push = compute_push_diff(&local, &remote, opts.force);
            let push_count = to_push.len();
            total_skipped += sub_total - push_count;

            if push_count == 0 {
                continue;
            }

            let dir_bar = progress.dir_bar(&format!("{sub}/"), push_count as u64, "files");

            let mut stream = futures_util::stream::iter(to_push.into_iter().map(|lf| {
                let bucket = &bucket;
                let prefix = &config.prefix;
                let key = lf.relative.clone();
                let file_bar = progress.file_bar(
                    lf.absolute
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or("?"),
                );
                async move {
                    let result = push_file(bucket, prefix, lf, &file_bar).await;
                    (key, result)
                }
            }))
            .buffer_unordered(opts.concurrency);

            while let Some((key, result)) = stream.next().await {
                if cancelled.load(Ordering::Relaxed) {
                    break;
                }
                match result {
                    Ok(size) => {
                        total_bytes += size;
                        total_transferred += 1;
                    }
                    Err(e) => {
                        tracing::error!("{key}: {e}");
                        failed_keys_out.push(key);
                    }
                }
                dir_bar.inc(1);
            }

            dir_bar.finish_and_clear();
        }

        Ok(TransferSummary {
            files_transferred: total_transferred,
            bytes_transferred: total_bytes,
            files_skipped: total_skipped,
            files_failed: failed_keys_out.len(),
            failed_keys: failed_keys_out,
        })
    })
}

/// Pull R2 files to local (download new/changed).
pub fn run_pull(
    config: &ResolvedUploadConfig,
    output_dir: &Path,
    opts: &TransferOpts,
    progress: &crate::progress::SharedProgress,
    cancelled: &Arc<AtomicBool>,
) -> Result<TransferSummary> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create tokio runtime")?;

    rt.block_on(async {
        let bucket = create_bucket(config)?;
        let subs = target_subs(&opts.targets);

        if opts.dry_run {
            let mut total = 0usize;
            let mut diff_count = 0usize;
            for sub in &subs {
                let remote = list_remote(&bucket, &config.prefix, sub).await?;
                let local = list_local(output_dir, sub)?;
                let diff = compute_pull_diff(&remote, &local, opts.force);
                total += remote.len();
                diff_count += diff.len();
                if !diff.is_empty() {
                    eprintln!("{sub}/: {} to transfer", diff.len());
                    for f in &diff {
                        eprintln!("  + {} ({})", f.relative, format_bytes(f.size as usize));
                    }
                }
            }
            eprintln!(
                "Pull: {diff_count} to transfer, {} up to date",
                total - diff_count
            );
            return Ok(TransferSummary {
                files_transferred: 0,
                bytes_transferred: 0,
                files_skipped: total,
                files_failed: 0,
                failed_keys: Vec::new(),
            });
        }

        let mut total_bytes = 0u64;
        let mut total_transferred = 0usize;
        let mut total_skipped = 0usize;
        let mut failed_keys_out = Vec::new();

        for sub in &subs {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }

            let remote = list_remote(&bucket, &config.prefix, sub).await?;
            let local = list_local(output_dir, sub)?;
            let sub_total = remote.len();
            let to_pull = compute_pull_diff(&remote, &local, opts.force);
            let pull_count = to_pull.len();
            total_skipped += sub_total - pull_count;

            if pull_count == 0 {
                continue;
            }

            let dir_bar = progress.dir_bar(&format!("{sub}/"), pull_count as u64, "files");

            let mut stream = futures_util::stream::iter(to_pull.into_iter().map(|ro| {
                let bucket = &bucket;
                let prefix = &config.prefix;
                let key = ro.relative.clone();
                let file_bar =
                    progress.file_bar(ro.relative.rsplit('/').next().unwrap_or(&ro.relative));
                async move {
                    let result = pull_file(bucket, prefix, output_dir, ro, &file_bar).await;
                    (key, result)
                }
            }))
            .buffer_unordered(opts.concurrency);

            while let Some((key, result)) = stream.next().await {
                if cancelled.load(Ordering::Relaxed) {
                    break;
                }
                match result {
                    Ok(size) => {
                        total_bytes += size;
                        total_transferred += 1;
                    }
                    Err(e) => {
                        tracing::error!("{key}: {e}");
                        failed_keys_out.push(key);
                    }
                }
                dir_bar.inc(1);
            }

            dir_bar.finish_and_clear();
        }

        Ok(TransferSummary {
            files_transferred: total_transferred,
            bytes_transferred: total_bytes,
            files_skipped: total_skipped,
            files_failed: failed_keys_out.len(),
            failed_keys: failed_keys_out,
        })
    })
}

fn target_subs(targets: &RemoteTargets) -> Vec<&'static str> {
    let mut subs = Vec::new();
    if targets.raw {
        subs.push("raw");
    }
    if targets.hive {
        subs.push("hive");
    }
    subs
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn local(relative: &str, size: u64) -> LocalFile {
        LocalFile {
            relative: relative.to_string(),
            absolute: PathBuf::from(relative),
            size,
        }
    }

    fn remote(relative: &str, size: u64) -> RemoteObject {
        RemoteObject {
            relative: relative.to_string(),
            size,
        }
    }

    #[test]
    fn push_diff_new_file() {
        let local_files = vec![local("raw/works/shard_0000.parquet", 1000)];
        let remote_files = vec![];
        let diff = compute_push_diff(&local_files, &remote_files, false);
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn push_diff_same_file_skipped() {
        let local_files = vec![local("raw/works/shard_0000.parquet", 1000)];
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let diff = compute_push_diff(&local_files, &remote_files, false);
        assert!(diff.is_empty());
    }

    #[test]
    fn push_diff_size_mismatch() {
        let local_files = vec![local("raw/works/shard_0000.parquet", 2000)];
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let diff = compute_push_diff(&local_files, &remote_files, false);
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn push_diff_state_always_pushed() {
        let local_files = vec![local("raw/.manifest.json", 500)];
        let remote_files = vec![remote("raw/.manifest.json", 500)];
        let diff = compute_push_diff(&local_files, &remote_files, false);
        assert_eq!(diff.len(), 1, "state files should always be pushed");
    }

    #[test]
    fn push_diff_force_includes_all() {
        let local_files = vec![local("raw/works/shard_0000.parquet", 1000)];
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let diff = compute_push_diff(&local_files, &remote_files, true);
        assert_eq!(diff.len(), 1, "force should include all files");
    }

    #[test]
    fn pull_diff_new_file() {
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let local_files = vec![];
        let diff = compute_pull_diff(&remote_files, &local_files, false);
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn pull_diff_same_file_skipped() {
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let local_files = vec![local("raw/works/shard_0000.parquet", 1000)];
        let diff = compute_pull_diff(&remote_files, &local_files, false);
        assert!(diff.is_empty());
    }

    #[test]
    fn pull_diff_state_always_pulled() {
        let remote_files = vec![remote("hive/.hive_state.json", 200)];
        let local_files = vec![local("hive/.hive_state.json", 200)];
        let diff = compute_pull_diff(&remote_files, &local_files, false);
        assert_eq!(diff.len(), 1, "state files should always be pulled");
    }

    #[test]
    fn pull_diff_force_includes_all() {
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let local_files = vec![local("raw/works/shard_0000.parquet", 1000)];
        let diff = compute_pull_diff(&remote_files, &local_files, true);
        assert_eq!(diff.len(), 1, "force should include all files");
    }

    #[test]
    fn is_state_file_matches() {
        assert!(is_state_file("raw/.manifest.json"));
        assert!(is_state_file("raw/.state.json"));
        assert!(is_state_file("raw/.sync_log.jsonl"));
        assert!(is_state_file("hive/.hive_state.json"));
        assert!(!is_state_file("raw/works/shard_0000.parquet"));
    }

    #[test]
    fn content_type_parquet() {
        assert_eq!(
            content_type("raw/works/shard_0.parquet"),
            "application/vnd.apache.parquet"
        );
        assert_eq!(content_type("raw/.manifest.json"), "application/json");
    }

    #[test]
    fn make_key_with_prefix() {
        assert_eq!(make_key("data", "raw/a.parquet"), "data/raw/a.parquet");
        assert_eq!(make_key("data/", "raw/a.parquet"), "data/raw/a.parquet");
        assert_eq!(make_key("", "raw/a.parquet"), "raw/a.parquet");
    }

    #[test]
    fn list_local_missing_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        let files = list_local(dir.path(), "raw").unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn list_local_with_state_and_parquet() {
        let dir = tempfile::TempDir::new().unwrap();
        let raw = dir.path().join("raw");
        std::fs::create_dir_all(raw.join("works")).unwrap();
        std::fs::write(raw.join(".manifest.json"), "{}").unwrap();
        std::fs::write(raw.join("works/shard_0000.parquet"), "fake").unwrap();

        let files = list_local(dir.path(), "raw").unwrap();
        assert_eq!(files.len(), 2);

        let relatives: Vec<&str> = files.iter().map(|f| f.relative.as_str()).collect();
        assert!(relatives.contains(&"raw/.manifest.json"));
        assert!(relatives.contains(&"raw/works/shard_0000.parquet"));
    }
}
