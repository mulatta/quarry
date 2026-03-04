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
use std::sync::mpsc;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use s3::creds::Credentials;
use s3::{Bucket, Region};

use crate::config::{ResolvedUploadConfig, format_bytes};
use crate::oa::TABLES;

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
            match stream_upload_file(&bucket, &config.prefix, &base_dir, &path).await {
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
            format_bytes(bytes as usize),
        );
        Ok(())
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

/// Result summary after push/pull completes.
pub struct TransferSummary {
    pub files_transferred: usize,
    pub bytes_transferred: u64,
    pub files_skipped: usize,
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
fn compute_push_diff<'a>(local: &'a [LocalFile], remote: &[RemoteObject]) -> Vec<&'a LocalFile> {
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
fn compute_pull_diff<'a>(remote: &'a [RemoteObject], local: &[LocalFile]) -> Vec<&'a RemoteObject> {
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
    _output_dir: &Path,
    lf: &LocalFile,
) -> Result<u64> {
    let key = make_key(prefix, &lf.relative);

    let mut file = tokio::fs::File::open(&lf.absolute)
        .await
        .with_context(|| format!("Cannot open {}", lf.absolute.display()))?;

    bucket
        .put_object_stream_with_content_type(&mut file, &key, content_type(&lf.relative))
        .await
        .map_err(|e| anyhow::anyhow!("S3 upload failed for {key}: {e}"))?;

    tracing::info!(
        "pushed {} ({})",
        lf.relative,
        format_bytes(lf.size as usize)
    );
    Ok(lf.size)
}

/// Download a single file from the bucket. Atomic: write to .tmp then rename.
async fn pull_file(
    bucket: &Bucket,
    prefix: &str,
    output_dir: &Path,
    ro: &RemoteObject,
) -> Result<u64> {
    let local_path = output_dir.join(&ro.relative);
    let tmp_path = local_path.with_extension("tmp");

    if let Some(parent) = local_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("Cannot create dir {}", parent.display()))?;
    }

    let key = make_key(prefix, &ro.relative);

    let response = bucket
        .get_object(&key)
        .await
        .map_err(|e| anyhow::anyhow!("S3 download failed for {key}: {e}"))?;

    tokio::fs::write(&tmp_path, response.bytes())
        .await
        .with_context(|| format!("Cannot write {}", tmp_path.display()))?;

    tokio::fs::rename(&tmp_path, &local_path)
        .await
        .with_context(|| {
            format!(
                "Cannot rename {} → {}",
                tmp_path.display(),
                local_path.display()
            )
        })?;

    tracing::info!(
        "pulled {} ({})",
        ro.relative,
        format_bytes(ro.size as usize)
    );
    Ok(ro.size)
}

// ============================================================
// Public API
// ============================================================

/// Push local files to R2 (upload new/changed).
pub fn run_push(
    config: &ResolvedUploadConfig,
    output_dir: &Path,
    targets: &RemoteTargets,
    dry_run: bool,
    concurrency: usize,
) -> Result<TransferSummary> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create tokio runtime")?;

    rt.block_on(async {
        let bucket = create_bucket(config)?;
        let subs = target_subs(targets);

        let mut all_local = Vec::new();
        let mut all_remote = Vec::new();
        for sub in &subs {
            all_local.extend(list_local(output_dir, sub)?);
            all_remote.extend(list_remote(&bucket, &config.prefix, sub).await?);
        }

        let to_push = compute_push_diff(&all_local, &all_remote);
        let total_files = all_local.len();
        let push_count = to_push.len();

        if dry_run {
            println!(
                "Push dry run: {} to transfer, {} up to date",
                push_count,
                total_files - push_count
            );
            for f in &to_push {
                println!("  + {} ({})", f.relative, format_bytes(f.size as usize));
            }
            return Ok(TransferSummary {
                files_transferred: 0,
                bytes_transferred: 0,
                files_skipped: total_files,
            });
        }

        if push_count == 0 {
            return Ok(TransferSummary {
                files_transferred: 0,
                bytes_transferred: 0,
                files_skipped: total_files,
            });
        }

        tracing::info!(
            "Pushing {} files ({} up to date)",
            push_count,
            total_files - push_count
        );

        let results: Vec<Result<u64>> = futures_util::stream::iter(to_push.into_iter().map(|lf| {
            let bucket = &bucket;
            let prefix = &config.prefix;
            async move { push_file(bucket, prefix, output_dir, lf).await }
        }))
        .buffer_unordered(concurrency)
        .collect()
        .await;

        let mut bytes = 0u64;
        let mut transferred = 0usize;
        for r in results {
            bytes += r?;
            transferred += 1;
        }

        Ok(TransferSummary {
            files_transferred: transferred,
            bytes_transferred: bytes,
            files_skipped: total_files - transferred,
        })
    })
}

/// Pull R2 files to local (download new/changed).
pub fn run_pull(
    config: &ResolvedUploadConfig,
    output_dir: &Path,
    targets: &RemoteTargets,
    dry_run: bool,
    concurrency: usize,
) -> Result<TransferSummary> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("Failed to create tokio runtime")?;

    rt.block_on(async {
        let bucket = create_bucket(config)?;
        let subs = target_subs(targets);

        let mut all_local = Vec::new();
        let mut all_remote = Vec::new();
        for sub in &subs {
            all_remote.extend(list_remote(&bucket, &config.prefix, sub).await?);
            all_local.extend(list_local(output_dir, sub)?);
        }

        let to_pull = compute_pull_diff(&all_remote, &all_local);
        let total_files = all_remote.len();
        let pull_count = to_pull.len();

        if dry_run {
            println!(
                "Pull dry run: {} to transfer, {} up to date",
                pull_count,
                total_files - pull_count
            );
            for f in &to_pull {
                println!("  + {} ({})", f.relative, format_bytes(f.size as usize));
            }
            return Ok(TransferSummary {
                files_transferred: 0,
                bytes_transferred: 0,
                files_skipped: total_files,
            });
        }

        if pull_count == 0 {
            return Ok(TransferSummary {
                files_transferred: 0,
                bytes_transferred: 0,
                files_skipped: total_files,
            });
        }

        tracing::info!(
            "Pulling {} files ({} up to date)",
            pull_count,
            total_files - pull_count
        );

        let results: Vec<Result<u64>> = futures_util::stream::iter(to_pull.into_iter().map(|ro| {
            let bucket = &bucket;
            let prefix = &config.prefix;
            async move { pull_file(bucket, prefix, output_dir, ro).await }
        }))
        .buffer_unordered(concurrency)
        .collect()
        .await;

        let mut bytes = 0u64;
        let mut transferred = 0usize;
        for r in results {
            bytes += r?;
            transferred += 1;
        }

        Ok(TransferSummary {
            files_transferred: transferred,
            bytes_transferred: bytes,
            files_skipped: total_files - transferred,
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
        let diff = compute_push_diff(&local_files, &remote_files);
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn push_diff_same_file_skipped() {
        let local_files = vec![local("raw/works/shard_0000.parquet", 1000)];
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let diff = compute_push_diff(&local_files, &remote_files);
        assert!(diff.is_empty());
    }

    #[test]
    fn push_diff_size_mismatch() {
        let local_files = vec![local("raw/works/shard_0000.parquet", 2000)];
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let diff = compute_push_diff(&local_files, &remote_files);
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn push_diff_state_always_pushed() {
        let local_files = vec![local("raw/.manifest.json", 500)];
        let remote_files = vec![remote("raw/.manifest.json", 500)];
        let diff = compute_push_diff(&local_files, &remote_files);
        assert_eq!(diff.len(), 1, "state files should always be pushed");
    }

    #[test]
    fn pull_diff_new_file() {
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let local_files = vec![];
        let diff = compute_pull_diff(&remote_files, &local_files);
        assert_eq!(diff.len(), 1);
    }

    #[test]
    fn pull_diff_same_file_skipped() {
        let remote_files = vec![remote("raw/works/shard_0000.parquet", 1000)];
        let local_files = vec![local("raw/works/shard_0000.parquet", 1000)];
        let diff = compute_pull_diff(&remote_files, &local_files);
        assert!(diff.is_empty());
    }

    #[test]
    fn pull_diff_state_always_pulled() {
        let remote_files = vec![remote("hive/.hive_state.json", 200)];
        let local_files = vec![local("hive/.hive_state.json", 200)];
        let diff = compute_pull_diff(&remote_files, &local_files);
        assert_eq!(diff.len(), 1, "state files should always be pulled");
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
