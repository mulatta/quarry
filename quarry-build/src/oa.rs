//! OpenAlex JSONL → Parquet pipeline.
//!
//! Supports two data sources:
//! - Local directory of .gz files (`build_oa_local`)
//! - S3 prefix (`build_oa_s3`)
//!
//! Architecture (S3 path):
//!   tokio async tasks (download) → spawn_blocking (decompress+parse)
//!   → mpsc channel → main thread (accumulate + sink write)
//!
//! Optional local cache (`--cache-dir`): downloaded .gz files are saved
//! with ETag sidecar files. On re-run, files with matching ETag are
//! read from cache instead of re-downloading.

use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use flate2::read::GzDecoder;

use crate::config::BuildConfig;
use crate::oa_json::{
    self, OaAuthor, OaCitation, OaCrosswalk, OaLineResult, OaTopic, OaWork,
};
use crate::parquet_sink::ParquetSink;
use crate::schema;

/// Stats from an OA build run.
pub struct OaBuildStats {
    pub num_works: usize,
    pub num_authors: usize,
    pub num_topics: usize,
    pub num_citations: usize,
    pub num_crosswalk: usize,
    pub num_files_processed: usize,
    pub num_failed_lines: usize,
    pub num_cache_hits: usize,
    pub elapsed_secs: f64,
}

// ── Internal helpers ──

struct OaSinks {
    works: ParquetSink,
    authors: ParquetSink,
    topics: ParquetSink,
    citations: ParquetSink,
    crosswalk: ParquetSink,
}

impl OaSinks {
    fn new(config: &BuildConfig) -> Self {
        let rg = config.row_group_size;
        let mr = config.max_rows_per_file;
        Self {
            works: ParquetSink::new(
                &config.works_dir(), "works",
                Arc::new(schema::works_schema()), rg, mr,
            ),
            authors: ParquetSink::new(
                &config.work_authors_dir(), "work_authors",
                Arc::new(schema::work_authors_schema()), rg, mr,
            ),
            topics: ParquetSink::new(
                &config.work_topics_dir(), "work_topics",
                Arc::new(schema::work_topics_schema()), rg, mr,
            ),
            citations: ParquetSink::new(
                &config.work_citations_dir(), "work_citations",
                Arc::new(schema::work_citations_schema()), rg, mr,
            ),
            crosswalk: ParquetSink::new(
                &config.id_crosswalk_dir(), "id_crosswalk",
                Arc::new(schema::id_crosswalk_schema()), rg, mr,
            ),
        }
    }

    fn finish(self) -> Result<(), Box<dyn std::error::Error>> {
        self.works.finish()?;
        self.authors.finish()?;
        self.topics.finish()?;
        self.citations.finish()?;
        self.crosswalk.finish()?;
        Ok(())
    }
}

#[derive(Default)]
struct OaCounters {
    works: usize,
    authors: usize,
    topics: usize,
    citations: usize,
    crosswalk: usize,
    failed_lines: usize,
    cache_hits: usize,
}

struct OaBatchAccumulator {
    works: Vec<OaWork>,
    authors: Vec<OaAuthor>,
    topics: Vec<OaTopic>,
    citations: Vec<OaCitation>,
    crosswalk: Vec<OaCrosswalk>,
    batch_size: usize,
}

impl OaBatchAccumulator {
    fn new(batch_size: usize) -> Self {
        Self {
            works: Vec::with_capacity(batch_size),
            authors: Vec::new(),
            topics: Vec::new(),
            citations: Vec::new(),
            crosswalk: Vec::new(),
            batch_size,
        }
    }

    fn push(&mut self, result: OaLineResult) {
        if let Some(cw) = result.crosswalk {
            self.crosswalk.push(cw);
        }
        self.citations.extend(result.citations);
        self.authors.extend(result.authors);
        self.topics.extend(result.topics);
        self.works.push(result.work);
    }

    fn should_flush(&self) -> bool {
        self.works.len() >= self.batch_size
    }

    fn flush(
        &mut self,
        sinks: &mut OaSinks,
        counters: &mut OaCounters,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.works.is_empty() {
            return Ok(());
        }

        sinks.works.write_batch(&oa_json::works_to_batch(&self.works))?;
        counters.works += self.works.len();
        self.works.clear();

        if !self.authors.is_empty() {
            sinks.authors.write_batch(&oa_json::authors_to_batch(&self.authors))?;
            counters.authors += self.authors.len();
            self.authors.clear();
        }

        if !self.topics.is_empty() {
            sinks.topics.write_batch(&oa_json::topics_to_batch(&self.topics))?;
            counters.topics += self.topics.len();
            self.topics.clear();
        }

        if !self.citations.is_empty() {
            sinks
                .citations
                .write_batch(&oa_json::citations_to_batch(&self.citations))?;
            counters.citations += self.citations.len();
            self.citations.clear();
        }

        if !self.crosswalk.is_empty() {
            sinks
                .crosswalk
                .write_batch(&oa_json::crosswalk_to_batch(&self.crosswalk))?;
            counters.crosswalk += self.crosswalk.len();
            self.crosswalk.clear();
        }

        Ok(())
    }
}

/// Process a gzip-compressed JSONL stream.
fn process_gz_reader<R: BufRead>(
    reader: R,
    t2_domains: &HashSet<String>,
    acc: &mut OaBatchAccumulator,
    sinks: &mut OaSinks,
    counters: &mut OaCounters,
) -> Result<(), Box<dyn std::error::Error>> {
    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => {
                counters.failed_lines += 1;
                continue;
            }
        };
        if line.is_empty() {
            continue;
        }

        match oa_json::parse_line(&line, t2_domains) {
            Ok(parsed) => {
                acc.push(parsed);
                if acc.should_flush() {
                    acc.flush(sinks, counters)?;
                }
            }
            Err(_) => {
                counters.failed_lines += 1;
            }
        }
    }
    Ok(())
}

/// Recursively collect .gz files from a directory.
fn collect_gz_files(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_dir() {
            collect_gz_files(&entry.path(), out)?;
        } else if ft.is_file() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.ends_with(".gz") {
                out.push(entry.path());
            }
        }
    }
    Ok(())
}

/// Parse a gz buffer (in-memory bytes) into parsed line results.
/// Runs decompression + JSON parsing — CPU-heavy, should be called
/// from spawn_blocking or a dedicated thread.
fn parse_gz_bytes(
    bytes: &[u8],
    t2_domains: &HashSet<String>,
) -> (Vec<OaLineResult>, usize) {
    let reader = BufReader::with_capacity(256 * 1024, GzDecoder::new(bytes));
    let mut results = Vec::new();
    let mut failed = 0usize;
    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => {
                failed += 1;
                continue;
            }
        };
        if line.is_empty() {
            continue;
        }
        match oa_json::parse_line(&line, t2_domains) {
            Ok(parsed) => results.push(parsed),
            Err(_) => {
                failed += 1;
            }
        }
    }
    (results, failed)
}

// ── Cache helpers ──

/// Derive a flat cache filename from an S3 object path.
/// e.g. "data/works/updated_date=2024-01-01/part_000.gz"
///   → "data__works__updated_date=2024-01-01__part_000.gz"
fn cache_filename(object_path: &str) -> String {
    object_path.replace('/', "__")
}

/// Check if a cached .gz file exists with a matching ETag.
fn cache_hit(cache_dir: &Path, object_path: &str, etag: &str) -> Option<PathBuf> {
    let name = cache_filename(object_path);
    let gz_path = cache_dir.join(&name);
    let etag_path = cache_dir.join(format!("{name}.etag"));

    if gz_path.exists() && etag_path.exists() {
        if let Ok(stored) = std::fs::read_to_string(&etag_path) {
            if stored.trim() == etag {
                return Some(gz_path);
            }
        }
    }
    None
}

/// Save downloaded bytes to cache with ETag sidecar.
fn cache_save(cache_dir: &Path, object_path: &str, etag: &str, bytes: &[u8]) {
    let name = cache_filename(object_path);
    let gz_path = cache_dir.join(&name);
    let etag_path = cache_dir.join(format!("{name}.etag"));

    // Best-effort: don't fail the pipeline if cache write fails
    if let Err(e) = std::fs::write(&gz_path, bytes) {
        eprintln!("oa: WARN: cache write failed for {name}: {e}");
        return;
    }
    if let Err(e) = std::fs::write(&etag_path, etag) {
        eprintln!("oa: WARN: etag write failed for {name}: {e}");
    }
}

// ── Public entry points ──

/// Build OA Parquet files from a local directory of .gz JSONL files.
pub fn build_oa_local(
    config: &BuildConfig,
    local_dir: &Path,
) -> Result<OaBuildStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let t2_domains = config.t2_domains_set();

    let mut gz_files: Vec<PathBuf> = Vec::new();
    collect_gz_files(local_dir, &mut gz_files)?;
    gz_files.sort();

    let num_files = gz_files.len();
    eprintln!("oa: found {num_files} .gz files in {}", local_dir.display());

    let mut sinks = OaSinks::new(config);
    let mut acc = OaBatchAccumulator::new(config.oa_batch_size);
    let mut counters = OaCounters::default();

    for (file_idx, gz_path) in gz_files.iter().enumerate() {
        let file = std::fs::File::open(gz_path)?;
        let reader = BufReader::with_capacity(256 * 1024, GzDecoder::new(file));
        process_gz_reader(reader, &t2_domains, &mut acc, &mut sinks, &mut counters)?;

        if (file_idx + 1) % 10 == 0 || file_idx + 1 == num_files {
            let elapsed = t0.elapsed().as_secs_f64();
            let total_w = counters.works + acc.works.len();
            let total_c = counters.citations + acc.citations.len();
            eprintln!(
                "oa: {}/{} files, {} works, {} citations ({:.1}s)",
                file_idx + 1,
                num_files,
                total_w,
                total_c,
                elapsed,
            );
        }
    }

    // Final flush
    acc.flush(&mut sinks, &mut counters)?;
    sinks.finish()?;

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "oa: done — {} works, {} authors, {} topics, {} citations, {} crosswalk in {:.1}s",
        counters.works,
        counters.authors,
        counters.topics,
        counters.citations,
        counters.crosswalk,
        elapsed,
    );
    if counters.failed_lines > 0 {
        eprintln!("oa: WARN: {} lines failed to parse", counters.failed_lines);
    }

    Ok(OaBuildStats {
        num_works: counters.works,
        num_authors: counters.authors,
        num_topics: counters.topics,
        num_citations: counters.citations,
        num_crosswalk: counters.crosswalk,
        num_files_processed: num_files,
        num_failed_lines: counters.failed_lines,
        num_cache_hits: 0,
        elapsed_secs: elapsed,
    })
}

/// Build OA Parquet files from S3.
///
/// Downloads .gz files concurrently (controlled by
/// `config.s3_download_concurrency`). Each file is:
/// 1. Downloaded (async I/O on tokio runtime)
/// 2. Decompressed + parsed (spawn_blocking — dedicated thread pool)
/// 3. Sent to main thread via mpsc for sequential sink writes
///
/// Optional `cache_dir`: downloaded .gz files are cached locally with
/// ETag sidecar files. On re-run, cache hits skip the S3 download.
pub fn build_oa_s3(
    config: &BuildConfig,
    s3_url: &str,
) -> Result<OaBuildStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let t2_domains: Arc<HashSet<String>> = Arc::new(config.t2_domains_set());
    let concurrency = config.s3_download_concurrency;
    let cache_dir = config.oa_cache_dir.clone();

    if let Some(ref dir) = cache_dir {
        std::fs::create_dir_all(dir)?;
        eprintln!("oa: cache dir: {}", dir.display());
    }

    let (bucket, prefix) = parse_s3_url(s3_url)?;
    eprintln!("oa: listing s3://{bucket}/{prefix}");

    let rt = tokio::runtime::Runtime::new()?;
    let store: Arc<dyn object_store::ObjectStore> = Arc::new(
        object_store::aws::AmazonS3Builder::from_env()
            .with_bucket_name(&bucket)
            .with_region("us-east-1")
            .with_skip_signature(true)
            .build()?,
    );

    // List all .gz objects under prefix
    let objects = rt.block_on(async {
        use futures::TryStreamExt;
        let prefix_path = object_store::path::Path::from(prefix.as_str());
        store
            .list(Some(&prefix_path))
            .try_collect::<Vec<_>>()
            .await
    })?;

    let gz_objects: Vec<_> = objects
        .into_iter()
        .filter(|o| o.location.as_ref().ends_with(".gz"))
        .collect();

    let num_files = gz_objects.len();
    eprintln!(
        "oa: found {num_files} .gz objects in s3://{bucket}/{prefix} (concurrency={concurrency})"
    );

    let mut sinks = OaSinks::new(config);
    let mut acc = OaBatchAccumulator::new(config.oa_batch_size);
    let mut counters = OaCounters::default();

    // Channel: download+parse tasks → main thread
    let (tx, mut rx) =
        tokio::sync::mpsc::channel::<(Vec<OaLineResult>, usize, bool)>(concurrency);

    // Spawn the producer: fans out downloads using buffer_unordered
    let store_clone = Arc::clone(&store);
    let t2_clone = Arc::clone(&t2_domains);
    let cache_clone = cache_dir.clone();
    rt.spawn(async move {
        use futures::StreamExt;

        let download_futures = gz_objects.into_iter().map(|obj| {
            let store = Arc::clone(&store_clone);
            let t2 = Arc::clone(&t2_clone);
            let cache = cache_clone.clone();
            async move {
                let obj_path = obj.location.as_ref().to_string();
                let etag = obj.e_tag.clone().unwrap_or_default();

                // Check cache first
                if let Some(ref cache_dir) = cache {
                    if let Some(cached_path) = cache_hit(cache_dir, &obj_path, &etag) {
                        // Cache hit: read from disk on blocking thread
                        let t2 = Arc::clone(&t2);
                        let result = tokio::task::spawn_blocking(move || {
                            match std::fs::read(&cached_path) {
                                Ok(bytes) => parse_gz_bytes(&bytes, &t2),
                                Err(_) => (Vec::new(), 1),
                            }
                        })
                        .await
                        .unwrap_or((Vec::new(), 1));
                        return (result.0, result.1, true); // cache_hit = true
                    }
                }

                // Download from S3
                let dl_result = async {
                    let result = store.get(&obj.location).await?;
                    result.bytes().await
                }
                .await;

                match dl_result {
                    Ok(bytes) => {
                        // Save to cache before parsing
                        if let Some(ref cache_dir) = cache {
                            cache_save(cache_dir, &obj_path, &etag, &bytes);
                        }

                        // Parse on blocking thread pool
                        let t2 = Arc::clone(&t2);
                        let result = tokio::task::spawn_blocking(move || {
                            parse_gz_bytes(&bytes, &t2)
                        })
                        .await
                        .unwrap_or((Vec::new(), 1));
                        (result.0, result.1, false)
                    }
                    Err(e) => {
                        eprintln!(
                            "oa: WARN: failed to download {}: {}",
                            obj.location, e
                        );
                        (Vec::new(), 1, false)
                    }
                }
            }
        });

        let mut stream =
            futures::stream::iter(download_futures).buffer_unordered(concurrency);

        while let Some(batch) = stream.next().await {
            if tx.send(batch).await.is_err() {
                break;
            }
        }
    });

    // Main thread: receive parsed batches, accumulate, flush to sinks
    let mut files_done = 0usize;
    while let Some((batch, failed, was_cached)) = rt.block_on(rx.recv()) {
        counters.failed_lines += failed;
        if was_cached {
            counters.cache_hits += 1;
        }
        for result in batch {
            acc.push(result);
            if acc.should_flush() {
                acc.flush(&mut sinks, &mut counters)?;
            }
        }
        files_done += 1;

        if files_done % 10 == 0 || files_done == num_files {
            let elapsed = t0.elapsed().as_secs_f64();
            let total_w = counters.works + acc.works.len();
            eprintln!(
                "oa: {}/{} s3 objects, {} works, {} cached ({:.1}s)",
                files_done, num_files, total_w, counters.cache_hits, elapsed,
            );
        }
    }

    acc.flush(&mut sinks, &mut counters)?;
    sinks.finish()?;

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "oa: done — {} works, {} citations, {} cache hits in {:.1}s",
        counters.works, counters.citations, counters.cache_hits, elapsed,
    );
    if counters.failed_lines > 0 {
        eprintln!("oa: WARN: {} lines failed to parse", counters.failed_lines);
    }

    Ok(OaBuildStats {
        num_works: counters.works,
        num_authors: counters.authors,
        num_topics: counters.topics,
        num_citations: counters.citations,
        num_crosswalk: counters.crosswalk,
        num_files_processed: num_files,
        num_failed_lines: counters.failed_lines,
        num_cache_hits: counters.cache_hits,
        elapsed_secs: elapsed,
    })
}

/// Parse "s3://bucket/prefix" → (bucket, prefix).
fn parse_s3_url(url: &str) -> Result<(String, String), Box<dyn std::error::Error>> {
    let stripped = url
        .strip_prefix("s3://")
        .ok_or("S3 URL must start with s3://")?;
    let (bucket, prefix) = stripped
        .split_once('/')
        .ok_or("S3 URL must have format s3://bucket/prefix")?;
    Ok((bucket.to_string(), prefix.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_s3_url() {
        let (b, p) = parse_s3_url("s3://openalex/data/works").unwrap();
        assert_eq!(b, "openalex");
        assert_eq!(p, "data/works");
    }

    #[test]
    fn test_parse_s3_url_error() {
        assert!(parse_s3_url("https://example.com").is_err());
        assert!(parse_s3_url("s3://nobucket").is_err());
    }

    #[test]
    fn test_cache_filename() {
        assert_eq!(
            cache_filename("data/works/updated_date=2024-01-01/part_000.gz"),
            "data__works__updated_date=2024-01-01__part_000.gz"
        );
    }

    #[test]
    fn test_cache_hit_miss() {
        let dir = std::env::temp_dir().join("quarry_test_cache");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        // No cached file → miss
        assert!(cache_hit(&dir, "test/file.gz", "etag123").is_none());

        // Write cache
        cache_save(&dir, "test/file.gz", "etag123", b"fake data");
        assert!(cache_hit(&dir, "test/file.gz", "etag123").is_some());

        // Different etag → miss
        assert!(cache_hit(&dir, "test/file.gz", "etag456").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
