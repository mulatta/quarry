//! OpenAlex JSONL → Parquet pipeline.
//!
//! Supports two data sources:
//! - Local directory of .gz files (`build_oa_local`)
//! - S3 prefix (`build_oa_s3`)
//!
//! Architecture (S3 path):
//!   tokio async tasks (download) → spawn_blocking (decompress+parse+Arrow)
//!   → mpsc channel → main thread (sequential sink writes)
//!
//! Optional Parquet cache (`--cache-dir`): parsed output is saved as
//! per-file Parquet files with ETag sidecar. On re-run, files with
//! matching ETag skip both download AND parsing.
//! A cache version prefix (`v{N}/`) invalidates stale cache when
//! schema or parse logic changes.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use arrow::compute;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use flate2::read::GzDecoder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::config::BuildConfig;
use crate::oa_json::{self, OaAuthor, OaCitation, OaCrosswalk, OaLineResult, OaTopic, OaWork};
use crate::parquet_sink::ParquetSink;
use crate::schema;

/// Bump when schema or parse logic changes to invalidate stale cache.
const CACHE_VERSION: u32 = 1;

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
                &config.works_dir(),
                "works",
                Arc::new(schema::works_schema()),
                rg,
                mr,
            ),
            authors: ParquetSink::new(
                &config.work_authors_dir(),
                "work_authors",
                Arc::new(schema::work_authors_schema()),
                rg,
                mr,
            ),
            topics: ParquetSink::new(
                &config.work_topics_dir(),
                "work_topics",
                Arc::new(schema::work_topics_schema()),
                rg,
                mr,
            ),
            citations: ParquetSink::new(
                &config.work_citations_dir(),
                "work_citations",
                Arc::new(schema::work_citations_schema()),
                rg,
                mr,
            ),
            crosswalk: ParquetSink::new(
                &config.id_crosswalk_dir(),
                "id_crosswalk",
                Arc::new(schema::id_crosswalk_schema()),
                rg,
                mr,
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

/// Parsed output from a single S3 file, ready for sink writes.
struct FileOutput {
    works: RecordBatch,
    authors: RecordBatch,
    topics: RecordBatch,
    citations: RecordBatch,
    crosswalk: RecordBatch,
    failed_lines: usize,
    was_cached: bool,
}

impl FileOutput {
    fn empty() -> Self {
        Self {
            works: RecordBatch::new_empty(Arc::new(schema::works_schema())),
            authors: RecordBatch::new_empty(Arc::new(schema::work_authors_schema())),
            topics: RecordBatch::new_empty(Arc::new(schema::work_topics_schema())),
            citations: RecordBatch::new_empty(Arc::new(schema::work_citations_schema())),
            crosswalk: RecordBatch::new_empty(Arc::new(schema::id_crosswalk_schema())),
            failed_lines: 0,
            was_cached: false,
        }
    }
}

// ── Accumulator (used by build_oa_local only) ──

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

        sinks
            .works
            .write_batch(&oa_json::works_to_batch(&self.works))?;
        counters.works += self.works.len();
        self.works.clear();

        if !self.authors.is_empty() {
            sinks
                .authors
                .write_batch(&oa_json::authors_to_batch(&self.authors))?;
            counters.authors += self.authors.len();
            self.authors.clear();
        }

        if !self.topics.is_empty() {
            sinks
                .topics
                .write_batch(&oa_json::topics_to_batch(&self.topics))?;
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

/// Process a gzip-compressed JSONL stream (used by build_oa_local).
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

// ── gz → RecordBatch conversion ──

/// Parse a gz buffer into RecordBatches.
/// CPU-heavy — should be called from spawn_blocking.
fn parse_gz_to_batches(bytes: &[u8], t2_domains: &HashSet<String>) -> FileOutput {
    let reader = BufReader::with_capacity(256 * 1024, GzDecoder::new(bytes));
    let mut works = Vec::new();
    let mut authors = Vec::new();
    let mut topics = Vec::new();
    let mut citations = Vec::new();
    let mut crosswalk = Vec::new();
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
            Ok(parsed) => {
                works.push(parsed.work);
                authors.extend(parsed.authors);
                topics.extend(parsed.topics);
                citations.extend(parsed.citations);
                if let Some(cw) = parsed.crosswalk {
                    crosswalk.push(cw);
                }
            }
            Err(_) => {
                failed += 1;
            }
        }
    }

    FileOutput {
        works: oa_json::works_to_batch(&works),
        authors: oa_json::authors_to_batch(&authors),
        topics: oa_json::topics_to_batch(&topics),
        citations: oa_json::citations_to_batch(&citations),
        crosswalk: oa_json::crosswalk_to_batch(&crosswalk),
        failed_lines: failed,
        was_cached: false,
    }
}

// ── Parquet cache helpers ──

/// Versioned cache subdirectory for a single S3 object.
/// e.g. "data/works/updated_date=2024-01-01/part_000.gz"
///   → "cache_dir/v1/data__works__updated_date=2024-01-01__part_000"
fn cache_obj_dir(cache_dir: &Path, object_path: &str) -> PathBuf {
    let flat = object_path.replace('/', "__");
    let stem = flat.strip_suffix(".gz").unwrap_or(&flat);
    cache_dir.join(format!("v{CACHE_VERSION}")).join(stem)
}

/// Check if cached Parquet output exists with matching ETag.
fn cache_hit(cache_dir: &Path, object_path: &str, etag: &str) -> Option<PathBuf> {
    let dir = cache_obj_dir(cache_dir, object_path);
    let etag_path = dir.join("etag");
    if let Ok(stored) = std::fs::read_to_string(etag_path) {
        if stored.trim() == etag && dir.join("works.parquet").exists() {
            return Some(dir);
        }
    }
    None
}

/// Write a single RecordBatch as a Parquet file (fast zstd-1).
fn write_cache_parquet(
    path: &Path,
    batch: &RecordBatch,
) -> Result<(), Box<dyn std::error::Error>> {
    if batch.num_rows() == 0 {
        return Ok(());
    }
    let file = File::create(path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(1).unwrap()))
        .build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    writer.write(batch)?;
    writer.close()?;
    Ok(())
}

/// Read a cached Parquet file into a single RecordBatch.
fn read_cache_parquet(path: &Path, schema: Arc<Schema>) -> RecordBatch {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return RecordBatch::new_empty(schema),
    };
    let reader = match ParquetRecordBatchReaderBuilder::try_new(file) {
        Ok(b) => match b.build() {
            Ok(r) => r,
            Err(_) => return RecordBatch::new_empty(schema),
        },
        Err(_) => return RecordBatch::new_empty(schema),
    };
    let batches: Vec<RecordBatch> = reader.filter_map(|r| r.ok()).collect();
    if batches.is_empty() {
        return RecordBatch::new_empty(schema);
    }
    compute::concat_batches(&schema, &batches)
        .unwrap_or_else(|_| RecordBatch::new_empty(schema))
}

/// Read all cached Parquet files for an S3 object.
fn cache_read(dir: &Path) -> FileOutput {
    FileOutput {
        works: read_cache_parquet(
            &dir.join("works.parquet"),
            Arc::new(schema::works_schema()),
        ),
        authors: read_cache_parquet(
            &dir.join("authors.parquet"),
            Arc::new(schema::work_authors_schema()),
        ),
        topics: read_cache_parquet(
            &dir.join("topics.parquet"),
            Arc::new(schema::work_topics_schema()),
        ),
        citations: read_cache_parquet(
            &dir.join("citations.parquet"),
            Arc::new(schema::work_citations_schema()),
        ),
        crosswalk: read_cache_parquet(
            &dir.join("crosswalk.parquet"),
            Arc::new(schema::id_crosswalk_schema()),
        ),
        failed_lines: 0,
        was_cached: true,
    }
}

/// Save parsed RecordBatches as cached Parquet files with ETag sidecar.
fn cache_save(dir: &Path, etag: &str, output: &FileOutput) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("oa: WARN: cache mkdir failed: {e}");
        return;
    }
    // Best-effort: don't fail the pipeline on cache write errors
    let _ = write_cache_parquet(&dir.join("works.parquet"), &output.works);
    let _ = write_cache_parquet(&dir.join("authors.parquet"), &output.authors);
    let _ = write_cache_parquet(&dir.join("topics.parquet"), &output.topics);
    let _ = write_cache_parquet(&dir.join("citations.parquet"), &output.citations);
    let _ = write_cache_parquet(&dir.join("crosswalk.parquet"), &output.crosswalk);
    let _ = std::fs::write(dir.join("etag"), etag);
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
/// 2. Decompressed + parsed + converted to Arrow batches
///    (spawn_blocking — dedicated thread pool)
/// 3. Sent to main thread via mpsc for sequential sink writes
///
/// Optional `cache_dir`: parsed output is cached as Parquet files
/// (one per table per S3 object) with ETag sidecar. On re-run,
/// cache hits skip both S3 download AND gz parsing.
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
        eprintln!("oa: parquet cache dir: {} (v{CACHE_VERSION})", dir.display());
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
    let mut counters = OaCounters::default();

    // Channel: download+parse tasks → main thread
    let (tx, mut rx) = tokio::sync::mpsc::channel::<FileOutput>(concurrency);

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

                // Check cache first — reads Parquet (no gz parsing)
                if let Some(ref cache_dir) = cache {
                    if let Some(cached_dir) = cache_hit(cache_dir, &obj_path, &etag) {
                        let output = tokio::task::spawn_blocking(move || cache_read(&cached_dir))
                            .await
                            .unwrap_or_else(|_| FileOutput::empty());
                        return output;
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
                        // Parse on blocking thread pool
                        let t2 = Arc::clone(&t2);
                        let cache = cache.clone();
                        let obj_path = obj_path.clone();
                        let etag = etag.clone();
                        tokio::task::spawn_blocking(move || {
                            let output = parse_gz_to_batches(&bytes, &t2);
                            // Save parsed output as Parquet cache
                            if let Some(ref cache_dir) = cache {
                                let dir = cache_obj_dir(cache_dir, &obj_path);
                                cache_save(&dir, &etag, &output);
                            }
                            output
                        })
                        .await
                        .unwrap_or_else(|_| FileOutput::empty())
                    }
                    Err(e) => {
                        eprintln!("oa: WARN: failed to download {}: {}", obj.location, e);
                        FileOutput::empty()
                    }
                }
            }
        });

        let mut stream =
            futures::stream::iter(download_futures).buffer_unordered(concurrency);

        while let Some(output) = stream.next().await {
            if tx.send(output).await.is_err() {
                break;
            }
        }
    });

    // Main thread: receive RecordBatches, write directly to sinks
    let mut files_done = 0usize;
    while let Some(output) = rt.block_on(rx.recv()) {
        counters.failed_lines += output.failed_lines;
        if output.was_cached {
            counters.cache_hits += 1;
        }

        sinks.works.write_batch(&output.works)?;
        counters.works += output.works.num_rows();

        sinks.authors.write_batch(&output.authors)?;
        counters.authors += output.authors.num_rows();

        sinks.topics.write_batch(&output.topics)?;
        counters.topics += output.topics.num_rows();

        sinks.citations.write_batch(&output.citations)?;
        counters.citations += output.citations.num_rows();

        sinks.crosswalk.write_batch(&output.crosswalk)?;
        counters.crosswalk += output.crosswalk.num_rows();

        files_done += 1;

        if files_done % 10 == 0 || files_done == num_files {
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!(
                "oa: {}/{} s3 objects, {} works, {} cached ({:.1}s)",
                files_done, num_files, counters.works, counters.cache_hits, elapsed,
            );
        }
    }

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
    fn test_cache_obj_dir() {
        let dir = PathBuf::from("/tmp/cache");
        let result = cache_obj_dir(&dir, "data/works/updated_date=2024-01-01/part_000.gz");
        assert_eq!(
            result,
            PathBuf::from("/tmp/cache/v1/data__works__updated_date=2024-01-01__part_000")
        );
    }

    #[test]
    fn test_cache_hit_miss() {
        let dir = std::env::temp_dir().join("quarry_test_pq_cache");
        let _ = std::fs::remove_dir_all(&dir);

        // No cached files → miss
        assert!(cache_hit(&dir, "test/file.gz", "etag123").is_none());

        // Create fake cache directory with etag + works.parquet
        let obj_dir = cache_obj_dir(&dir, "test/file.gz");
        std::fs::create_dir_all(&obj_dir).unwrap();
        std::fs::write(obj_dir.join("etag"), "etag123").unwrap();

        // Still miss — no works.parquet
        assert!(cache_hit(&dir, "test/file.gz", "etag123").is_none());

        // Create works.parquet (can be empty for this test)
        std::fs::write(obj_dir.join("works.parquet"), b"fake").unwrap();
        assert!(cache_hit(&dir, "test/file.gz", "etag123").is_some());

        // Different etag → miss
        assert!(cache_hit(&dir, "test/file.gz", "etag456").is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
