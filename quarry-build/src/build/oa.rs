//! OpenAlex JSONL → PostgreSQL pipeline.
//!
//! Supports two data sources:
//! - Local directory of .gz files (`build_oa_local`)
//! - S3 prefix (`build_oa_s3`)
//!
//! Architecture (S3 path):
//!   tokio async tasks (download) → spawn_blocking (decompress+parse)
//!   → mpsc channel → main thread (sequential PG COPY writes)

use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use flate2::read::GzDecoder;

use crate::build::config::BuildConfig;
use crate::build::oa_json::{self, OaAuthor, OaCitation, OaCrosswalk, OaTopic, OaWork};
use crate::build::pg_sink::PgSink;

/// Stats from an OA build run.
pub struct OaBuildStats {
    pub num_works: usize,
    pub num_authors: usize,
    pub num_topics: usize,
    pub num_citations: usize,
    pub num_crosswalk: usize,
    pub num_files_processed: usize,
    pub num_failed_lines: usize,
    pub elapsed_secs: f64,
}

/// Parsed output from a single file, ready for PG writes.
struct FileOutput {
    works: Vec<OaWork>,
    authors: Vec<OaAuthor>,
    topics: Vec<OaTopic>,
    citations: Vec<OaCitation>,
    crosswalk: Vec<OaCrosswalk>,
    failed_lines: usize,
    filename: String,
}

impl FileOutput {
    fn empty(filename: String) -> Self {
        Self {
            works: Vec::new(),
            authors: Vec::new(),
            topics: Vec::new(),
            citations: Vec::new(),
            crosswalk: Vec::new(),
            failed_lines: 0,
            filename,
        }
    }
}

/// Parse a gz buffer into struct vecs.
fn parse_gz_to_structs(
    bytes: &[u8],
    t2_domains: &HashSet<String>,
    filename: String,
) -> FileOutput {
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
        works,
        authors,
        topics,
        citations,
        crosswalk,
        failed_lines: failed,
        filename,
    }
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

/// Write a FileOutput to PG via COPY within a transaction.
fn write_file_output(
    sink: &mut PgSink,
    output: &FileOutput,
    counters: &mut OaCounters,
) -> Result<(), Box<dyn std::error::Error>> {
    sink.begin()?;
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        sink.copy_works(&output.works)?;
        sink.copy_work_authors(&output.authors)?;
        sink.copy_work_topics(&output.topics)?;
        sink.copy_work_citations(&output.citations)?;
        sink.copy_id_crosswalk(&output.crosswalk)?;
        sink.mark_progress("oa", &output.filename, output.works.len() as i64)?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            sink.commit()?;
            counters.works += output.works.len();
            counters.authors += output.authors.len();
            counters.topics += output.topics.len();
            counters.citations += output.citations.len();
            counters.crosswalk += output.crosswalk.len();
            counters.failed_lines += output.failed_lines;
            Ok(())
        }
        Err(e) => {
            eprintln!("oa: WARN: write failed for {}: {}", output.filename, crate::err_chain(&*e));
            sink.rollback()?;
            counters.failed_lines += output.failed_lines;
            Err(e)
        }
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
}

/// Build OA from a local directory of .gz JSONL files → PG.
pub fn build_oa_local(
    config: &BuildConfig,
    local_dir: &Path,
    pg_conninfo: &str,
) -> Result<OaBuildStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let t2_domains = config.t2_domains_set();

    let mut gz_files: Vec<PathBuf> = Vec::new();
    collect_gz_files(local_dir, &mut gz_files)?;
    gz_files.sort();

    let num_files = gz_files.len();
    eprintln!("oa: found {num_files} .gz files in {}", local_dir.display());

    let mut sink = PgSink::connect(pg_conninfo)?;
    let mut counters = OaCounters::default();

    for (file_idx, gz_path) in gz_files.iter().enumerate() {
        let filename = gz_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();

        if sink.is_done("oa", &filename)? {
            if (file_idx + 1) % 100 == 0 {
                eprintln!("oa: {}/{num_files} files (skipping done)", file_idx + 1);
            }
            continue;
        }

        let bytes = std::fs::read(gz_path)?;
        let output = parse_gz_to_structs(&bytes, &t2_domains, filename);
        write_file_output(&mut sink, &output, &mut counters)?;

        if (file_idx + 1).is_multiple_of(10) || file_idx + 1 == num_files {
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!(
                "oa: {}/{num_files} files, {} works, {} citations ({elapsed:.1}s)",
                file_idx + 1,
                counters.works,
                counters.citations,
            );
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "oa: done — {} works, {} citations in {elapsed:.1}s",
        counters.works, counters.citations,
    );

    Ok(OaBuildStats {
        num_works: counters.works,
        num_authors: counters.authors,
        num_topics: counters.topics,
        num_citations: counters.citations,
        num_crosswalk: counters.crosswalk,
        num_files_processed: num_files,
        num_failed_lines: counters.failed_lines,
        elapsed_secs: elapsed,
    })
}

/// Build OA from S3 → PG.
pub fn build_oa_s3(
    config: &BuildConfig,
    s3_url: &str,
    pg_conninfo: &str,
) -> Result<OaBuildStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    let t2_domains: Arc<HashSet<String>> = Arc::new(config.t2_domains_set());
    let concurrency = config.effective_s3_concurrency();

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

    let mut sink = PgSink::connect(pg_conninfo)?;
    let mut counters = OaCounters::default();

    // Channel: download+parse tasks → main thread
    let (tx, mut rx) = tokio::sync::mpsc::channel::<FileOutput>(concurrency);

    let max_retries = config.fetch_max_retries;
    let initial_backoff_ms = config.fetch_initial_backoff_ms;
    let max_backoff_ms = config.fetch_max_backoff_ms;

    let store_clone = Arc::clone(&store);
    let t2_clone = Arc::clone(&t2_domains);
    let pg_conninfo_owned = pg_conninfo.to_string();
    let producer_handle: tokio::task::JoinHandle<Result<(), String>> = rt.spawn(async move {
        use futures::StreamExt;

        // Check progress upfront to skip already-done files.
        // Fail-closed: if we can't read progress, abort rather than risk PK conflicts.
        let pg_conn = pg_conninfo_owned.clone();
        let done_files: Arc<HashSet<String>> = Arc::new(
            tokio::task::spawn_blocking(move || -> Result<HashSet<String>, String> {
                use postgres::{Client, NoTls};
                let mut client = Client::connect(&pg_conn, NoTls)
                    .map_err(|e| format!("progress check connect failed: {e}"))?;
                let rows = client.query(
                    "SELECT filename FROM _build_progress WHERE source = 'oa'",
                    &[],
                ).map_err(|e| format!("progress check query failed: {e}"))?;
                Ok(rows.iter().map(|r| r.get::<_, String>(0)).collect())
            })
            .await
            .map_err(|e| format!("progress check task panicked: {e}"))??,
        );

        let download_futures = gz_objects.into_iter().map(|obj| {
            let store = Arc::clone(&store_clone);
            let t2 = Arc::clone(&t2_clone);
            let done = Arc::clone(&done_files);
            async move {
                let obj_path = obj.location.as_ref().to_string();
                let filename = obj_path
                    .rsplit('/')
                    .next()
                    .unwrap_or(&obj_path)
                    .to_string();

                if done.contains(&filename) {
                    return FileOutput::empty(filename);
                }

                let total_attempts = max_retries + 1;
                let mut last_err = String::new();
                for attempt in 0..total_attempts {
                    if attempt > 0 {
                        let shift = (attempt - 1).min(63);
                        let delay_ms = initial_backoff_ms
                            .saturating_mul(1u64 << shift)
                            .min(max_backoff_ms);
                        let delay = std::time::Duration::from_millis(delay_ms);
                        eprintln!("oa: retry {attempt}/{max_retries} for {obj_path} in {delay:?}");
                        tokio::time::sleep(delay).await;
                    }

                    let dl_result = async {
                        let result = store.get(&obj.location).await?;
                        result.bytes().await
                    }
                    .await;

                    match dl_result {
                        Ok(bytes) => {
                            let t2 = Arc::clone(&t2);
                            let fname = filename.clone();
                            let parsed = tokio::task::spawn_blocking(move || {
                                parse_gz_to_structs(&bytes, &t2, fname)
                            })
                            .await
                            .unwrap_or_else(|e| {
                                eprintln!("oa: WARN: parse task panicked for {obj_path}: {e}");
                                FileOutput::empty(obj_path.rsplit('/').next().unwrap_or(&obj_path).to_string())
                            });
                            return parsed;
                        }
                        Err(e) => {
                            last_err = format!("{e}");
                        }
                    }
                }
                eprintln!("oa: WARN: failed to download {obj_path} after {total_attempts} attempts: {last_err}");
                FileOutput::empty(filename)
            }
        });

        let mut stream =
            futures::stream::iter(download_futures).buffer_unordered(concurrency);

        while let Some(output) = stream.next().await {
            if tx.send(output).await.is_err() {
                break;
            }
        }
        Ok(())
    });

    // Main thread: receive structs, write to PG
    let mut files_done = 0usize;
    while let Some(output) = rt.block_on(rx.recv()) {
        if output.works.is_empty() && output.failed_lines == 0 {
            files_done += 1;
            continue;
        }
        write_file_output(&mut sink, &output, &mut counters)?;
        files_done += 1;

        if files_done.is_multiple_of(10) || files_done == num_files {
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!(
                "oa: {files_done}/{num_files} s3 objects, {} works ({elapsed:.1}s)",
                counters.works,
            );
        }
    }

    // Ensure the producer task completed successfully
    rt.block_on(producer_handle)
        .map_err(|e| format!("oa: S3 producer task panicked: {e}"))?
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "oa: done — {} works, {} citations in {elapsed:.1}s",
        counters.works, counters.citations,
    );

    Ok(OaBuildStats {
        num_works: counters.works,
        num_authors: counters.authors,
        num_topics: counters.topics,
        num_citations: counters.citations,
        num_crosswalk: counters.crosswalk,
        num_files_processed: num_files,
        num_failed_lines: counters.failed_lines,
        elapsed_secs: elapsed,
    })
}

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
}
