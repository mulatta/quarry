//! OpenAlex JSONL → PostgreSQL pipeline.
//!
//! Supports two data sources:
//! - Local directory of .gz files (`build_oa_local`)
//! - S3 prefix (`build_oa_s3`)
//!
//! Architecture (S3 path) — 3-stage pipeline:
//!   Stage 1: tokio async download (N concurrent) → dl_channel (prefetch buffer)
//!   Stage 2: tokio spawn_blocking parse pool → write_channel
//!   Stage 3: N PG writer threads (sync COPY)
//!
//! Stages overlap: downloads continue while parsing, parsing continues while
//! PG writes are in flight. Bounded channels provide backpressure at each boundary.

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
    /// Pipeline data permit — released when PG write completes, bounding total
    /// in-flight memory across all pipeline stages (download → parse → write).
    _data_permit: Option<tokio::sync::OwnedSemaphorePermit>,
}

/// Parse a gz buffer into struct vecs.
fn parse_gz_to_structs(
    bytes: &[u8],
    t2_domains: &HashSet<String>,
    filename: String,
    data_permit: Option<tokio::sync::OwnedSemaphorePermit>,
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
        _data_permit: data_permit,
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
        // Delete existing work_ids to handle cross-partition duplicates.
        let work_ids: Vec<&str> = output.works.iter().map(|w| w.work_id.as_str()).collect();
        sink.delete_work_ids(&work_ids)?;
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
        let output = parse_gz_to_structs(&bytes, &t2_domains, filename, None);
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
            .with_client_options(
                object_store::ClientOptions::new()
                    .with_connect_timeout(std::time::Duration::from_secs(30))
                    .with_timeout(std::time::Duration::from_secs(600)),
            )
            .with_retry(object_store::RetryConfig {
                max_retries: 10,
                retry_timeout: std::time::Duration::from_secs(600),
                ..Default::default()
            })
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

    let n_writers = config.effective_pg_writer_threads();
    let chan_buf = config.effective_channel_buffer();
    let prefetch = config.effective_prefetch_buffer();

    let max_retries = config.fetch_max_retries;
    let initial_backoff_ms = config.fetch_initial_backoff_ms;
    let max_backoff_ms = config.fetch_max_backoff_ms;

    // Strip "data/works/" prefix so filenames become "updated_date=.../part_XXXX.gz"
    let prefix_with_slash = if prefix.ends_with('/') {
        prefix.clone()
    } else {
        format!("{prefix}/")
    };

    // Check progress upfront to skip already-done files.
    let done_files: Arc<HashSet<String>> = {
        let mut sink = PgSink::connect(pg_conninfo)?;
        Arc::new(sink.done_filenames("oa")?)
    };
    let num_done = done_files.len();
    let num_todo = num_files.saturating_sub(num_done);
    eprintln!("oa: {num_done} files already done, {num_todo} to process");

    // ── Stage 1: S3 download (independent tasks) → dl_channel ─────────
    // Two semaphores decouple network concurrency from memory pressure:
    //   dl_sem:   bounds concurrent network connections (= concurrency)
    //   data_sem: bounds total in-flight data items (downloading + waiting-to-send
    //             + in channel). Caps memory at ~max_in_flight * file_size.
    //
    // dl_permit released immediately after download → network always at full speed.
    // data_permit released after channel send → memory stays bounded.
    let max_in_flight = concurrency + prefetch;
    let (dl_tx, dl_rx) = tokio::sync::mpsc::channel::<(String, bytes::Bytes, tokio::sync::OwnedSemaphorePermit)>(prefetch);

    let store_dl = Arc::clone(&store);
    let done_dl = Arc::clone(&done_files);
    let pfx_dl = prefix_with_slash.clone();
    let dl_t0 = t0;
    let dl_handle: tokio::task::JoinHandle<Result<(), String>> = rt.spawn(async move {
        let dl_sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let data_sem = Arc::new(tokio::sync::Semaphore::new(max_in_flight));
        let dl_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut join_set = tokio::task::JoinSet::new();

        for obj in gz_objects {
            let obj_path = obj.location.as_ref().to_string();
            let filename = obj_path
                .strip_prefix(&pfx_dl)
                .unwrap_or(&obj_path)
                .to_string();

            // Skip done files before acquiring semaphore — no slot wasted.
            if done_dl.contains(&filename) {
                continue;
            }

            // Acquire data budget first (memory bound), then download slot (network bound).
            // Order matters: don't hold a download slot while waiting for memory budget.
            let data_permit = Arc::clone(&data_sem).acquire_owned().await.unwrap();
            let dl_permit = Arc::clone(&dl_sem).acquire_owned().await.unwrap();
            let store = Arc::clone(&store_dl);
            let tx = dl_tx.clone();
            let count = Arc::clone(&dl_count);

            join_set.spawn(async move {
                let total_attempts = max_retries + 1;
                let mut last_err = String::new();

                for attempt in 0..total_attempts {
                    if attempt > 0 {
                        let shift = (attempt - 1).min(63);
                        let delay_ms = initial_backoff_ms
                            .saturating_mul(1u64 << shift)
                            .min(max_backoff_ms);
                        eprintln!(
                            "oa: retry {attempt}/{max_retries} for {obj_path} in {delay_ms}ms"
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    }

                    match async {
                        let result = store.get(&obj.location).await?;
                        result.bytes().await
                    }
                    .await
                    {
                        Ok(bytes) => {
                            let n =
                                count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            if n.is_multiple_of(10) {
                                let elapsed = dl_t0.elapsed().as_secs_f64();
                                eprintln!("oa: {n}/{num_todo} downloaded ({elapsed:.1}s)");
                            }
                            // Release network slot immediately — next download starts.
                            drop(dl_permit);
                            // Send data + permit to parse stage. Permit flows through
                            // the entire pipeline and is released when PG write completes.
                            let _ = tx.send((filename, bytes, data_permit)).await;
                            return;
                        }
                        Err(e) => last_err = format!("{e}"),
                    }
                }

                eprintln!(
                    "oa: WARN: failed {obj_path} after {total_attempts} attempts: {last_err}"
                );
                drop(dl_permit);
                drop(data_permit);
            });
        }

        // Wait for all download tasks.
        while let Some(result) = join_set.join_next().await {
            if let Err(e) = result {
                eprintln!("oa: download task panicked: {e}");
            }
        }
        Ok(())
    });

    // ── Stage 2: Parse (spawn_blocking) → write_channel ────────────────
    // Consumes downloaded bytes, decompresses+parses on blocking threads,
    // sends parsed FileOutput to PG writers via crossbeam.
    let (write_tx, write_rx) = crossbeam_channel::bounded::<FileOutput>(chan_buf);

    let t2_parse = Arc::clone(&t2_domains);
    let (parse_tx, mut parse_rx) = tokio::sync::mpsc::channel::<FileOutput>(chan_buf);
    let mut dl_rx = dl_rx;
    // Limit concurrent parse tasks to avoid unbounded spawn_blocking growth.
    // Permit released immediately after parse (before send) so parsing never
    // stalls waiting for PG writers. The bounded parse_tx channel provides
    // backpressure between parse and PG write stages.
    let parse_concurrency = concurrency;
    let parse_sem = Arc::new(tokio::sync::Semaphore::new(parse_concurrency));
    let parse_handle: tokio::task::JoinHandle<Result<(), String>> = rt.spawn(async move {
        while let Some((filename, bytes, data_permit)) = dl_rx.recv().await {
            let t2 = Arc::clone(&t2_parse);
            let tx = parse_tx.clone();
            let permit = Arc::clone(&parse_sem).acquire_owned().await.unwrap();
            tokio::task::spawn_blocking(move || {
                let output = parse_gz_to_structs(&bytes, &t2, filename, Some(data_permit));
                drop(permit); // release parse slot immediately — don't hold during send
                let _ = tx.blocking_send(output); // backpressure from PG writers via channel
            });
        }
        Ok(())
    });

    // ── Stage 3: PG writer threads ─────────────────────────────────────
    let files_written = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    eprintln!(
        "oa: 3-stage pipeline — {concurrency} downloaders, prefetch {prefetch}, \
         max_in_flight {max_in_flight}, {n_writers} writers, channel buffer {chan_buf}"
    );
    let writer_handles: Vec<_> = (0..n_writers)
        .map(|i| {
            let rx = write_rx.clone();
            let conninfo = pg_conninfo.to_string();
            let written = Arc::clone(&files_written);
            std::thread::Builder::new()
                .name(format!("oa-writer-{i}"))
                .spawn(move || -> Result<OaCounters, Box<dyn std::error::Error + Send + Sync>> {
                    let mut sink = PgSink::connect(&conninfo)
                        .map_err(|e| format!("writer-{i} connect: {e}"))?;
                    let mut counters = OaCounters::default();
                    while let Ok(output) = rx.recv() {
                        if output.works.is_empty() && output.failed_lines == 0 {
                            continue;
                        }
                        let n_works = output.works.len();
                        let fname = output.filename.clone();
                        if let Err(e) = write_file_output(&mut sink, &output, &mut counters) {
                            eprintln!("oa: writer-{i} error: {e}");
                        }
                        let done = written.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if done.is_multiple_of(100) || n_works > 100_000 {
                            let elapsed = t0.elapsed().as_secs_f64();
                            eprintln!(
                                "oa: writer-{i} wrote {fname} ({n_works} works), {done} files total ({elapsed:.1}s)",
                            );
                        }
                    }
                    Ok(counters)
                })
                .expect("failed to spawn writer thread")
        })
        .collect();
    // Drop our copy so writers see disconnect when parse stage stops sending.
    drop(write_rx);

    // Bridge: tokio mpsc (parse output) → crossbeam channel (PG writers).
    let mut files_done = 0usize;
    while let Some(output) = rt.block_on(parse_rx.recv()) {
        files_done += 1;
        if write_tx.send(output).is_err() {
            eprintln!("oa: all writer threads died, aborting");
            break;
        }
        if files_done.is_multiple_of(10) || files_done == num_todo {
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!("oa: {files_done}/{num_todo} files parsed+dispatched ({elapsed:.1}s)");
        }
    }
    // Signal writers to finish.
    drop(write_tx);

    // Ensure async stages completed successfully.
    rt.block_on(dl_handle)
        .map_err(|e| format!("oa: download task panicked: {e}"))?
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    rt.block_on(parse_handle)
        .map_err(|e| format!("oa: parse task panicked: {e}"))?
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Join writer threads and merge counters.
    let mut counters = OaCounters::default();
    for (i, handle) in writer_handles.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(c)) => {
                counters.works += c.works;
                counters.authors += c.authors;
                counters.topics += c.topics;
                counters.citations += c.citations;
                counters.crosswalk += c.crosswalk;
                counters.failed_lines += c.failed_lines;
            }
            Ok(Err(e)) => eprintln!("oa: writer-{i} error: {e}"),
            Err(_) => eprintln!("oa: writer-{i} panicked"),
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "oa: done — {} works, {} citations in {elapsed:.1}s ({n_writers} writers)",
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
