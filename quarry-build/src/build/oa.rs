//! OpenAlex JSONL → PostgreSQL pipeline.
//!
//! Supports two data sources:
//! - Local directory of .gz files (`build_oa_local`)
//! - S3 prefix (`build_oa_s3`)
//!
//! Architecture (S3 path):
//!   tokio async tasks (download) → spawn_blocking (decompress+parse)
//!   → mpsc channel → main thread (sequential PG COPY writes)

use std::collections::{HashMap, HashSet};
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
    pub num_cache_hits: usize,
    pub elapsed_secs: f64,
    pub skipped: bool,
}

/// PubMed fields for inline enrichment during OA build.
struct PmFields {
    journal_abbr: Option<String>,
    country: Option<String>,
    medline_status: Option<String>,
    pub_type: Vec<String>,
    created_date: Option<String>,
    revised_date: Option<String>,
    indexed_date: Option<String>,
}

/// Load papers from PG into a HashMap<pmid, PmFields> for enrichment.
fn load_pm_lookup(
    pg_conninfo: &str,
) -> Result<HashMap<i32, PmFields>, Box<dyn std::error::Error>> {
    use postgres::{Client, NoTls};

    let mut client = Client::connect(pg_conninfo, NoTls)?;
    let mut map = HashMap::new();

    // Check if papers table has data
    let count: i64 = client
        .query_one("SELECT count(*) FROM papers", &[])?
        .get(0);
    if count == 0 {
        eprintln!("oa: papers table empty, skipping pm enrichment");
        return Ok(map);
    }

    eprintln!("oa: loading {count} papers from PG for pm enrichment");

    for row in client.query(
        "SELECT pmid, journal_abbr, country, medline_status, pub_type, \
         created_date::text, revised_date::text, indexed_date::text FROM papers",
        &[],
    )? {
        let pmid: i32 = row.get(0);
        let pub_type: Option<Vec<String>> = row.get(4);
        map.insert(
            pmid,
            PmFields {
                journal_abbr: row.get(1),
                country: row.get(2),
                medline_status: row.get(3),
                pub_type: pub_type.unwrap_or_default(),
                created_date: row.get(5),
                revised_date: row.get(6),
                indexed_date: row.get(7),
            },
        );
    }

    eprintln!("oa: loaded {} papers for pm enrichment", map.len());
    Ok(map)
}

/// Enrich an OaWork with PubMed fields from the lookup HashMap.
fn enrich_work(work: &mut oa_json::OaWork, pm_lookup: &HashMap<i32, PmFields>) {
    if let Some(pmid) = work.pmid
        && let Some(pm) = pm_lookup.get(&pmid)
    {
        work.pm_journal_abbr = pm.journal_abbr.clone();
        work.pm_country = pm.country.clone();
        work.pm_medline_status = pm.medline_status.clone();
        work.pm_pub_type = pm.pub_type.clone();
        work.pm_created_date = pm.created_date.clone();
        work.pm_revised_date = pm.revised_date.clone();
        work.pm_indexed_date = pm.indexed_date.clone();
    }
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
    pm_lookup: &HashMap<i32, PmFields>,
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
            Ok(mut parsed) => {
                enrich_work(&mut parsed.work, pm_lookup);
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
            eprintln!("oa: WARN: write failed for {}: {e}", output.filename);
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

    let pm_lookup = load_pm_lookup(pg_conninfo)?;
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
        let output = parse_gz_to_structs(&bytes, &t2_domains, &pm_lookup, filename);
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
        num_cache_hits: 0,
        elapsed_secs: elapsed,
        skipped: false,
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

    let pm_lookup: Arc<HashMap<i32, PmFields>> = Arc::new(load_pm_lookup(pg_conninfo)?);
    let mut sink = PgSink::connect(pg_conninfo)?;
    let mut counters = OaCounters::default();

    // Channel: download+parse tasks → main thread
    let (tx, mut rx) = tokio::sync::mpsc::channel::<FileOutput>(concurrency);

    let store_clone = Arc::clone(&store);
    let t2_clone = Arc::clone(&t2_domains);
    let pm_clone = Arc::clone(&pm_lookup);
    let pg_conninfo_owned = pg_conninfo.to_string();
    let producer_handle = rt.spawn(async move {
        use futures::StreamExt;

        // Check progress upfront to skip already-done files.
        // Uses spawn_blocking to avoid blocking the tokio runtime.
        let pg_conn = pg_conninfo_owned.clone();
        let done_files: Arc<HashSet<String>> = Arc::new(
            tokio::task::spawn_blocking(move || {
                use postgres::{Client, NoTls};
                match Client::connect(&pg_conn, NoTls) {
                    Ok(mut client) => match client.query(
                        "SELECT filename FROM _build_progress WHERE source = 'oa'",
                        &[],
                    ) {
                        Ok(rows) => rows.iter().map(|r| r.get::<_, String>(0)).collect(),
                        Err(e) => {
                            eprintln!("oa: WARN: failed to query _build_progress: {e}");
                            HashSet::new()
                        }
                    },
                    Err(e) => {
                        eprintln!("oa: WARN: failed to connect for progress check: {e}");
                        HashSet::new()
                    }
                }
            })
            .await
            .unwrap_or_default(),
        );

        let download_futures = gz_objects.into_iter().map(|obj| {
            let store = Arc::clone(&store_clone);
            let t2 = Arc::clone(&t2_clone);
            let pm = Arc::clone(&pm_clone);
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

                let dl_result = async {
                    let result = store.get(&obj.location).await?;
                    result.bytes().await
                }
                .await;

                match dl_result {
                    Ok(bytes) => {
                        let t2 = Arc::clone(&t2);
                        let pm = Arc::clone(&pm);
                        tokio::task::spawn_blocking(move || {
                            parse_gz_to_structs(&bytes, &t2, &pm, filename)
                        })
                        .await
                        .unwrap_or_else(|e| {
                            eprintln!("oa: WARN: parse task panicked for {obj_path}: {e}");
                            FileOutput::empty(obj_path.rsplit('/').next().unwrap_or(&obj_path).to_string())
                        })
                    }
                    Err(e) => {
                        eprintln!("oa: WARN: failed to download {obj_path}: {e}");
                        FileOutput::empty(filename)
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

    // Ensure the producer task completed without panicking
    rt.block_on(producer_handle)
        .map_err(|e| format!("oa: S3 producer task failed: {e}"))?;

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
        num_cache_hits: 0,
        elapsed_secs: elapsed,
        skipped: false,
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
