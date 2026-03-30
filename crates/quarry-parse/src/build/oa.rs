//! OpenAlex JSONL → Parquet pipeline.
//!
//! Reads local .gz files and writes Hive-partitioned Parquet output:
//!   works/updated_date=YYYY-MM-DD/part_NNNN.parquet
//!   work_authors/updated_date=YYYY-MM-DD/...
//!   work_topics/updated_date=YYYY-MM-DD/...
//!   work_citations/updated_date=YYYY-MM-DD/...
//!   id_crosswalk/updated_date=YYYY-MM-DD/...
//!
//! 1 input file = 1 set of Parquet files (file-level reprocessing).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use flate2::read::GzDecoder;
use rayon::prelude::*;

use serde::Serialize;

use crate::build::config::ParseConfig;
use crate::build::oa_json::{self, OaAuthor, OaCitation, OaCrosswalk, OaTopic, OaWork};
use crate::build::parquet_writer;

/// Lines per chunk before flushing to Parquet.
/// 100K lines ≈ 200MB per thread (works+authors+topics+citations).
const FLUSH_CHUNK_SIZE: usize = 100_000;

/// Stats from an OA parse run.
#[derive(Serialize)]
pub struct OaParseStats {
    pub num_works: usize,
    pub num_authors: usize,
    pub num_topics: usize,
    pub num_citations: usize,
    pub num_crosswalk: usize,
    pub num_files_total: usize,
    pub num_files_new: usize,
    pub num_files_skipped: usize,
    pub num_failed_files: usize,
    pub num_failed_lines: usize,
    pub elapsed_secs: f64,
}

/// Accumulated parsed data for one chunk, flushed periodically.
#[derive(Default)]
struct ChunkBuf {
    works: Vec<OaWork>,
    authors: Vec<OaAuthor>,
    topics: Vec<OaTopic>,
    citations: Vec<OaCitation>,
    crosswalk: Vec<OaCrosswalk>,
}

impl ChunkBuf {
    fn clear(&mut self) {
        self.works.clear();
        self.authors.clear();
        self.topics.clear();
        self.citations.clear();
        self.crosswalk.clear();
    }

    fn is_empty(&self) -> bool {
        self.works.is_empty()
    }
}

/// Aggregate counts from processing a single file.
#[derive(Default)]
struct FileStats {
    works: usize,
    authors: usize,
    topics: usize,
    citations: usize,
    crosswalk: usize,
    failed_lines: usize,
}

impl FileStats {
    fn accum(&mut self, buf: &ChunkBuf) {
        self.works += buf.works.len();
        self.authors += buf.authors.len();
        self.topics += buf.topics.len();
        self.citations += buf.citations.len();
        self.crosswalk += buf.crosswalk.len();
    }
}

/// Extract Hive partition key from OA path.
/// Input: "updated_date=2025-03-01/part_000.gz" → "updated_date=2025-03-01"
fn extract_partition(filename: &str) -> String {
    if let Some(idx) = filename.find('/') {
        filename[..idx].to_string()
    } else {
        "updated_date=unknown".to_string()
    }
}

/// Extract part name: "updated_date=2025-03-01/part_000.gz" → "part_000"
fn extract_part_name(filename: &str) -> String {
    let base = filename.rsplit('/').next().unwrap_or(filename);
    base.strip_suffix(".gz").unwrap_or(base).to_string()
}

/// Flush one chunk of parsed data to Parquet files.
fn flush_chunk(
    partition: &str,
    part_name: &str,
    chunk_idx: u32,
    buf: &ChunkBuf,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let suffix = format!("{part_name}_c{chunk_idx:04}");

    if !buf.works.is_empty() {
        parquet_writer::write_oa_works(
            &buf.works,
            &output_dir.join(format!("works/{partition}/{suffix}.parquet")),
        )?;
    }
    if !buf.authors.is_empty() {
        parquet_writer::write_oa_work_authors(
            &buf.authors,
            &output_dir.join(format!("work_authors/{partition}/{suffix}.parquet")),
        )?;
    }
    if !buf.topics.is_empty() {
        parquet_writer::write_oa_work_topics(
            &buf.topics,
            &output_dir.join(format!("work_topics/{partition}/{suffix}.parquet")),
        )?;
    }
    if !buf.citations.is_empty() {
        parquet_writer::write_oa_work_citations(
            &buf.citations,
            &output_dir.join(format!("work_citations/{partition}/{suffix}.parquet")),
        )?;
    }
    if !buf.crosswalk.is_empty() {
        parquet_writer::write_oa_id_crosswalk(
            &buf.crosswalk,
            &output_dir.join(format!("id_crosswalk/{partition}/{suffix}.parquet")),
        )?;
    }
    Ok(())
}

/// Parse a .gz file and write Parquet in bounded-memory chunks.
///
/// Every FLUSH_CHUNK_SIZE lines, accumulated data is flushed to disk
/// and vecs are cleared. Peak memory per thread ≈ FLUSH_CHUNK_SIZE × ~2KB.
fn process_gz_chunked(
    path: &Path,
    filename: &str,
    output_dir: &Path,
) -> Result<FileStats, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(256 * 1024, GzDecoder::new(BufReader::new(file)));

    let partition = extract_partition(filename);
    let part_name = extract_part_name(filename);

    let mut buf = ChunkBuf::default();
    let mut line_count = 0usize;
    let mut chunk_idx = 0u32;
    let mut stats = FileStats::default();

    for line_result in reader.lines() {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => {
                stats.failed_lines += 1;
                continue;
            }
        };
        if line.is_empty() {
            continue;
        }
        match oa_json::parse_line(&line) {
            Ok(Some(parsed)) => {
                buf.works.push(parsed.work);
                buf.authors.extend(parsed.authors);
                buf.topics.extend(parsed.topics);
                buf.citations.extend(parsed.citations);
                if let Some(cw) = parsed.crosswalk {
                    buf.crosswalk.push(cw);
                }
                line_count += 1;
            }
            Ok(None) => {} // intentionally skipped (paratext)
            Err(_) => {
                stats.failed_lines += 1;
            }
        }
        if line_count >= FLUSH_CHUNK_SIZE {
            stats.accum(&buf);
            flush_chunk(&partition, &part_name, chunk_idx, &buf, output_dir)?;
            buf.clear();
            line_count = 0;
            chunk_idx += 1;
        }
    }

    // Flush remaining lines
    if !buf.is_empty() {
        stats.accum(&buf);
        flush_chunk(&partition, &part_name, chunk_idx, &buf, output_dir)?;
    }

    // Write sentinel file to mark this input as fully processed.
    // Without this, a crash mid-file would leave partial chunks that get
    // skipped on retry, causing silent data loss.
    let sentinel = output_dir.join(format!("works/{partition}/{part_name}.done"));
    if let Some(parent) = sentinel.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&sentinel, [])?;

    Ok(stats)
}

/// Recursively collect .gz files from a directory, returning relative paths.
fn collect_gz_files(dir: &Path) -> std::io::Result<Vec<(PathBuf, String)>> {
    let mut out = Vec::new();
    collect_gz_files_inner(dir, dir, &mut out)?;
    out.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(out)
}

fn collect_gz_files_inner(
    base: &Path,
    dir: &Path,
    out: &mut Vec<(PathBuf, String)>,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        if ft.is_dir() {
            collect_gz_files_inner(base, &entry.path(), out)?;
        } else if ft.is_file() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".gz") {
                let rel = entry
                    .path()
                    .strip_prefix(base)
                    .unwrap_or(&entry.path())
                    .to_string_lossy()
                    .to_string();
                out.push((entry.path(), rel));
            }
        }
    }
    Ok(())
}

/// Parse OA from a local directory of .gz JSONL files → Parquet.
pub fn parse_oa(
    config: &ParseConfig,
    input_dir: &Path,
    output_dir: &Path,
) -> Result<OaParseStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let gz_files = collect_gz_files(input_dir)?;
    let num_files = gz_files.len();
    eprintln!("oa: found {num_files} .gz files in {}", input_dir.display());

    // Skip files whose .done sentinel exists (written after all chunks complete).
    // Partial chunks from a crashed run are ignored and overwritten on retry.
    let todo: Vec<_> = gz_files
        .into_iter()
        .filter(|(_, rel)| {
            let part = extract_partition(rel);
            let name = extract_part_name(rel);
            let done = output_dir.join(format!("works/{part}/{name}.done"));
            !done.exists()
        })
        .collect();

    let num_todo = todo.len();
    let num_skipped = num_files - num_todo;
    eprintln!("oa: {num_skipped} already done, {num_todo} to process");

    let n_threads = config.effective_parse_threads();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()?;

    let works_count = Arc::new(AtomicUsize::new(0));
    let authors_count = Arc::new(AtomicUsize::new(0));
    let topics_count = Arc::new(AtomicUsize::new(0));
    let citations_count = Arc::new(AtomicUsize::new(0));
    let crosswalk_count = Arc::new(AtomicUsize::new(0));
    let failed_lines_count = Arc::new(AtomicUsize::new(0));
    let failed_files_count = Arc::new(AtomicUsize::new(0));
    let files_done = Arc::new(AtomicUsize::new(0));
    let out_dir = output_dir.to_path_buf();

    pool.install(|| {
        todo.par_iter().for_each(|(abs_path, rel_path)| {
            let stats = match process_gz_chunked(abs_path, rel_path, &out_dir) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("oa: WARN: failed {rel_path}: {e}");
                    failed_files_count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };

            works_count.fetch_add(stats.works, Ordering::Relaxed);
            authors_count.fetch_add(stats.authors, Ordering::Relaxed);
            topics_count.fetch_add(stats.topics, Ordering::Relaxed);
            citations_count.fetch_add(stats.citations, Ordering::Relaxed);
            crosswalk_count.fetch_add(stats.crosswalk, Ordering::Relaxed);
            failed_lines_count.fetch_add(stats.failed_lines, Ordering::Relaxed);

            let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done.is_multiple_of(10) || done == num_todo {
                let elapsed = t0.elapsed().as_secs_f64();
                let w = works_count.load(Ordering::Relaxed);
                eprintln!("oa: {done}/{num_todo} files, {w} works ({elapsed:.1}s)");
            }
        });
    });

    let elapsed = t0.elapsed().as_secs_f64();
    let total_works = works_count.load(Ordering::Relaxed);
    let total_citations = citations_count.load(Ordering::Relaxed);
    eprintln!("oa: done — {total_works} works, {total_citations} citations in {elapsed:.1}s");

    Ok(OaParseStats {
        num_works: total_works,
        num_authors: authors_count.load(Ordering::Relaxed),
        num_topics: topics_count.load(Ordering::Relaxed),
        num_citations: total_citations,
        num_crosswalk: crosswalk_count.load(Ordering::Relaxed),
        num_files_total: num_files,
        num_files_new: num_todo,
        num_files_skipped: num_skipped,
        num_failed_files: failed_files_count.load(Ordering::Relaxed),
        num_failed_lines: failed_lines_count.load(Ordering::Relaxed),
        elapsed_secs: elapsed,
    })
}
