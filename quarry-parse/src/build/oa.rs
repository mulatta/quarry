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

/// Parsed output from a single file.
struct FileOutput {
    works: Vec<OaWork>,
    authors: Vec<OaAuthor>,
    topics: Vec<OaTopic>,
    citations: Vec<OaCitation>,
    crosswalk: Vec<OaCrosswalk>,
    failed_lines: usize,
    /// Hive partition key: extracted from filename (e.g. "updated_date=2025-03-01")
    partition: String,
    /// Base filename for Parquet part naming
    part_name: String,
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

/// Parse a gz file into struct vecs (streaming from disk, no full-file load).
fn parse_gz_to_structs(
    path: &Path,
    filename: &str,
) -> Result<FileOutput, std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::with_capacity(256 * 1024, GzDecoder::new(BufReader::new(file)));
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
        match oa_json::parse_line(&line) {
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

    Ok(FileOutput {
        works,
        authors,
        topics,
        citations,
        crosswalk,
        failed_lines: failed,
        partition: extract_partition(filename),
        part_name: extract_part_name(filename),
    })
}

/// Write a FileOutput to Parquet files in output_dir.
fn write_file_output(
    output: &FileOutput,
    output_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let part = &output.partition;
    let name = &output.part_name;

    if !output.works.is_empty() {
        parquet_writer::write_oa_works(
            &output.works,
            &output_dir.join(format!("works/{part}/{name}.parquet")),
        )?;
    }
    if !output.authors.is_empty() {
        parquet_writer::write_oa_work_authors(
            &output.authors,
            &output_dir.join(format!("work_authors/{part}/{name}.parquet")),
        )?;
    }
    if !output.topics.is_empty() {
        parquet_writer::write_oa_work_topics(
            &output.topics,
            &output_dir.join(format!("work_topics/{part}/{name}.parquet")),
        )?;
    }
    if !output.citations.is_empty() {
        parquet_writer::write_oa_work_citations(
            &output.citations,
            &output_dir.join(format!("work_citations/{part}/{name}.parquet")),
        )?;
    }
    if !output.crosswalk.is_empty() {
        parquet_writer::write_oa_id_crosswalk(
            &output.crosswalk,
            &output_dir.join(format!("id_crosswalk/{part}/{name}.parquet")),
        )?;
    }
    Ok(())
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

    // Skip files whose Parquet output already exists
    let todo: Vec<_> = gz_files
        .into_iter()
        .filter(|(_, rel)| {
            let part = extract_partition(rel);
            let name = extract_part_name(rel);
            let pq_path = output_dir.join(format!("works/{part}/{name}.parquet"));
            !pq_path.exists()
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
            let output = match parse_gz_to_structs(abs_path, rel_path) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("oa: WARN: cannot read {}: {e}", abs_path.display());
                    failed_files_count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };
            if let Err(e) = write_file_output(&output, &out_dir) {
                eprintln!("oa: WARN: write failed for {rel_path}: {e}");
                failed_files_count.fetch_add(1, Ordering::Relaxed);
                return;
            }

            works_count.fetch_add(output.works.len(), Ordering::Relaxed);
            authors_count.fetch_add(output.authors.len(), Ordering::Relaxed);
            topics_count.fetch_add(output.topics.len(), Ordering::Relaxed);
            citations_count.fetch_add(output.citations.len(), Ordering::Relaxed);
            crosswalk_count.fetch_add(output.crosswalk.len(), Ordering::Relaxed);
            failed_lines_count.fetch_add(output.failed_lines, Ordering::Relaxed);

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
