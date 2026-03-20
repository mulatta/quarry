//! PubMed XML → PostgreSQL pipeline.
//!
//! Architecture: rayon parallel parse per file → channel → main thread
//! sequential PG COPY writes. Parsing and writing are pipelined.
//!
//! Build-time merge: when `updates_dir` is provided, update XMLs are
//! parsed first to collect a HashSet<i32> of PMIDs. Baseline parsing
//! then skips those PMIDs, producing merged output with no duplicates.
//! PubMed uses INSERT via COPY (no upsert needed — baseline + updates
//! are pre-deduplicated).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::parse::xml;
use rayon::prelude::*;

use crate::build::config::BuildConfig;
use crate::build::pg_sink::PgSink;

/// Stats from a PubMed build run.
pub struct PubmedBuildStats {
    pub num_papers: usize,
    pub num_authors: usize,
    pub num_mesh: usize,
    pub num_grants: usize,
    pub num_chemicals: usize,
    pub num_delete_pmids: usize,
    pub num_files_processed: usize,
    pub num_failed_files: usize,
    pub elapsed_secs: f64,
    pub skipped: bool,
}

/// Parsed output from a single XML file, ready for PG writes.
struct FileData {
    result: xml::ParseResult,
    filename: String,
}

/// Build PubMed data from XML baseline + optional updates → PG directly.
pub fn build_pubmed(
    config: &BuildConfig,
    xml_dir: &Path,
    updates_dir: Option<&Path>,
    pg_conninfo: &str,
) -> Result<PubmedBuildStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let xml_files = find_xml_files(xml_dir)?;
    let num_baseline = xml_files.len();
    eprintln!("pubmed: found {num_baseline} baseline XML files in {}", xml_dir.display());

    let update_files = if let Some(udir) = updates_dir {
        let files = find_xml_files(udir)?;
        eprintln!("pubmed: found {} update XML files in {}", files.len(), udir.display());
        files
    } else {
        vec![]
    };
    let num_files = num_baseline + update_files.len();

    let mut sink = PgSink::connect(pg_conninfo)?;

    let mut total_papers = 0usize;
    let mut total_authors = 0usize;
    let mut total_mesh = 0usize;
    let mut total_grants = 0usize;
    let mut total_chemicals = 0usize;
    let mut total_deletes = 0usize;
    let mut failed_files = 0usize;

    // Phase 1: Parse update XMLs → collect PMIDs + write to PG.
    // Process in REVERSE order (newest first) so that when a PMID appears
    // in multiple update files, only the latest version is kept.
    let update_pmids: HashSet<i32> = if !update_files.is_empty() {
        let mut pmids = HashSet::new();
        for path in update_files.iter().rev() {
            let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
            if sink.is_done("updates", &filename)? {
                eprintln!("pubmed: skip (done): {filename}");
                // Still collect PMIDs for dedup even if already loaded
                if let Ok(pr) = xml::parse_file(path) {
                    for p in &pr.papers {
                        pmids.insert(p.pmid);
                    }
                }
                continue;
            }
            match xml::parse_file(path) {
                Ok(mut pr) => {
                    pr.papers.retain(|p| !pmids.contains(&p.pmid));
                    let mut file_pmids = HashSet::new();
                    pr.papers.retain(|p| file_pmids.insert(p.pmid));
                    pr.authors.retain(|a| file_pmids.contains(&a.pmid));
                    pr.mesh_headings.retain(|m| file_pmids.contains(&m.pmid));
                    pr.grants.retain(|g| file_pmids.contains(&g.pmid));
                    pr.chemicals.retain(|c| file_pmids.contains(&c.pmid));
                    pmids.extend(&file_pmids);

                    let n_papers = pr.papers.len();
                    total_papers += n_papers;
                    total_authors += pr.authors.len();
                    total_mesh += pr.mesh_headings.len();
                    total_grants += pr.grants.len();
                    total_chemicals += pr.chemicals.len();
                    total_deletes += pr.delete_pmids.len();

                    sink.copy_papers(&pr.papers)?;
                    sink.copy_authors(&pr.authors)?;
                    sink.copy_mesh_headings(&pr.mesh_headings)?;
                    sink.copy_grants(&pr.grants)?;
                    sink.copy_chemicals(&pr.chemicals)?;
                    if !pr.delete_pmids.is_empty() {
                        sink.soft_delete_pmids(&pr.delete_pmids)?;
                    }
                    sink.mark_progress("updates", &filename, n_papers as i64)?;
                }
                Err(e) => {
                    eprintln!("pubmed: WARN: update {}: {e}", path.display());
                    failed_files += 1;
                }
            }
        }
        eprintln!(
            "pubmed: phase 1 done — {} unique update PMIDs, {} papers written",
            pmids.len(),
            total_papers,
        );
        pmids
    } else {
        HashSet::new()
    };
    let update_pmids = Arc::new(update_pmids);

    // Phase 2: Parse baseline XMLs with dedup filter (rayon parallel).
    let n_threads = config.effective_parse_threads();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .expect("failed to build rayon thread pool");

    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<FileData, String>>(n_threads);

    let files = xml_files;
    let skip_pmids = Arc::clone(&update_pmids);
    let producer = std::thread::spawn(move || {
        pool.install(|| {
            files.par_iter().for_each_with(tx, |tx, path| {
                let filename = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let result = match xml::parse_file(path) {
                    Ok(mut pr) => {
                        if !skip_pmids.is_empty() {
                            pr.papers.retain(|p| !skip_pmids.contains(&p.pmid));
                            pr.authors.retain(|a| !skip_pmids.contains(&a.pmid));
                            pr.mesh_headings.retain(|m| !skip_pmids.contains(&m.pmid));
                            pr.grants.retain(|g| !skip_pmids.contains(&g.pmid));
                            pr.chemicals.retain(|c| !skip_pmids.contains(&c.pmid));
                        }
                        Ok(FileData { result: pr, filename })
                    }
                    Err(e) => Err(format!("{}: {e}", path.display())),
                };
                let _ = tx.send(result);
            });
        });
    });

    // Main thread: receive and write to PG (file-level transactions)
    let mut files_done = 0usize;
    for result in rx {
        match result {
            Ok(data) => {
                if sink.is_done("baseline", &data.filename)? {
                    files_done += 1;
                    continue;
                }
                let pr = &data.result;
                let n_papers = pr.papers.len();

                sink.begin()?;
                let write_result = (|| -> Result<(), Box<dyn std::error::Error>> {
                    sink.copy_papers(&pr.papers)?;
                    sink.copy_authors(&pr.authors)?;
                    sink.copy_mesh_headings(&pr.mesh_headings)?;
                    sink.copy_grants(&pr.grants)?;
                    sink.copy_chemicals(&pr.chemicals)?;
                    if !pr.delete_pmids.is_empty() {
                        sink.soft_delete_pmids(&pr.delete_pmids)?;
                    }
                    sink.mark_progress("baseline", &data.filename, n_papers as i64)?;
                    Ok(())
                })();
                match write_result {
                    Ok(()) => {
                        sink.commit()?;
                        total_papers += n_papers;
                        total_authors += pr.authors.len();
                        total_mesh += pr.mesh_headings.len();
                        total_grants += pr.grants.len();
                        total_chemicals += pr.chemicals.len();
                        total_deletes += pr.delete_pmids.len();
                    }
                    Err(e) => {
                        eprintln!("pubmed: WARN: write failed for {}: {e}", data.filename);
                        sink.rollback()?;
                        failed_files += 1;
                    }
                }
            }
            Err(e) => {
                eprintln!("pubmed: WARN: {e}");
                failed_files += 1;
            }
        }
        files_done += 1;

        if files_done.is_multiple_of(100) || files_done == num_baseline {
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!(
                "pubmed: {files_done}/{num_baseline} baseline files, {total_papers} papers ({elapsed:.1}s)"
            );
        }
    }

    producer
        .join()
        .map_err(|_| "rayon producer thread panicked")?;

    let elapsed = t0.elapsed().as_secs_f64();
    let skipped = update_pmids.len();
    eprintln!(
        "pubmed: wrote {total_papers} papers in {elapsed:.1}s (skipped {skipped} baseline dupes)"
    );
    if failed_files > 0 {
        eprintln!("pubmed: WARN: {failed_files} files failed to parse");
    }

    Ok(PubmedBuildStats {
        num_papers: total_papers,
        num_authors: total_authors,
        num_mesh: total_mesh,
        num_grants: total_grants,
        num_chemicals: total_chemicals,
        num_delete_pmids: total_deletes,
        num_files_processed: num_files,
        num_failed_files: failed_files,
        elapsed_secs: elapsed,
        skipped: false,
    })
}

/// Find all XML files (.xml.gz or .xml) in a directory, sorted.
fn find_xml_files(dir: &Path) -> Result<Vec<std::path::PathBuf>, Box<dyn std::error::Error>> {
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            name.ends_with(".xml.gz") || name.ends_with(".xml")
        })
        .collect();
    files.sort();
    Ok(files)
}
