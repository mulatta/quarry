//! PubMed XML → Parquet pipeline.
//!
//! Reads local XML files (plain or gzipped) and writes Parquet output:
//!   papers/part_NNNN.parquet
//!   authors/part_NNNN.parquet
//!   mesh_headings/part_NNNN.parquet
//!   grants/part_NNNN.parquet
//!   chemicals/part_NNNN.parquet
//!
//! 1 input file = 1 set of Parquet files (file-level reprocessing).

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use serde::Serialize;

use crate::build::config::ParseConfig;
use crate::build::parquet_writer;
use crate::parse::xml;

/// Stats from a PubMed parse run.
#[derive(Serialize)]
pub struct PubmedParseStats {
    pub num_papers: usize,
    pub num_authors: usize,
    pub num_mesh: usize,
    pub num_grants: usize,
    pub num_chemicals: usize,
    pub num_delete_pmids: usize,
    pub num_files_processed: usize,
    pub num_failed_files: usize,
    pub elapsed_secs: f64,
}

/// Parse PubMed XML from baseline + optional updates → Parquet.
pub fn parse_pubmed(
    config: &ParseConfig,
    xml_dir: &Path,
    updates_dir: Option<&Path>,
    output_dir: &Path,
) -> Result<PubmedParseStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    let mut xml_files = find_xml_files(xml_dir)?;
    let num_baseline = xml_files.len();
    eprintln!(
        "pubmed: found {num_baseline} baseline XML files in {}",
        xml_dir.display()
    );

    if let Some(udir) = updates_dir {
        let update_files = find_xml_files(udir)?;
        eprintln!(
            "pubmed: found {} update XML files in {}",
            update_files.len(),
            udir.display()
        );
        xml_files.extend(update_files);
    }
    let todo = xml_files;
    let num_todo = todo.len();
    eprintln!("pubmed: {num_todo} files to process");

    let n_threads = config.effective_parse_threads();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()?;

    let papers_count = Arc::new(AtomicUsize::new(0));
    let authors_count = Arc::new(AtomicUsize::new(0));
    let mesh_count = Arc::new(AtomicUsize::new(0));
    let grants_count = Arc::new(AtomicUsize::new(0));
    let chemicals_count = Arc::new(AtomicUsize::new(0));
    let delete_count = Arc::new(AtomicUsize::new(0));
    let failed_count = Arc::new(AtomicUsize::new(0));
    let files_done = Arc::new(AtomicUsize::new(0));
    let out_dir = output_dir.to_path_buf();

    pool.install(|| {
        todo.par_iter().for_each(|path| {
            let part_name = part_name_from_path(path);

            let pr = match xml::parse_file(path) {
                Ok(pr) => pr,
                Err(e) => {
                    eprintln!("pubmed: WARN: {}: {e}", path.display());
                    failed_count.fetch_add(1, Ordering::Relaxed);
                    return;
                }
            };

            if !pr.delete_pmids.is_empty() {
                delete_count.fetch_add(pr.delete_pmids.len(), Ordering::Relaxed);
            }

            if let Err(e) = write_pubmed_output(&pr, &out_dir, &part_name) {
                eprintln!("pubmed: WARN: write failed for {part_name}: {e}");
                failed_count.fetch_add(1, Ordering::Relaxed);
                return;
            }

            papers_count.fetch_add(pr.papers.len(), Ordering::Relaxed);
            authors_count.fetch_add(pr.authors.len(), Ordering::Relaxed);
            mesh_count.fetch_add(pr.mesh_headings.len(), Ordering::Relaxed);
            grants_count.fetch_add(pr.grants.len(), Ordering::Relaxed);
            chemicals_count.fetch_add(pr.chemicals.len(), Ordering::Relaxed);

            let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
            if done.is_multiple_of(10) || done == num_todo {
                let elapsed = t0.elapsed().as_secs_f64();
                let p = papers_count.load(Ordering::Relaxed);
                eprintln!("pubmed: {done}/{num_todo} files, {p} papers ({elapsed:.1}s)");
            }
        });
    });

    let elapsed = t0.elapsed().as_secs_f64();
    let total_papers = papers_count.load(Ordering::Relaxed);
    eprintln!("pubmed: done — {total_papers} papers in {elapsed:.1}s");

    Ok(PubmedParseStats {
        num_papers: total_papers,
        num_authors: authors_count.load(Ordering::Relaxed),
        num_mesh: mesh_count.load(Ordering::Relaxed),
        num_grants: grants_count.load(Ordering::Relaxed),
        num_chemicals: chemicals_count.load(Ordering::Relaxed),
        num_delete_pmids: delete_count.load(Ordering::Relaxed),
        num_files_processed: num_todo,
        num_failed_files: failed_count.load(Ordering::Relaxed),
        elapsed_secs: elapsed,
    })
}

fn write_pubmed_output(
    pr: &xml::ParseResult,
    output_dir: &Path,
    part_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if !pr.papers.is_empty() {
        parquet_writer::write_pm_papers(
            &pr.papers,
            &output_dir.join(format!("papers/{part_name}.parquet")),
        )?;
    }
    if !pr.authors.is_empty() {
        parquet_writer::write_pm_authors(
            &pr.authors,
            &output_dir.join(format!("authors/{part_name}.parquet")),
        )?;
    }
    if !pr.mesh_headings.is_empty() {
        parquet_writer::write_pm_mesh_headings(
            &pr.mesh_headings,
            &output_dir.join(format!("mesh_headings/{part_name}.parquet")),
        )?;
    }
    if !pr.grants.is_empty() {
        parquet_writer::write_pm_grants(
            &pr.grants,
            &output_dir.join(format!("grants/{part_name}.parquet")),
        )?;
    }
    if !pr.chemicals.is_empty() {
        parquet_writer::write_pm_chemicals(
            &pr.chemicals,
            &output_dir.join(format!("chemicals/{part_name}.parquet")),
        )?;
    }
    Ok(())
}

/// Extract part name from path: "pubmed25n0001.xml.gz" → "pubmed25n0001"
fn part_name_from_path(path: &Path) -> String {
    let name = path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    name.strip_suffix(".xml.gz")
        .or_else(|| name.strip_suffix(".xml"))
        .unwrap_or(&name)
        .to_string()
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
