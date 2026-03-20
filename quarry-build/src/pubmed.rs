//! PubMed XML → Parquet pipeline.
//!
//! Architecture: rayon parallel parse per file → Arrow batch conversion
//! → channel → main thread sequential sink writes. Parsing and writing
//! are pipelined: while the main thread writes batches, rayon threads
//! parse the next files.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::Int32Array;
use arrow::record_batch::RecordBatch;
use quarry_parse::{arrow as qp_arrow, xml};
use rayon::prelude::*;

use crate::config::BuildConfig;
use crate::parquet_sink::ParquetSink;
use crate::schema;

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
}

/// Batches produced from a single parsed file, ready for sink writes.
struct FileBatches {
    papers: RecordBatch,
    authors: RecordBatch,
    mesh: RecordBatch,
    grants: RecordBatch,
    chemicals: RecordBatch,
    delete_pmids: Vec<i32>,
    // counts (avoid re-counting from batches)
    n_papers: usize,
    n_authors: usize,
    n_mesh: usize,
    n_grants: usize,
    n_chemicals: usize,
}

/// Build PubMed Parquet files from XML baseline directory.
///
/// Files are parsed in parallel (rayon) and converted to Arrow batches
/// per-file. A channel pipelines parsing with Parquet sink writes on
/// the main thread.
pub fn build_pubmed(
    config: &BuildConfig,
    xml_dir: &Path,
) -> Result<PubmedBuildStats, Box<dyn std::error::Error>> {
    let t0 = Instant::now();

    // Find all XML files
    let mut xml_files: Vec<_> = std::fs::read_dir(xml_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            name.ends_with(".xml.gz") || name.ends_with(".xml")
        })
        .collect();
    xml_files.sort();

    let num_files = xml_files.len();
    eprintln!("pubmed: found {num_files} XML files in {}", xml_dir.display());

    let rg_size = config.row_group_size;
    let max_rows = config.max_rows_per_file;

    let mut papers_sink =
        ParquetSink::new(&config.papers_dir(), "papers", Arc::new(schema::papers_schema()), rg_size, max_rows);
    let mut authors_sink =
        ParquetSink::new(&config.pubmed_authors_dir(), "authors", Arc::new(schema::authors_schema()), rg_size, max_rows);
    let mut mesh_sink = ParquetSink::new(
        &config.mesh_headings_dir(), "mesh_headings", Arc::new(schema::mesh_headings_schema()), rg_size, max_rows,
    );
    let mut grants_sink =
        ParquetSink::new(&config.grants_dir(), "grants", Arc::new(schema::grants_schema()), rg_size, max_rows);
    let mut chemicals_sink = ParquetSink::new(
        &config.chemicals_dir(), "chemicals", Arc::new(schema::chemicals_schema()), rg_size, max_rows,
    );
    let mut delete_sink = ParquetSink::new(
        &config.delete_pmids_dir(), "delete_pmids", Arc::new(schema::delete_pmids_schema()), rg_size, max_rows,
    );

    let mut total_papers = 0usize;
    let mut total_authors = 0usize;
    let mut total_mesh = 0usize;
    let mut total_grants = 0usize;
    let mut total_chemicals = 0usize;
    let mut total_deletes = 0usize;
    let mut failed_files = 0usize;

    // Channel: rayon parse threads → main thread sink writer
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<FileBatches, String>>(
        rayon::current_num_threads(),
    );

    // Spawn rayon producer in a separate thread so main thread can
    // consume from the channel concurrently.
    let files = xml_files.clone();
    let producer = std::thread::spawn(move || {
        files.par_iter().for_each_with(tx, |tx, path| {
            let result = match xml::parse_file(path) {
                Ok(pr) => {
                    let n_papers = pr.papers.len();
                    let n_authors = pr.authors.len();
                    let n_mesh = pr.mesh_headings.len();
                    let n_grants = pr.grants.len();
                    let n_chemicals = pr.chemicals.len();
                    let delete_pmids = pr.delete_pmids.clone();

                    let (papers, authors, mesh, grants, chemicals) =
                        qp_arrow::result_to_batches(&pr);

                    Ok(FileBatches {
                        papers,
                        authors,
                        mesh,
                        grants,
                        chemicals,
                        delete_pmids,
                        n_papers,
                        n_authors,
                        n_mesh,
                        n_grants,
                        n_chemicals,
                    })
                }
                Err(e) => Err(format!("{}: {e}", path.display())),
            };
            // Ignore send error (receiver dropped)
            let _ = tx.send(result);
        });
    });

    // Main thread: receive and write to sinks
    let mut files_done = 0usize;
    for result in rx {
        match result {
            Ok(batches) => {
                total_papers += batches.n_papers;
                total_authors += batches.n_authors;
                total_mesh += batches.n_mesh;
                total_grants += batches.n_grants;
                total_chemicals += batches.n_chemicals;
                total_deletes += batches.delete_pmids.len();

                papers_sink.write_batch(&batches.papers)?;
                authors_sink.write_batch(&batches.authors)?;
                mesh_sink.write_batch(&batches.mesh)?;
                grants_sink.write_batch(&batches.grants)?;
                chemicals_sink.write_batch(&batches.chemicals)?;

                if !batches.delete_pmids.is_empty() {
                    let del_batch = RecordBatch::try_new(
                        Arc::new(schema::delete_pmids_schema()),
                        vec![Arc::new(Int32Array::from(batches.delete_pmids))],
                    )?;
                    delete_sink.write_batch(&del_batch)?;
                }
            }
            Err(e) => {
                eprintln!("pubmed: WARN: {e}");
                failed_files += 1;
            }
        }
        files_done += 1;

        if files_done % 100 == 0 || files_done == num_files {
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!(
                "pubmed: {files_done}/{num_files} files, {total_papers} papers ({elapsed:.1}s)"
            );
        }
    }

    producer.join().expect("rayon producer panicked");

    let papers_stats = papers_sink.finish()?;
    authors_sink.finish()?;
    mesh_sink.finish()?;
    grants_sink.finish()?;
    chemicals_sink.finish()?;
    delete_sink.finish()?;

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "pubmed: wrote {} papers to {} files in {elapsed:.1}s",
        papers_stats.total_rows, papers_stats.num_files,
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
    })
}
