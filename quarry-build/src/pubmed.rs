//! PubMed XML → Parquet pipeline.
//!
//! Streaming architecture: parse files in parallel chunks, convert to
//! RecordBatches, and write to Parquet incrementally. Peak memory is
//! bounded to CHUNK_SIZE files worth of data, not the entire dataset.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use arrow::array::Int32Array;
use arrow::record_batch::RecordBatch;
use quarry_parse::{arrow as qp_arrow, xml};

use crate::config::BuildConfig;
use crate::parquet_sink::ParquetSink;
use crate::schema;

/// Number of XML files to parse per chunk before flushing to Parquet.
const CHUNK_SIZE: usize = 20;

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

/// Build PubMed Parquet files from XML baseline directory.
///
/// Scans `xml_dir` for *.xml.gz files, parses in parallel chunks,
/// and streams results to Parquet files incrementally.
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

    // Process in chunks: parallel parse → merge chunk → write to sinks
    for (chunk_idx, chunk) in xml_files.chunks(CHUNK_SIZE).enumerate() {
        let results: Vec<Result<xml::ParseResult, String>> = chunk
            .par_iter()
            .map(|p| xml::parse_file(p).map_err(|e| format!("{}: {e}", p.display())))
            .collect();

        // Merge chunk, collecting errors instead of aborting
        let mut merged = xml::ParseResult::default();
        for r in results {
            match r {
                Ok(pr) => merged.extend(pr),
                Err(e) => {
                    eprintln!("pubmed: WARN: {e}");
                    failed_files += 1;
                }
            }
        }

        total_papers += merged.papers.len();
        total_authors += merged.authors.len();
        total_mesh += merged.mesh_headings.len();
        total_grants += merged.grants.len();
        total_chemicals += merged.chemicals.len();
        total_deletes += merged.delete_pmids.len();

        // Convert to RecordBatches and write
        let (papers_batch, authors_batch, mesh_batch, grants_batch, chemicals_batch) =
            qp_arrow::result_to_batches(&merged);

        papers_sink.write_batch(&papers_batch)?;
        authors_sink.write_batch(&authors_batch)?;
        mesh_sink.write_batch(&mesh_batch)?;
        grants_sink.write_batch(&grants_batch)?;
        chemicals_sink.write_batch(&chemicals_batch)?;

        // Write delete PMIDs
        if !merged.delete_pmids.is_empty() {
            let del_batch = RecordBatch::try_new(
                Arc::new(schema::delete_pmids_schema()),
                vec![Arc::new(Int32Array::from(merged.delete_pmids.clone()))],
            )?;
            delete_sink.write_batch(&del_batch)?;
        }

        // merged + batches dropped here — memory freed before next chunk

        let processed = (chunk_idx + 1) * CHUNK_SIZE;
        let processed = processed.min(num_files);
        if chunk_idx % 5 == 0 || processed == num_files {
            let elapsed = t0.elapsed().as_secs_f64();
            eprintln!(
                "pubmed: {processed}/{num_files} files, {total_papers} papers ({elapsed:.1}s)"
            );
        }
    }

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
