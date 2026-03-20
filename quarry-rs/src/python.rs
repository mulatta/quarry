//! PyO3 bindings — only compiled with `python` feature.

use std::path::Path;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_arrow::PyRecordBatch;
use rayon::prelude::*;

use crate::build::{config::BuildConfig, oa, pg_sink::PgSink, pubmed};
use crate::parse::{abstract_recon, arrow, mesh, normalize, xml};

/// Parse a single PubMed XML file (.xml or .xml.gz).
///
/// Returns a dict with Arrow RecordBatch tables:
///   { "papers": RecordBatch, "authors": RecordBatch, "mesh_headings": RecordBatch,
///     "grants": RecordBatch, "chemicals": RecordBatch, "delete_pmids": list[int],
///     "stats": { "num_papers": int, "num_authors": int, ... } }
#[pyfunction]
pub fn parse_pubmed_file(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let t0 = Instant::now();
    let result = xml::parse_file(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let num_papers = result.papers.len();
    let num_authors = result.authors.len();
    let num_mesh = result.mesh_headings.len();
    let num_grants = result.grants.len();
    let num_chemicals = result.chemicals.len();
    let num_deletes = result.delete_pmids.len();

    let (papers, authors, mesh, grants, chemicals) = arrow::result_to_batches(&result);

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "quarry_parse: {path} → {num_papers} papers, {num_authors} authors, \
         {num_mesh} mesh, {num_grants} grants, {num_chemicals} chemicals, \
         {num_deletes} deletes in {elapsed:.1}s"
    );

    let dict = PyDict::new(py);
    dict.set_item("papers", PyRecordBatch::new(papers).to_pyarrow(py)?)?;
    dict.set_item("authors", PyRecordBatch::new(authors).to_pyarrow(py)?)?;
    dict.set_item("mesh_headings", PyRecordBatch::new(mesh).to_pyarrow(py)?)?;
    dict.set_item("grants", PyRecordBatch::new(grants).to_pyarrow(py)?)?;
    dict.set_item("chemicals", PyRecordBatch::new(chemicals).to_pyarrow(py)?)?;
    dict.set_item("delete_pmids", result.delete_pmids)?;

    let stats = PyDict::new(py);
    stats.set_item("num_papers", num_papers)?;
    stats.set_item("num_authors", num_authors)?;
    stats.set_item("num_mesh", num_mesh)?;
    stats.set_item("num_grants", num_grants)?;
    stats.set_item("num_chemicals", num_chemicals)?;
    stats.set_item("num_deletes", num_deletes)?;
    stats.set_item("elapsed_secs", elapsed)?;
    dict.set_item("stats", stats)?;

    Ok(dict.into())
}

/// Parse multiple PubMed XML files in parallel using rayon.
///
/// Returns a list of dicts (same structure as parse_pubmed_file).
#[pyfunction]
pub fn parse_pubmed_files(py: Python<'_>, paths: Vec<String>) -> PyResult<PyObject> {
    let t0 = Instant::now();
    let num_files = paths.len();

    // Parse all files in parallel (rayon)
    let results: Vec<Result<xml::ParseResult, String>> = paths
        .par_iter()
        .map(|p| xml::parse_file(Path::new(p)).map_err(|e| format!("{p}: {e}")))
        .collect();

    // Merge into single result
    let mut merged = xml::ParseResult::default();
    for r in results {
        let r = r.map_err(pyo3::exceptions::PyIOError::new_err)?;
        merged.extend(r);
    }

    let num_papers = merged.papers.len();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "quarry_parse: {num_files} files → {num_papers} papers in {elapsed:.1}s \
         ({:.0} papers/s)",
        num_papers as f64 / elapsed
    );

    let num_authors = merged.authors.len();
    let num_mesh = merged.mesh_headings.len();
    let num_grants = merged.grants.len();
    let num_chemicals = merged.chemicals.len();
    let num_deletes = merged.delete_pmids.len();

    let (papers, authors, mesh, grants, chemicals) = arrow::result_to_batches(&merged);

    let dict = PyDict::new(py);
    dict.set_item("papers", PyRecordBatch::new(papers).to_pyarrow(py)?)?;
    dict.set_item("authors", PyRecordBatch::new(authors).to_pyarrow(py)?)?;
    dict.set_item("mesh_headings", PyRecordBatch::new(mesh).to_pyarrow(py)?)?;
    dict.set_item("grants", PyRecordBatch::new(grants).to_pyarrow(py)?)?;
    dict.set_item("chemicals", PyRecordBatch::new(chemicals).to_pyarrow(py)?)?;
    dict.set_item("delete_pmids", merged.delete_pmids)?;

    let stats = PyDict::new(py);
    stats.set_item("num_papers", num_papers)?;
    stats.set_item("num_authors", num_authors)?;
    stats.set_item("num_mesh", num_mesh)?;
    stats.set_item("num_grants", num_grants)?;
    stats.set_item("num_chemicals", num_chemicals)?;
    stats.set_item("num_deletes", num_deletes)?;
    stats.set_item("num_files", num_files)?;
    stats.set_item("elapsed_secs", elapsed)?;
    dict.set_item("stats", stats)?;

    Ok(dict.into())
}

/// Batch reconstruct OpenAlex abstract_inverted_index JSON → plaintext.
///
/// Input: list of JSON strings like '{"word": [0, 2], "other": [1]}'
/// Output: list of plaintext strings
#[pyfunction]
pub fn reconstruct_abstracts(jsons: Vec<String>) -> Vec<String> {
    abstract_recon::reconstruct_abstracts(jsons)
}

/// Batch strip HTML tags + collapse whitespace.
///
/// Input: list of raw text strings (may contain HTML)
/// Output: list of normalized plaintext strings
#[pyfunction]
pub fn normalize_texts(texts: Vec<String>) -> Vec<String> {
    normalize::normalize_texts(texts)
}

/// Parse MeSH descriptor XML → Arrow RecordBatch.
///
/// Returns a RecordBatch with columns: descriptor_ui, descriptor_name, tree_number.
#[pyfunction]
pub fn parse_mesh_xml(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let t0 = Instant::now();
    let batch = mesh::parse_mesh_xml(Path::new(path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let num_rows = batch.num_rows();
    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!("quarry_parse: MeSH {path} → {num_rows} entries in {elapsed:.1}s");
    PyRecordBatch::new(batch).to_pyarrow(py)
}

// ── Build pipeline bindings ──────────────────────────────────────

/// Initialize PostgreSQL schema (create tables + indexes).
#[pyfunction]
pub fn init_schema(pg_conninfo: &str) -> PyResult<()> {
    let mut sink = PgSink::connect(pg_conninfo)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    sink.init_schema()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Build PubMed data from XML baseline + optional updates → PG.
///
/// Returns a dict with build stats.
#[pyfunction]
#[pyo3(signature = (pg_conninfo, xml_dir, updates_dir=None, threads=None))]
pub fn build_pubmed_pg(
    pg_conninfo: &str,
    xml_dir: &str,
    updates_dir: Option<&str>,
    threads: Option<usize>,
) -> PyResult<PyObject> {
    let config = BuildConfig {
        parse_threads: threads,
        ..Default::default()
    };
    let stats = pubmed::build_pubmed(
        &config,
        Path::new(xml_dir),
        updates_dir.map(Path::new),
        pg_conninfo,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let d = PyDict::new(py);
        d.set_item("num_papers", stats.num_papers)?;
        d.set_item("num_authors", stats.num_authors)?;
        d.set_item("num_mesh", stats.num_mesh)?;
        d.set_item("num_grants", stats.num_grants)?;
        d.set_item("num_chemicals", stats.num_chemicals)?;
        d.set_item("num_files_processed", stats.num_files_processed)?;
        d.set_item("elapsed_secs", stats.elapsed_secs)?;
        Ok(d.into())
    })
}

/// Build OpenAlex from local .gz JSONL directory → PG.
///
/// Returns a dict with build stats.
#[pyfunction]
pub fn build_oa_local_pg(pg_conninfo: &str, local_dir: &str) -> PyResult<PyObject> {
    let config = BuildConfig::default();
    let stats = oa::build_oa_local(&config, Path::new(local_dir), pg_conninfo)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    oa_stats_to_py(&stats)
}

/// Build OpenAlex from S3 → PG.
///
/// Returns a dict with build stats.
#[pyfunction]
#[pyo3(signature = (pg_conninfo, s3_prefix, s3_concurrency=8))]
pub fn build_oa_s3_pg(
    pg_conninfo: &str,
    s3_prefix: &str,
    s3_concurrency: usize,
) -> PyResult<PyObject> {
    let config = BuildConfig {
        s3_download_concurrency: s3_concurrency,
        ..Default::default()
    };
    let stats = oa::build_oa_s3(&config, s3_prefix, pg_conninfo)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    oa_stats_to_py(&stats)
}

/// Enrich: UPDATE works SET pm_* FROM papers; generate work_mesh.
///
/// Returns a dict with row counts.
#[pyfunction]
pub fn enrich_pg(pg_conninfo: &str) -> PyResult<PyObject> {
    let mut sink = PgSink::connect(pg_conninfo)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let n_enriched = sink
        .enrich_works_from_papers()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    sink.execute_static("DELETE FROM work_mesh")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let n_mesh = sink
        .generate_work_mesh()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Python::with_gil(|py| {
        let d = PyDict::new(py);
        d.set_item("works_enriched", n_enriched)?;
        d.set_item("work_mesh_rows", n_mesh)?;
        Ok(d.into())
    })
}

fn oa_stats_to_py(stats: &oa::OaBuildStats) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let d = PyDict::new(py);
        d.set_item("num_works", stats.num_works)?;
        d.set_item("num_authors", stats.num_authors)?;
        d.set_item("num_topics", stats.num_topics)?;
        d.set_item("num_citations", stats.num_citations)?;
        d.set_item("num_crosswalk", stats.num_crosswalk)?;
        d.set_item("num_files_processed", stats.num_files_processed)?;
        d.set_item("num_failed_lines", stats.num_failed_lines)?;
        d.set_item("elapsed_secs", stats.elapsed_secs)?;
        Ok(d.into())
    })
}
