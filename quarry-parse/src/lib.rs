mod arrow;
mod xml;

use std::path::Path;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_arrow::PyRecordBatch;
use rayon::prelude::*;

/// Parse a single PubMed XML file (.xml or .xml.gz).
///
/// Returns a dict with Arrow RecordBatch tables:
///   { "papers": RecordBatch, "authors": RecordBatch, "mesh_headings": RecordBatch,
///     "grants": RecordBatch, "chemicals": RecordBatch, "delete_pmids": list[int],
///     "stats": { "num_papers": int, "num_authors": int, ... } }
#[pyfunction]
fn parse_pubmed_file(py: Python<'_>, path: &str) -> PyResult<PyObject> {
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
fn parse_pubmed_files(py: Python<'_>, paths: Vec<String>) -> PyResult<PyObject> {
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
        let r = r.map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;
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

#[pymodule]
fn quarry_parse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_pubmed_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_pubmed_files, m)?)?;
    Ok(())
}
