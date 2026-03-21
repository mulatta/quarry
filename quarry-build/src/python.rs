//! PyO3 bindings — only compiled with `python` feature.

use std::path::Path;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::build::{config::BuildConfig, oa, pg_sink::PgSink, pubmed};
use crate::parse::mesh;

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

/// Parse MeSH descriptor XML and load directly into PG mesh_tree table.
///
/// Returns number of rows written.
#[pyfunction]
pub fn mesh_stage_pg(pg_conninfo: &str, xml_path: &str) -> PyResult<u64> {
    let t0 = Instant::now();
    let entries = mesh::parse_mesh_xml(Path::new(xml_path))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let n = entries.len();
    eprintln!("mesh: parsed {n} entries from {xml_path} in {:.1}s", t0.elapsed().as_secs_f64());

    let mut sink = PgSink::connect(pg_conninfo)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let rows = sink
        .write_mesh_tree(&entries)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    eprintln!("mesh: {rows} rows written to PG in {:.1}s", t0.elapsed().as_secs_f64());
    Ok(rows)
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
