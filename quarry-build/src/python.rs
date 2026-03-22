//! PyO3 bindings — only compiled with `python` feature.

use std::error::Error;
use std::path::Path;
use std::time::Instant;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::build::{config::BuildConfig, oa, pg_sink::PgSink, pubmed};
use crate::err_chain;
use crate::parse::mesh;

fn to_py_err(e: Box<dyn Error>) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err_chain(&*e))
}

/// Initialize PostgreSQL schema (create tables + indexes).
#[pyfunction]
pub fn init_schema(pg_conninfo: &str) -> PyResult<()> {
    let mut sink = PgSink::connect(pg_conninfo)
        .map_err(to_py_err)?;
    sink.init_schema()
        .map_err(to_py_err)
}

/// TRUNCATE all data tables + reset build progress for full re-load.
#[pyfunction]
pub fn reset_all(pg_conninfo: &str) -> PyResult<()> {
    let mut sink = PgSink::connect(pg_conninfo)
        .map_err(to_py_err)?;
    sink.reset_all()
        .map_err(to_py_err)
}

/// Build PubMed data from XML baseline + optional updates → PG.
///
/// Returns a dict with build stats.
#[pyfunction]
#[pyo3(signature = (pg_conninfo, xml_dir, updates_dir=None, threads=None, pg_writers=4, channel_buffer=0))]
pub fn build_pubmed_pg(
    pg_conninfo: &str,
    xml_dir: &str,
    updates_dir: Option<&str>,
    threads: Option<usize>,
    pg_writers: usize,
    channel_buffer: usize,
) -> PyResult<PyObject> {
    let config = BuildConfig {
        parse_threads: threads,
        pg_writer_threads: pg_writers,
        channel_buffer,
        ..Default::default()
    };
    let stats = pubmed::build_pubmed(
        &config,
        Path::new(xml_dir),
        updates_dir.map(Path::new),
        pg_conninfo,
    )
    .map_err(to_py_err)?;

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
        .map_err(to_py_err)?;
    oa_stats_to_py(&stats)
}

/// Build OpenAlex from S3 → PG.
///
/// Returns a dict with build stats.
#[pyfunction]
#[pyo3(signature = (
    pg_conninfo,
    s3_prefix,
    s3_concurrency=32,
    pg_writers=4,
    channel_buffer=0,
    fetch_max_retries=3,
    fetch_initial_backoff_ms=2000,
    fetch_max_backoff_ms=30000,
))]
#[allow(clippy::too_many_arguments)] // PyO3 kwargs require individual params for Python IDE support
pub fn build_oa_s3_pg(
    pg_conninfo: &str,
    s3_prefix: &str,
    s3_concurrency: usize,
    pg_writers: usize,
    channel_buffer: usize,
    fetch_max_retries: u32,
    fetch_initial_backoff_ms: u64,
    fetch_max_backoff_ms: u64,
) -> PyResult<PyObject> {
    let config = BuildConfig {
        s3_download_concurrency: s3_concurrency,
        pg_writer_threads: pg_writers,
        channel_buffer,
        fetch_max_retries,
        fetch_initial_backoff_ms,
        fetch_max_backoff_ms,
        ..Default::default()
    };
    let stats = oa::build_oa_s3(&config, s3_prefix, pg_conninfo)
        .map_err(to_py_err)?;
    oa_stats_to_py(&stats)
}

/// Enrich: UPDATE works SET pm_* FROM papers; generate work_mesh.
///
/// Returns a dict with row counts.
#[pyfunction]
pub fn enrich_pg(pg_conninfo: &str) -> PyResult<PyObject> {
    let mut sink = PgSink::connect(pg_conninfo)
        .map_err(to_py_err)?;

    let n_enriched = sink
        .enrich_works_from_papers()
        .map_err(to_py_err)?;

    // Atomic: DELETE + regenerate work_mesh in one transaction
    sink.begin().map_err(to_py_err)?;
    let mesh_result = (|| -> Result<u64, Box<dyn std::error::Error>> {
        sink.execute_static("DELETE FROM work_mesh")?;
        sink.generate_work_mesh()
    })();
    let n_mesh = match mesh_result {
        Ok(n) => {
            sink.commit().map_err(to_py_err)?;
            n
        }
        Err(e) => {
            let _ = sink.rollback();
            return Err(to_py_err(e));
        }
    };

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
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(err_chain(&*e)))?;
    let n = entries.len();
    eprintln!("mesh: parsed {n} entries from {xml_path} in {:.1}s", t0.elapsed().as_secs_f64());

    let mut sink = PgSink::connect(pg_conninfo)
        .map_err(to_py_err)?;
    let rows = sink
        .write_mesh_tree(&entries)
        .map_err(to_py_err)?;

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
