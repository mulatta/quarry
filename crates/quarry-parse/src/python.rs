//! PyO3 bindings for quarry-parse.
//!
//! Exposes parse functions directly to Python, eliminating the need for
//! subprocess calls from Dagster assets.

use std::path::PathBuf;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::build::config::ParseConfig;
use crate::build::{oa, pubmed};
use crate::build::parquet_writer;
use crate::parse::mesh;

fn to_py_err(e: Box<dyn std::error::Error>) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(crate::err_chain(&*e))
}

/// Parse OpenAlex .gz JSONL files into Parquet.
///
/// Returns dict with stats: num_works, num_authors, etc.
#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, partition=None, threads=None))]
pub fn parse_oa(
    py: Python<'_>,
    input_dir: &str,
    output_dir: &str,
    partition: Option<&str>,
    threads: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let config = make_config(threads);
    let input = PathBuf::from(input_dir);
    let output = PathBuf::from(output_dir);
    let part = partition.map(|s| s.to_owned());

    let stats = py.allow_threads(move || {
        oa::parse_oa(&config, &input, &output, part.as_deref())
            .map_err(|e| e.to_string())
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    stats_to_dict(py, &stats)
}

/// Parse PubMed XML files into Parquet.
///
/// Returns dict with stats: num_papers, num_authors, etc.
#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, updates_dir=None, threads=None))]
pub fn parse_pubmed(
    py: Python<'_>,
    input_dir: &str,
    output_dir: &str,
    updates_dir: Option<&str>,
    threads: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let config = make_config(threads);
    let input = PathBuf::from(input_dir);
    let output = PathBuf::from(output_dir);
    let updates = updates_dir.map(PathBuf::from);

    let stats = py.allow_threads(move || {
        pubmed::parse_pubmed(&config, &input, updates.as_deref(), &output)
            .map_err(|e| e.to_string())
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    stats_to_dict(py, &stats)
}

/// Parse MeSH descriptor XML into Parquet.
///
/// Returns dict with tree_rows and term_rows counts.
#[pyfunction]
#[pyo3(signature = (xml_path, output_dir))]
pub fn parse_mesh(
    py: Python<'_>,
    xml_path: &str,
    output_dir: &str,
) -> PyResult<Py<PyDict>> {
    let xml = PathBuf::from(xml_path);
    let out = PathBuf::from(output_dir);

    let result = py.allow_threads(move || {
        mesh::parse_mesh_xml(&xml).map_err(|e| e.to_string())
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let tree_path = out.join("mesh_tree.parquet");
    let tree_rows = parquet_writer::write_mesh_tree(&result.tree_entries, &tree_path)
        .map_err(|e| to_py_err(e))?;

    let terms_path = out.join("mesh_terms.parquet");
    let term_rows = parquet_writer::write_mesh_terms(&result.term_entries, &terms_path)
        .map_err(|e| to_py_err(e))?;

    let dict = PyDict::new(py);
    dict.set_item("tree_rows", tree_rows)?;
    dict.set_item("term_rows", term_rows)?;
    Ok(dict.into())
}

fn make_config(threads: Option<usize>) -> ParseConfig {
    let mut config = ParseConfig::default();
    if let Some(v) = threads {
        config.parse_threads = if v == 0 { None } else { Some(v) };
    }
    config
}

fn stats_to_dict<T: serde::Serialize>(py: Python<'_>, stats: &T) -> PyResult<Py<PyDict>> {
    let json = serde_json::to_string(stats)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let json_mod = py.import("json")?;
    let dict = json_mod.call_method1("loads", (json,))?;
    Ok(dict.extract::<Py<PyDict>>()?)
}
