//! PyO3 bindings — only compiled with `python` feature.

use pyo3::prelude::*;

use crate::{abstract_recon, normalize};

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
