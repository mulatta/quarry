pub mod parse;
pub mod build;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn quarry_parse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::parse_oa, m)?)?;
    m.add_function(wrap_pyfunction!(python::parse_pubmed, m)?)?;
    m.add_function(wrap_pyfunction!(python::parse_mesh, m)?)?;
    Ok(())
}

/// Format error with full source chain.
pub fn err_chain(e: &dyn std::error::Error) -> String {
    let mut msg = e.to_string();
    let mut source = e.source();
    while let Some(cause) = source {
        msg.push_str(": ");
        msg.push_str(&cause.to_string());
        source = cause.source();
    }
    msg
}
