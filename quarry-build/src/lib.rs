pub mod parse;
pub mod build;

/// Format error with full source chain (postgres errors hide details behind `.source()`).
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

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn quarry_build(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Build pipeline
    m.add_function(wrap_pyfunction!(python::init_schema, m)?)?;
    m.add_function(wrap_pyfunction!(python::reset_all, m)?)?;
    m.add_function(wrap_pyfunction!(python::build_pubmed_pg, m)?)?;
    m.add_function(wrap_pyfunction!(python::build_oa_local_pg, m)?)?;
    m.add_function(wrap_pyfunction!(python::build_oa_s3_pg, m)?)?;
    m.add_function(wrap_pyfunction!(python::enrich_pg, m)?)?;
    m.add_function(wrap_pyfunction!(python::mesh_stage_pg, m)?)?;
    Ok(())
}
