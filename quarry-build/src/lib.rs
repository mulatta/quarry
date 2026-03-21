pub mod parse;
pub mod build;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn quarry_build(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Build pipeline
    m.add_function(wrap_pyfunction!(python::init_schema, m)?)?;
    m.add_function(wrap_pyfunction!(python::build_pubmed_pg, m)?)?;
    m.add_function(wrap_pyfunction!(python::build_oa_local_pg, m)?)?;
    m.add_function(wrap_pyfunction!(python::build_oa_s3_pg, m)?)?;
    m.add_function(wrap_pyfunction!(python::enrich_pg, m)?)?;
    m.add_function(wrap_pyfunction!(python::mesh_stage_pg, m)?)?;
    Ok(())
}
