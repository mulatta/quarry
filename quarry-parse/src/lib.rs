pub mod abstract_recon;
pub mod arrow;
pub mod mesh;
pub mod normalize;
pub mod schema;
pub mod xml;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn quarry_parse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python::parse_pubmed_file, m)?)?;
    m.add_function(wrap_pyfunction!(python::parse_pubmed_files, m)?)?;
    m.add_function(wrap_pyfunction!(python::reconstruct_abstracts, m)?)?;
    m.add_function(wrap_pyfunction!(python::normalize_texts, m)?)?;
    m.add_function(wrap_pyfunction!(python::parse_mesh_xml, m)?)?;
    Ok(())
}
