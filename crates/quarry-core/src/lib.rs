pub mod abstract_recon;
pub mod normalize;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn quarry_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Text processing
    m.add_function(wrap_pyfunction!(python::reconstruct_abstracts, m)?)?;
    m.add_function(wrap_pyfunction!(python::normalize_texts, m)?)?;
    Ok(())
}
