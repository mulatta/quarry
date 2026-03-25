mod algo;
mod build;
mod graph;
mod pattern;

use pyo3::prelude::*;

pub use graph::Graph;

#[pymodule]
fn quarry_graph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build::build_from_csv, m)?)?;
    m.add_class::<Graph>()?;
    Ok(())
}
