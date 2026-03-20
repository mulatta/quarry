pub mod abstract_recon;
pub mod normalize;
pub mod xml;

// Arrow-dependent modules — only needed for PyO3 bindings
#[cfg(feature = "python")]
pub mod arrow;
#[cfg(feature = "python")]
pub mod mesh;
#[cfg(feature = "python")]
pub mod schema;

#[cfg(test)]
mod xml_tests;
