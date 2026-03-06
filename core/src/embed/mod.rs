//! Embedding pipeline: read hive parquet → encode → write embedding parquet.

pub mod embedder;
pub mod http_backend;
pub mod io;

#[cfg(feature = "local")]
pub mod ort_backend;
#[cfg(feature = "local")]
pub mod tokenize;

pub use embedder::{Embedder, PoolingStrategy};
pub use http_backend::HttpEmbedder;
