//! Embedder trait and shared types.

use anyhow::Result;
use indicatif::ProgressBar;

/// Pooling strategy for extracting a fixed-size vector from token-level hidden states.
#[derive(Debug, Clone, Copy, clap::ValueEnum, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingStrategy {
    /// Average over non-padding tokens (weighted by attention_mask).
    Mean,
    /// First token (CLS) hidden state.
    Cls,
    /// Last non-padding token hidden state.
    LastToken,
}

/// Unified embedding interface.
///
/// Implementations: `HttpEmbedder` (OpenAI-compatible API), `OrtEmbedder` (local ONNX Runtime).
pub trait Embedder {
    /// Encode texts into embeddings.
    ///
    /// Returns `(flat_embeddings, dim)` — flat f32 vec in row-major order,
    /// where `flat_embeddings.len() == texts.len() * dim`.
    fn encode(
        &self,
        texts: &[String],
        batch_size: usize,
        bar: &ProgressBar,
    ) -> Result<(Vec<f32>, usize)>;
}
