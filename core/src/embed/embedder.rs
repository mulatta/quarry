//! Embedder trait and shared types.

use anyhow::Result;
use indicatif::ProgressBar;

/// Pooling strategy for extracting a fixed-size vector from token-level hidden states.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolingStrategy {
    /// Average over non-padding tokens (weighted by attention_mask).
    Mean,
    /// First token (CLS) hidden state.
    Cls,
    /// Last non-padding token hidden state.
    LastToken,
}

impl std::fmt::Display for PoolingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mean => write!(f, "mean"),
            Self::Cls => write!(f, "cls"),
            Self::LastToken => write!(f, "last_token"),
        }
    }
}

impl std::str::FromStr for PoolingStrategy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "mean" => Ok(Self::Mean),
            "cls" => Ok(Self::Cls),
            "last_token" => Ok(Self::LastToken),
            _ => anyhow::bail!("unknown pooling strategy: {s} (expected mean|cls|last_token)"),
        }
    }
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

// ============================================================
// Pooling & normalization (pure math, no backend dependency)
// ============================================================

/// Apply pooling strategy to transformer hidden states.
///
/// `hidden`: flat `[batch, seq, dim]` row-major f32 array.
/// `attention_mask`: flat `[batch, seq]` i64 array (1=real, 0=pad).
#[cfg_attr(not(feature = "local"), allow(dead_code))]
pub fn pool(
    hidden: &[f32],
    attention_mask: &[i64],
    batch: usize,
    seq: usize,
    dim: usize,
    strategy: PoolingStrategy,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * dim];

    match strategy {
        PoolingStrategy::Mean => {
            for b in 0..batch {
                let mut count = 0.0f32;
                for s in 0..seq {
                    if attention_mask[b * seq + s] > 0 {
                        count += 1.0;
                        let h = (b * seq + s) * dim;
                        let o = b * dim;
                        for d in 0..dim {
                            out[o + d] += hidden[h + d];
                        }
                    }
                }
                let denom = count.max(1e-9);
                for d in 0..dim {
                    out[b * dim + d] /= denom;
                }
            }
        }
        PoolingStrategy::Cls => {
            for b in 0..batch {
                let h = b * seq * dim;
                out[b * dim..(b + 1) * dim].copy_from_slice(&hidden[h..h + dim]);
            }
        }
        PoolingStrategy::LastToken => {
            for b in 0..batch {
                let last = attention_mask[b * seq..(b + 1) * seq]
                    .iter()
                    .filter(|&&m| m > 0)
                    .count()
                    .saturating_sub(1);
                let h = (b * seq + last) * dim;
                out[b * dim..(b + 1) * dim].copy_from_slice(&hidden[h..h + dim]);
            }
        }
    }

    out
}

/// L2 normalize each embedding vector in the batch.
#[cfg_attr(not(feature = "local"), allow(dead_code))]
pub fn l2_normalize(embeddings: &[f32], batch: usize, dim: usize) -> Vec<f32> {
    let mut out = embeddings.to_vec();
    for b in 0..batch {
        let row = &mut out[b * dim..(b + 1) * dim];
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // hidden states: batch=2, seq=3, dim=2
    // row-major: [b0s0d0, b0s0d1, b0s1d0, b0s1d1, b0s2d0, b0s2d1,
    //             b1s0d0, b1s0d1, b1s1d0, b1s1d1, b1s2d0, b1s2d1]
    fn sample_hidden() -> Vec<f32> {
        vec![
            1.0, 2.0, // b0 s0
            3.0, 4.0, // b0 s1
            5.0, 6.0, // b0 s2 (padding)
            7.0, 8.0, // b1 s0
            9.0, 10.0, // b1 s1 (padding)
            11.0, 12.0, // b1 s2 (padding)
        ]
    }

    fn sample_mask() -> Vec<i64> {
        vec![
            1, 1, 0, // b0: 2 real tokens
            1, 0, 0, // b1: 1 real token
        ]
    }

    #[test]
    fn mean_pooling() {
        let hidden = sample_hidden();
        let mask = sample_mask();
        let out = pool(&hidden, &mask, 2, 3, 2, PoolingStrategy::Mean);

        // b0: mean of s0+s1 = [(1+3)/2, (2+4)/2] = [2.0, 3.0]
        assert_eq!(out[0], 2.0);
        assert_eq!(out[1], 3.0);

        // b1: mean of s0 only = [7.0, 8.0]
        assert_eq!(out[2], 7.0);
        assert_eq!(out[3], 8.0);
    }

    #[test]
    fn cls_pooling() {
        let hidden = sample_hidden();
        let mask = sample_mask();
        let out = pool(&hidden, &mask, 2, 3, 2, PoolingStrategy::Cls);

        // b0: s0 = [1.0, 2.0]
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);

        // b1: s0 = [7.0, 8.0]
        assert_eq!(out[2], 7.0);
        assert_eq!(out[3], 8.0);
    }

    #[test]
    fn last_token_pooling() {
        let hidden = sample_hidden();
        let mask = sample_mask();
        let out = pool(&hidden, &mask, 2, 3, 2, PoolingStrategy::LastToken);

        // b0: last real = s1 = [3.0, 4.0]
        assert_eq!(out[0], 3.0);
        assert_eq!(out[1], 4.0);

        // b1: last real = s0 = [7.0, 8.0]
        assert_eq!(out[2], 7.0);
        assert_eq!(out[3], 8.0);
    }

    #[test]
    fn l2_normalize_unit_vectors() {
        let embeddings = vec![3.0, 4.0, 0.0, 5.0];
        let out = l2_normalize(&embeddings, 2, 2);

        // b0: norm = 5.0, [0.6, 0.8]
        assert!((out[0] - 0.6).abs() < 1e-6);
        assert!((out[1] - 0.8).abs() < 1e-6);

        // b1: norm = 5.0, [0.0, 1.0]
        assert!((out[2] - 0.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector() {
        let embeddings = vec![0.0, 0.0];
        let out = l2_normalize(&embeddings, 1, 2);
        // zero vector stays near zero (divided by 1e-9)
        assert!(out[0].abs() < 1e-3);
        assert!(out[1].abs() < 1e-3);
    }

    #[test]
    fn mean_pooling_all_masked() {
        // Edge case: all tokens masked
        let hidden = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![0, 0];
        let out = pool(&hidden, &mask, 1, 2, 2, PoolingStrategy::Mean);
        // Should produce zeros (divided by 1e-9 denominator)
        assert!(out[0].abs() < 1e-3);
        assert!(out[1].abs() < 1e-3);
    }
}
