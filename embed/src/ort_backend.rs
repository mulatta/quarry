//! ONNX Runtime embedding backend (local model inference).
//!
//! Supports any transformer model exported to ONNX format with
//! `{input_ids, attention_mask, [token_type_ids]} → hidden_states` interface.
//! Pooling (mean/cls/last_token) and L2 normalization applied post-inference.

use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;
use indicatif::ProgressBar;
use ort::session::Session;

use crate::embedder::{Embedder, PoolingStrategy};
use crate::tokenize::{BatchEncoding, tokenize_batch};

/// Local ONNX Runtime embedding backend.
pub struct OrtEmbedder {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    pooling: PoolingStrategy,
    max_length: usize,
    prompt: String,
    needs_token_type_ids: bool,
}

impl OrtEmbedder {
    pub fn new(
        model_dir: &Path,
        pooling: PoolingStrategy,
        max_length: usize,
        prompt: String,
        device: &str,
    ) -> Result<Self> {
        let onnx_path = find_onnx_model(model_dir)?;

        let tokenizer_path = model_dir.join("tokenizer.json");
        anyhow::ensure!(
            tokenizer_path.exists(),
            "tokenizer.json not found in {}",
            model_dir.display()
        );

        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        // Disable built-in padding/truncation — we handle manually in tokenize_batch
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!("failed to disable truncation: {e}"))?;

        let builder = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(ort_err)?;

        #[allow(unused_mut)]
        let mut builder = builder;

        if device == "cuda" {
            #[cfg(feature = "cuda")]
            {
                builder = builder
                    .with_execution_providers([
                        ort::execution_providers::CUDAExecutionProvider::default().build(),
                    ])
                    .map_err(ort_err)?;
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA not compiled — rebuild with --features cuda");
        } else if device == "coreml" {
            #[cfg(feature = "coreml")]
            {
                builder = builder
                    .with_execution_providers([
                        ort::execution_providers::CoreMLExecutionProvider::default().build(),
                    ])
                    .map_err(ort_err)?;
            }
            #[cfg(not(feature = "coreml"))]
            anyhow::bail!("CoreML not compiled — rebuild with --features coreml");
        }

        let session = builder.commit_from_file(&onnx_path).map_err(ort_err)?;

        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        tracing::info!(
            "Loaded {} (device={device}, token_type_ids={needs_token_type_ids})",
            onnx_path.display()
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            pooling,
            max_length,
            prompt,
            needs_token_type_ids,
        })
    }

    fn run_batch(&self, enc: &BatchEncoding) -> Result<(Vec<f32>, usize)> {
        let shape = [enc.batch_size, enc.seq_len];

        let ids =
            ort::value::Tensor::from_array((shape, enc.input_ids.clone())).map_err(ort_err)?;
        let mask =
            ort::value::Tensor::from_array((shape, enc.attention_mask.clone())).map_err(ort_err)?;

        let mut session = self.session.lock().unwrap();

        let outputs = if self.needs_token_type_ids {
            let types = ort::value::Tensor::from_array((shape, enc.token_type_ids.clone()))
                .map_err(ort_err)?;
            session
                .run(ort::inputs! {
                    "input_ids" => ids,
                    "attention_mask" => mask,
                    "token_type_ids" => types,
                })
                .map_err(ort_err)?
        } else {
            session
                .run(ort::inputs! {
                    "input_ids" => ids,
                    "attention_mask" => mask,
                })
                .map_err(ort_err)?
        };

        // Hidden states: [batch_size, seq_len, dim]
        let (_, raw) = outputs[0].try_extract_tensor::<f32>().map_err(ort_err)?;
        let dim = raw.len() / (enc.batch_size * enc.seq_len);

        let pooled = pool(
            raw,
            &enc.attention_mask,
            enc.batch_size,
            enc.seq_len,
            dim,
            self.pooling,
        );
        let normalized = l2_normalize(&pooled, enc.batch_size, dim);

        Ok((normalized, dim))
    }
}

impl Embedder for OrtEmbedder {
    fn encode(
        &self,
        texts: &[String],
        batch_size: usize,
        bar: &ProgressBar,
    ) -> Result<(Vec<f32>, usize)> {
        let n = texts.len();
        let mut all: Vec<f32> = Vec::new();
        let mut dim = 0usize;

        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
            let batch_texts: Vec<String> = if self.prompt.is_empty() {
                texts[start..end].to_vec()
            } else {
                texts[start..end]
                    .iter()
                    .map(|t| format!("{}{t}", self.prompt))
                    .collect()
            };

            let encoding = tokenize_batch(&self.tokenizer, &batch_texts, self.max_length)?;
            let (emb, d) = self.run_batch(&encoding)?;

            if dim == 0 {
                dim = d;
                all.reserve(n * dim);
            }
            all.extend_from_slice(&emb);
            bar.inc((end - start) as u64);
        }

        Ok((all, dim))
    }

    fn dim(&self) -> usize {
        0 // unknown until first encode
    }
}

// ============================================================
// Helpers
// ============================================================

/// Convert ort::Error (not Send+Sync due to recovery type) to anyhow::Error.
fn ort_err(e: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{e}")
}

fn find_onnx_model(dir: &Path) -> Result<std::path::PathBuf> {
    for candidate in ["model.onnx", "onnx/model.onnx"] {
        let p = dir.join(candidate);
        if p.exists() {
            return Ok(p);
        }
    }
    anyhow::bail!(
        "no model.onnx found in {} (checked root and onnx/ subdir)",
        dir.display()
    )
}

/// Apply pooling strategy to hidden states.
pub(crate) fn pool(
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
pub(crate) fn l2_normalize(embeddings: &[f32], batch: usize, dim: usize) -> Vec<f32> {
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
