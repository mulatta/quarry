//! ONNX Runtime embedding backend (local model inference).
//!
//! Supports any transformer model exported to ONNX format with
//! `{input_ids, attention_mask, [token_type_ids]} → hidden_states` interface.
//! Pooling (mean/cls/last_token) and L2 normalization applied post-inference.
//!
//! Model metadata (pooling, max_length, prompt) is auto-detected from
//! HuggingFace/SentenceTransformer JSON config files in the model directory.
//! CLI arguments override auto-detected values.

use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;
use indicatif::ProgressBar;
use ort::session::Session;

use super::embedder::{Embedder, PoolingStrategy, l2_normalize, pool};
use super::tokenize::{BatchEncoding, tokenize_batch};

/// Local ONNX Runtime embedding backend.
pub struct OrtEmbedder {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    pooling: PoolingStrategy,
    max_length: usize,
    prompt: String,
    needs_token_type_ids: bool,
}

/// Options for `OrtEmbedder::new`. `None` fields are auto-detected from model dir.
pub struct OrtEmbedderOpts {
    pub pooling: Option<PoolingStrategy>,
    pub max_length: Option<usize>,
    pub prompt: Option<String>,
    pub device: String,
}

impl OrtEmbedder {
    pub fn new(model_dir: &Path, opts: OrtEmbedderOpts) -> Result<Self> {
        let onnx_path = find_onnx_model(model_dir)?;

        // Auto-detect from model config files, CLI overrides
        let detected = ModelConfig::load(model_dir);
        let pooling = opts.pooling.or(detected.pooling).unwrap_or_else(|| {
            tracing::warn!("pooling not detected, defaulting to mean");
            PoolingStrategy::Mean
        });
        const DEFAULT_MAX_LENGTH: usize = 512;
        let max_length = opts.max_length.unwrap_or_else(|| {
            let detected_len = detected.max_length.unwrap_or(DEFAULT_MAX_LENGTH);
            if detected_len > DEFAULT_MAX_LENGTH {
                tracing::info!(
                    "model max_position_embeddings={detected_len}, capping to {DEFAULT_MAX_LENGTH} \
                     (override with --max-length)"
                );
                DEFAULT_MAX_LENGTH
            } else {
                detected_len
            }
        });
        let prompt = opts.prompt.or(detected.prompt).unwrap_or_default();

        tracing::info!(
            "Model config: pooling={pooling}, max_length={max_length}, prompt={:?}",
            if prompt.is_empty() { "(none)" } else { &prompt }
        );

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

        let device = &opts.device;
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

// ============================================================
// Model config auto-detection from HuggingFace JSON files
// ============================================================

/// Auto-detected model configuration from JSON files in model directory.
struct ModelConfig {
    pooling: Option<PoolingStrategy>,
    max_length: Option<usize>,
    prompt: Option<String>,
}

impl ModelConfig {
    /// Read `1_Pooling/config.json`, `config.json`, and
    /// `config_sentence_transformers.json` to detect pooling, max_length, prompt.
    fn load(model_dir: &Path) -> Self {
        Self {
            pooling: detect_pooling(model_dir),
            max_length: detect_max_length(model_dir),
            prompt: detect_prompt(model_dir),
        }
    }
}

/// Read `1_Pooling/config.json` for pooling strategy.
fn detect_pooling(model_dir: &Path) -> Option<PoolingStrategy> {
    #[derive(serde::Deserialize)]
    struct PoolingConfig {
        #[serde(default)]
        pooling_mode_cls_token: bool,
        #[serde(default)]
        pooling_mode_mean_tokens: bool,
        #[serde(default)]
        pooling_mode_lasttoken: bool,
    }

    let path = model_dir.join("1_Pooling/config.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let cfg: PoolingConfig = sonic_rs::from_str(&text).ok()?;

    if cfg.pooling_mode_cls_token {
        Some(PoolingStrategy::Cls)
    } else if cfg.pooling_mode_lasttoken {
        Some(PoolingStrategy::LastToken)
    } else if cfg.pooling_mode_mean_tokens {
        Some(PoolingStrategy::Mean)
    } else {
        None
    }
}

/// Read `config.json` for `max_position_embeddings`.
fn detect_max_length(model_dir: &Path) -> Option<usize> {
    #[derive(serde::Deserialize)]
    struct ModelConfig_ {
        max_position_embeddings: Option<usize>,
    }

    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let cfg: ModelConfig_ = sonic_rs::from_str(&text).ok()?;
    cfg.max_position_embeddings
}

/// Read `config_sentence_transformers.json` for default prompt.
fn detect_prompt(model_dir: &Path) -> Option<String> {
    #[derive(serde::Deserialize)]
    struct StConfig {
        #[serde(default)]
        prompts: std::collections::HashMap<String, String>,
    }

    let path = model_dir.join("config_sentence_transformers.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let cfg: StConfig = sonic_rs::from_str(&text).ok()?;
    cfg.prompts.into_values().next()
}
