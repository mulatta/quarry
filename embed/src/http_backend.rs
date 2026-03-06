//! HTTP embedding backend (OpenAI-compatible API).

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};

use crate::embedder::Embedder;

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
}

/// Embedding backend that calls an OpenAI-compatible HTTP API.
pub struct HttpEmbedder {
    client: reqwest::Client,
    endpoint: String,
    model: String,
}

impl HttpEmbedder {
    pub fn new(endpoint: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            endpoint,
            model,
        }
    }

    async fn encode_async(
        &self,
        texts: &[String],
        batch_size: usize,
        bar: &ProgressBar,
    ) -> Result<(Vec<f32>, usize)> {
        let n = texts.len();
        let mut all_embeddings: Vec<f32> = Vec::new();
        let mut dim = 0usize;

        for start in (0..n).step_by(batch_size) {
            let end = (start + batch_size).min(n);
            let batch_texts = &texts[start..end];

            let req = EmbedRequest {
                model: &self.model,
                input: batch_texts,
            };

            let resp = self
                .client
                .post(&self.endpoint)
                .json(&req)
                .send()
                .await
                .with_context(|| format!("POST {}", self.endpoint))?
                .error_for_status()
                .context("embedding API error")?
                .json::<EmbedResponse>()
                .await
                .context("failed to parse embedding response")?;

            anyhow::ensure!(
                resp.data.len() == batch_texts.len(),
                "response count mismatch: expected {}, got {}",
                batch_texts.len(),
                resp.data.len(),
            );

            if dim == 0 {
                dim = resp.data[0].embedding.len();
                all_embeddings.reserve(n * dim);
            }

            for item in &resp.data {
                anyhow::ensure!(
                    item.embedding.len() == dim,
                    "dimension mismatch: expected {dim}, got {}",
                    item.embedding.len(),
                );
                all_embeddings.extend_from_slice(&item.embedding);
            }

            bar.inc((end - start) as u64);
        }

        Ok((all_embeddings, dim))
    }
}

impl Embedder for HttpEmbedder {
    fn encode(
        &self,
        texts: &[String],
        batch_size: usize,
        bar: &ProgressBar,
    ) -> Result<(Vec<f32>, usize)> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(self.encode_async(texts, batch_size, bar))
    }
}
