//! HTTP embedding backend (OpenAI-compatible API).

use std::time::Duration;

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use serde::{Deserialize, Serialize};

use super::embedder::Embedder;

const MAX_RETRIES: u32 = 3;
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

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
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_default();
        Self {
            client,
            endpoint,
            model,
        }
    }

    /// POST a single batch with retry on transient errors (429, 5xx, network).
    async fn post_with_retry(&self, req: &EmbedRequest<'_>) -> Result<EmbedResponse> {
        let mut last_err = None;

        for attempt in 0..=MAX_RETRIES {
            if attempt > 0 {
                let delay = INITIAL_BACKOFF * 2u32.pow(attempt - 1);
                tracing::warn!(
                    "embed API retry {attempt}/{MAX_RETRIES} after {:.1}s",
                    delay.as_secs_f64()
                );
                tokio::time::sleep(delay).await;
            }

            let result = self
                .client
                .post(&self.endpoint)
                .json(req)
                .send()
                .await
                .with_context(|| format!("POST {}", self.endpoint));

            let resp = match result {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };

            let status = resp.status();
            if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
                last_err = Some(anyhow::anyhow!("HTTP {status}"));
                continue;
            }

            return resp
                .error_for_status()
                .context("embedding API error")?
                .json::<EmbedResponse>()
                .await
                .context("failed to parse embedding response");
        }

        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("embed API failed after retries")))
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

            let resp = self.post_with_retry(&req).await?;

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
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(self.encode_async(texts, batch_size, bar))
    }
}
