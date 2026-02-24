//! OpenAlex manifest client -- fetches shard URLs from the bulk data manifest.
//!
//! OpenAlex bulk data is hosted on AWS Open Data Program (no auth required).
//! Manifest at `https://openalex.s3.amazonaws.com/data/{entity}/manifest`
//! lists all shard URLs as `s3://openalex/...`, which we convert to HTTPS.

use anyhow::Context;
use serde::Deserialize;

use crate::oa::{OAShard, parse_updated_date};
use crate::stream::{http_client, shared_runtime};

const MANIFEST_BASE: &str = "https://openalex.s3.amazonaws.com/data";

#[derive(Deserialize)]
struct ManifestResponse {
    entries: Vec<ManifestEntry>,
}

#[derive(Deserialize)]
struct ManifestEntry {
    url: String,
    meta: EntryMeta,
}

#[derive(Deserialize)]
struct EntryMeta {
    content_length: u64,
    record_count: u64,
}

/// Fetch the OpenAlex manifest for `entity` (e.g., "works") and return shards
/// with HTTPS URLs. Shard indices are assigned by manifest array order.
pub fn fetch_manifest(entity: &str) -> anyhow::Result<Vec<OAShard>> {
    let url = format!("{MANIFEST_BASE}/{entity}/manifest");
    tracing::info!("Fetching OpenAlex {entity} manifest...");

    let body: String = shared_runtime()
        .handle()
        .block_on(async {
            http_client()
                .get(&url)
                .send()
                .await?
                .error_for_status()?
                .text()
                .await
        })
        .context("Failed to fetch OpenAlex manifest")?;

    let manifest: ManifestResponse = sonic_rs::from_str(&body).context("Invalid manifest JSON")?;

    let shards: Vec<OAShard> = manifest
        .entries
        .into_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let url = s3_to_https(&entry.url);
            let updated_date = parse_updated_date(&url);
            OAShard {
                shard_idx: idx,
                url,
                content_length: Some(entry.meta.content_length),
                record_count: entry.meta.record_count,
                updated_date,
            }
        })
        .collect();

    anyhow::ensure!(!shards.is_empty(), "Empty manifest for {entity}");
    let total_records: u64 = shards.iter().map(|s| s.record_count).sum();
    tracing::info!(
        "{entity}: {} shards, {} total records",
        shards.len(),
        total_records
    );
    Ok(shards)
}

/// Convert S3 URI to HTTPS URL.
/// `s3://openalex/path` → `https://openalex.s3.amazonaws.com/path`
fn s3_to_https(url: &str) -> String {
    url.replace("s3://openalex/", "https://openalex.s3.amazonaws.com/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_url_conversion() {
        assert_eq!(
            s3_to_https("s3://openalex/data/works/updated_date=2024-01-01/part_0000.gz"),
            "https://openalex.s3.amazonaws.com/data/works/updated_date=2024-01-01/part_0000.gz"
        );
    }

    #[test]
    fn non_s3_url_passthrough() {
        let url = "https://example.com/file.gz";
        assert_eq!(s3_to_https(url), url);
    }
}
