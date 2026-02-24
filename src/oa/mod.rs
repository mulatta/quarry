//! OpenAlex provider -- domain types and public API.
//!
//! Processing internals live in the `pipeline` submodule.

mod pipeline;

use std::path::Path;

use indicatif::ProgressBar;

use crate::error::ShardError;
use crate::provider::{RunContext, ShardStats};
use crate::sink::is_valid_parquet;
use crate::transform::Filter;

// ============================================================
// Constants
// ============================================================

/// The 12 output table names (one parquet per shard per table).
pub const TABLES: &[&str] = &[
    "works",
    "works_keys",
    "citations",
    "work_authorships",
    "work_topics",
    "work_keywords",
    "work_mesh",
    "work_locations",
    "work_funders",
    "work_awards",
    "work_sdgs",
    "work_counts_by_year",
];

// ============================================================
// OAShard
// ============================================================

/// A shard from the OpenAlex manifest.
#[derive(Debug, Clone)]
pub struct OAShard {
    pub shard_idx: usize,
    pub url: String,
    pub content_length: Option<u64>,
    pub record_count: u64,
    /// Parsed from URL pattern: `.../updated_date=YYYY-MM-DD/...`
    pub updated_date: Option<String>,
}

/// Parse updated_date from shard URL.
///
/// URL pattern: `https://...data/works/updated_date=2024-01-15/part_0000.gz`
pub fn parse_updated_date(url: &str) -> Option<String> {
    url.find("updated_date=").and_then(|start| {
        let rest = &url[start + "updated_date=".len()..];
        let date = rest.split('/').next()?;
        // Basic validation: YYYY-MM-DD
        if date.len() == 10 && date.chars().filter(|&c| c == '-').count() == 2 {
            Some(date.to_string())
        } else {
            None
        }
    })
}

/// Check if all 12 output parquet files exist and have valid footers for a shard.
///
/// Fast path: check all paths exist first (cheap stat calls) before
/// opening files to validate parquet footers.
pub fn is_shard_complete(output_dir: &Path, shard_idx: usize) -> bool {
    let paths: Vec<_> = TABLES
        .iter()
        .map(|table| {
            output_dir
                .join(table)
                .join(format!("shard_{shard_idx:04}.parquet"))
        })
        .collect();

    // Fast path: if any file is missing, skip expensive footer validation
    if !paths.iter().all(|p| p.exists()) {
        return false;
    }

    paths.iter().all(|p| is_valid_parquet(p))
}

// ============================================================
// Provider
// ============================================================

/// OpenAlex provider for the papeline pipeline.
pub struct OAProvider {
    pub filter: Filter,
}

impl crate::provider::Provider for OAProvider {
    type Shard = OAShard;
    type Err = ShardError;

    fn shard_label(&self, shard: &OAShard) -> String {
        format!("works_{:04}", shard.shard_idx)
    }

    fn process_shard(
        &self,
        shard: &OAShard,
        ctx: &RunContext,
        pb: &ProgressBar,
    ) -> Result<ShardStats, ShardError> {
        pipeline::process_works_shard(shard, ctx, &self.filter, pb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_updated_date_valid() {
        let url =
            "https://openalex.s3.amazonaws.com/data/works/updated_date=2024-01-15/part_0000.gz";
        assert_eq!(parse_updated_date(url), Some("2024-01-15".to_string()));
    }

    #[test]
    fn parse_updated_date_missing() {
        assert_eq!(parse_updated_date("https://example.com/file.gz"), None);
    }

    #[test]
    fn parse_updated_date_no_slash() {
        let url = "https://example.com/updated_date=2024-01-15";
        assert_eq!(parse_updated_date(url), Some("2024-01-15".to_string()));
    }

    #[test]
    fn parse_updated_date_invalid_format() {
        let url = "https://example.com/updated_date=2024-1-5/part.gz";
        assert_eq!(parse_updated_date(url), None);
    }

    #[test]
    fn parse_updated_date_no_dashes() {
        let url = "https://example.com/updated_date=20240115xx/part.gz";
        assert_eq!(parse_updated_date(url), None);
    }

    #[test]
    fn shard_complete_nonexistent() {
        assert!(!is_shard_complete(std::path::Path::new("/nonexistent"), 0));
    }

    /// Create a minimal valid parquet file at the given path
    fn write_minimal_parquet(path: &std::path::Path) {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![1]))])
            .unwrap();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn is_shard_complete_partial() {
        let dir = tempfile::TempDir::new().unwrap();
        for table in &TABLES[..11] {
            let path = dir.path().join(table).join("shard_0000.parquet");
            write_minimal_parquet(&path);
        }
        assert!(!is_shard_complete(dir.path(), 0));
    }

    #[test]
    fn is_shard_complete_all_valid() {
        let dir = tempfile::TempDir::new().unwrap();
        for table in TABLES {
            let path = dir.path().join(table).join("shard_0000.parquet");
            write_minimal_parquet(&path);
        }
        assert!(is_shard_complete(dir.path(), 0));
    }
}
