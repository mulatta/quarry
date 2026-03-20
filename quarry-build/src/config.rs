//! Build configuration: paths, S3 prefixes, tier domains.

use std::collections::HashSet;
use std::path::PathBuf;

/// Default T2 domains — mirrors quarry/config.py:oa_t2_domains.
pub const DEFAULT_T2_DOMAINS: &[&str] = &[
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Engineering",
];

/// Top-level build configuration.
pub struct BuildConfig {
    /// Output directory for all Parquet files.
    pub output_dir: PathBuf,
    /// Parquet row group size.
    pub row_group_size: usize,
    /// Max rows per output Parquet file.
    pub max_rows_per_file: usize,
    /// T2 tier domain names.
    pub t2_domains: Vec<String>,
    /// OA batch size — flush to Parquet every N works.
    pub oa_batch_size: usize,
    /// Number of concurrent S3 downloads.
    pub s3_download_concurrency: usize,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("build_output"),
            row_group_size: 128 * 1024,    // 128K rows
            max_rows_per_file: 500_000,
            t2_domains: DEFAULT_T2_DOMAINS.iter().map(|s| s.to_string()).collect(),
            oa_batch_size: 50_000,
            s3_download_concurrency: 8,
        }
    }
}

impl BuildConfig {
    /// Return T2 domains as a HashSet for O(1) lookup.
    pub fn t2_domains_set(&self) -> HashSet<String> {
        self.t2_domains.iter().cloned().collect()
    }
}

/// Parquet output subdirectories.
///
/// Each table gets its own directory so DataFusion / DuckDB can
/// register a directory as a single logical table.
impl BuildConfig {
    // PubMed tables (one dir per table)
    pub fn papers_dir(&self) -> PathBuf {
        self.output_dir.join("papers")
    }
    pub fn pubmed_authors_dir(&self) -> PathBuf {
        self.output_dir.join("pubmed_authors")
    }
    pub fn mesh_headings_dir(&self) -> PathBuf {
        self.output_dir.join("mesh_headings")
    }
    pub fn grants_dir(&self) -> PathBuf {
        self.output_dir.join("grants")
    }
    pub fn chemicals_dir(&self) -> PathBuf {
        self.output_dir.join("chemicals")
    }
    pub fn delete_pmids_dir(&self) -> PathBuf {
        self.output_dir.join("delete_pmids")
    }

    // OA tables
    pub fn works_dir(&self) -> PathBuf {
        self.output_dir.join("works")
    }
    pub fn work_authors_dir(&self) -> PathBuf {
        self.output_dir.join("work_authors")
    }
    pub fn work_topics_dir(&self) -> PathBuf {
        self.output_dir.join("work_topics")
    }
    pub fn work_citations_dir(&self) -> PathBuf {
        self.output_dir.join("work_citations")
    }
    pub fn id_crosswalk_dir(&self) -> PathBuf {
        self.output_dir.join("id_crosswalk")
    }

    // Enrich outputs
    pub fn enriched_works_dir(&self) -> PathBuf {
        self.output_dir.join("enriched_works")
    }
    pub fn work_mesh_dir(&self) -> PathBuf {
        self.output_dir.join("work_mesh")
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        for dir in [
            self.output_dir.clone(),
            self.papers_dir(),
            self.pubmed_authors_dir(),
            self.mesh_headings_dir(),
            self.grants_dir(),
            self.chemicals_dir(),
            self.delete_pmids_dir(),
            self.works_dir(),
            self.work_authors_dir(),
            self.work_topics_dir(),
            self.work_citations_dir(),
            self.id_crosswalk_dir(),
            self.enriched_works_dir(),
            self.work_mesh_dir(),
        ] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }
}
