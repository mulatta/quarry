//! Build configuration: paths, S3 prefixes, tier domains.

use std::path::PathBuf;

/// Top-level build configuration.
pub struct BuildConfig {
    /// Output directory for all Parquet files.
    pub output_dir: PathBuf,
    /// Parquet row group size.
    pub row_group_size: usize,
    /// Max rows per output Parquet file.
    pub max_rows_per_file: usize,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("build_output"),
            row_group_size: 128 * 1024,    // 128K rows
            max_rows_per_file: 500_000,
        }
    }
}

/// Parquet output subdirectories.
impl BuildConfig {
    pub fn papers_dir(&self) -> PathBuf {
        self.output_dir.join("papers")
    }
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
    pub fn work_mesh_dir(&self) -> PathBuf {
        self.output_dir.join("work_mesh")
    }
    pub fn id_crosswalk_dir(&self) -> PathBuf {
        self.output_dir.join("id_crosswalk")
    }

    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        for dir in [
            self.output_dir.clone(),
            self.papers_dir(),
            self.works_dir(),
            self.work_authors_dir(),
            self.work_topics_dir(),
            self.work_citations_dir(),
            self.work_mesh_dir(),
            self.id_crosswalk_dir(),
        ] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }
}
