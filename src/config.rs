//! Configuration loading (TOML) and state/sync-log persistence.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::transform::Filter;

// ============================================================
// TOML config file
// ============================================================

/// Parsed from `papeline.toml`.
///
/// Run settings live at the top level (no `[run]` section).
/// Hive settings live under `[hive]`, filter settings under `[filter]`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    pub output: Option<String>,
    /// ZSTD level for run output. Also used as fallback for `[hive]` if unset.
    pub zstd_level: Option<i32>,
    pub concurrency: Option<usize>,
    pub max_retries: Option<u32>,
    pub read_timeout: Option<u64>,
    pub outer_retries: Option<u32>,
    pub retry_delay: Option<u64>,
    #[serde(default)]
    pub filter: FilterConfig,
    #[serde(default)]
    pub hive: HiveConfig,
    #[serde(default)]
    pub upload: UploadConfig,
}

/// `[hive]` section — settings for `papeline hive`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HiveConfig {
    /// Auto-run hive after `papeline run` (same as `--hive` CLI flag).
    pub enable: Option<bool>,
    /// Remove raw parquet after hive (same as `--clean-raw` CLI flag).
    pub clean_raw: Option<bool>,
    pub zstd_level: Option<i32>,
    pub row_group_size: Option<usize>,
    pub num_shards: Option<usize>,
    pub threads: Option<usize>,
    pub memory_limit: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FilterConfig {
    #[serde(default)]
    pub domains: Vec<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub work_types: Vec<String>,
    #[serde(default)]
    pub require_abstract: bool,
}

/// `[upload]` section — S3-compatible upload (e.g. Cloudflare R2).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UploadConfig {
    /// S3 bucket name
    pub bucket: Option<String>,
    /// S3 endpoint URL (e.g. "https://<account>.r2.cloudflarestorage.com")
    pub endpoint: Option<String>,
    /// S3 region (default: "auto" for R2)
    pub region: Option<String>,
    /// Access key (or use $R2_ACCESS_KEY_ID / $AWS_ACCESS_KEY_ID env var)
    pub access_key: Option<String>,
    /// Secret key (or use $R2_SECRET_ACCESS_KEY / $AWS_SECRET_ACCESS_KEY env var)
    pub secret_key: Option<String>,
    /// Key prefix in bucket (e.g. "papeline/hive")
    pub prefix: Option<String>,
}

/// Fully resolved upload configuration.
#[derive(Debug, Clone)]
pub struct ResolvedUploadConfig {
    pub bucket: String,
    pub endpoint: String,
    pub region: String,
    pub access_key: String,
    pub secret_key: String,
    pub prefix: String,
}

impl ResolvedUploadConfig {
    /// Resolve upload config from TOML + env vars.
    ///
    /// Env vars: R2_ACCESS_KEY_ID / AWS_ACCESS_KEY_ID,
    ///           R2_SECRET_ACCESS_KEY / AWS_SECRET_ACCESS_KEY,
    ///           R2_ENDPOINT, R2_BUCKET.
    pub fn from_config(cfg: &UploadConfig) -> anyhow::Result<Self> {
        let bucket = cfg
            .bucket
            .clone()
            .or_else(|| std::env::var("R2_BUCKET").ok())
            .ok_or_else(|| anyhow::anyhow!("upload.bucket or $R2_BUCKET required"))?;

        let endpoint = cfg
            .endpoint
            .clone()
            .or_else(|| std::env::var("R2_ENDPOINT").ok())
            .ok_or_else(|| anyhow::anyhow!("upload.endpoint or $R2_ENDPOINT required"))?;

        let region = cfg.region.clone().unwrap_or_else(|| "auto".to_string());

        let access_key = cfg
            .access_key
            .clone()
            .or_else(|| std::env::var("R2_ACCESS_KEY_ID").ok())
            .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "upload.access_key or $R2_ACCESS_KEY_ID / $AWS_ACCESS_KEY_ID required"
                )
            })?;

        let secret_key = cfg
            .secret_key
            .clone()
            .or_else(|| std::env::var("R2_SECRET_ACCESS_KEY").ok())
            .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "upload.secret_key or $R2_SECRET_ACCESS_KEY / $AWS_SECRET_ACCESS_KEY required"
                )
            })?;

        let prefix = cfg.prefix.clone().unwrap_or_default();

        Ok(Self {
            bucket,
            endpoint,
            region,
            access_key,
            secret_key,
            prefix,
        })
    }
}

/// Try loading config from explicit `--config` path or `./papeline.toml` in CWD.
pub fn load_config(explicit: Option<&Path>) -> anyhow::Result<FileConfig> {
    let path = if let Some(p) = explicit {
        if !p.exists() {
            anyhow::bail!("Config file not found: {}", p.display());
        }
        Some(p.to_path_buf())
    } else {
        let auto = PathBuf::from("./papeline.toml");
        if auto.exists() {
            tracing::info!("Using config: {}", auto.display());
            Some(auto)
        } else {
            None
        }
    };

    match path {
        Some(p) => {
            let text = fs::read_to_string(&p)
                .map_err(|e| anyhow::anyhow!("Failed to read {}: {e}", p.display()))?;
            let cfg: FileConfig = toml::from_str(&text)
                .map_err(|e| anyhow::anyhow!("Invalid TOML in {}: {e}", p.display()))?;
            Ok(cfg)
        }
        None => Ok(FileConfig::default()),
    }
}

// ============================================================
// Resolved config (CLI > config file > defaults)
// ============================================================

/// Fully resolved configuration with all defaults applied.
#[derive(Debug)]
pub struct ResolvedConfig {
    pub output_dir: PathBuf,
    pub zstd_level: i32,
    pub concurrency: usize,
    pub max_retries: u32,
    pub read_timeout: u64,
    pub outer_retries: u32,
    pub retry_delay: u64,
    pub filter: Filter,
    pub force: bool,
    pub since: Option<String>,
    pub max_shards: Option<usize>,
    pub dry_run: bool,
}

/// Fully resolved hive configuration with all defaults applied.
#[derive(Debug)]
pub struct ResolvedHiveConfig {
    pub raw_dir: PathBuf,
    pub hive_dir: PathBuf,
    pub staging_dir: PathBuf,
    pub zstd_level: i32,
    pub row_group_size: usize,
    pub num_shards: usize,
    pub threads: usize,
    pub memory_limit_bytes: usize,
}

impl ResolvedHiveConfig {
    /// Resolve hive config with fallback chain:
    /// CLI > `[hive]` section > `fallback_zstd` (top-level) > defaults.
    ///
    /// `output_dir` is the root (e.g. `./outputs`).
    /// Raw shards are read from `output_dir/raw/`, hive written to `output_dir/hive/`.
    pub fn from_config(
        output_dir: &Path,
        hive: &HiveConfig,
        fallback_zstd: Option<i32>,
    ) -> anyhow::Result<Self> {
        let raw_dir = output_dir.join("raw");
        let hive_dir = output_dir.join("hive");
        let staging_dir = hive_dir.join(".staging");

        let memory_limit_bytes = match &hive.memory_limit {
            Some(s) => parse_memory_limit(s)?,
            None => (system_memory_bytes() * 65 / 100) as usize,
        };

        // 3/4 of cores: avoids saturating efficiency cores on heterogeneous CPUs (e.g. M4 Max)
        let threads = hive
            .threads
            .unwrap_or_else(|| (num_cpus::get() * 3 / 4).max(2));

        Ok(Self {
            raw_dir,
            hive_dir,
            staging_dir,
            zstd_level: hive.zstd_level.or(fallback_zstd).unwrap_or(8),
            row_group_size: hive.row_group_size.unwrap_or(500_000),
            num_shards: hive.num_shards.unwrap_or(4),
            threads,
            memory_limit_bytes,
        })
    }
}

/// Format byte count as human-readable string (e.g. "41GB", "512MB").
pub fn format_bytes(bytes: usize) -> String {
    let gb = bytes / (1024 * 1024 * 1024);
    if gb > 0 {
        format!("{gb}GB")
    } else {
        let mb = bytes / (1024 * 1024);
        format!("{mb}MB")
    }
}

/// Parse human-readable memory limit string (e.g. "32GB", "512MB") to bytes.
pub fn parse_memory_limit(s: &str) -> anyhow::Result<usize> {
    let s = s.trim();
    let (num_str, multiplier) = if let Some(n) = s.strip_suffix("GB") {
        (n, 1024 * 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("MB") {
        (n, 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("KB") {
        (n, 1024)
    } else {
        (s, 1usize)
    };
    let num: usize = num_str
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid memory limit: {s}"))?;
    Ok(num * multiplier)
}

fn system_memory_bytes() -> u64 {
    // /proc/meminfo on Linux, sysctl on macOS; fallback to 8GB
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse::<u64>().ok())
                            .map(|kb| kb * 1024)
                    })
            })
            .unwrap_or(8 * 1024 * 1024 * 1024)
    }
    #[cfg(target_os = "macos")]
    {
        // sysctl hw.memsize returns total physical memory in bytes
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .trim()
                    .parse::<u64>()
                    .ok()
            })
            .unwrap_or(8 * 1024 * 1024 * 1024)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        8 * 1024 * 1024 * 1024
    }
}

/// Build filter from merged CLI + config domains/topics/languages.
pub fn build_filter(
    cli_domains: &[String],
    cli_topics: &[String],
    cli_languages: &[String],
    cli_work_types: &[String],
    cli_require_abstract: bool,
    cfg: &FilterConfig,
) -> Filter {
    let mut filter = Filter::default();
    // CLI overrides config if any CLI values given; otherwise use config
    let domains = if cli_domains.is_empty() {
        &cfg.domains
    } else {
        cli_domains
    };
    let topics = if cli_topics.is_empty() {
        &cfg.topics
    } else {
        cli_topics
    };
    let languages = if cli_languages.is_empty() {
        &cfg.languages
    } else {
        cli_languages
    };
    let work_types = if cli_work_types.is_empty() {
        &cfg.work_types
    } else {
        cli_work_types
    };
    for d in domains {
        filter.domains.insert(d.clone());
    }
    for t in topics {
        filter.topic_ids.insert(t.clone());
    }
    for l in languages {
        filter.languages.insert(l.clone());
    }
    for w in work_types {
        filter.work_types.insert(w.clone());
    }
    filter.require_abstract = cli_require_abstract || cfg.require_abstract;
    filter
}

// ============================================================
// State file (.state.json)
// ============================================================

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    pub last_sync_date: Option<String>,
    #[serde(default)]
    pub completed_shards: Vec<usize>,
    #[serde(default)]
    pub manifest_shard_count: usize,
    #[serde(default)]
    pub filter: StateFilter,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub finished_at: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct StateFilter {
    #[serde(default)]
    pub domains: Vec<String>,
    #[serde(default)]
    pub topics: Vec<String>,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub work_types: Vec<String>,
    #[serde(default)]
    pub require_abstract: bool,
}

impl State {
    pub fn load(output_dir: &Path) -> Self {
        let path = output_dir.join(".state.json");
        if !path.exists() {
            return Self::default();
        }
        match fs::read_to_string(&path) {
            Ok(text) => match sonic_rs::from_str(&text) {
                Ok(state) => state,
                Err(e) => {
                    tracing::warn!("Corrupt state file {}: {e}", path.display());
                    Self::default()
                }
            },
            Err(e) => {
                tracing::warn!("Failed to read state file {}: {e}", path.display());
                Self::default()
            }
        }
    }

    /// Atomic write: .state.json.tmp -> rename .state.json
    pub fn save(&self, output_dir: &Path) -> anyhow::Result<()> {
        let path = output_dir.join(".state.json");
        let tmp = output_dir.join(".state.json.tmp");
        let json = sonic_rs::to_string_pretty(self)?;
        fs::write(&tmp, json)?;
        fs::rename(&tmp, &path)?;
        Ok(())
    }
}

// ============================================================
// Sync log (.sync_log.jsonl)
// ============================================================

#[derive(Debug, Serialize)]
pub struct SyncLogEntry {
    pub timestamp: String,
    pub shards_processed: Vec<usize>,
    pub shards_removed: Vec<usize>,
    pub updated_dates: Vec<String>,
    pub rows_written: usize,
    pub failed: usize,
}

impl SyncLogEntry {
    pub fn append(output_dir: &Path, entry: &SyncLogEntry) -> anyhow::Result<()> {
        use std::io::Write;
        let path = output_dir.join(".sync_log.jsonl");
        let line = sonic_rs::to_string(entry)?;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ---- load_config ----

    #[test]
    fn load_config_explicit_path() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("my.toml");
        fs::write(
            &path,
            r#"
output = "/data/out"
zstd_level = 5
concurrency = 16

[filter]
domains = ["Health Sciences"]
topics = ["T123"]
"#,
        )
        .unwrap();

        let cfg = load_config(Some(path.as_path())).unwrap();
        assert_eq!(cfg.output.as_deref(), Some("/data/out"));
        assert_eq!(cfg.zstd_level, Some(5));
        assert_eq!(cfg.filter.domains, vec!["Health Sciences"]);
        assert_eq!(cfg.filter.topics, vec!["T123"]);
        assert_eq!(cfg.concurrency, Some(16));
    }

    #[test]
    fn load_config_missing_explicit() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("nope.toml");
        let err = load_config(Some(missing.as_path()));
        assert!(err.is_err());
        assert!(
            err.unwrap_err().to_string().contains("not found"),
            "error should mention 'not found'"
        );
    }

    #[test]
    fn load_config_invalid_toml() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.toml");
        fs::write(&path, "this is not [[[valid toml").unwrap();

        let err = load_config(Some(path.as_path()));
        assert!(err.is_err());
    }

    #[test]
    fn load_config_rejects_old_format() {
        let dir = TempDir::new().unwrap();
        // Old format: [run] section is no longer valid
        let path = dir.path().join("old.toml");
        fs::write(&path, "[run]\nconcurrency = 8\n").unwrap();
        let err = load_config(Some(path.as_path()));
        assert!(err.is_err());
        assert!(
            err.unwrap_err().to_string().contains("unknown field"),
            "should mention unknown field"
        );
    }

    #[test]
    fn load_config_all_sections() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("full.toml");
        fs::write(
            &path,
            r#"
output = "/data"
zstd_level = 9
concurrency = 16

[filter]
domains = ["Health Sciences"]

[hive]
enable = true
clean_raw = true
zstd_level = 8
threads = 12
memory_limit = "32GB"

[upload]
bucket = "my-bucket"
endpoint = "https://acct.r2.cloudflarestorage.com"
prefix = "papeline/hive"
"#,
        )
        .unwrap();
        let cfg = load_config(Some(path.as_path())).unwrap();
        assert_eq!(cfg.zstd_level, Some(9));
        assert_eq!(cfg.concurrency, Some(16));
        assert_eq!(cfg.filter.domains, vec!["Health Sciences"]);
        assert_eq!(cfg.hive.enable, Some(true));
        assert_eq!(cfg.hive.clean_raw, Some(true));
        assert_eq!(cfg.upload.bucket.as_deref(), Some("my-bucket"));
        assert_eq!(
            cfg.upload.endpoint.as_deref(),
            Some("https://acct.r2.cloudflarestorage.com")
        );
        assert_eq!(cfg.upload.prefix.as_deref(), Some("papeline/hive"));
        assert_eq!(cfg.hive.zstd_level, Some(8));
        assert_eq!(cfg.hive.threads, Some(12));
        assert_eq!(cfg.hive.memory_limit.as_deref(), Some("32GB"));
    }

    #[test]
    fn resolved_hive_inherits_global_zstd() {
        let hive = HiveConfig::default();
        let r = ResolvedHiveConfig::from_config(Path::new("/tmp"), &hive, Some(9)).unwrap();
        assert_eq!(r.zstd_level, 9);
        assert_eq!(r.raw_dir, PathBuf::from("/tmp/raw"));
        assert_eq!(r.hive_dir, PathBuf::from("/tmp/hive"));
    }

    #[test]
    fn resolved_hive_section_overrides_global() {
        let hive = HiveConfig {
            zstd_level: Some(5),
            ..Default::default()
        };
        let r = ResolvedHiveConfig::from_config(Path::new("/tmp"), &hive, Some(9)).unwrap();
        assert_eq!(r.zstd_level, 5);
    }

    #[test]
    fn resolved_hive_default_without_global() {
        let hive = HiveConfig::default();
        let r = ResolvedHiveConfig::from_config(Path::new("/tmp"), &hive, None).unwrap();
        assert_eq!(r.zstd_level, 8);
    }

    #[test]
    fn load_config_no_config() {
        // No explicit path, and CWD auto-discovery may or may not find a file.
        // Just verify that None doesn't panic.
        let cfg = load_config(None).unwrap();
        // If CWD has no papeline.toml, we get defaults.
        // If it does (e.g. repo root), we get that file's config.
        // Either way, it must succeed.
        let _ = cfg;
    }

    // ---- build_filter ----

    #[test]
    fn build_filter_cli_overrides() {
        let cli_domains = vec!["D1".to_string()];
        let cli_topics = vec!["T1".to_string()];
        let cfg = FilterConfig {
            domains: vec!["D2".to_string()],
            topics: vec!["T2".to_string()],
            languages: vec![],
            work_types: vec![],
            require_abstract: false,
        };
        let f = build_filter(&cli_domains, &cli_topics, &[], &[], false, &cfg);
        assert!(f.domains.contains("D1"));
        assert!(!f.domains.contains("D2"));
        assert!(f.topic_ids.contains("T1"));
        assert!(!f.topic_ids.contains("T2"));
    }

    #[test]
    fn build_filter_config_fallback() {
        let cfg = FilterConfig {
            domains: vec!["D2".to_string()],
            topics: vec!["T2".to_string()],
            languages: vec![],
            work_types: vec![],
            require_abstract: false,
        };
        let f = build_filter(&[], &[], &[], &[], false, &cfg);
        assert!(f.domains.contains("D2"));
        assert!(f.topic_ids.contains("T2"));
    }

    #[test]
    fn build_filter_empty() {
        let cfg = FilterConfig::default();
        let f = build_filter(&[], &[], &[], &[], false, &cfg);
        assert!(f.is_empty());
    }

    // ---- State ----

    #[test]
    fn state_save_load_roundtrip() {
        let dir = TempDir::new().unwrap();
        let state = State {
            completed_shards: vec![0, 1, 2],
            last_sync_date: Some("2025-01-01".to_string()),
            manifest_shard_count: 100,
            ..Default::default()
        };
        state.save(dir.path()).unwrap();
        let loaded = State::load(dir.path());
        assert_eq!(loaded.completed_shards, vec![0, 1, 2]);
        assert_eq!(loaded.last_sync_date.as_deref(), Some("2025-01-01"));
        assert_eq!(loaded.manifest_shard_count, 100);
    }

    #[test]
    fn state_load_missing() {
        let dir = TempDir::new().unwrap();
        let s = State::load(dir.path());
        assert!(s.completed_shards.is_empty());
        assert!(s.last_sync_date.is_none());
    }

    #[test]
    fn state_load_corrupted() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join(".state.json"), "NOT JSON {{{{").unwrap();
        let s = State::load(dir.path());
        // Should return default, not panic
        assert!(s.completed_shards.is_empty());
    }

    // ---- SyncLogEntry ----

    #[test]
    fn sync_log_append() {
        let dir = TempDir::new().unwrap();
        let entry1 = SyncLogEntry {
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            shards_processed: vec![0, 1],
            shards_removed: vec![],
            updated_dates: vec!["2025-01-01".to_string()],
            rows_written: 100,
            failed: 0,
        };
        let entry2 = SyncLogEntry {
            timestamp: "2025-01-02T00:00:00Z".to_string(),
            shards_processed: vec![2],
            shards_removed: vec![5],
            updated_dates: vec!["2025-01-02".to_string()],
            rows_written: 50,
            failed: 1,
        };
        SyncLogEntry::append(dir.path(), &entry1).unwrap();
        SyncLogEntry::append(dir.path(), &entry2).unwrap();

        let content = fs::read_to_string(dir.path().join(".sync_log.jsonl")).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("2025-01-01T00:00:00Z"));
        assert!(lines[1].contains("2025-01-02T00:00:00Z"));
    }

    // ---- ResolvedUploadConfig ----

    #[test]
    fn resolved_upload_from_config_all_fields() {
        let cfg = UploadConfig {
            bucket: Some("test-bucket".to_string()),
            endpoint: Some("https://example.com".to_string()),
            region: Some("us-east-1".to_string()),
            access_key: Some("AK".to_string()),
            secret_key: Some("SK".to_string()),
            prefix: Some("data/hive".to_string()),
        };
        let resolved = ResolvedUploadConfig::from_config(&cfg).unwrap();
        assert_eq!(resolved.bucket, "test-bucket");
        assert_eq!(resolved.endpoint, "https://example.com");
        assert_eq!(resolved.region, "us-east-1");
        assert_eq!(resolved.access_key, "AK");
        assert_eq!(resolved.secret_key, "SK");
        assert_eq!(resolved.prefix, "data/hive");
    }

    #[test]
    fn resolved_upload_defaults_region_auto() {
        let cfg = UploadConfig {
            bucket: Some("b".to_string()),
            endpoint: Some("https://e.com".to_string()),
            region: None,
            access_key: Some("AK".to_string()),
            secret_key: Some("SK".to_string()),
            prefix: None,
        };
        let resolved = ResolvedUploadConfig::from_config(&cfg).unwrap();
        assert_eq!(resolved.region, "auto");
        assert!(resolved.prefix.is_empty());
    }

    #[test]
    fn resolved_upload_missing_bucket_errors() {
        let cfg = UploadConfig::default();
        let err = ResolvedUploadConfig::from_config(&cfg);
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("bucket"));
    }
}
