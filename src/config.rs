//! Configuration loading (TOML) and state/sync-log persistence.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::transform::Filter;

// ============================================================
// TOML config file
// ============================================================

/// Parsed from `papeline.toml`.
#[derive(Debug, Default, Deserialize)]
pub struct FileConfig {
    pub output: Option<String>,
    pub zstd_level: Option<i32>,
    #[serde(default)]
    pub filter: FilterConfig,
    #[serde(default)]
    pub http: HttpConfig,
    #[serde(default)]
    pub hive: HiveConfig,
}

#[derive(Debug, Default, Deserialize)]
pub struct HiveConfig {
    pub zstd_level: Option<i32>,
    pub row_group_size: Option<usize>,
    pub threads: Option<usize>,
    pub memory_limit: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct FilterConfig {
    #[serde(default)]
    pub domains: Vec<String>,
    #[serde(default)]
    pub topics: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct HttpConfig {
    pub concurrency: Option<usize>,
    pub max_retries: Option<u32>,
    pub read_timeout: Option<u64>,
    pub outer_retries: Option<u32>,
    pub retry_delay: Option<u64>,
}

/// Try loading config from explicit path or `{output_dir}/papeline.toml`.
pub fn load_config(explicit: Option<&Path>, output_dir: &Path) -> anyhow::Result<FileConfig> {
    let path = if let Some(p) = explicit {
        if !p.exists() {
            anyhow::bail!("Config file not found: {}", p.display());
        }
        Some(p.to_path_buf())
    } else {
        let auto = output_dir.join("papeline.toml");
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
    pub threads: usize,
    pub memory_limit_bytes: usize,
}

impl ResolvedHiveConfig {
    pub fn from_config(output_dir: &Path, hive: &HiveConfig) -> anyhow::Result<Self> {
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
            raw_dir: output_dir.to_path_buf(),
            hive_dir,
            staging_dir,
            zstd_level: hive.zstd_level.unwrap_or(8),
            row_group_size: hive.row_group_size.unwrap_or(500_000),
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

/// Build filter from merged CLI + config domains/topics.
pub fn build_filter(cli_domains: &[String], cli_topics: &[String], cfg: &FilterConfig) -> Filter {
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
    for d in domains {
        filter.domains.insert(d.clone());
    }
    for t in topics {
        filter.topic_ids.insert(t.clone());
    }
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

[filter]
domains = ["Health Sciences"]
topics = ["T123"]

[http]
concurrency = 16
"#,
        )
        .unwrap();

        let cfg = load_config(Some(path.as_path()), dir.path()).unwrap();
        assert_eq!(cfg.output.as_deref(), Some("/data/out"));
        assert_eq!(cfg.zstd_level, Some(5));
        assert_eq!(cfg.filter.domains, vec!["Health Sciences"]);
        assert_eq!(cfg.filter.topics, vec!["T123"]);
        assert_eq!(cfg.http.concurrency, Some(16));
    }

    #[test]
    fn load_config_auto_discovery() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("papeline.toml");
        fs::write(&path, "zstd_level = 9\n").unwrap();

        let cfg = load_config(None, dir.path()).unwrap();
        assert_eq!(cfg.zstd_level, Some(9));
    }

    #[test]
    fn load_config_missing_explicit() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("nope.toml");
        let err = load_config(Some(missing.as_path()), dir.path());
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

        let err = load_config(Some(path.as_path()), dir.path());
        assert!(err.is_err());
    }

    #[test]
    fn load_config_no_config() {
        let dir = TempDir::new().unwrap();
        // No papeline.toml, no explicit path
        let cfg = load_config(None, dir.path()).unwrap();
        assert!(cfg.output.is_none());
        assert!(cfg.zstd_level.is_none());
        assert!(cfg.filter.domains.is_empty());
    }

    // ---- build_filter ----

    #[test]
    fn build_filter_cli_overrides() {
        let cli_domains = vec!["D1".to_string()];
        let cli_topics = vec!["T1".to_string()];
        let cfg = FilterConfig {
            domains: vec!["D2".to_string()],
            topics: vec!["T2".to_string()],
        };
        let f = build_filter(&cli_domains, &cli_topics, &cfg);
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
        };
        let f = build_filter(&[], &[], &cfg);
        assert!(f.domains.contains("D2"));
        assert!(f.topic_ids.contains("T2"));
    }

    #[test]
    fn build_filter_empty() {
        let cfg = FilterConfig::default();
        let f = build_filter(&[], &[], &cfg);
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
}
