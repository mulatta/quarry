//! Build configuration: thread counts, S3 settings, tier domains.
//!
//! Priority: CLI args > env vars > config.toml > defaults.

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;

/// Default T2 domains — mirrors quarry/config.py:oa_t2_domains.
pub const DEFAULT_T2_DOMAINS: &[&str] = &[
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Engineering",
];

/// Maximum concurrent S3 downloads to prevent resource exhaustion.
const MAX_S3_CONCURRENCY: usize = 64;

/// Maximum PG writer threads — beyond this PG I/O contention dominates.
const MAX_PG_WRITERS: usize = 16;

/// Maximum prefetch buffer to bound memory usage (~100MB per slot).
const MAX_PREFETCH: usize = 64;

/// Top-level build configuration.
///
/// Deserializable from TOML. All fields have serde defaults matching
/// `Default::default()`, so partial TOML files work.
#[derive(Deserialize)]
#[serde(default)]
pub struct BuildConfig {
    /// T2 tier domain names.
    pub t2_domains: Vec<String>,
    /// Number of concurrent S3 downloads (clamped to MAX_S3_CONCURRENCY).
    pub s3_download_concurrency: usize,
    /// Downloaded-but-not-yet-parsed file buffer slots.
    /// Decouples download from parse — larger values keep downloads flowing
    /// while parse is CPU-bound. Each slot holds ~50-100MB.
    /// 0 = auto (s3_download_concurrency).
    pub prefetch_buffer: usize,
    /// Max parallel parse threads (rayon). Bounds memory during PubMed baseline
    /// parsing: each thread holds ~400MB. `None` = auto from available memory.
    pub parse_threads: Option<usize>,
    /// Max retry attempts for remote fetch operations (0 = no retry).
    pub fetch_max_retries: u32,
    /// Initial backoff duration in milliseconds (doubles each retry).
    pub fetch_initial_backoff_ms: u64,
    /// Maximum backoff duration in milliseconds (cap for exponential growth).
    pub fetch_max_backoff_ms: u64,
    /// Number of parallel PG COPY writer threads (clamped to MAX_PG_WRITERS).
    pub pg_writer_threads: usize,
    /// Bounded channel buffer between download/parse and PG writers.
    /// 0 = auto (s3_download_concurrency * 2).
    pub channel_buffer: usize,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            t2_domains: DEFAULT_T2_DOMAINS.iter().map(|s| s.to_string()).collect(),
            s3_download_concurrency: 8,
            prefetch_buffer: 0, // auto
            parse_threads: None,
            fetch_max_retries: 3,
            fetch_initial_backoff_ms: 2_000,
            fetch_max_backoff_ms: 30_000,
            pg_writer_threads: 4,
            channel_buffer: 0, // auto
        }
    }
}

impl BuildConfig {
    /// Load config from a TOML file. Missing fields use defaults.
    pub fn from_toml(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read config {}: {e}", path.display()))?;
        let config: Self = toml::from_str(&content)
            .map_err(|e| format!("invalid config {}: {e}", path.display()))?;
        Ok(config)
    }

    /// Return T2 domains as a HashSet for O(1) lookup.
    pub fn t2_domains_set(&self) -> HashSet<String> {
        self.t2_domains.iter().cloned().collect()
    }

    /// Compute backoff duration for a given attempt (0-indexed).
    /// Exponential: initial_ms * 2^attempt, capped at max_ms.
    pub fn backoff_duration(&self, attempt: u32) -> std::time::Duration {
        let shift = attempt.min(63);
        let ms = self
            .fetch_initial_backoff_ms
            .saturating_mul(1u64 << shift)
            .min(self.fetch_max_backoff_ms);
        std::time::Duration::from_millis(ms)
    }

    /// Effective S3 concurrency, clamped to a safe upper bound.
    pub fn effective_s3_concurrency(&self) -> usize {
        self.s3_download_concurrency.clamp(1, MAX_S3_CONCURRENCY)
    }

    /// Effective prefetch buffer: downloaded-but-not-yet-parsed files in memory.
    /// Auto = s3_concurrency (allows full download parallelism while parsing).
    pub fn effective_prefetch_buffer(&self) -> usize {
        if self.prefetch_buffer == 0 {
            self.effective_s3_concurrency()
        } else {
            self.prefetch_buffer.clamp(1, MAX_PREFETCH)
        }
    }

    /// Effective PG writer thread count.
    pub fn effective_pg_writer_threads(&self) -> usize {
        self.pg_writer_threads.clamp(1, MAX_PG_WRITERS)
    }

    /// Effective channel buffer size.
    pub fn effective_channel_buffer(&self) -> usize {
        if self.channel_buffer == 0 {
            self.effective_s3_concurrency() * 2
        } else {
            self.channel_buffer.max(1)
        }
    }

    /// Effective parse thread count: explicit value, or auto-detect from
    /// available system memory. Each parse thread holds ~400MB.
    pub fn effective_parse_threads(&self) -> usize {
        if let Some(n) = self.parse_threads {
            return n.max(1);
        }
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let mem_threads = available_memory_gb() / 0.4;
        let n = (mem_threads as usize).min(cpus).max(1);
        eprintln!(
            "parse_threads: auto={n} (cpus={cpus}, mem_limit={})",
            (mem_threads as usize)
        );
        n
    }
}

/// Read available memory in GB from /proc/meminfo (Linux).
/// Falls back to 16GB if unavailable.
fn available_memory_gb() -> f64 {
    let content = match std::fs::read_to_string("/proc/meminfo") {
        Ok(c) => c,
        Err(_) => return 16.0,
    };
    for line in content.lines() {
        if line.starts_with("MemAvailable:") {
            let kb: f64 = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(16_000_000.0);
            return kb / 1_048_576.0;
        }
    }
    16.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_t2_domains() {
        let cfg = BuildConfig::default();
        let set = cfg.t2_domains_set();
        assert!(set.contains("Health Sciences"));
        assert!(set.contains("Engineering"));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_s3_concurrency_clamp_zero() {
        let cfg = BuildConfig {
            s3_download_concurrency: 0,
            ..Default::default()
        };
        assert_eq!(cfg.effective_s3_concurrency(), 1);
    }

    #[test]
    fn test_s3_concurrency_clamp_high() {
        let cfg = BuildConfig {
            s3_download_concurrency: 1000,
            ..Default::default()
        };
        assert_eq!(cfg.effective_s3_concurrency(), 64);
    }

    #[test]
    fn test_s3_concurrency_normal() {
        let cfg = BuildConfig::default();
        assert_eq!(cfg.effective_s3_concurrency(), 8);
    }

    #[test]
    fn test_prefetch_buffer_auto() {
        let cfg = BuildConfig::default();
        assert_eq!(cfg.effective_prefetch_buffer(), 8); // = s3_concurrency
    }

    #[test]
    fn test_prefetch_buffer_explicit() {
        let cfg = BuildConfig {
            prefetch_buffer: 16,
            ..Default::default()
        };
        assert_eq!(cfg.effective_prefetch_buffer(), 16);
    }

    #[test]
    fn test_prefetch_buffer_clamp() {
        let cfg = BuildConfig {
            prefetch_buffer: 200,
            ..Default::default()
        };
        assert_eq!(cfg.effective_prefetch_buffer(), 64);
    }

    #[test]
    fn test_parse_threads_explicit() {
        let cfg = BuildConfig {
            parse_threads: Some(4),
            ..Default::default()
        };
        assert_eq!(cfg.effective_parse_threads(), 4);
    }

    #[test]
    fn test_parse_threads_explicit_zero_becomes_one() {
        let cfg = BuildConfig {
            parse_threads: Some(0),
            ..Default::default()
        };
        assert_eq!(cfg.effective_parse_threads(), 1);
    }

    #[test]
    fn test_parse_threads_auto() {
        let cfg = BuildConfig::default();
        let n = cfg.effective_parse_threads();
        assert!(n >= 1);
    }

    #[test]
    fn test_pg_writers_default() {
        let cfg = BuildConfig::default();
        assert_eq!(cfg.effective_pg_writer_threads(), 4);
    }

    #[test]
    fn test_pg_writers_clamp() {
        let cfg = BuildConfig {
            pg_writer_threads: 100,
            ..Default::default()
        };
        assert_eq!(cfg.effective_pg_writer_threads(), 16);

        let cfg = BuildConfig {
            pg_writer_threads: 0,
            ..Default::default()
        };
        assert_eq!(cfg.effective_pg_writer_threads(), 1);
    }

    #[test]
    fn test_channel_buffer_auto() {
        let cfg = BuildConfig::default();
        assert_eq!(cfg.effective_channel_buffer(), 16); // 8 * 2
    }

    #[test]
    fn test_channel_buffer_explicit() {
        let cfg = BuildConfig {
            channel_buffer: 128,
            ..Default::default()
        };
        assert_eq!(cfg.effective_channel_buffer(), 128);
    }

    #[test]
    fn test_backoff_exponential() {
        let cfg = BuildConfig {
            fetch_initial_backoff_ms: 1000,
            fetch_max_backoff_ms: 30_000,
            ..Default::default()
        };
        assert_eq!(cfg.backoff_duration(0), std::time::Duration::from_millis(1000));
        assert_eq!(cfg.backoff_duration(1), std::time::Duration::from_millis(2000));
        assert_eq!(cfg.backoff_duration(2), std::time::Duration::from_millis(4000));
    }

    #[test]
    fn test_backoff_capped_at_max() {
        let cfg = BuildConfig {
            fetch_initial_backoff_ms: 2000,
            fetch_max_backoff_ms: 5000,
            ..Default::default()
        };
        assert_eq!(cfg.backoff_duration(2), std::time::Duration::from_millis(5000));
    }

    #[test]
    fn test_backoff_overflow_safe() {
        let cfg = BuildConfig {
            fetch_initial_backoff_ms: u64::MAX,
            fetch_max_backoff_ms: 30_000,
            ..Default::default()
        };
        assert_eq!(cfg.backoff_duration(5), std::time::Duration::from_millis(30_000));
    }

    #[test]
    fn test_from_toml_partial() {
        let toml_str = r#"
            s3_download_concurrency = 16
            prefetch_buffer = 32
        "#;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.toml");
        std::fs::write(&path, toml_str).unwrap();

        let cfg = BuildConfig::from_toml(&path).unwrap();
        assert_eq!(cfg.s3_download_concurrency, 16);
        assert_eq!(cfg.prefetch_buffer, 32);
        // defaults preserved for unspecified fields
        assert_eq!(cfg.pg_writer_threads, 4);
        assert_eq!(cfg.fetch_max_retries, 3);
    }

    #[test]
    fn test_from_toml_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.toml");
        std::fs::write(&path, "").unwrap();

        let cfg = BuildConfig::from_toml(&path).unwrap();
        assert_eq!(cfg.s3_download_concurrency, 8);
        assert_eq!(cfg.prefetch_buffer, 0);
    }
}
