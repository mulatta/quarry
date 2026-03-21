//! Build configuration: thread counts, S3 settings, tier domains.

use std::collections::HashSet;

/// Default T2 domains — mirrors quarry/config.py:oa_t2_domains.
pub const DEFAULT_T2_DOMAINS: &[&str] = &[
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Engineering",
];

/// Maximum concurrent S3 downloads to prevent resource exhaustion.
const MAX_S3_CONCURRENCY: usize = 64;

/// Top-level build configuration.
pub struct BuildConfig {
    /// T2 tier domain names.
    pub t2_domains: Vec<String>,
    /// Number of concurrent S3 downloads (clamped to MAX_S3_CONCURRENCY).
    pub s3_download_concurrency: usize,
    /// Max parallel parse threads (rayon). Bounds memory during PubMed baseline
    /// parsing: each thread holds ~400MB. `None` = auto from available memory.
    pub parse_threads: Option<usize>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            t2_domains: DEFAULT_T2_DOMAINS.iter().map(|s| s.to_string()).collect(),
            s3_download_concurrency: 8,
            parse_threads: None,
        }
    }
}

impl BuildConfig {
    /// Return T2 domains as a HashSet for O(1) lookup.
    pub fn t2_domains_set(&self) -> HashSet<String> {
        self.t2_domains.iter().cloned().collect()
    }

    /// Effective S3 concurrency, clamped to a safe upper bound.
    pub fn effective_s3_concurrency(&self) -> usize {
        self.s3_download_concurrency.clamp(1, MAX_S3_CONCURRENCY)
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
        let cfg = BuildConfig {
            s3_download_concurrency: 16,
            ..Default::default()
        };
        assert_eq!(cfg.effective_s3_concurrency(), 16);
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
}
