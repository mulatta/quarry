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
    /// Max retry attempts for remote fetch operations (0 = no retry).
    pub fetch_max_retries: u32,
    /// Initial backoff duration in milliseconds (doubles each retry).
    pub fetch_initial_backoff_ms: u64,
    /// Maximum backoff duration in milliseconds (cap for exponential growth).
    pub fetch_max_backoff_ms: u64,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            t2_domains: DEFAULT_T2_DOMAINS.iter().map(|s| s.to_string()).collect(),
            s3_download_concurrency: 32,
            parse_threads: None,
            fetch_max_retries: 3,
            fetch_initial_backoff_ms: 2_000,
            fetch_max_backoff_ms: 30_000,
        }
    }
}

impl BuildConfig {
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
        assert_eq!(cfg.effective_s3_concurrency(), 32);
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
        // 2000 * 2^2 = 8000, but capped at 5000
        assert_eq!(cfg.backoff_duration(2), std::time::Duration::from_millis(5000));
    }

    #[test]
    fn test_backoff_overflow_safe() {
        let cfg = BuildConfig {
            fetch_initial_backoff_ms: u64::MAX,
            fetch_max_backoff_ms: 30_000,
            ..Default::default()
        };
        // saturating_mul overflow → capped at max
        assert_eq!(cfg.backoff_duration(5), std::time::Duration::from_millis(30_000));
    }
}
