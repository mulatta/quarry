//! Parse configuration: thread counts, tier domains.

use std::collections::HashSet;

/// Default T2 domains — mirrors quarry/config.py:oa_t2_domains.
pub const DEFAULT_T2_DOMAINS: &[&str] = &[
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Engineering",
];

/// Parse configuration — thread count and tier classification.
pub struct ParseConfig {
    /// T2 tier domain names.
    pub t2_domains: Vec<String>,
    /// Max parallel parse threads (rayon). Each thread holds ~400MB.
    /// `None` = auto from available memory.
    pub parse_threads: Option<usize>,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            t2_domains: DEFAULT_T2_DOMAINS.iter().map(|s| s.to_string()).collect(),
            parse_threads: None,
        }
    }
}

impl ParseConfig {
    /// Return T2 domains as a HashSet for O(1) lookup.
    pub fn t2_domains_set(&self) -> HashSet<String> {
        self.t2_domains.iter().cloned().collect()
    }

    /// Effective parse thread count: explicit value, or auto-detect from
    /// available system memory. Each parse thread holds ~2GB peak
    /// (decompressed data + parsed structs + Arrow batch).
    pub fn effective_parse_threads(&self) -> usize {
        if let Some(n) = self.parse_threads {
            return n.max(1);
        }
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let mem_threads = available_memory_gb() / 2.0;
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
        let cfg = ParseConfig::default();
        let set = cfg.t2_domains_set();
        assert!(set.contains("Health Sciences"));
        assert!(set.contains("Engineering"));
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_parse_threads_explicit() {
        let cfg = ParseConfig {
            parse_threads: Some(4),
            ..Default::default()
        };
        assert_eq!(cfg.effective_parse_threads(), 4);
    }

    #[test]
    fn test_parse_threads_explicit_zero_becomes_one() {
        let cfg = ParseConfig {
            parse_threads: Some(0),
            ..Default::default()
        };
        assert_eq!(cfg.effective_parse_threads(), 1);
    }

    #[test]
    fn test_parse_threads_auto() {
        let cfg = ParseConfig::default();
        let n = cfg.effective_parse_threads();
        assert!(n >= 1);
    }
}
