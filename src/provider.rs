//! Generic provider trait and unified runner for shard-based pipelines.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use indicatif::ProgressBar;
use rayon::prelude::*;

use crate::progress::SharedProgress;

/// Statistics returned from processing a single shard.
#[derive(Debug, Default)]
pub struct ShardStats {
    pub rows_written: usize,
}

/// Aggregated summary from [`run_provider`].
#[derive(Debug)]
pub struct RunSummary {
    pub completed: usize,
    pub failed: usize,
    pub total_rows: usize,
    pub elapsed: Duration,
    /// Indices (into the input vec) of shards that failed.
    pub failed_indices: Vec<usize>,
}

/// Runtime context shared across all providers.
pub struct RunContext {
    /// Output directory for parquet files.
    pub output_dir: PathBuf,
    /// Zstd compression level for parquet output.
    pub zstd_level: i32,
    /// Number of concurrent shard workers (rayon thread pool size).
    pub concurrency: usize,
}

/// Trait implemented by each data-source provider.
pub trait Provider: Sync {
    /// A single unit of work (manifest entry, URL, ...).
    type Shard: Send + Sync;
    /// Provider-specific error type.
    type Err: std::fmt::Display + Send;

    /// Human-readable label for a shard (shown in progress bars).
    fn shard_label(&self, shard: &Self::Shard) -> String;

    /// Process one shard end-to-end (download -> parse -> write).
    fn process_shard(
        &self,
        shard: &Self::Shard,
        ctx: &RunContext,
        pb: &ProgressBar,
    ) -> Result<ShardStats, Self::Err>;
}

/// Run a provider's full pipeline: parallel process -> summarise.
pub fn run_provider<P: Provider>(
    provider: &P,
    shards: &[P::Shard],
    ctx: &RunContext,
    progress: &SharedProgress,
) -> RunSummary {
    let start = Instant::now();
    let rows = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let failed_indices: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

    progress.init_global(shards.len() as u64);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(ctx.concurrency)
        .build()
        .expect("failed to build provider thread pool");

    pool.install(|| {
        shards.par_iter().enumerate().for_each(|(idx, shard)| {
            let label = provider.shard_label(shard);
            let pb = progress.shard_bar(&label);

            match provider.process_shard(shard, ctx, &pb) {
                Ok(stats) => {
                    rows.fetch_add(stats.rows_written, Ordering::Relaxed);
                    completed.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!("{label}: {} rows", stats.rows_written);
                }
                Err(e) => {
                    failed.fetch_add(1, Ordering::Relaxed);
                    failed_indices
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(idx);
                    tracing::error!("{label}: {e}");
                }
            }

            progress.inc_global();
            pb.finish_and_clear();
        });
    });

    let f = failed.load(Ordering::Relaxed);
    if f > 0 {
        progress.set_global_message(format!("{f} failed"));
    }
    progress.finish_global();

    let mut fi = failed_indices.into_inner().unwrap();
    fi.sort_unstable();

    RunSummary {
        completed: completed.load(Ordering::Relaxed),
        failed: failed.load(Ordering::Relaxed),
        total_rows: rows.load(Ordering::Relaxed),
        elapsed: start.elapsed(),
        failed_indices: fi,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::progress::ProgressContext;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    struct DummyProvider {
        fail_indices: Vec<usize>,
        call_count: AtomicUsize,
    }

    impl Provider for DummyProvider {
        type Shard = usize;
        type Err = String;

        fn shard_label(&self, shard: &usize) -> String {
            format!("shard_{shard:04}")
        }

        fn process_shard(
            &self,
            shard: &usize,
            _ctx: &RunContext,
            _pb: &ProgressBar,
        ) -> Result<ShardStats, String> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            if self.fail_indices.contains(shard) {
                Err(format!("shard {shard} failed"))
            } else {
                Ok(ShardStats { rows_written: 10 })
            }
        }
    }

    fn test_ctx() -> RunContext {
        RunContext {
            output_dir: PathBuf::from("/tmp/test"),
            zstd_level: 3,
            concurrency: 2,
        }
    }

    #[test]
    fn run_all_shards() {
        let provider = DummyProvider {
            fail_indices: vec![],
            call_count: AtomicUsize::new(0),
        };
        let progress: SharedProgress = Arc::new(ProgressContext::new());
        let summary = run_provider(&provider, &[0, 1, 2, 3], &test_ctx(), &progress);

        assert_eq!(summary.completed, 4);
        assert_eq!(summary.failed, 0);
        assert_eq!(summary.total_rows, 40);
        assert!(summary.failed_indices.is_empty());
    }

    #[test]
    fn run_with_failures() {
        let provider = DummyProvider {
            fail_indices: vec![1, 3],
            call_count: AtomicUsize::new(0),
        };
        let progress: SharedProgress = Arc::new(ProgressContext::new());
        let summary = run_provider(&provider, &[0, 1, 2, 3], &test_ctx(), &progress);

        assert_eq!(summary.completed, 2);
        assert_eq!(summary.failed, 2);
        assert_eq!(summary.total_rows, 20);
        assert_eq!(summary.failed_indices.len(), 2);
    }

    #[test]
    fn empty_shards() {
        let provider = DummyProvider {
            fail_indices: vec![],
            call_count: AtomicUsize::new(0),
        };
        let progress: SharedProgress = Arc::new(ProgressContext::new());
        let summary = run_provider(&provider, &[], &test_ctx(), &progress);

        assert_eq!(summary.completed, 0);
    }
}
