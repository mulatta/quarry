//! PyO3 bindings for papeline-core.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use indicatif::MultiProgress;
use pyo3::exceptions::{PyKeyboardInterrupt, PyRuntimeError};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use papeline_core::config::{self, ResolvedHiveConfig, ResolvedUploadConfig};
use papeline_core::oa::{self, OAProvider, OAShard};
use papeline_core::progress::{IndicatifMakeWriter, ProgressContext};
use papeline_core::provider::{RunContext, run_provider};
use papeline_core::remote::RemoteTargets;
use papeline_core::stream::HttpPool;
use papeline_core::{api, hive, remote, transform};

// ============================================================
// Helper
// ============================================================

fn to_pyerr(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Set up tracing subscriber routed through indicatif for TTY/non-TTY compat.
fn init_tracing(progress: &ProgressContext) {
    let writer = IndicatifMakeWriter::new(progress.multi().clone());
    let _ = tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(writer)
                .without_time(),
        )
        .try_init();
}

// ============================================================
// Classes
// ============================================================

/// Shared HTTP client pool backed by a Tokio runtime.
#[pyclass(name = "HttpPool")]
struct PyHttpPool {
    inner: Arc<HttpPool>,
}

#[pymethods]
impl PyHttpPool {
    #[new]
    #[pyo3(signature = (concurrency=8, read_timeout_secs=30, max_retries=3))]
    fn new(concurrency: usize, read_timeout_secs: u64, max_retries: u32) -> PyResult<Self> {
        let pool = HttpPool::new(
            concurrency,
            Duration::from_secs(read_timeout_secs),
            max_retries,
        )
        .map_err(to_pyerr)?;
        Ok(Self {
            inner: Arc::new(pool),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "HttpPool(read_timeout={:?}, max_retries={})",
            self.inner.read_timeout, self.inner.max_retries,
        )
    }
}

/// Filter for domains, topics, languages, work types, and abstract requirement.
#[pyclass(name = "Filter")]
#[derive(Clone)]
struct PyFilter {
    inner: transform::Filter,
}

#[pymethods]
impl PyFilter {
    #[new]
    #[pyo3(signature = (*, domains=vec![], topics=vec![], languages=vec![], work_types=vec![], require_abstract=false))]
    fn new(
        domains: Vec<String>,
        topics: Vec<String>,
        languages: Vec<String>,
        work_types: Vec<String>,
        require_abstract: bool,
    ) -> Self {
        Self {
            inner: transform::Filter {
                domains: domains.into_iter().collect::<FxHashSet<_>>(),
                topic_ids: topics.into_iter().collect::<FxHashSet<_>>(),
                languages: languages.into_iter().collect::<FxHashSet<_>>(),
                work_types: work_types.into_iter().collect::<FxHashSet<_>>(),
                require_abstract,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Filter(domains={}, topics={}, languages={}, work_types={}, require_abstract={})",
            self.inner.domains.len(),
            self.inner.topic_ids.len(),
            self.inner.languages.len(),
            self.inner.work_types.len(),
            self.inner.require_abstract,
        )
    }
}

/// A shard entry from the OpenAlex manifest.
#[pyclass(name = "OAShard", frozen)]
struct PyOAShard {
    inner: OAShard,
}

#[pymethods]
impl PyOAShard {
    #[getter]
    fn shard_idx(&self) -> usize {
        self.inner.shard_idx
    }
    #[getter]
    fn url(&self) -> &str {
        &self.inner.url
    }
    #[getter]
    fn content_length(&self) -> Option<u64> {
        self.inner.content_length
    }
    #[getter]
    fn record_count(&self) -> u64 {
        self.inner.record_count
    }
    #[getter]
    fn updated_date(&self) -> Option<&str> {
        self.inner.updated_date.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "OAShard(idx={}, records={}, date={:?})",
            self.inner.shard_idx, self.inner.record_count, self.inner.updated_date,
        )
    }
}

/// Summary returned by `run()`.
#[pyclass(name = "RunSummary", frozen)]
struct PyRunSummary {
    #[pyo3(get)]
    completed: usize,
    #[pyo3(get)]
    failed: usize,
    #[pyo3(get)]
    total_rows: usize,
    #[pyo3(get)]
    total_scanned: usize,
    #[pyo3(get)]
    elapsed_secs: f64,
    #[pyo3(get)]
    failed_indices: Vec<usize>,
}

#[pymethods]
impl PyRunSummary {
    fn __repr__(&self) -> String {
        format!(
            "RunSummary(completed={}, failed={}, rows={}, scanned={}, elapsed={:.1}s)",
            self.completed, self.failed, self.total_rows, self.total_scanned, self.elapsed_secs,
        )
    }
}

/// Summary returned by `push()` / `pull()`.
#[pyclass(name = "TransferSummary", frozen)]
struct PyTransferSummary {
    #[pyo3(get)]
    files_transferred: usize,
    #[pyo3(get)]
    bytes_transferred: u64,
    #[pyo3(get)]
    files_skipped: usize,
    #[pyo3(get)]
    files_failed: usize,
    #[pyo3(get)]
    failed_keys: Vec<String>,
}

#[pymethods]
impl PyTransferSummary {
    fn __repr__(&self) -> String {
        format!(
            "TransferSummary(transferred={}, skipped={}, failed={}, bytes={})",
            self.files_transferred, self.files_skipped, self.files_failed, self.bytes_transferred,
        )
    }
}

// ============================================================
// Functions
// ============================================================

/// Fetch the OpenAlex manifest and return a list of shards.
#[pyfunction]
#[pyo3(signature = (pool, entity="works"))]
fn fetch_manifest(py: Python<'_>, pool: &PyHttpPool, entity: &str) -> PyResult<Vec<PyOAShard>> {
    let pool = Arc::clone(&pool.inner);
    let entity = entity.to_owned();
    py.allow_threads(move || {
        api::fetch_manifest(&pool, &entity)
            .map(|shards| shards.into_iter().map(|s| PyOAShard { inner: s }).collect())
            .map_err(to_pyerr)
    })
}

/// Run the OpenAlex pipeline: download, transform, and write Parquet files.
#[pyfunction]
#[pyo3(signature = (shards, output_dir, pool, filter=None, zstd_level=3, concurrency=8))]
fn run(
    py: Python<'_>,
    shards: Vec<PyRef<'_, PyOAShard>>,
    output_dir: &str,
    pool: &PyHttpPool,
    filter: Option<&PyFilter>,
    zstd_level: i32,
    concurrency: usize,
) -> PyResult<PyRunSummary> {
    let oa_shards: Vec<OAShard> = shards.iter().map(|s| s.inner.clone()).collect();
    let filter = filter.map_or_else(transform::Filter::default, |f| f.inner.clone());
    let pool_arc = Arc::clone(&pool.inner);
    let output_dir = PathBuf::from(output_dir);

    let cancelled = Arc::new(AtomicBool::new(false));

    // Register SIGINT handler so Ctrl+C sets the flag while GIL is released
    let sig_flag = Arc::clone(&cancelled);
    let sig_id = unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGINT, move || {
            sig_flag.store(true, Ordering::Relaxed);
        })
    }
    .map_err(to_pyerr)?;

    let result = py.allow_threads(move || {
        let progress: Arc<ProgressContext> = Arc::new(ProgressContext::new());
        let ctx = RunContext {
            output_dir,
            zstd_level,
            concurrency,
            cancelled: Arc::clone(&cancelled),
        };
        let provider = OAProvider {
            filter,
            pool: pool_arc,
        };

        let summary = run_provider(&provider, &oa_shards, &ctx, &progress, None);

        (summary, cancelled)
    });

    // Unregister our signal handler, restore default Python SIGINT handling
    signal_hook::low_level::unregister(sig_id);

    let (summary, cancelled) = result;
    if cancelled.load(Ordering::Relaxed) {
        return Err(PyKeyboardInterrupt::new_err("interrupted"));
    }

    Ok(PyRunSummary {
        completed: summary.completed,
        failed: summary.failed,
        total_rows: summary.total_rows,
        total_scanned: summary.total_scanned,
        elapsed_secs: summary.elapsed.as_secs_f64(),
        failed_indices: summary.failed_indices,
    })
}

/// Run hive partitioning on raw Parquet files.
#[pyfunction]
#[pyo3(signature = (
    output_dir,
    zstd_level = 3,
    row_group_size = 500_000,
    num_shards = 64,
    threads = 0,
    memory_limit = "32GB",
    force = false,
    dry_run = false,
))]
fn run_hive(
    py: Python<'_>,
    output_dir: &str,
    zstd_level: i32,
    row_group_size: usize,
    num_shards: usize,
    threads: usize,
    memory_limit: &str,
    force: bool,
    dry_run: bool,
) -> PyResult<()> {
    let output_dir = PathBuf::from(output_dir);
    let memory_limit = memory_limit.to_owned();

    py.allow_threads(move || {
        let threads = if threads == 0 { num_cpus() } else { threads };
        let memory_limit_bytes = config::parse_memory_limit(&memory_limit).map_err(to_pyerr)?;
        let hive_config = ResolvedHiveConfig {
            raw_dir: output_dir.join("raw"),
            hive_dir: output_dir.join("hive"),
            staging_dir: output_dir.join("hive/.staging"),
            zstd_level,
            row_group_size,
            num_shards,
            threads,
            memory_limit_bytes,
        };
        let multi = MultiProgress::new();
        hive::run_hive(&hive_config, force, dry_run, &multi, None).map_err(to_pyerr)
    })
}

/// Push local files to S3-compatible remote storage.
#[pyfunction]
#[pyo3(signature = (
    output_dir,
    bucket,
    endpoint,
    access_key,
    secret_key,
    region = "auto",
    prefix = "",
    raw = true,
    hive = true,
    dry_run = false,
    concurrency = 4,
))]
fn push(
    py: Python<'_>,
    output_dir: &str,
    bucket: &str,
    endpoint: &str,
    access_key: &str,
    secret_key: &str,
    region: &str,
    prefix: &str,
    raw: bool,
    hive: bool,
    dry_run: bool,
    concurrency: usize,
) -> PyResult<PyTransferSummary> {
    let config = ResolvedUploadConfig {
        bucket: bucket.to_owned(),
        endpoint: endpoint.to_owned(),
        region: region.to_owned(),
        access_key: access_key.to_owned(),
        secret_key: secret_key.to_owned(),
        prefix: prefix.to_owned(),
    };
    let output_dir = PathBuf::from(output_dir);
    let targets = RemoteTargets { raw, hive };

    py.allow_threads(move || {
        let progress: Arc<ProgressContext> = Arc::new(ProgressContext::new());
        init_tracing(&progress);
        let summary = remote::run_push(
            &config,
            &output_dir,
            &targets,
            dry_run,
            concurrency,
            &progress,
        )
        .map_err(to_pyerr)?;
        Ok(PyTransferSummary {
            files_transferred: summary.files_transferred,
            bytes_transferred: summary.bytes_transferred,
            files_skipped: summary.files_skipped,
            files_failed: summary.files_failed,
            failed_keys: summary.failed_keys,
        })
    })
}

/// Pull files from S3-compatible remote storage to local.
#[pyfunction]
#[pyo3(signature = (
    output_dir,
    bucket,
    endpoint,
    access_key,
    secret_key,
    region = "auto",
    prefix = "",
    raw = true,
    hive = true,
    dry_run = false,
    concurrency = 4,
))]
fn pull(
    py: Python<'_>,
    output_dir: &str,
    bucket: &str,
    endpoint: &str,
    access_key: &str,
    secret_key: &str,
    region: &str,
    prefix: &str,
    raw: bool,
    hive: bool,
    dry_run: bool,
    concurrency: usize,
) -> PyResult<PyTransferSummary> {
    let config = ResolvedUploadConfig {
        bucket: bucket.to_owned(),
        endpoint: endpoint.to_owned(),
        region: region.to_owned(),
        access_key: access_key.to_owned(),
        secret_key: secret_key.to_owned(),
        prefix: prefix.to_owned(),
    };
    let output_dir = PathBuf::from(output_dir);
    let targets = RemoteTargets { raw, hive };

    py.allow_threads(move || {
        let progress: Arc<ProgressContext> = Arc::new(ProgressContext::new());
        init_tracing(&progress);
        let summary = remote::run_pull(
            &config,
            &output_dir,
            &targets,
            dry_run,
            concurrency,
            &progress,
        )
        .map_err(to_pyerr)?;
        Ok(PyTransferSummary {
            files_transferred: summary.files_transferred,
            bytes_transferred: summary.bytes_transferred,
            files_skipped: summary.files_skipped,
            files_failed: summary.files_failed,
            failed_keys: summary.failed_keys,
        })
    })
}

/// Check if a shard has all 12 output tables.
#[pyfunction]
fn is_shard_complete(output_dir: &str, shard_idx: usize) -> bool {
    oa::is_shard_complete(&PathBuf::from(output_dir), shard_idx)
}

/// Return set of shard indices that are complete (parallel check using rayon).
#[pyfunction]
fn complete_shards(py: Python<'_>, output_dir: &str, shard_indices: Vec<usize>) -> Vec<usize> {
    let output_dir = PathBuf::from(output_dir);
    py.allow_threads(move || {
        shard_indices
            .into_par_iter()
            .filter(|&idx| oa::is_shard_complete(&output_dir, idx))
            .collect()
    })
}

/// List of all output table names.
#[pyfunction]
fn tables() -> Vec<&'static str> {
    papeline_core::oa::TABLES.to_vec()
}

// ============================================================
// Helpers
// ============================================================

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// ============================================================
// Module
// ============================================================

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHttpPool>()?;
    m.add_class::<PyFilter>()?;
    m.add_class::<PyOAShard>()?;
    m.add_class::<PyRunSummary>()?;
    m.add_class::<PyTransferSummary>()?;
    m.add_function(wrap_pyfunction!(fetch_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_hive, m)?)?;
    m.add_function(wrap_pyfunction!(push, m)?)?;
    m.add_function(wrap_pyfunction!(pull, m)?)?;
    m.add_function(wrap_pyfunction!(is_shard_complete, m)?)?;
    m.add_function(wrap_pyfunction!(complete_shards, m)?)?;
    m.add_function(wrap_pyfunction!(tables, m)?)?;
    Ok(())
}
