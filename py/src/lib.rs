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

use papeline_core::config::{self, ResolvedHiveConfig, ResolvedUploadConfig};
use papeline_core::oa::{self, OAProvider, OAShard};
use papeline_core::progress::ProgressContext;
use papeline_core::provider::{RunContext, run_provider};
use papeline_core::remote::{RemoteTargets, TransferOpts};
use papeline_core::stream::HttpPool;
use papeline_core::{api, hive, remote, transform};

// ============================================================
// Helper
// ============================================================

fn to_pyerr(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ============================================================
// Config (PyO3 wrappers for Rust serde-validated config)
// ============================================================

/// Parsed TOML configuration — validated by Rust serde with deny_unknown_fields.
#[pyclass(name = "Config", frozen)]
struct PyConfig {
    #[pyo3(get)]
    output: Option<String>,
    #[pyo3(get)]
    zstd_level: Option<i32>,
    #[pyo3(get)]
    concurrency: Option<usize>,
    #[pyo3(get)]
    max_retries: Option<u32>,
    #[pyo3(get)]
    read_timeout: Option<u64>,
    #[pyo3(get)]
    outer_retries: Option<u32>,
    #[pyo3(get)]
    retry_delay: Option<u64>,
    #[pyo3(get)]
    filter: PyFilterConfig,
    #[pyo3(get)]
    hive: PyHiveConfig,
    #[pyo3(get)]
    upload: PyUploadConfig,
}

#[pyclass(name = "FilterConfig", frozen)]
#[derive(Clone)]
struct PyFilterConfig {
    #[pyo3(get)]
    domains: Vec<String>,
    #[pyo3(get)]
    topics: Vec<String>,
    #[pyo3(get)]
    languages: Vec<String>,
    #[pyo3(get)]
    work_types: Vec<String>,
    #[pyo3(get)]
    require_abstract: bool,
}

#[pyclass(name = "HiveConfig", frozen)]
#[derive(Clone)]
struct PyHiveConfig {
    #[pyo3(get)]
    enable: Option<bool>,
    #[pyo3(get)]
    clean_raw: Option<bool>,
    #[pyo3(get)]
    zstd_level: Option<i32>,
    #[pyo3(get)]
    row_group_size: Option<usize>,
    #[pyo3(get)]
    num_shards: Option<usize>,
    #[pyo3(get)]
    threads: Option<usize>,
    #[pyo3(get)]
    memory_limit: Option<String>,
}

#[pyclass(name = "UploadConfig", frozen)]
#[derive(Clone)]
struct PyUploadConfig {
    #[pyo3(get)]
    bucket: Option<String>,
    #[pyo3(get)]
    endpoint: Option<String>,
    #[pyo3(get)]
    region: Option<String>,
    #[pyo3(get)]
    access_key: Option<String>,
    #[pyo3(get)]
    secret_key: Option<String>,
    #[pyo3(get)]
    prefix: Option<String>,
    #[pyo3(get)]
    force: Option<bool>,
    #[pyo3(get)]
    concurrency: Option<usize>,
    #[pyo3(get)]
    auto_push: Option<bool>,
}

impl From<config::FileConfig> for PyConfig {
    fn from(c: config::FileConfig) -> Self {
        Self {
            output: c.output,
            zstd_level: c.zstd_level,
            concurrency: c.concurrency,
            max_retries: c.max_retries,
            read_timeout: c.read_timeout,
            outer_retries: c.outer_retries,
            retry_delay: c.retry_delay,
            filter: PyFilterConfig {
                domains: c.filter.domains,
                topics: c.filter.topics,
                languages: c.filter.languages,
                work_types: c.filter.work_types,
                require_abstract: c.filter.require_abstract,
            },
            hive: PyHiveConfig {
                enable: c.hive.enable,
                clean_raw: c.hive.clean_raw,
                zstd_level: c.hive.zstd_level,
                row_group_size: c.hive.row_group_size,
                num_shards: c.hive.num_shards,
                threads: c.hive.threads,
                memory_limit: c.hive.memory_limit,
            },
            upload: PyUploadConfig {
                bucket: c.upload.bucket,
                endpoint: c.upload.endpoint,
                region: c.upload.region,
                access_key: c.upload.access_key,
                secret_key: c.upload.secret_key,
                prefix: c.upload.prefix,
                force: c.upload.force,
                concurrency: c.upload.concurrency,
                auto_push: c.upload.auto_push,
            },
        }
    }
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

/// Load and validate TOML config. Returns default config if no file found.
#[pyfunction]
#[pyo3(signature = (path=None))]
fn load_config(path: Option<&str>) -> PyResult<PyConfig> {
    let p = path.map(std::path::Path::new);
    let cfg = config::load_config(p).map_err(to_pyerr)?;
    Ok(PyConfig::from(cfg))
}

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
    force = false,
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
    force: bool,
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
    let opts = TransferOpts {
        targets: RemoteTargets { raw, hive },
        dry_run,
        force,
        concurrency,
    };
    let cancelled = Arc::new(AtomicBool::new(false));

    let sig_flag = Arc::clone(&cancelled);
    let sig_id = unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGINT, move || {
            sig_flag.store(true, Ordering::Relaxed);
        })
    }
    .map_err(to_pyerr)?;

    let result = py.allow_threads(move || {
        let progress: Arc<ProgressContext> = Arc::new(ProgressContext::new());
        let summary = remote::run_push(&config, &output_dir, &opts, &progress, &cancelled);
        (summary, cancelled)
    });

    signal_hook::low_level::unregister(sig_id);

    let (summary, cancelled) = result;
    if cancelled.load(Ordering::Relaxed) {
        return Err(PyKeyboardInterrupt::new_err("interrupted"));
    }

    let summary = summary.map_err(to_pyerr)?;
    Ok(PyTransferSummary {
        files_transferred: summary.files_transferred,
        bytes_transferred: summary.bytes_transferred,
        files_skipped: summary.files_skipped,
        files_failed: summary.files_failed,
        failed_keys: summary.failed_keys,
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
    force = false,
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
    force: bool,
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
    let opts = TransferOpts {
        targets: RemoteTargets { raw, hive },
        dry_run,
        force,
        concurrency,
    };
    let cancelled = Arc::new(AtomicBool::new(false));

    let sig_flag = Arc::clone(&cancelled);
    let sig_id = unsafe {
        signal_hook::low_level::register(signal_hook::consts::SIGINT, move || {
            sig_flag.store(true, Ordering::Relaxed);
        })
    }
    .map_err(to_pyerr)?;

    let result = py.allow_threads(move || {
        let progress: Arc<ProgressContext> = Arc::new(ProgressContext::new());
        let summary = remote::run_pull(&config, &output_dir, &opts, &progress, &cancelled);
        (summary, cancelled)
    });

    signal_hook::low_level::unregister(sig_id);

    let (summary, cancelled) = result;
    if cancelled.load(Ordering::Relaxed) {
        return Err(PyKeyboardInterrupt::new_err("interrupted"));
    }

    let summary = summary.map_err(to_pyerr)?;
    Ok(PyTransferSummary {
        files_transferred: summary.files_transferred,
        bytes_transferred: summary.bytes_transferred,
        files_skipped: summary.files_skipped,
        files_failed: summary.files_failed,
        failed_keys: summary.failed_keys,
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
    m.add_class::<PyConfig>()?;
    m.add_class::<PyFilterConfig>()?;
    m.add_class::<PyHiveConfig>()?;
    m.add_class::<PyUploadConfig>()?;
    m.add_class::<PyHttpPool>()?;
    m.add_class::<PyFilter>()?;
    m.add_class::<PyOAShard>()?;
    m.add_class::<PyRunSummary>()?;
    m.add_class::<PyTransferSummary>()?;
    m.add_function(wrap_pyfunction!(load_config, m)?)?;
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
