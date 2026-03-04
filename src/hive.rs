//! Hive-partitioned Parquet output via streaming arrow-rs writers.
//!
//! Raw shard parquets → `pub_year=YYYY/shard_N.parquet` layout with
//! deterministic `work_id % num_shards` assignment.
//!
//! Uses direct `ArrowWriter` for bounded memory: reads raw shards one at a time,
//! routes each batch to per-partition writers that flush at `row_group_size`.
//! Memory ≈ O(partition_map + writer_buffers) instead of O(total_data).

use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, IsTerminal};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use arrow::array::{Array, ArrayRef, Int32Array, StringArray, UInt32Array};
use arrow::compute;
use arrow::record_batch::RecordBatch;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::{ArrowWriter, ProjectionMask};
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::{ResolvedHiveConfig, format_bytes};
use crate::oa::TABLES;

/// Coerce a RecordBatch to match the target schema.
///
/// Raw shards may have minor schema differences (e.g., List inner field named
/// "item" vs "element"). This casts mismatched columns to the target types.
fn coerce_batch(
    batch: RecordBatch,
    target_schema: &arrow::datatypes::SchemaRef,
) -> Result<RecordBatch> {
    if batch.schema() == *target_schema {
        return Ok(batch);
    }
    let columns: Vec<ArrayRef> = target_schema
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(tgt_field, col)| {
            if col.data_type() == tgt_field.data_type() {
                Ok(col.clone())
            } else {
                compute::cast(col, tgt_field.data_type()).with_context(|| {
                    format!(
                        "Cannot coerce column '{}': {:?} → {:?}",
                        tgt_field.name(),
                        col.data_type(),
                        tgt_field.data_type()
                    )
                })
            }
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new(target_schema.clone(), columns).context("Schema coercion failed")
}

/// Parquet reader batch size (rows per RecordBatch from reader).
const READER_BATCH_SIZE: usize = 65_536;

// ============================================================
// Idempotency state
// ============================================================

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct HiveState {
    pub source_manifest_hash: String,
    /// Settings that affect output data (changing any triggers rebuild).
    pub zstd_level: i32,
    #[serde(default)]
    pub row_group_size: usize,
    #[serde(default)]
    pub num_shards: usize,
    #[serde(default)]
    pub completed_at: String,
    pub tables: Vec<String>,
    /// Tables successfully written to hive dir (for resume on interrupted runs).
    #[serde(default)]
    pub completed_tables: Vec<String>,
}

impl HiveState {
    const FILENAME: &str = ".hive_state.json";

    fn load(hive_dir: &Path) -> Option<Self> {
        let path = hive_dir.join(Self::FILENAME);
        let text = fs::read_to_string(&path).ok()?;
        sonic_rs::from_str(&text).ok()
    }

    fn save(&self, hive_dir: &Path) -> Result<()> {
        let path = hive_dir.join(Self::FILENAME);
        let tmp = hive_dir.join(format!("{}.tmp", Self::FILENAME));
        let json = sonic_rs::to_string_pretty(self)?;
        fs::write(&tmp, &json)?;
        fs::rename(&tmp, &path)?;
        Ok(())
    }
}

// ============================================================
// Manifest hashing
// ============================================================

fn manifest_hash(raw_dir: &Path) -> Result<String> {
    let path = raw_dir.join(".manifest.json");
    let content = fs::read(&path).with_context(|| format!("Cannot read {}", path.display()))?;
    let hash = Sha256::digest(&content);
    Ok(format!("{hash:x}"))
}

// ============================================================
// Parquet writer properties
// ============================================================

// ============================================================
// Work ID helpers
// ============================================================

/// Extract numeric part from work_id (e.g. "W1234567890" → 1234567890).
fn work_id_num(work_id: &str) -> i64 {
    if work_id.len() > 1 {
        work_id[1..].parse::<i64>().unwrap_or(0)
    } else {
        0
    }
}

// ============================================================
// Work partition map
// ============================================================

/// Maps work_id numeric part → (pub_year, shard_bucket).
/// Built during works processing, used by child table routing.
struct WorkPartitionMap {
    inner: FxHashMap<i64, u64>,
}

impl WorkPartitionMap {
    fn new() -> Self {
        Self {
            inner: FxHashMap::default(),
        }
    }

    /// Pack (year, bucket) into a single u64 for compact storage.
    fn insert(&mut self, wid_num: i64, year: i32, bucket: u32) {
        let packed = ((year as u64) << 32) | (bucket as u64);
        self.inner.insert(wid_num, packed);
    }

    fn extend(&mut self, other: Self) {
        self.inner.extend(other.inner);
    }

    /// Collect all unique (year, bucket) pairs present in the map.
    fn unique_partitions(&self) -> Vec<(i32, usize)> {
        let mut seen = FxHashSet::default();
        for &packed in self.inner.values() {
            let year = (packed >> 32) as i32;
            let bucket = (packed & 0xFFFF_FFFF) as usize;
            seen.insert((year, bucket));
        }
        seen.into_iter().collect()
    }

    fn get(&self, wid_num: i64) -> Option<(i32, usize)> {
        self.inner.get(&wid_num).map(|&packed| {
            let year = (packed >> 32) as i32;
            let bucket = (packed & 0xFFFF_FFFF) as usize;
            (year, bucket)
        })
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

// ============================================================
// Shard file listing
// ============================================================

type PartitionWriter = ArrowWriter<BufWriter<fs::File>>;

fn list_shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "parquet"))
        .collect();
    files.sort();
    Ok(files)
}

// ============================================================
// Per-partition parallel write
// ============================================================

/// Pre-create ArrowWriters for all known partitions.
///
/// Each (year, bucket) gets its own file. Writers are wrapped in Mutex so
/// rayon reader threads can write directly without a channel bottleneck.
fn create_partition_writers(
    partitions: &[(i32, usize)],
    schema: &arrow::datatypes::SchemaRef,
    props: &WriterProperties,
    staging: &Path,
) -> Result<HashMap<(i32, usize), std::sync::Mutex<PartitionWriter>>> {
    let mut writers = HashMap::with_capacity(partitions.len());
    for &(year, bucket) in partitions {
        let dir = staging.join(format!("pub_year={year}"));
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("shard_{bucket}.parquet"));
        let file = BufWriter::new(fs::File::create(&path)?);
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props.clone()))?;
        writers.insert((year, bucket), std::sync::Mutex::new(writer));
    }
    Ok(writers)
}

/// Parallel flush + close all partition writers.
fn close_writers_parallel(
    writers: HashMap<(i32, usize), std::sync::Mutex<PartitionWriter>>,
    pool: &rayon::ThreadPool,
) -> Result<()> {
    let entries: Vec<_> = writers
        .into_iter()
        .map(|(key, mutex)| (key, mutex.into_inner().expect("mutex not poisoned")))
        .collect();

    if entries.is_empty() {
        return Ok(());
    }

    // Parallel close — each writer flushes its buffered data (ZSTD compression) and writes footer.
    // This is the CPU-heavy phase; parallelising across all partitions saturates cores.
    pool.install(|| {
        entries
            .into_par_iter()
            .try_for_each(|((year, bucket), writer)| {
                writer
                    .close()
                    .map(|_| ())
                    .with_context(|| format!("Failed to close pub_year={year}/shard_{bucket}"))
            })
    })
}

/// Read one shard, route each batch via `route_fn`, and write sub-batches
/// directly to the per-partition Mutex writers.
///
/// `route_fn` returns a map of (year, bucket) → row indices for each batch.
fn read_and_write_shard<F>(
    shard_path: &Path,
    schema: &arrow::datatypes::SchemaRef,
    writers: &HashMap<(i32, usize), std::sync::Mutex<PartitionWriter>>,
    per_writer_mem_limit: usize,
    route_fn: &F,
) -> Result<()>
where
    F: Fn(&RecordBatch) -> Result<HashMap<(i32, usize), Vec<u32>>>,
{
    let file = fs::File::open(shard_path)
        .with_context(|| format!("Cannot open {}", shard_path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("Invalid parquet: {}", shard_path.display()))?
        .with_batch_size(READER_BATCH_SIZE)
        .build()
        .with_context(|| format!("Cannot build reader: {}", shard_path.display()))?;

    for batch_result in reader {
        let batch = coerce_batch(batch_result?, schema)?;
        let groups = route_fn(&batch)?;

        for ((year, bucket), indices) in &groups {
            let indices_array = UInt32Array::from(indices.clone());
            let columns: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .map(|col| compute::take(col.as_ref(), &indices_array, None))
                .collect::<Result<Vec<_>, _>>()
                .context("arrow take failed")?;
            let sub_batch = RecordBatch::try_new(batch.schema(), columns)?;

            let writer_mutex = writers
                .get(&(*year, *bucket))
                .with_context(|| format!("No writer for pub_year={year}/shard_{bucket}"))?;
            let mut guard = writer_mutex.lock().expect("mutex not poisoned");
            guard.write(&sub_batch)?;
            if guard.memory_size() > per_writer_mem_limit {
                guard.flush()?;
            }
        }
    }
    Ok(())
}

/// Process shards in parallel: readers decode + route + write directly to
/// per-partition Mutex<ArrowWriter>. No channel or dedicated writer thread.
fn parallel_process<F>(
    shard_files: &[PathBuf],
    schema: &arrow::datatypes::SchemaRef,
    writers: &HashMap<(i32, usize), std::sync::Mutex<PartitionWriter>>,
    per_writer_mem_limit: usize,
    pool: &rayon::ThreadPool,
    pb: &ProgressBar,
    route_fn: F,
) -> Result<()>
where
    F: Fn(&RecordBatch) -> Result<HashMap<(i32, usize), Vec<u32>>> + Send + Sync,
{
    pb.set_length(shard_files.len() as u64);

    let read_error: std::sync::Mutex<Option<anyhow::Error>> = std::sync::Mutex::new(None);

    pool.install(|| {
        shard_files.par_iter().for_each(|shard_path| {
            if read_error.lock().expect("not poisoned").is_some() {
                return;
            }

            if let Err(e) =
                read_and_write_shard(shard_path, schema, writers, per_writer_mem_limit, &route_fn)
            {
                let mut guard = read_error.lock().expect("not poisoned");
                if guard.is_none() {
                    *guard = Some(e.context(format!("Failed: {}", shard_path.display())));
                }
            }
            pb.inc(1);
        });
    });

    if let Some(e) = read_error.into_inner().expect("not poisoned") {
        return Err(e);
    }

    Ok(())
}

fn writer_properties(config: &ResolvedHiveConfig) -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(config.zstd_level).expect("valid zstd level"),
        ))
        .set_max_row_group_size(config.row_group_size)
        .build()
}

/// Build partition map from raw works shards (projection: 2 columns only).
///
/// Parallelised with rayon: each thread builds a thread-local map, then merged.
fn rebuild_partition_map_from(
    config: &ResolvedHiveConfig,
    shard_files: &[PathBuf],
    pool: &rayon::ThreadPool,
    pb: &ProgressBar,
) -> Result<WorkPartitionMap> {
    let num_shards = config.num_shards;

    pb.set_length(shard_files.len() as u64);

    let map = pool.install(|| {
        shard_files
            .par_iter()
            .map(|shard_path| -> Result<WorkPartitionMap> {
                let mut local = WorkPartitionMap::new();
                let file = fs::File::open(shard_path)?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

                let arrow_schema = builder.schema();
                let work_id_root = arrow_schema
                    .index_of("work_id")
                    .map_err(|e| anyhow::anyhow!("works schema missing work_id: {e}"))?;
                let pub_year_root = arrow_schema
                    .index_of("publication_year")
                    .map_err(|e| anyhow::anyhow!("works schema missing publication_year: {e}"))?;
                let mask =
                    ProjectionMask::roots(builder.parquet_schema(), [work_id_root, pub_year_root]);

                let reader = builder
                    .with_projection(mask)
                    .with_batch_size(READER_BATCH_SIZE)
                    .build()?;

                for batch_result in reader {
                    let batch = batch_result?;
                    let work_id_col = batch
                        .column_by_name("work_id")
                        .context("projected batch missing work_id")?
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .context("work_id must be Utf8")?;
                    let pub_year_col = batch
                        .column_by_name("publication_year")
                        .context("projected batch missing publication_year")?;
                    let pub_year_array = pub_year_col.as_any().downcast_ref::<Int32Array>();

                    for i in 0..batch.num_rows() {
                        if work_id_col.is_null(i) {
                            continue;
                        }
                        let year = if pub_year_col.is_null(i) {
                            0i32
                        } else {
                            pub_year_array
                                .expect("publication_year should be Int32")
                                .value(i)
                        };
                        let wid_num = work_id_num(work_id_col.value(i));
                        let bucket = (wid_num.unsigned_abs() % num_shards as u64) as usize;
                        local.insert(wid_num, year, bucket as u32);
                    }
                }

                pb.inc(1);
                Ok(local)
            })
            .try_reduce(WorkPartitionMap::new, |mut a, b| {
                a.extend(b);
                Ok(a)
            })
    })?;

    Ok(map)
}

// ============================================================
// Per-table processing
// ============================================================

/// Process works table: 2-pass (build partition map, then parallel read+write).
fn process_works(
    config: &ResolvedHiveConfig,
    pool: &rayon::ThreadPool,
    pb: &ProgressBar,
) -> Result<WorkPartitionMap> {
    let staging = config.staging_dir.join("works");
    fs::create_dir_all(&staging)?;

    let shard_files = list_shard_files(&config.raw_dir.join("works"))?;
    if shard_files.is_empty() {
        anyhow::bail!("No works shard files found in {}", config.raw_dir.display());
    }

    // Pass 1: build partition map (projection: work_id + publication_year only)
    pb.set_message("building partition map...");
    let t_map = Instant::now();
    let partition_map = rebuild_partition_map_from(config, &shard_files, pool, pb)?;
    tracing::info!(
        "{:<20} partition map: {} entries from {} shards [{}]",
        "works",
        partition_map.len(),
        shard_files.len(),
        fmt_elapsed(t_map.elapsed()),
    );

    // Pass 2: parallel read + write via per-partition Mutex writers
    let schema = {
        let file = fs::File::open(&shard_files[0])?;
        ParquetRecordBatchReaderBuilder::try_new(file)?
            .schema()
            .clone()
    };

    let pub_year_idx = schema
        .index_of("publication_year")
        .map_err(|e| anyhow::anyhow!("works schema missing publication_year: {e}"))?;
    let work_id_idx = schema
        .index_of("work_id")
        .map_err(|e| anyhow::anyhow!("works schema missing work_id: {e}"))?;
    let num_shards = config.num_shards;

    let props = writer_properties(config);
    let partitions = partition_map.unique_partitions();
    let writers = create_partition_writers(&partitions, &schema, &props, &staging)?;
    let per_writer_mem_limit = config.memory_limit_bytes * 9 / 10 / partitions.len().max(1);
    tracing::info!(
        "{:<20} writing {} shards \u{2192} {} partitions (mem limit {}/writer)",
        "works",
        shard_files.len(),
        partitions.len(),
        format_bytes(per_writer_mem_limit),
    );

    pb.set_position(0);
    pb.set_message("writing partitions...");

    parallel_process(
        &shard_files,
        &schema,
        &writers,
        per_writer_mem_limit,
        pool,
        pb,
        |batch| {
            let num_rows = batch.num_rows();
            let work_id_col = batch
                .column(work_id_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("work_id column must be Utf8")?;
            let pub_year_col = batch.column(pub_year_idx);
            let pub_year_array = pub_year_col.as_any().downcast_ref::<Int32Array>();

            let mut groups: HashMap<(i32, usize), Vec<u32>> = HashMap::new();

            for i in 0..num_rows {
                let year = if pub_year_col.is_null(i) {
                    0i32
                } else {
                    pub_year_array
                        .expect("publication_year should be Int32")
                        .value(i)
                };
                let bucket = if work_id_col.is_null(i) {
                    0usize
                } else {
                    let num = work_id_num(work_id_col.value(i));
                    (num.unsigned_abs() % num_shards as u64) as usize
                };
                groups.entry((year, bucket)).or_default().push(i as u32);
            }

            Ok(groups)
        },
    )?;

    tracing::info!(
        "{:<20} read done, compressing {} partitions",
        "works",
        partitions.len(),
    );
    pb.set_message("compressing...");
    close_writers_parallel(writers, pool)?;

    Ok(partition_map)
}

/// Process a child table: parallel read with partition_map lookup.
fn process_child(
    table: &str,
    config: &ResolvedHiveConfig,
    partition_map: &WorkPartitionMap,
    pool: &rayon::ThreadPool,
    pb: &ProgressBar,
) -> Result<()> {
    let staging = config.staging_dir.join(table);
    fs::create_dir_all(&staging)?;

    let shard_files = list_shard_files(&config.raw_dir.join(table))?;
    if shard_files.is_empty() {
        return Ok(());
    }

    let schema = {
        let file = fs::File::open(&shard_files[0])?;
        ParquetRecordBatchReaderBuilder::try_new(file)?
            .schema()
            .clone()
    };

    let work_id_idx = schema
        .index_of("work_id")
        .map_err(|e| anyhow::anyhow!("{table} schema missing work_id: {e}"))?;

    let unmatched = std::sync::atomic::AtomicUsize::new(0);

    let props = writer_properties(config);
    let partitions = partition_map.unique_partitions();
    let writers = create_partition_writers(&partitions, &schema, &props, &staging)?;
    let per_writer_mem_limit = config.memory_limit_bytes * 9 / 10 / partitions.len().max(1);
    tracing::info!(
        "{:<20} writing {} shards \u{2192} {} partitions",
        table,
        shard_files.len(),
        partitions.len(),
    );

    parallel_process(
        &shard_files,
        &schema,
        &writers,
        per_writer_mem_limit,
        pool,
        pb,
        |batch| {
            let num_rows = batch.num_rows();
            let work_id_col = batch
                .column(work_id_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("work_id column must be Utf8")?;

            let mut groups: HashMap<(i32, usize), Vec<u32>> = HashMap::new();

            for i in 0..num_rows {
                if work_id_col.is_null(i) {
                    unmatched.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
                let wid_num = work_id_num(work_id_col.value(i));
                match partition_map.get(wid_num) {
                    Some((year, bucket)) => {
                        groups.entry((year, bucket)).or_default().push(i as u32);
                    }
                    None => {
                        unmatched.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }

            Ok(groups)
        },
    )?;

    tracing::info!(
        "{:<20} read done, compressing {} partitions",
        table,
        partitions.len(),
    );
    pb.set_message("compressing...");
    close_writers_parallel(writers, pool)?;

    let u = unmatched.load(std::sync::atomic::Ordering::Relaxed);
    if u > 0 {
        tracing::warn!("{:<20} {} rows had no matching work_id (skipped)", table, u,);
    }

    Ok(())
}

// ============================================================
// Atomic swap
// ============================================================

/// Swap staging/{table} -> {hive_dir}/{table} atomically via rename.
fn atomic_swap_table(hive_dir: &Path, staging_dir: &Path, table: &str) -> Result<()> {
    let target = hive_dir.join(table);
    let staged = staging_dir.join(table);
    let old = hive_dir.join(format!(".old_{table}"));

    if target.exists() {
        fs::rename(&target, &old).with_context(|| format!("Failed to rename old {table}"))?;
    }
    fs::rename(&staged, &target).with_context(|| format!("Failed to swap staged {table}"))?;
    if old.exists()
        && let Err(e) = fs::remove_dir_all(&old)
    {
        tracing::warn!("Failed to remove old {table} dir: {e}");
    }
    Ok(())
}

/// Remove staging directory for a table on failure.
fn cleanup_staging(staging_dir: &Path, table: &str) {
    let staged = staging_dir.join(table);
    if staged.exists()
        && let Err(e) = fs::remove_dir_all(&staged)
    {
        tracing::warn!("Failed to clean staging for {table}: {e}");
    }
}

// ============================================================
// State persistence helpers
// ============================================================

fn save_partial_state(
    hash: &str,
    config: &ResolvedHiveConfig,
    all_tables: &[&str],
    completed: &[String],
) -> Result<()> {
    let state = HiveState {
        source_manifest_hash: hash.to_string(),
        zstd_level: config.zstd_level,
        row_group_size: config.row_group_size,
        num_shards: config.num_shards,
        completed_at: String::new(),
        tables: all_tables.iter().map(|s| (*s).to_string()).collect(),
        completed_tables: completed.to_vec(),
    };
    state.save(&config.hive_dir)
}

fn save_final_state(hash: &str, config: &ResolvedHiveConfig, all_tables: &[&str]) -> Result<()> {
    let all: Vec<String> = all_tables.iter().map(|s| (*s).to_string()).collect();
    let state = HiveState {
        source_manifest_hash: hash.to_string(),
        zstd_level: config.zstd_level,
        row_group_size: config.row_group_size,
        num_shards: config.num_shards,
        completed_at: chrono::Utc::now().to_rfc3339(),
        tables: all.clone(),
        completed_tables: all,
    };
    state.save(&config.hive_dir)
}

// ============================================================
// Progress styles
// ============================================================

fn hive_global_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("{prefix:.bold} {bar:30.cyan/black.dim} {pos}/{len} tables [{elapsed_precise}]")
        .expect("invalid template")
        .progress_chars("=>-")
}

fn hive_table_style() -> ProgressStyle {
    ProgressStyle::default_bar()
        .template("{prefix:<20} {bar:20.cyan/black.dim} {pos}/{len} shards [{elapsed}]")
        .expect("invalid template")
        .progress_chars("=>-")
}

fn make_table_bar(multi: &MultiProgress, is_tty: bool, name: &str) -> ProgressBar {
    if !is_tty {
        return ProgressBar::hidden();
    }
    let pb = multi.add(ProgressBar::new(0));
    pb.set_style(hive_table_style());
    pb.set_prefix(name.to_string());
    pb
}

// ============================================================
// Utility
// ============================================================

/// Check if raw shard directory has data for a table.
fn has_raw_shards(raw_dir: &Path, table: &str) -> bool {
    let dir = raw_dir.join(table);
    dir.exists()
        && fs::read_dir(&dir)
            .map(|mut rd| {
                rd.any(|e| {
                    e.ok()
                        .is_some_and(|e| e.path().extension().is_some_and(|ext| ext == "parquet"))
                })
            })
            .unwrap_or(false)
}

fn fmt_elapsed(elapsed: std::time::Duration) -> String {
    let secs = elapsed.as_secs();
    if secs >= 60 {
        format!("{}m{:02}s", secs / 60, secs % 60)
    } else {
        format!("{:.1}s", elapsed.as_secs_f64())
    }
}

// ============================================================
// Public entry point
// ============================================================

pub fn run_hive(
    config: &ResolvedHiveConfig,
    force: bool,
    dry_run: bool,
    multi: &MultiProgress,
) -> Result<()> {
    let is_tty = std::io::stderr().is_terminal();
    let total_start = Instant::now();

    // 1. Compute source manifest hash
    let hash = manifest_hash(&config.raw_dir)?;

    // 2. Check idempotency / resume
    fs::create_dir_all(&config.hive_dir)?;
    let mut completed_tables: Vec<String> = Vec::new();

    if !force
        && let Some(state) = HiveState::load(&config.hive_dir)
        && state.source_manifest_hash == hash
        && state.zstd_level == config.zstd_level
        && state.row_group_size == config.row_group_size
        && state.num_shards == config.num_shards
    {
        if !state.completed_at.is_empty() {
            println!("Up to date (manifest hash unchanged)");
            return Ok(());
        }
        // Partial run detected: resume from completed tables (verify they exist)
        completed_tables = state
            .completed_tables
            .into_iter()
            .filter(|t| config.hive_dir.join(t).exists())
            .collect();
    }

    // 3. Determine tables with raw data
    let tables: Vec<&str> = TABLES
        .iter()
        .copied()
        .filter(|t| has_raw_shards(&config.raw_dir, t))
        .collect();

    if tables.is_empty() {
        anyhow::bail!(
            "No raw parquet shards found in {}",
            config.raw_dir.display()
        );
    }

    // 4. Filter out already completed tables
    let remaining: Vec<&str> = tables
        .iter()
        .copied()
        .filter(|t| !completed_tables.iter().any(|c| c == *t))
        .collect();

    if remaining.is_empty() {
        // All tables done but completed_at wasn't saved (e.g., crash after last table)
        save_final_state(&hash, config, &tables)?;
        println!("Up to date (all tables already completed)");
        return Ok(());
    }

    if !completed_tables.is_empty() {
        tracing::info!(
            "Resuming: {}/{} tables already done",
            completed_tables.len(),
            tables.len()
        );
    }

    // 5. Dry run
    if dry_run {
        println!("Hive partitioning plan:");
        println!("  Source:      {}", config.raw_dir.display());
        println!("  Destination: {}", config.hive_dir.display());
        println!("  ZSTD level:  {}", config.zstd_level);
        println!("  Row group:   {}", config.row_group_size);
        println!("  Shards/year: {}", config.num_shards);
        println!("  Threads:     {}", config.threads);
        println!("  Memory:      {}", format_bytes(config.memory_limit_bytes));
        println!("  Tables:      {}", tables.len());
        for t in &tables {
            let status = if completed_tables.iter().any(|c| c == *t) {
                " (done)"
            } else {
                ""
            };
            println!("    - {t}{status}");
        }
        if !completed_tables.is_empty() {
            println!("  Resume:      {} remaining", remaining.len());
        }
        return Ok(());
    }

    tracing::info!(
        "Hive config: threads={}, memory={}, zstd={}, row_group={}, shards/year={}",
        config.threads,
        format_bytes(config.memory_limit_bytes),
        config.zstd_level,
        config.row_group_size,
        config.num_shards,
    );

    // 6. Progress bars
    let global_bar = if is_tty {
        let pb = multi.add(ProgressBar::new(tables.len() as u64));
        pb.set_style(hive_global_style());
        pb.set_prefix("Hive");
        pb.set_position(completed_tables.len() as u64);
        pb
    } else {
        ProgressBar::hidden()
    };

    // 7. Build rayon thread pool for parallel compression
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build()
        .context("Failed to build thread pool")?;

    // Clean stale staging from any previous interrupted run, then create fresh
    if config.staging_dir.exists() {
        fs::remove_dir_all(&config.staging_dir)
            .context("Failed to clean stale staging directory")?;
    }
    fs::create_dir_all(&config.staging_dir)?;

    // 8. Process works first (builds partition map for child tables)
    let partition_map = if remaining.contains(&"works") {
        let pb = make_table_bar(multi, is_tty, "works");
        let t = Instant::now();
        match process_works(config, &pool, &pb) {
            Ok(map) => {
                atomic_swap_table(&config.hive_dir, &config.staging_dir, "works")?;
                completed_tables.push("works".to_string());
                save_partial_state(&hash, config, &tables, &completed_tables)?;
                let elapsed = t.elapsed();
                pb.finish_with_message(format!(
                    "done ({}, {} works)",
                    fmt_elapsed(elapsed),
                    map.len()
                ));
                global_bar.inc(1);
                tracing::info!(
                    "{:<20} completed in {} ({} works)",
                    "works",
                    fmt_elapsed(elapsed),
                    map.len(),
                );
                map
            }
            Err(e) => {
                cleanup_staging(&config.staging_dir, "works");
                pb.finish_with_message("FAILED");
                return Err(e).context("Failed to build works hive");
            }
        }
    } else {
        // Works already done — rebuild partition map from existing hive works
        let pb = make_table_bar(multi, is_tty, "works (map)");
        let t = Instant::now();
        let map = rebuild_partition_map(config, &pool, &pb)?;
        let elapsed = t.elapsed();
        pb.finish_with_message(format!(
            "map rebuilt ({}, {} works)",
            fmt_elapsed(elapsed),
            map.len()
        ));
        tracing::info!(
            "{:<20} partition map rebuilt ({} entries) [{}]",
            "works",
            map.len(),
            fmt_elapsed(elapsed),
        );
        map
    };

    // 9. Process remaining child tables sequentially
    for table in &remaining {
        if *table == "works" {
            continue;
        }
        let pb = make_table_bar(multi, is_tty, table);
        let t = Instant::now();
        match process_child(table, config, &partition_map, &pool, &pb) {
            Ok(()) => {
                atomic_swap_table(&config.hive_dir, &config.staging_dir, table)?;
                completed_tables.push(table.to_string());
                save_partial_state(&hash, config, &tables, &completed_tables)?;
                let elapsed = t.elapsed();
                pb.finish_with_message(format!("done ({})", fmt_elapsed(elapsed)));
                global_bar.inc(1);
                tracing::info!("{:<20} completed in {}", table, fmt_elapsed(elapsed));
            }
            Err(e) => {
                cleanup_staging(&config.staging_dir, table);
                pb.finish_with_message("FAILED");
                return Err(e).with_context(|| format!("Failed to build {table} hive"));
            }
        }
    }

    // 10. Save final state (marks run as complete)
    save_final_state(&hash, config, &tables)?;

    // 11. Cleanup staging dir
    if config.staging_dir.exists()
        && let Err(e) = fs::remove_dir_all(&config.staging_dir)
    {
        tracing::warn!("Failed to remove staging dir: {e}");
    }

    global_bar.finish_and_clear();

    println!(
        "Done: {} tables, {}",
        tables.len(),
        fmt_elapsed(total_start.elapsed()),
    );

    Ok(())
}

/// Rebuild partition map from raw works shards (for resume).
///
/// Delegates to `rebuild_partition_map_from` which parallelises the read.
fn rebuild_partition_map(
    config: &ResolvedHiveConfig,
    pool: &rayon::ThreadPool,
    pb: &ProgressBar,
) -> Result<WorkPartitionMap> {
    let shard_files = list_shard_files(&config.raw_dir.join("works"))?;
    rebuild_partition_map_from(config, &shard_files, pool, pb)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::parse_memory_limit;

    #[test]
    fn parse_memory_limit_gb() {
        assert_eq!(parse_memory_limit("32GB").unwrap(), 32 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_memory_limit_mb() {
        assert_eq!(parse_memory_limit("512MB").unwrap(), 512 * 1024 * 1024);
    }

    #[test]
    fn parse_memory_limit_bytes() {
        assert_eq!(parse_memory_limit("1024").unwrap(), 1024);
    }

    #[test]
    fn parse_memory_limit_invalid() {
        assert!(parse_memory_limit("notanumber").is_err());
    }

    #[test]
    fn hive_state_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let state = HiveState {
            source_manifest_hash: "abc123".to_string(),
            zstd_level: 8,
            row_group_size: 500_000,
            num_shards: 4,
            completed_at: "2026-01-01T00:00:00Z".to_string(),
            tables: vec!["works".to_string(), "citations".to_string()],
            completed_tables: vec!["works".to_string(), "citations".to_string()],
        };
        state.save(dir.path()).unwrap();
        let loaded = HiveState::load(dir.path()).unwrap();
        assert_eq!(loaded.source_manifest_hash, "abc123");
        assert_eq!(loaded.zstd_level, 8);
        assert_eq!(loaded.tables.len(), 2);
        assert_eq!(loaded.completed_tables.len(), 2);
    }

    #[test]
    fn hive_state_load_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(HiveState::load(dir.path()).is_none());
    }

    #[test]
    fn hive_state_backward_compat() {
        // Old state files without completed_tables should deserialize with empty vec
        let dir = tempfile::TempDir::new().unwrap();
        let json = r#"{"source_manifest_hash":"abc","zstd_level":8,"completed_at":"2026-01-01","tables":["works"]}"#;
        fs::write(dir.path().join(".hive_state.json"), json).unwrap();
        let loaded = HiveState::load(dir.path()).unwrap();
        assert!(loaded.completed_tables.is_empty());
        assert_eq!(loaded.tables, vec!["works"]);
    }

    #[test]
    fn hive_state_resume_partial() {
        // Partial state: completed_at is empty, completed_tables has some entries
        let dir = tempfile::TempDir::new().unwrap();
        let state = HiveState {
            source_manifest_hash: "hash1".to_string(),
            zstd_level: 8,
            row_group_size: 500_000,
            num_shards: 4,
            completed_at: String::new(),
            tables: vec![
                "works".to_string(),
                "citations".to_string(),
                "work_topics".to_string(),
            ],
            completed_tables: vec!["works".to_string()],
        };
        state.save(dir.path()).unwrap();
        let loaded = HiveState::load(dir.path()).unwrap();
        assert!(loaded.completed_at.is_empty());
        assert_eq!(loaded.completed_tables, vec!["works"]);
    }

    #[test]
    fn manifest_hash_deterministic() {
        let dir = tempfile::TempDir::new().unwrap();
        fs::write(
            dir.path().join(".manifest.json"),
            r#"{"saved_at":"2026-01-01","entries":[]}"#,
        )
        .unwrap();
        let h1 = manifest_hash(dir.path()).unwrap();
        let h2 = manifest_hash(dir.path()).unwrap();
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // SHA-256 hex
    }

    #[test]
    fn manifest_hash_missing_file() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(manifest_hash(dir.path()).is_err());
    }

    #[test]
    fn has_raw_shards_empty_dir() {
        let dir = tempfile::TempDir::new().unwrap();
        assert!(!has_raw_shards(dir.path(), "works"));
    }

    #[test]
    fn has_raw_shards_with_parquet() {
        let dir = tempfile::TempDir::new().unwrap();
        let works_dir = dir.path().join("works");
        fs::create_dir_all(&works_dir).unwrap();
        fs::write(works_dir.join("shard_0000.parquet"), b"fake").unwrap();
        assert!(has_raw_shards(dir.path(), "works"));
    }

    #[test]
    fn has_raw_shards_no_parquet() {
        let dir = tempfile::TempDir::new().unwrap();
        let works_dir = dir.path().join("works");
        fs::create_dir_all(&works_dir).unwrap();
        fs::write(works_dir.join("readme.txt"), b"not parquet").unwrap();
        assert!(!has_raw_shards(dir.path(), "works"));
    }

    #[test]
    fn fmt_elapsed_seconds() {
        assert_eq!(fmt_elapsed(std::time::Duration::from_secs_f64(3.5)), "3.5s");
    }

    #[test]
    fn fmt_elapsed_minutes() {
        assert_eq!(fmt_elapsed(std::time::Duration::from_secs(125)), "2m05s");
    }

    #[test]
    fn work_id_num_basic() {
        assert_eq!(work_id_num("W1234567890"), 1234567890);
        assert_eq!(work_id_num("W0"), 0);
    }

    #[test]
    fn work_id_num_empty() {
        assert_eq!(work_id_num(""), 0);
        assert_eq!(work_id_num("W"), 0); // "W"[1..] = "" which fails to parse
    }

    #[test]
    fn work_partition_map_roundtrip() {
        let mut map = WorkPartitionMap::new();
        map.insert(12345, 2023, 2);
        map.insert(99999, 0, 0); // year=0 for null pub_year

        assert_eq!(map.get(12345), Some((2023, 2)));
        assert_eq!(map.get(99999), Some((0, 0)));
        assert_eq!(map.get(11111), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn partition_writers_basic() {
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::TempDir::new().unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let props = WriterProperties::builder().build();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();

        let partitions = vec![(2020, 0), (2021, 1)];
        let writers = create_partition_writers(&partitions, &schema, &props, dir.path()).unwrap();

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

        // Route rows 0,2 → (2020,0), row 1 → (2021,1)
        let groups: HashMap<(i32, usize), Vec<u32>> =
            HashMap::from([((2020, 0), vec![0, 2]), ((2021, 1), vec![1])]);

        for ((year, bucket), indices) in &groups {
            let indices_array = UInt32Array::from(indices.clone());
            let columns: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .map(|col| compute::take(col.as_ref(), &indices_array, None).unwrap())
                .collect();
            let sub_batch = RecordBatch::try_new(batch.schema(), columns).unwrap();
            writers
                .get(&(*year, *bucket))
                .unwrap()
                .lock()
                .unwrap()
                .write(&sub_batch)
                .unwrap();
        }

        close_writers_parallel(writers, &pool).unwrap();

        assert!(dir.path().join("pub_year=2020/shard_0.parquet").exists());
        assert!(dir.path().join("pub_year=2021/shard_1.parquet").exists());
    }
}
