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
use rustc_hash::FxHashMap;
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

fn writer_properties(config: &ResolvedHiveConfig) -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(config.zstd_level).expect("valid zstd level"),
        ))
        .set_max_row_group_size(config.row_group_size)
        .build()
}

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
// Partitioned writer
// ============================================================

/// Manages per-partition ArrowWriter instances.
///
/// Each (pub_year, shard_bucket) combination gets its own parquet file.
/// Writers are lazily created and auto-flush row groups at the configured size.
///
/// Memory tracking uses actual Arrow buffer sizes (not `in_progress_size()` which
/// returns encoded/compressed sizes that underestimate real memory).
struct PartitionedWriter {
    writers: HashMap<(i32, usize), ArrowWriter<BufWriter<fs::File>>>,
    /// Unflushed rows per partition — reset to 0 on flush.
    unflushed_rows: HashMap<(i32, usize), usize>,
    /// Total unflushed rows across all writers.
    total_unflushed_rows: usize,
    /// Estimated bytes per row in Arrow memory (computed from first batch).
    est_bytes_per_row: usize,
    schema: arrow::datatypes::SchemaRef,
    props: WriterProperties,
    base_dir: PathBuf,
    max_buffer_bytes: usize,
}

impl PartitionedWriter {
    fn new(
        schema: arrow::datatypes::SchemaRef,
        props: WriterProperties,
        base_dir: PathBuf,
        max_buffer_bytes: usize,
    ) -> Self {
        Self {
            writers: HashMap::new(),
            unflushed_rows: HashMap::new(),
            total_unflushed_rows: 0,
            est_bytes_per_row: 0,
            schema,
            props,
            base_dir,
            max_buffer_bytes,
        }
    }

    fn get_or_create(
        &mut self,
        year: i32,
        bucket: usize,
    ) -> Result<&mut ArrowWriter<BufWriter<fs::File>>> {
        // Use entry API to avoid double lookup
        use std::collections::hash_map::Entry;
        match self.writers.entry((year, bucket)) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let dir = self.base_dir.join(format!("pub_year={year}"));
                fs::create_dir_all(&dir)?;
                let path = dir.join(format!("shard_{bucket}.parquet"));
                let file = BufWriter::new(fs::File::create(&path)?);
                let writer =
                    ArrowWriter::try_new(file, self.schema.clone(), Some(self.props.clone()))?;
                Ok(entry.insert(writer))
            }
        }
    }

    /// Write a batch, splitting rows by partition assignment.
    fn write_partitioned(
        &mut self,
        batch: &RecordBatch,
        years: &[i32],
        buckets: &[usize],
    ) -> Result<()> {
        let num_rows = batch.num_rows();
        debug_assert_eq!(num_rows, years.len());
        debug_assert_eq!(num_rows, buckets.len());

        // Estimate bytes-per-row from the first batch using actual Arrow buffer sizes
        if self.est_bytes_per_row == 0 && num_rows > 0 {
            let total_mem: usize = batch
                .columns()
                .iter()
                .map(|c| c.get_buffer_memory_size())
                .sum();
            self.est_bytes_per_row = (total_mem / num_rows).max(64);
        }

        // Group row indices by (year, bucket)
        let mut groups: HashMap<(i32, usize), Vec<u32>> = HashMap::new();
        for i in 0..num_rows {
            groups
                .entry((years[i], buckets[i]))
                .or_default()
                .push(i as u32);
        }

        // For each group, take rows and write to the partition's writer
        for ((year, bucket), indices) in &groups {
            let n = indices.len();
            let indices_array = UInt32Array::from(indices.clone());
            let columns: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .map(|col| compute::take(col.as_ref(), &indices_array, None))
                .collect::<Result<Vec<_>, _>>()
                .context("arrow take failed")?;
            let sub_batch = RecordBatch::try_new(batch.schema(), columns)?;
            let writer = self.get_or_create(*year, *bucket)?;
            writer.write(&sub_batch)?;

            // Track unflushed rows
            *self.unflushed_rows.entry((*year, *bucket)).or_default() += n;
            self.total_unflushed_rows += n;
        }

        // Flush largest writers if estimated memory exceeds budget
        self.maybe_flush()?;
        Ok(())
    }

    fn maybe_flush(&mut self) -> Result<()> {
        let bpr = self.est_bytes_per_row.max(64);
        loop {
            let est_bytes = self.total_unflushed_rows * bpr;
            if est_bytes <= self.max_buffer_bytes {
                break;
            }
            // Find the partition with the most unflushed rows
            let key = self
                .unflushed_rows
                .iter()
                .filter(|(_, rows)| **rows > 0)
                .max_by_key(|(_, rows)| **rows)
                .map(|(k, _)| *k);
            match key {
                Some(key) => {
                    let rows = self.unflushed_rows[&key];
                    self.writers
                        .get_mut(&key)
                        .unwrap()
                        .flush()
                        .context("flush failed during memory management")?;
                    self.total_unflushed_rows -= rows;
                    *self.unflushed_rows.get_mut(&key).unwrap() = 0;
                }
                None => break,
            }
        }
        Ok(())
    }

    /// Flush all buffers in parallel, then close writers (just footers).
    fn close_all(self, pool: &rayon::ThreadPool) -> Result<()> {
        let mut writers: Vec<_> = self.writers.into_iter().collect();
        if writers.is_empty() {
            return Ok(());
        }

        // Phase 1: parallel flush — compress buffered data, free Arrow memory
        pool.install(|| {
            writers
                .par_iter_mut()
                .try_for_each(|(_, writer)| -> Result<()> {
                    writer.flush().context("flush failed during close")
                })
        })?;

        // Phase 2: parallel close — write footers only (fast, no compression)
        pool.install(|| {
            writers
                .into_par_iter()
                .try_for_each(|((year, bucket), writer)| {
                    writer
                        .close()
                        .map(|_| ())
                        .with_context(|| format!("Failed to close pub_year={year}/shard_{bucket}"))
                })
        })
    }
}

// ============================================================
// Shard file listing
// ============================================================

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
// Per-table processing
// ============================================================

/// Process works table: build partition map + write hive parquet.
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

    // Read schema from first file
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

    let max_buffer = config.memory_limit_bytes * 2 / 5; // 40% for writer buffers
    let mut writer = PartitionedWriter::new(
        schema.clone(),
        writer_properties(config),
        staging,
        max_buffer,
    );
    let mut partition_map = WorkPartitionMap::new();

    pb.set_length(shard_files.len() as u64);
    for shard_path in &shard_files {
        let file = fs::File::open(shard_path)
            .with_context(|| format!("Cannot open {}", shard_path.display()))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("Invalid parquet: {}", shard_path.display()))?
            .with_batch_size(READER_BATCH_SIZE)
            .build()
            .with_context(|| format!("Cannot build reader: {}", shard_path.display()))?;

        for batch_result in reader {
            let batch = coerce_batch(batch_result?, &schema)?;
            let num_rows = batch.num_rows();

            let work_id_col = batch
                .column(work_id_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("work_id column must be Utf8")?;

            let pub_year_col = batch.column(pub_year_idx);
            let pub_year_array = pub_year_col.as_any().downcast_ref::<Int32Array>();

            let mut years = Vec::with_capacity(num_rows);
            let mut buckets = Vec::with_capacity(num_rows);

            for i in 0..num_rows {
                let year = if pub_year_col.is_null(i) {
                    0i32
                } else {
                    pub_year_array
                        .expect("publication_year should be Int32")
                        .value(i)
                };

                let (wid_num, bucket) = if work_id_col.is_null(i) {
                    (0i64, 0usize)
                } else {
                    let wid = work_id_col.value(i);
                    let num = work_id_num(wid);
                    let b = (num.unsigned_abs() % config.num_shards as u64) as usize;
                    (num, b)
                };

                years.push(year);
                buckets.push(bucket);

                if !work_id_col.is_null(i) {
                    partition_map.insert(wid_num, year, bucket as u32);
                }
            }

            writer.write_partitioned(&batch, &years, &buckets)?;
        }

        pb.inc(1);
    }

    writer.close_all(pool)?;
    Ok(partition_map)
}

/// Process a child table: look up partition assignment from works map.
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

    let max_buffer = config.memory_limit_bytes * 2 / 5;
    let mut writer = PartitionedWriter::new(
        schema.clone(),
        writer_properties(config),
        staging,
        max_buffer,
    );
    let mut unmatched = 0usize;

    pb.set_length(shard_files.len() as u64);
    for shard_path in &shard_files {
        let file = fs::File::open(shard_path)
            .with_context(|| format!("Cannot open {}", shard_path.display()))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("Invalid parquet: {}", shard_path.display()))?
            .with_batch_size(READER_BATCH_SIZE)
            .build()
            .with_context(|| format!("Cannot build reader: {}", shard_path.display()))?;

        for batch_result in reader {
            let batch = coerce_batch(batch_result?, &schema)?;
            let num_rows = batch.num_rows();

            let work_id_col = batch
                .column(work_id_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("work_id column must be Utf8")?;

            let mut years = Vec::with_capacity(num_rows);
            let mut buckets = Vec::with_capacity(num_rows);
            let mut matched = Vec::with_capacity(num_rows);

            for i in 0..num_rows {
                if work_id_col.is_null(i) {
                    unmatched += 1;
                    continue;
                }
                let wid_num = work_id_num(work_id_col.value(i));
                match partition_map.get(wid_num) {
                    Some((year, bucket)) => {
                        years.push(year);
                        buckets.push(bucket);
                        matched.push(i as u32);
                    }
                    None => {
                        unmatched += 1;
                    }
                }
            }

            if matched.len() == num_rows {
                // All rows matched — write directly
                writer.write_partitioned(&batch, &years, &buckets)?;
            } else if !matched.is_empty() {
                // Filter to matched rows only
                let indices = UInt32Array::from(matched);
                let columns: Vec<ArrayRef> = batch
                    .columns()
                    .iter()
                    .map(|col| compute::take(col.as_ref(), &indices, None))
                    .collect::<Result<Vec<_>, _>>()
                    .context("arrow take failed")?;
                let filtered = RecordBatch::try_new(batch.schema(), columns)?;
                writer.write_partitioned(&filtered, &years, &buckets)?;
            }
        }

        pb.inc(1);
    }

    if unmatched > 0 {
        tracing::warn!(
            "{table}: {unmatched} rows had no matching work_id in works table (skipped)"
        );
    }

    writer.close_all(pool)?;
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
                    "works completed in {} ({} entries in partition map)",
                    fmt_elapsed(elapsed),
                    map.len()
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
        let map = rebuild_partition_map(config, &pb)?;
        let elapsed = t.elapsed();
        pb.finish_with_message(format!(
            "map rebuilt ({}, {} works)",
            fmt_elapsed(elapsed),
            map.len()
        ));
        tracing::info!(
            "Partition map rebuilt in {} ({} entries)",
            fmt_elapsed(elapsed),
            map.len()
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
                tracing::info!("{table} completed in {}", fmt_elapsed(elapsed));
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
/// Uses column projection to read only `work_id` + `publication_year` (2 of 84
/// columns), reducing I/O by ~97% compared to a full read.
fn rebuild_partition_map(
    config: &ResolvedHiveConfig,
    pb: &ProgressBar,
) -> Result<WorkPartitionMap> {
    let shard_files = list_shard_files(&config.raw_dir.join("works"))?;
    let mut map = WorkPartitionMap::new();

    pb.set_length(shard_files.len() as u64);
    for shard_path in &shard_files {
        let file = fs::File::open(shard_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;

        // Project only the 2 columns needed for partition routing
        let arrow_schema = builder.schema();
        let work_id_root = arrow_schema
            .index_of("work_id")
            .map_err(|e| anyhow::anyhow!("works schema missing work_id: {e}"))?;
        let pub_year_root = arrow_schema
            .index_of("publication_year")
            .map_err(|e| anyhow::anyhow!("works schema missing publication_year: {e}"))?;
        let mask = ProjectionMask::roots(builder.parquet_schema(), [work_id_root, pub_year_root]);

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
                let bucket = (wid_num.unsigned_abs() % config.num_shards as u64) as usize;
                map.insert(wid_num, year, bucket as u32);
            }
        }

        pb.inc(1);
    }

    Ok(map)
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
    fn partitioned_writer_basic() {
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

        let mut writer = PartitionedWriter::new(
            schema.clone(),
            props,
            dir.path().to_path_buf(),
            1_000_000_000,
        );

        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]).unwrap();

        let years = vec![2020, 2021, 2020];
        let buckets = vec![0, 1, 0];
        writer.write_partitioned(&batch, &years, &buckets).unwrap();
        writer.close_all(&pool).unwrap();

        assert!(dir.path().join("pub_year=2020/shard_0.parquet").exists());
        assert!(dir.path().join("pub_year=2021/shard_1.parquet").exists());
    }
}
