//! Parquet I/O for the embedding pipeline.
//!
//! Streaming design: `process_hive_chunks` reads hive parquet files in
//! bounded chunks and calls a callback per chunk, so the caller can
//! encode + write incrementally without holding the full dataset in memory.
//!
//! Incremental mode: when an existing embeddings parquet is provided,
//! `process_hive_chunks_incremental` compares `content_hash` values to skip
//! re-encoding works whose text (title + abstract) has not changed.

use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use arrow::array::{Array, ArrayRef, AsArray, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use rustc_hash::FxHashMap;

const READ_BATCH_SIZE: usize = 65_536;

/// Count total rows and files in hive directory (parquet metadata only, no data read).
pub fn count_hive_rows(hive_dir: &Path) -> Result<(usize, usize)> {
    let mut files = Vec::new();
    walk_parquet_files(hive_dir, &mut files)?;

    let mut total_rows = 0usize;
    for path in &files {
        let file = std::fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        total_rows += builder.metadata().file_metadata().num_rows() as usize;
    }

    Ok((total_rows, files.len()))
}

// ============================================================
// Existing embeddings loader (for incremental mode)
// ============================================================

/// Cached embedding for a single work: content_hash + flat vector.
pub struct CachedEmbedding {
    pub content_hash: String,
    pub vector: Vec<f32>,
}

/// Load existing embeddings parquet into a HashMap for incremental diffing.
///
/// Returns `(map, dim)` where map is `work_id → CachedEmbedding`.
/// If the file has no `content_hash` column (old format), returns empty map
/// so all works will be re-encoded.
pub fn load_existing_embeddings(
    path: &Path,
) -> Result<(FxHashMap<String, CachedEmbedding>, usize)> {
    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();

    let wid_idx = schema.index_of("work_id").ok();
    let hash_idx = schema.index_of("content_hash").ok();
    let emb_idx = schema.index_of("embedding").ok();

    let (Some(wid_idx), Some(hash_idx), Some(emb_idx)) = (wid_idx, hash_idx, emb_idx) else {
        tracing::warn!("Existing embeddings missing content_hash column — full rebuild required");
        return Ok((FxHashMap::default(), 0));
    };

    // Extract embedding dimension from schema
    let dim = match schema.field(emb_idx).data_type() {
        DataType::FixedSizeList(_, d) => *d as usize,
        _ => {
            tracing::warn!("Unexpected embedding column type — full rebuild required");
            return Ok((FxHashMap::default(), 0));
        }
    };

    let proj = ProjectionMask::roots(builder.parquet_schema(), vec![wid_idx, hash_idx, emb_idx]);
    let reader = builder
        .with_projection(proj)
        .with_batch_size(READ_BATCH_SIZE)
        .build()?;

    let mut map = FxHashMap::default();
    for batch_result in reader {
        let batch = batch_result?;
        let wid_col = batch.column(0).as_string::<i32>();
        let hash_col = batch.column(1).as_string::<i32>();
        let emb_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("embedding column should be FixedSizeList");

        for row in 0..batch.num_rows() {
            if wid_col.is_null(row) || hash_col.is_null(row) {
                continue;
            }
            let work_id = wid_col.value(row).to_string();
            let content_hash = hash_col.value(row).to_string();
            let row_array = emb_col.value(row);
            let values = row_array
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("embedding values should be Float32");
            let vector: Vec<f32> = values.values().to_vec();

            map.insert(
                work_id,
                CachedEmbedding {
                    content_hash,
                    vector,
                },
            );
        }
    }

    tracing::info!("Loaded {} existing embeddings (dim={dim})", map.len());
    Ok((map, dim))
}

// ============================================================
// Hive chunk processing
// ============================================================

/// Extract a nullable string column value.
fn str_val(col: Option<&StringArray>, row: usize) -> &str {
    col.and_then(|c| {
        if c.is_null(row) {
            None
        } else {
            Some(c.value(row))
        }
    })
    .unwrap_or("")
}

/// Stream hive parquet files in chunks, calling `on_chunk` for each chunk.
///
/// Each chunk contains `(work_ids, texts)` with up to `chunk_size` rows.
/// Rows with empty title and abstract are skipped.
/// Checks `cancelled` between chunks and returns early if set.
/// Returns total rows processed.
pub fn process_hive_chunks(
    hive_dir: &Path,
    chunk_size: usize,
    max_rows: Option<usize>,
    cancelled: &Arc<AtomicBool>,
    mut on_chunk: impl FnMut(&[String], &[String]) -> Result<()>,
) -> Result<usize> {
    process_hive_chunks_incremental(
        hive_dir,
        chunk_size,
        max_rows,
        cancelled,
        None,
        |work_ids, texts, _hashes, _cache| on_chunk(work_ids, texts),
    )
}

/// Incremental chunk processing stats.
pub struct IncrementalStats {
    pub total: usize,
    pub encoded: usize,
    pub carried: usize,
}

/// Stream hive chunks with content_hash-based incremental support.
///
/// When `existing` is Some, compares each work's content_hash against cached
/// embeddings. The callback receives:
/// - `work_ids`: work IDs that need encoding (new or changed)
/// - `texts`: corresponding texts
/// - `content_hashes`: content_hash for each work_id (for writing to output)
/// - `carried`: Vec of (work_id, content_hash, Vec<f32>) for unchanged works
///
/// The caller should encode the texts, then write both encoded + carried to output.
pub fn process_hive_chunks_incremental(
    hive_dir: &Path,
    chunk_size: usize,
    max_rows: Option<usize>,
    cancelled: &Arc<AtomicBool>,
    existing: Option<&FxHashMap<String, CachedEmbedding>>,
    mut on_chunk: impl FnMut(
        &[String],                     // work_ids to encode
        &[String],                     // texts to encode
        &[String],                     // content_hashes for encode batch
        &[(String, String, Vec<f32>)], // carried: (work_id, hash, vec)
    ) -> Result<()>,
) -> Result<usize> {
    let mut parquet_files = Vec::new();
    walk_parquet_files(hive_dir, &mut parquet_files)?;
    parquet_files.sort();
    anyhow::ensure!(
        !parquet_files.is_empty(),
        "No parquet files in {}",
        hive_dir.display()
    );

    let limit = max_rows.unwrap_or(usize::MAX);
    let mut buf_ids = Vec::with_capacity(chunk_size);
    let mut buf_texts = Vec::with_capacity(chunk_size);
    let mut buf_hashes = Vec::with_capacity(chunk_size);
    let mut buf_carried: Vec<(String, String, Vec<f32>)> = Vec::with_capacity(chunk_size);
    let mut total = 0usize;

    'outer: for path in &parquet_files {
        let file = std::fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema().clone();

        let Some(wid_root) = schema.index_of("work_id").ok() else {
            continue;
        };
        let title_root = schema.index_of("title").ok();
        let abstract_root = schema.index_of("abstract_text").ok();
        let hash_root = schema.index_of("content_hash").ok();

        let mut proj = vec![wid_root];
        let title_local = title_root.map(|i| {
            let idx = proj.len();
            proj.push(i);
            idx
        });
        let abstract_local = abstract_root.map(|i| {
            let idx = proj.len();
            proj.push(i);
            idx
        });
        let hash_local = hash_root.map(|i| {
            let idx = proj.len();
            proj.push(i);
            idx
        });

        let mask = ProjectionMask::roots(builder.parquet_schema(), proj);
        let reader = builder
            .with_projection(mask)
            .with_batch_size(READ_BATCH_SIZE)
            .build()?;

        for batch_result in reader {
            let batch = batch_result?;
            let wid_col = batch.column(0).as_string::<i32>();
            let title_col = title_local.map(|i| batch.column(i).as_string::<i32>());
            let abstract_col = abstract_local.map(|i| batch.column(i).as_string::<i32>());
            let hash_col = hash_local.map(|i| batch.column(i).as_string::<i32>());

            for row in 0..batch.num_rows() {
                if total >= limit {
                    break 'outer;
                }

                let t = str_val(title_col, row).trim();
                let a = str_val(abstract_col, row).trim();

                let text = match (t.is_empty(), a.is_empty()) {
                    (false, false) => format!("{t}\n{a}"),
                    (false, true) => t.to_string(),
                    (true, false) => a.to_string(),
                    (true, true) => continue,
                };

                let work_id = wid_col.value(row).to_string();
                let content_hash = str_val(hash_col, row).to_string();

                // Check if we can carry over existing embedding
                if let Some(cache) = existing
                    && let Some(cached) = cache.get(&work_id)
                    && !content_hash.is_empty()
                    && cached.content_hash == content_hash
                {
                    buf_carried.push((work_id, content_hash, cached.vector.clone()));
                    total += 1;
                    // Flush if combined buffer is full
                    if buf_ids.len() + buf_carried.len() >= chunk_size {
                        on_chunk(&buf_ids, &buf_texts, &buf_hashes, &buf_carried)?;
                        buf_ids.clear();
                        buf_texts.clear();
                        buf_hashes.clear();
                        buf_carried.clear();
                        if cancelled.load(Ordering::Relaxed) {
                            return Ok(total);
                        }
                    }
                    continue;
                }

                buf_ids.push(work_id);
                buf_texts.push(text);
                buf_hashes.push(content_hash);
                total += 1;

                if buf_ids.len() + buf_carried.len() >= chunk_size {
                    on_chunk(&buf_ids, &buf_texts, &buf_hashes, &buf_carried)?;
                    buf_ids.clear();
                    buf_texts.clear();
                    buf_hashes.clear();
                    buf_carried.clear();
                    if cancelled.load(Ordering::Relaxed) {
                        return Ok(total);
                    }
                }
            }
        }
    }

    // Flush remaining
    if !buf_ids.is_empty() || !buf_carried.is_empty() {
        on_chunk(&buf_ids, &buf_texts, &buf_hashes, &buf_carried)?;
    }

    Ok(total)
}

// ============================================================
// Parquet writer
// ============================================================

/// Incremental parquet writer for embedding output.
///
/// Writes to `{path}.tmp`, renames to `path` on `close()`.
/// Removes `.tmp` on drop if `close()` was not called (failure path).
pub struct EmbedWriter {
    writer: Option<ArrowWriter<BufWriter<std::fs::File>>>,
    schema: Arc<Schema>,
    item_field: Arc<Field>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    dim: i32,
    rows_written: usize,
    committed: bool,
}

impl EmbedWriter {
    pub fn new(path: &Path, dim: usize) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let tmp_path = path.with_extension("parquet.tmp");

        let item_field = Arc::new(Field::new("item", DataType::Float32, false));
        let schema = Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(item_field.clone(), dim as i32),
                false,
            ),
            Field::new("content_hash", DataType::Utf8, false),
        ]));

        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();

        let file = BufWriter::new(std::fs::File::create(&tmp_path)?);
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        Ok(Self {
            writer: Some(writer),
            schema,
            item_field,
            tmp_path,
            final_path: path.to_path_buf(),
            dim: dim as i32,
            rows_written: 0,
            committed: false,
        })
    }

    /// Write a chunk of newly encoded embeddings with their content hashes.
    pub fn write_chunk(
        &mut self,
        work_ids: &[String],
        embeddings: Vec<f32>,
        content_hashes: &[String],
    ) -> Result<()> {
        if work_ids.is_empty() {
            return Ok(());
        }
        let wid_array: ArrayRef = Arc::new(StringArray::from_iter_values(work_ids));
        let values = Float32Array::from(embeddings);
        let emb_array: ArrayRef = Arc::new(FixedSizeListArray::try_new(
            self.item_field.clone(),
            self.dim,
            Arc::new(values),
            None,
        )?);
        let hash_array: ArrayRef = Arc::new(StringArray::from_iter_values(content_hashes));

        let batch =
            RecordBatch::try_new(self.schema.clone(), vec![wid_array, emb_array, hash_array])?;
        self.writer
            .as_mut()
            .expect("write_chunk after close")
            .write(&batch)?;
        self.rows_written += work_ids.len();
        Ok(())
    }

    /// Write carried-over embeddings (unchanged works).
    pub fn write_carried(&mut self, carried: &[(String, String, Vec<f32>)]) -> Result<()> {
        if carried.is_empty() {
            return Ok(());
        }
        let dim = self.dim as usize;
        let work_ids: Vec<&str> = carried.iter().map(|(id, _, _)| id.as_str()).collect();
        let hashes: Vec<&str> = carried.iter().map(|(_, h, _)| h.as_str()).collect();
        let mut flat_vecs = Vec::with_capacity(carried.len() * dim);
        for (_, _, vec) in carried {
            flat_vecs.extend_from_slice(vec);
        }

        let wid_array: ArrayRef = Arc::new(StringArray::from_iter_values(work_ids));
        let values = Float32Array::from(flat_vecs);
        let emb_array: ArrayRef = Arc::new(FixedSizeListArray::try_new(
            self.item_field.clone(),
            self.dim,
            Arc::new(values),
            None,
        )?);
        let hash_array: ArrayRef = Arc::new(StringArray::from_iter_values(hashes));

        let batch =
            RecordBatch::try_new(self.schema.clone(), vec![wid_array, emb_array, hash_array])?;
        self.writer
            .as_mut()
            .expect("write_carried after close")
            .write(&batch)?;
        self.rows_written += carried.len();
        Ok(())
    }

    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    /// Flush the parquet footer and atomically rename `.tmp` → final path.
    pub fn close(mut self) -> Result<()> {
        if let Some(w) = self.writer.take() {
            w.close()?;
        }
        std::fs::rename(&self.tmp_path, &self.final_path)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for EmbedWriter {
    fn drop(&mut self) {
        if !self.committed {
            drop(self.writer.take());
            let _ = std::fs::remove_file(&self.tmp_path);
        }
    }
}

/// Recursively collect .parquet files under a directory.
fn walk_parquet_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            walk_parquet_files(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "parquet") {
            out.push(path);
        }
    }
    Ok(())
}
