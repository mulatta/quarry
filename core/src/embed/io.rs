//! Parquet I/O for the embedding pipeline.
//!
//! Streaming design: `process_hive_chunks` reads hive parquet files in
//! bounded chunks and calls a callback per chunk, so the caller can
//! encode + write incrementally without holding the full dataset in memory.

use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{Array, ArrayRef, AsArray, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

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

/// Stream hive parquet files in chunks, calling `on_chunk` for each chunk.
///
/// Each chunk contains `(work_ids, texts)` with up to `chunk_size` rows.
/// Rows with empty title and abstract are skipped.
/// Returns total rows processed.
pub fn process_hive_chunks(
    hive_dir: &Path,
    chunk_size: usize,
    max_rows: Option<usize>,
    mut on_chunk: impl FnMut(&[String], &[String]) -> Result<()>,
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

            for row in 0..batch.num_rows() {
                if total >= limit {
                    break 'outer;
                }

                let t = title_col
                    .and_then(|c| {
                        if c.is_null(row) {
                            None
                        } else {
                            Some(c.value(row))
                        }
                    })
                    .unwrap_or("")
                    .trim();
                let a = abstract_col
                    .and_then(|c| {
                        if c.is_null(row) {
                            None
                        } else {
                            Some(c.value(row))
                        }
                    })
                    .unwrap_or("")
                    .trim();

                let text = match (t.is_empty(), a.is_empty()) {
                    (false, false) => format!("{t}\n{a}"),
                    (false, true) => t.to_string(),
                    (true, false) => a.to_string(),
                    (true, true) => continue,
                };

                buf_ids.push(wid_col.value(row).to_string());
                buf_texts.push(text);
                total += 1;

                if buf_ids.len() >= chunk_size {
                    on_chunk(&buf_ids, &buf_texts)?;
                    buf_ids.clear();
                    buf_texts.clear();
                }
            }
        }
    }

    // Flush remaining
    if !buf_ids.is_empty() {
        on_chunk(&buf_ids, &buf_texts)?;
    }

    Ok(total)
}

/// Incremental parquet writer for embedding output.
///
/// Each `write_chunk` call adds a row group, keeping memory bounded.
pub struct EmbedWriter {
    writer: ArrowWriter<BufWriter<std::fs::File>>,
    schema: Arc<Schema>,
    item_field: Arc<Field>,
    dim: i32,
    rows_written: usize,
}

impl EmbedWriter {
    pub fn new(path: &Path, dim: usize) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let item_field = Arc::new(Field::new("item", DataType::Float32, false));
        let schema = Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, false),
            Field::new(
                "embedding",
                DataType::FixedSizeList(item_field.clone(), dim as i32),
                false,
            ),
        ]));

        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
            .build();

        let file = BufWriter::new(std::fs::File::create(path)?);
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

        Ok(Self {
            writer,
            schema,
            item_field,
            dim: dim as i32,
            rows_written: 0,
        })
    }

    pub fn write_chunk(&mut self, work_ids: &[String], embeddings: Vec<f32>) -> Result<()> {
        let wid_array: ArrayRef = Arc::new(StringArray::from_iter_values(work_ids));
        let values = Float32Array::from(embeddings);
        let emb_array: ArrayRef = Arc::new(FixedSizeListArray::try_new(
            self.item_field.clone(),
            self.dim,
            Arc::new(values),
            None,
        )?);

        let batch = RecordBatch::try_new(self.schema.clone(), vec![wid_array, emb_array])?;
        self.writer.write(&batch)?;
        self.rows_written += work_ids.len();
        Ok(())
    }

    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    pub fn close(self) -> Result<()> {
        self.writer.close()?;
        Ok(())
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
