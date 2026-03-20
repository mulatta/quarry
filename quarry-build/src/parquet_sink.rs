//! Generic batched Parquet writer.
//!
//! Accumulates RecordBatches and flushes to numbered Parquet files
//! when a row threshold is reached. Handles batch splitting for
//! oversized batches.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

/// Batched Parquet writer that splits output into multiple files.
pub struct ParquetSink {
    output_dir: PathBuf,
    prefix: String,
    schema: Arc<Schema>,
    props: WriterProperties,
    /// Max rows per output file.
    max_rows_per_file: usize,
    /// Current file index.
    file_index: usize,
    /// Current writer (lazily created).
    writer: Option<ArrowWriter<File>>,
    /// Rows written to current file.
    current_rows: usize,
    /// Total rows written across all files.
    total_rows: usize,
}

impl ParquetSink {
    pub fn new(
        output_dir: &Path,
        prefix: &str,
        schema: Arc<Schema>,
        row_group_size: usize,
        max_rows_per_file: usize,
    ) -> Self {
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
            .set_max_row_group_size(row_group_size)
            .build();

        Self {
            output_dir: output_dir.to_path_buf(),
            prefix: prefix.to_string(),
            schema,
            props,
            max_rows_per_file,
            file_index: 0,
            writer: None,
            current_rows: 0,
            total_rows: 0,
        }
    }

    /// Write a RecordBatch, splitting across files as needed.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), Box<dyn std::error::Error>> {
        if batch.num_rows() == 0 {
            return Ok(());
        }

        let mut offset = 0;
        while offset < batch.num_rows() {
            // How many rows can fit in the current file?
            let remaining_in_file = self.max_rows_per_file.saturating_sub(self.current_rows);
            let remaining_in_batch = batch.num_rows() - offset;

            // If current file is full, rotate
            if remaining_in_file == 0 {
                self.flush_current()?;
                continue;
            }

            let take = remaining_in_batch.min(remaining_in_file);
            let slice = batch.slice(offset, take);

            let writer = self.ensure_writer()?;
            writer.write(&slice)?;
            self.current_rows += take;
            self.total_rows += take;
            offset += take;

            if self.current_rows >= self.max_rows_per_file {
                self.flush_current()?;
            }
        }

        Ok(())
    }

    /// Flush current file and finalize all output.
    pub fn finish(mut self) -> Result<SinkStats, Box<dyn std::error::Error>> {
        self.flush_current()?;
        // Prevent Drop from double-closing
        self.writer = None;
        Ok(SinkStats {
            total_rows: self.total_rows,
            num_files: self.file_index,
        })
    }

    fn ensure_writer(&mut self) -> Result<&mut ArrowWriter<File>, Box<dyn std::error::Error>> {
        if self.writer.is_none() {
            self.writer = Some(self.new_writer()?);
        }
        Ok(self.writer.as_mut().unwrap())
    }

    fn flush_current(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(writer) = self.writer.take() {
            writer.close()?;
            self.file_index += 1;
            self.current_rows = 0;
        }
        Ok(())
    }

    fn new_writer(&self) -> Result<ArrowWriter<File>, Box<dyn std::error::Error>> {
        let path = self
            .output_dir
            .join(format!("{}_{:05}.parquet", self.prefix, self.file_index));
        let file = File::create(&path)?;
        let writer = ArrowWriter::try_new(file, self.schema.clone(), Some(self.props.clone()))?;
        Ok(writer)
    }
}

/// Best-effort flush on drop (e.g. when early `?` return skips `finish()`).
impl Drop for ParquetSink {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            let _ = writer.close();
        }
    }
}

/// Stats returned after sink is finished.
pub struct SinkStats {
    pub total_rows: usize,
    pub num_files: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field};
    use tempfile::TempDir;

    fn test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn make_batch(schema: &Arc<Schema>, ids: &[i32], names: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names.to_vec())),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_single_file() {
        let dir = TempDir::new().unwrap();
        let schema = test_schema();
        let mut sink = ParquetSink::new(dir.path(), "test", schema.clone(), 1024, 500_000);

        let batch = make_batch(&schema, &[1, 2, 3], &["a", "b", "c"]);
        sink.write_batch(&batch).unwrap();

        let stats = sink.finish().unwrap();
        assert_eq!(stats.total_rows, 3);
        assert_eq!(stats.num_files, 1);
    }

    #[test]
    fn test_file_rotation() {
        let dir = TempDir::new().unwrap();
        let schema = test_schema();
        // max 2 rows per file
        let mut sink = ParquetSink::new(dir.path(), "rot", schema.clone(), 1024, 2);

        let batch1 = make_batch(&schema, &[1, 2], &["a", "b"]);
        let batch2 = make_batch(&schema, &[3, 4], &["c", "d"]);
        let batch3 = make_batch(&schema, &[5], &["e"]);

        sink.write_batch(&batch1).unwrap();
        sink.write_batch(&batch2).unwrap();
        sink.write_batch(&batch3).unwrap();

        let stats = sink.finish().unwrap();
        assert_eq!(stats.total_rows, 5);
        assert_eq!(stats.num_files, 3);
    }

    #[test]
    fn test_oversized_batch_split() {
        let dir = TempDir::new().unwrap();
        let schema = test_schema();
        // max 2 rows per file, but write a batch of 5
        let mut sink = ParquetSink::new(dir.path(), "split", schema.clone(), 1024, 2);

        let batch = make_batch(&schema, &[1, 2, 3, 4, 5], &["a", "b", "c", "d", "e"]);
        sink.write_batch(&batch).unwrap();

        let stats = sink.finish().unwrap();
        assert_eq!(stats.total_rows, 5);
        assert_eq!(stats.num_files, 3); // 2+2+1
    }

    #[test]
    fn test_empty_batch() {
        let dir = TempDir::new().unwrap();
        let schema = test_schema();
        let mut sink = ParquetSink::new(dir.path(), "empty", schema.clone(), 1024, 500_000);

        let batch = make_batch(&schema, &[], &[]);
        sink.write_batch(&batch).unwrap();

        let stats = sink.finish().unwrap();
        assert_eq!(stats.total_rows, 0);
        assert_eq!(stats.num_files, 0);
    }
}
