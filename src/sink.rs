//! Output sinks -- Parquet file writer with atomic tmp->rename.

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

/// Buffered parquet writer with atomic tmp->rename.
///
/// On drop, removes the `.tmp` file if [`finalize`](Self::finalize) was never called.
pub struct ParquetSink {
    writer: Option<ArrowWriter<File>>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    row_count: usize,
}

impl std::fmt::Debug for ParquetSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParquetSink")
            .field("final_path", &self.final_path)
            .field("row_count", &self.row_count)
            .finish_non_exhaustive()
    }
}

impl ParquetSink {
    /// Create a new sink writing to a temporary file.
    pub fn new(
        dataset: &str,
        shard_idx: usize,
        output_dir: &Path,
        schema: &Schema,
        zstd_level: i32,
    ) -> Result<Self, std::io::Error> {
        let filename = format!("{dataset}_{shard_idx:04}.parquet");
        let final_path = output_dir.join(&filename);
        let tmp_path = output_dir.join(format!("{filename}.tmp"));

        if tmp_path.exists() {
            fs::remove_file(&tmp_path)?;
        }

        let file = File::create(&tmp_path)?;
        let level = ZstdLevel::try_new(zstd_level)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(level))
            .set_max_row_group_size(500_000)
            .build();

        let writer = ArrowWriter::try_new(file, Arc::new(schema.clone()), Some(props))
            .map_err(std::io::Error::other)?;

        Ok(Self {
            writer: Some(writer),
            tmp_path,
            final_path,
            row_count: 0,
        })
    }

    /// Write a record batch
    pub fn write_batch(&mut self, batch: &RecordBatch) -> Result<(), std::io::Error> {
        self.row_count += batch.num_rows();
        self.writer
            .as_mut()
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "write after finalize")
            })?
            .write(batch)
            .map_err(std::io::Error::other)
    }

    /// Finalize: flush footer and atomically rename tmp -> final.
    ///
    /// After this call the `.tmp` file is gone and the final parquet file is
    /// in place.  Returns an error if called more than once.
    pub fn finalize(&mut self) -> Result<usize, std::io::Error> {
        let writer = self.writer.take().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "finalize called twice")
        })?;
        // writer.take() already set self.writer = None, so Drop won't clean
        // up the .tmp file. We must handle cleanup on failure ourselves.
        if let Err(e) = writer.close().map_err(std::io::Error::other) {
            let _ = fs::remove_file(&self.tmp_path);
            return Err(e);
        }
        if let Err(e) = fs::rename(&self.tmp_path, &self.final_path) {
            let _ = fs::remove_file(&self.tmp_path);
            return Err(e);
        }
        Ok(self.row_count)
    }
}

impl Drop for ParquetSink {
    fn drop(&mut self) {
        // writer is Some → finalize was never called; clean up the .tmp file.
        if self.writer.is_some() {
            // Drop the writer first so the file handle is released.
            self.writer.take();
            if self.tmp_path.exists() {
                tracing::warn!("Removing unfinalized tmp file: {}", self.tmp_path.display());
                let _ = fs::remove_file(&self.tmp_path);
            }
        }
    }
}

/// Check if a completed parquet file exists and has a valid footer
pub fn is_valid_parquet(path: &Path) -> bool {
    if !path.exists() {
        return false;
    }
    let Ok(file) = File::open(path) else {
        return false;
    };
    parquet::file::reader::SerializedFileReader::new(file).is_ok()
}

/// Remove stale .tmp files in a directory
pub fn cleanup_tmp_files(dir: &Path) -> std::io::Result<usize> {
    let mut count = 0;
    if !dir.exists() {
        return Ok(0);
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "tmp") {
            tracing::warn!("Removing stale tmp file: {}", path.display());
            fs::remove_file(&path)?;
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn is_valid_parquet_missing_file() {
        let dir = TempDir::new().unwrap();
        assert!(!is_valid_parquet(&dir.path().join("nope.parquet")));
    }

    #[test]
    fn is_valid_parquet_not_parquet() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.parquet");
        std::fs::write(&path, b"this is not parquet").unwrap();
        assert!(!is_valid_parquet(&path));
    }

    #[test]
    fn is_valid_parquet_real_file() {
        let dir = TempDir::new().unwrap();
        let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
            "id",
            arrow::datatypes::DataType::Int64,
            false,
        )]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(arrow::array::Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        let path = dir.path().join("valid.parquet");
        let file = File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, Arc::new(schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        assert!(is_valid_parquet(&path));
    }

    #[test]
    fn cleanup_tmp_files_removes_only_tmp() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.tmp"), b"stale").unwrap();
        std::fs::write(dir.path().join("b.parquet"), b"keep").unwrap();
        std::fs::write(dir.path().join("c.tmp"), b"stale2").unwrap();

        let count = cleanup_tmp_files(dir.path()).unwrap();

        assert_eq!(count, 2);
        assert!(!dir.path().join("a.tmp").exists());
        assert!(dir.path().join("b.parquet").exists());
        assert!(!dir.path().join("c.tmp").exists());
    }

    /// Helper: create a minimal test schema + batch
    fn test_schema_and_batch() -> (Schema, RecordBatch) {
        let schema = Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![1, 2, 3])),
                Arc::new(arrow::array::StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();
        (schema, batch)
    }

    #[test]
    fn parquet_sink_roundtrip() {
        let dir = TempDir::new().unwrap();
        let (schema, batch) = test_schema_and_batch();

        let mut sink = ParquetSink::new("test_ds", 0, dir.path(), &schema, 3).unwrap();
        sink.write_batch(&batch).unwrap();
        let rows = sink.finalize().unwrap();

        assert_eq!(rows, 3);
        let final_path = dir.path().join("test_ds_0000.parquet");
        assert!(is_valid_parquet(&final_path));
    }

    #[test]
    fn parquet_sink_tmp_cleanup() {
        let dir = TempDir::new().unwrap();
        let (schema, batch) = test_schema_and_batch();

        let tmp_path = dir.path().join("test_ds_0001.parquet.tmp");
        let final_path = dir.path().join("test_ds_0001.parquet");

        let mut sink = ParquetSink::new("test_ds", 1, dir.path(), &schema, 3).unwrap();
        // Before finalize: tmp exists, final doesn't
        assert!(tmp_path.exists());
        assert!(!final_path.exists());

        sink.write_batch(&batch).unwrap();
        sink.finalize().unwrap();

        // After finalize: tmp gone, final exists
        assert!(!tmp_path.exists());
        assert!(final_path.exists());
    }

    #[test]
    fn parquet_sink_overwrites_stale_tmp() {
        let dir = TempDir::new().unwrap();
        let (schema, batch) = test_schema_and_batch();

        let tmp_path = dir.path().join("test_ds_0002.parquet.tmp");
        // Pre-existing stale .tmp file
        std::fs::write(&tmp_path, b"stale data").unwrap();

        let mut sink = ParquetSink::new("test_ds", 2, dir.path(), &schema, 3).unwrap();
        sink.write_batch(&batch).unwrap();
        let rows = sink.finalize().unwrap();

        assert_eq!(rows, 3);
        assert!(is_valid_parquet(&dir.path().join("test_ds_0002.parquet")));
    }
}
