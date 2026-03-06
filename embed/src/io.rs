//! Parquet I/O for the embedding pipeline.

use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{Array, AsArray};
use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

/// Read work_id and text (title + abstract) from hive works parquet files.
pub fn read_inputs(hive_dir: &Path, max_rows: Option<usize>) -> Result<(Vec<String>, Vec<String>)> {
    let mut parquet_files = Vec::new();
    walk_parquet_files(hive_dir, &mut parquet_files)?;
    parquet_files.sort();
    anyhow::ensure!(
        !parquet_files.is_empty(),
        "No parquet files in {}",
        hive_dir.display()
    );

    let bar = ProgressBar::new(parquet_files.len() as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{prefix:.bold} {bar:30.cyan/black.dim} {pos}/{len} files [{elapsed}]")
            .expect("valid template")
            .progress_chars("=>-"),
    );
    bar.set_prefix("Reading");

    let limit = max_rows.unwrap_or(usize::MAX);
    let mut work_ids = Vec::new();
    let mut texts = Vec::new();

    'outer: for path in &parquet_files {
        let file = std::fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema().clone();

        let wid_root = schema.index_of("work_id").ok();
        let title_root = schema.index_of("title").ok();
        let abstract_root = schema.index_of("abstract_text").ok();

        let Some(wid_root) = wid_root else {
            bar.inc(1);
            continue;
        };

        // Build projection mask for needed columns only
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
            .with_batch_size(65_536)
            .build()?;

        for batch_result in reader {
            let batch = batch_result?;
            let wid_col = batch.column(0).as_string::<i32>();
            let title_col = title_local.map(|i| batch.column(i).as_string::<i32>());
            let abstract_col = abstract_local.map(|i| batch.column(i).as_string::<i32>());

            for row in 0..batch.num_rows() {
                if work_ids.len() >= limit {
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
                    (true, true) => continue, // skip empty
                };

                work_ids.push(wid_col.value(row).to_string());
                texts.push(text);
            }
        }

        bar.inc(1);
    }

    bar.finish_and_clear();
    tracing::info!(
        "{} works loaded from {} files",
        work_ids.len(),
        parquet_files.len()
    );
    Ok((work_ids, texts))
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

/// Write `{work_id, embedding}` to a parquet file.
pub fn write_output(
    out_path: &Path,
    work_ids: &[String],
    embeddings: Vec<f32>,
    dim: usize,
) -> Result<()> {
    if let Some(parent) = out_path.parent() {
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

    let wid_array: ArrayRef = Arc::new(StringArray::from_iter_values(work_ids));
    let values = Float32Array::from(embeddings);
    let emb_array: ArrayRef = Arc::new(FixedSizeListArray::try_new(
        item_field,
        dim as i32,
        Arc::new(values),
        None,
    )?);

    let batch = RecordBatch::try_new(schema.clone(), vec![wid_array, emb_array])?;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();

    let file = BufWriter::new(std::fs::File::create(out_path)?);
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    let size = out_path.metadata().map(|m| m.len()).unwrap_or(0);
    tracing::info!(
        "Written {} ({} rows, dim={dim}, {:.1} MiB)",
        out_path.display(),
        work_ids.len(),
        size as f64 / 1024.0 / 1024.0,
    );
    Ok(())
}
