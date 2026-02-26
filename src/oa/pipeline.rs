//! Shard processing pipeline -- download, parse, fan-out, write.
//!
//! Internal to the `oa` module. Only [`process_works_shard`] is exposed.

use std::io::BufRead;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow::array::RecordBatch;
use indicatif::ProgressBar;

use crate::accumulator::Accumulator;
use crate::error::ShardError;
use crate::progress::fmt_num;
use crate::provider::{RunContext, ShardStats};
use crate::sink::ParquetSink;
use crate::stream::ByteCounter;

use crate::schema;
use crate::transform::{
    AuthorshipsAccumulator, AwardsAccumulator, CitationsAccumulator, CountsByYearAccumulator,
    Filter, FundersAccumulator, KeywordsAccumulator, LocationsAccumulator, MeshAccumulator,
    SdgsAccumulator, TopicsAccumulator, WorkRow, WorksAccumulator, WorksKeysAccumulator,
    strip_oa_prefix,
};

use super::OAShard;

// ============================================================
// Fan-out accumulator trait (object-safe for dynamic dispatch)
// ============================================================

/// Unified interface for the 11 fan-out accumulators.
///
/// Each accumulator extracts a subset of fields from `WorkRow` and produces
/// `RecordBatch`es for its corresponding parquet table. The trait is
/// object-safe so that [`ShardPipeline`] can iterate over them in a loop
/// instead of enumerating each one manually.
trait FanoutAccumulator {
    fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow);
    fn fanout_len(&self) -> usize;
    fn fanout_is_full(&self) -> bool;
    fn fanout_take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError>;
}

macro_rules! impl_fanout {
    ($ty:ty) => {
        impl FanoutAccumulator for $ty {
            fn push_from_work(&mut self, work_id: &Arc<str>, row: &WorkRow) {
                self.push_from_work(work_id, row);
            }
            fn fanout_len(&self) -> usize {
                self.len()
            }
            fn fanout_is_full(&self) -> bool {
                self.is_full()
            }
            fn fanout_take_batch(&mut self) -> Result<RecordBatch, arrow::error::ArrowError> {
                self.take_batch()
            }
        }
    };
}

impl_fanout!(WorksKeysAccumulator);
impl_fanout!(CitationsAccumulator);
impl_fanout!(AuthorshipsAccumulator);
impl_fanout!(TopicsAccumulator);
impl_fanout!(KeywordsAccumulator);
impl_fanout!(MeshAccumulator);
impl_fanout!(LocationsAccumulator);
impl_fanout!(FundersAccumulator);
impl_fanout!(AwardsAccumulator);
impl_fanout!(SdgsAccumulator);
impl_fanout!(CountsByYearAccumulator);

// ============================================================
// ShardPipeline — unified sinks + accumulators
// ============================================================

/// All 12 sinks and accumulators for a single shard, paired for loop access.
///
/// The `works` table uses the `Accumulator` trait (push consumes `WorkRow`),
/// while the 11 fan-out tables use `FanoutAccumulator` (push borrows `&WorkRow`).
/// Fan-out push runs first, then works push consumes the row.
struct ShardPipeline {
    works_acc: WorksAccumulator,
    works_sink: ParquetSink,
    fanouts: Vec<(Box<dyn FanoutAccumulator>, ParquetSink)>,
}

impl ShardPipeline {
    fn create(
        shard_idx: usize,
        ctx: &RunContext,
        shard_updated_date: Option<String>,
        oa_shard_idx: u16,
    ) -> Result<Self, ShardError> {
        let z = ctx.zstd_level;
        let mk =
            |name: &str, schema: &arrow::datatypes::Schema| -> Result<ParquetSink, ShardError> {
                let dir = ctx.output_dir.join(name);
                std::fs::create_dir_all(&dir).map_err(ShardError::Io)?;
                ParquetSink::new("shard", shard_idx, &dir, schema, z).map_err(ShardError::Io)
            };

        // Adding a new table: add one entry here + define its accumulator.
        let fanouts: Vec<(Box<dyn FanoutAccumulator>, ParquetSink)> = vec![
            (
                Box::new(WorksKeysAccumulator::new()),
                mk("works_keys", schema::works_keys())?,
            ),
            (
                Box::new(CitationsAccumulator::new()),
                mk("citations", schema::citations())?,
            ),
            (
                Box::new(AuthorshipsAccumulator::new()),
                mk("work_authorships", schema::work_authorships())?,
            ),
            (
                Box::new(TopicsAccumulator::new()),
                mk("work_topics", schema::work_topics())?,
            ),
            (
                Box::new(KeywordsAccumulator::new()),
                mk("work_keywords", schema::work_keywords())?,
            ),
            (
                Box::new(MeshAccumulator::new()),
                mk("work_mesh", schema::work_mesh())?,
            ),
            (
                Box::new(LocationsAccumulator::new()),
                mk("work_locations", schema::work_locations())?,
            ),
            (
                Box::new(FundersAccumulator::new()),
                mk("work_funders", schema::work_funders())?,
            ),
            (
                Box::new(AwardsAccumulator::new()),
                mk("work_awards", schema::work_awards())?,
            ),
            (
                Box::new(SdgsAccumulator::new()),
                mk("work_sdgs", schema::work_sdgs())?,
            ),
            (
                Box::new(CountsByYearAccumulator::new()),
                mk("work_counts_by_year", schema::work_counts_by_year())?,
            ),
        ];

        Ok(Self {
            works_acc: WorksAccumulator::with_shard_metadata(shard_updated_date, oa_shard_idx),
            works_sink: mk("works", schema::works())?,
            fanouts,
        })
    }

    /// Push a parsed work row through all 12 accumulators, flushing when full.
    fn push_work(&mut self, work_id: &Arc<str>, row: Box<WorkRow>) -> std::io::Result<()> {
        // Fan-out accumulators borrow the row first
        for (acc, _) in &mut self.fanouts {
            acc.push_from_work(work_id, &row);
        }
        // Works accumulator consumes the row (computes content_hash)
        self.works_acc.push(*row);

        // Flush works if full
        if self.works_acc.is_full() {
            let batch = self.works_acc.take_batch().map_err(std::io::Error::other)?;
            self.works_sink.write_batch(&batch)?;
        }
        // Flush fan-outs if full
        for (acc, sink) in &mut self.fanouts {
            if acc.fanout_is_full() {
                let batch = acc.fanout_take_batch().map_err(std::io::Error::other)?;
                sink.write_batch(&batch)?;
            }
        }
        Ok(())
    }

    /// Flush any remaining buffered rows from all accumulators.
    fn flush_remaining(&mut self) -> std::io::Result<()> {
        if self.works_acc.len() > 0 {
            let batch = self.works_acc.take_batch().map_err(std::io::Error::other)?;
            self.works_sink.write_batch(&batch)?;
        }
        for (acc, sink) in &mut self.fanouts {
            if acc.fanout_len() > 0 {
                let batch = acc.fanout_take_batch().map_err(std::io::Error::other)?;
                sink.write_batch(&batch)?;
            }
        }
        Ok(())
    }

    /// Finalize all parquet files (write footer, atomic rename).
    fn finalize(&mut self) -> Result<(), ShardError> {
        self.works_sink.finalize().map_err(ShardError::Io)?;
        for (_, sink) in &mut self.fanouts {
            sink.finalize().map_err(ShardError::Io)?;
        }
        Ok(())
    }
}

// ============================================================
// Shard processor
// ============================================================

pub(super) fn process_works_shard(
    shard: &OAShard,
    ctx: &RunContext,
    filter: &Filter,
    pb: &ProgressBar,
) -> Result<ShardStats, ShardError> {
    let shard_label = format!("works_{:04}", shard.shard_idx);

    crate::retry::retry_with_backoff(&shard_label, pb, || {
        let t0 = std::time::Instant::now();

        let (mut reader, counter, total_bytes) =
            crate::stream::open_gzip_reader(&shard.url).map_err(ShardError::Stream)?;

        if let Some(total) = total_bytes.or(shard.content_length) {
            crate::progress::upgrade_to_bar(pb, total);
        }
        pb.set_message("processing...");

        let t_connect = t0.elapsed();

        let mut pipeline = ShardPipeline::create(
            shard.shard_idx,
            ctx,
            shard.updated_date.clone(),
            u16::try_from(shard.shard_idx).expect("shard_idx overflow"),
        )?;

        let result = process_works_lines(&mut reader, &counter, &mut pipeline, filter, pb)
            .map_err(ShardError::Io)?;

        if result.parse_failures > 0 {
            let total = result.rows_written + result.parse_failures;
            let pct = result.parse_failures as f64 / total as f64 * 100.0;
            tracing::warn!(
                "{shard_label}: {}/{total} lines failed to parse ({pct:.1}%)",
                result.parse_failures,
            );
        }
        if result.null_institution_ids > 0 {
            tracing::warn!(
                "{shard_label}: {} rows had null elements in affiliation.institution_ids (preserved with nulls removed)",
                result.null_institution_ids,
            );
        }

        let t_process = t0.elapsed() - t_connect;

        pipeline.finalize()?;

        let t_total = t0.elapsed();
        let t_finalize = t_total - t_connect - t_process;
        let rows = result.rows_written;
        let rows_per_sec = if t_process.as_secs_f64() > 0.0 {
            rows as f64 / t_process.as_secs_f64()
        } else {
            0.0
        };

        tracing::debug!(
            "{shard_label}: connect={:.1}s process={:.1}s ({:.0} rows/s) finalize={:.1}s total={:.1}s rows={}",
            t_connect.as_secs_f64(),
            t_process.as_secs_f64(),
            rows_per_sec,
            t_finalize.as_secs_f64(),
            t_total.as_secs_f64(),
            rows,
        );

        Ok(ShardStats { rows_written: rows })
    })
}

const UPDATE_INTERVAL: usize = 10_000;

/// Result of processing lines within a shard.
struct LinesResult {
    rows_written: usize,
    parse_failures: usize,
    /// Rows where `affiliation.institution_ids` contained null elements.
    null_institution_ids: usize,
}

fn process_works_lines(
    reader: &mut impl BufRead,
    counter: &ByteCounter,
    pipeline: &mut ShardPipeline,
    filter: &Filter,
    pb: &ProgressBar,
) -> std::io::Result<LinesResult> {
    let mut buf = String::with_capacity(4096);
    let mut rows_written = 0usize;
    let mut lines_scanned = 0usize;
    let mut parse_failures = 0usize;
    let mut null_institution_ids = 0usize;

    loop {
        buf.clear();
        if reader.read_line(&mut buf)? == 0 {
            break;
        }
        lines_scanned += 1;

        if lines_scanned.is_multiple_of(UPDATE_INTERVAL) {
            pb.set_position(counter.load(Ordering::Relaxed));
            if filter.is_empty() {
                pb.set_message(format!("{} rows", fmt_num(rows_written)));
            } else {
                let match_pct = rows_written as f64 / lines_scanned as f64 * 100.0;
                pb.set_message(format!(
                    "{} rows ({:.1}% match)",
                    fmt_num(rows_written),
                    match_pct
                ));
            }
        }

        let trimmed = buf.trim();
        if trimmed.is_empty() {
            continue;
        }
        match parse_work(trimmed, filter) {
            ParseResult::Matched(row) => {
                if row.authorships.iter().any(|a| {
                    a.affiliations
                        .iter()
                        .any(|af| af.institution_ids.iter().any(|id| id.is_none()))
                }) {
                    null_institution_ids += 1;
                }
                let work_id: Arc<str> = Arc::from(strip_oa_prefix(row.id.as_deref().unwrap_or("")));
                pipeline.push_work(&work_id, row)?;
                rows_written += 1;
            }
            ParseResult::Filtered => {}
            ParseResult::ParseError => {
                parse_failures += 1;
            }
        }
    }

    pipeline.flush_remaining()?;

    Ok(LinesResult {
        rows_written,
        parse_failures,
        null_institution_ids,
    })
}

// ============================================================
// Parsing + filtering
// ============================================================

#[derive(Debug)]
enum ParseResult {
    Matched(Box<WorkRow>),
    Filtered,
    ParseError,
}

fn parse_work(line: &str, filter: &Filter) -> ParseResult {
    match sonic_rs::from_str::<WorkRow>(line) {
        Ok(row) => {
            if filter.matches(&row) {
                ParseResult::Matched(Box::new(row))
            } else {
                ParseResult::Filtered
            }
        }
        Err(e) => {
            tracing::debug!(
                "parse error: {e} | line[..200]: {}",
                &line[..line.floor_char_boundary(200)]
            );
            ParseResult::ParseError
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::atomic::AtomicU64;

    use arrow::array::{Array, Int32Array, StringArray};
    use arrow::record_batch::RecordBatch;

    use super::super::TABLES;

    // ============================================================
    // Rich fixture — all major fields populated
    // ============================================================

    const RICH_WORK_JSON: &str = r#"{"id":"https://openalex.org/W2741809807","doi":"https://doi.org/10.1038/s41586-020-2649-2","display_name":"Array programming with NumPy","title":"Array programming with NumPy","publication_year":2020,"publication_date":"2020-09-17","language":"en","type":"journal-article","cited_by_count":5000,"is_retracted":false,"is_paratext":false,"has_content":{"pdf":true,"grobid_xml":false},"ids":{"openalex":"https://openalex.org/W2741809807","doi":"https://doi.org/10.1038/s41586-020-2649-2","mag":2741809807,"pmid":"https://pubmed.ncbi.nlm.nih.gov/32939066","pmcid":"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7759461"},"referenced_works":["https://openalex.org/W100","https://openalex.org/W200"],"counts_by_year":[{"year":2023,"cited_by_count":1500},{"year":2022,"cited_by_count":1200}],"keywords":[{"id":"K1","display_name":"NumPy","score":0.95}],"mesh":[{"descriptor_ui":"D000123","descriptor_name":"Software","qualifier_ui":"Q000456","qualifier_name":"standards","is_major_topic":true}],"topics":[{"id":"https://openalex.org/T10101","display_name":"Scientific Computing","score":0.99,"subfield":{"id":"SF1","display_name":"Computational Science"},"field":{"id":"F1","display_name":"Computer Science"},"domain":{"id":"D1","display_name":"Physical Sciences"}}],"authorships":[{"author_position":"first","author":{"id":"https://openalex.org/A1","display_name":"Charles Harris","orcid":"0000-0001-0002-0003"},"institutions":[{"id":"https://openalex.org/I1","display_name":"MIT","ror":"https://ror.org/012345","country_code":"US","type":"education"}],"countries":["US"],"is_corresponding":true,"raw_affiliation_strings":["MIT, Cambridge, MA"]}],"locations":[{"is_oa":true,"landing_page_url":"https://www.nature.com/articles/s41586-020-2649-2","pdf_url":"https://www.nature.com/articles/s41586-020-2649-2.pdf","license":"cc-by","version":"publishedVersion","source":{"id":"https://openalex.org/S1","display_name":"Nature","issn_l":"0028-0836","issn":["0028-0836","1476-4687"],"host_organization":"https://openalex.org/P1","type":"journal"}}],"sustainable_development_goals":[{"id":"SDG9","display_name":"Industry and Innovation","score":0.8}],"funders":[{"id":"https://openalex.org/F1","display_name":"NSF"}],"awards":[{"id":"https://openalex.org/G1234567","funder_id":"https://openalex.org/F1","funder_display_name":"NSF","funder_award_id":"1234567"}],"open_access":{"is_oa":true,"oa_status":"gold","oa_url":"https://example.com/oa"},"abstract_inverted_index":{"Array":[0],"programming":[1],"with":[2],"NumPy":[3]}}"#;

    // ============================================================
    // Test helpers
    // ============================================================

    fn read_parquet(path: &Path) -> RecordBatch {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let file = std::fs::File::open(path).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap()
    }

    fn assert_str_col(batch: &RecordBatch, col: &str, expected: &[&str]) {
        let arr = batch
            .column_by_name(col)
            .unwrap_or_else(|| panic!("column '{col}' not found"))
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap_or_else(|| panic!("column '{col}' is not StringArray"));
        let actual: Vec<&str> = (0..arr.len()).map(|i| arr.value(i)).collect();
        assert_eq!(actual, expected, "column '{col}' mismatch");
    }

    fn assert_i32_col(batch: &RecordBatch, col: &str, expected: &[i32]) {
        let arr = batch
            .column_by_name(col)
            .unwrap_or_else(|| panic!("column '{col}' not found"))
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap_or_else(|| panic!("column '{col}' is not Int32Array"));
        let actual: Vec<i32> = (0..arr.len()).map(|i| arr.value(i)).collect();
        assert_eq!(actual, expected, "column '{col}' mismatch");
    }

    // ============================================================
    // End-to-end: JSON → process_works_lines → Parquet → read back
    // ============================================================

    #[test]
    fn end_to_end_json_to_parquet() {
        let dir = tempfile::TempDir::new().unwrap();
        let ctx = crate::provider::RunContext {
            output_dir: dir.path().to_path_buf(),
            zstd_level: 3,
            concurrency: 1,
        };

        let mut pipeline =
            ShardPipeline::create(0, &ctx, Some("2024-01-15".to_string()), 0u16).unwrap();

        let input = format!("{}\n", RICH_WORK_JSON);
        let mut reader = std::io::BufReader::new(input.as_bytes());
        let counter: ByteCounter = Arc::new(AtomicU64::new(0));
        let pb = ProgressBar::hidden();
        let filter = Filter::default();

        let result =
            process_works_lines(&mut reader, &counter, &mut pipeline, &filter, &pb).unwrap();
        assert_eq!(result.rows_written, 1);
        assert_eq!(result.parse_failures, 0);

        pipeline.finalize().unwrap();

        for table in TABLES {
            assert!(
                dir.path().join(table).join("shard_0000.parquet").exists(),
                "missing table: {table}"
            );
        }

        // Verify works table
        let works = read_parquet(&dir.path().join("works/shard_0000.parquet"));
        assert_eq!(works.num_rows(), 1);
        assert_str_col(&works, "work_id", &["W2741809807"]);
        assert_str_col(
            &works,
            "doi",
            &["https://doi.org/10.1038/s41586-020-2649-2"],
        );
        assert_str_col(&works, "display_name", &["Array programming with NumPy"]);
        assert_str_col(&works, "language", &["en"]);
        assert_i32_col(&works, "publication_year", &[2020]);
        assert_i32_col(&works, "cited_by_count", &[5000]);
        assert_str_col(&works, "abstract_text", &["Array programming with NumPy"]);
        assert_str_col(&works, "ids_pmid", &["32939066"]);
        assert_str_col(&works, "ids_pmcid", &["PMC7759461"]);
        assert_str_col(&works, "ids_mag", &["2741809807"]);
        assert_str_col(&works, "shard_updated_date", &["2024-01-15"]);

        // oa_shard_idx
        let idx_col = works
            .column_by_name("oa_shard_idx")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::UInt16Array>()
            .unwrap();
        assert_eq!(idx_col.value(0), 0);

        // content_hash = blake3(title + \0 + abstract)
        let expected_hash = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"Array programming with NumPy");
            hasher.update(b"\0");
            hasher.update(b"Array programming with NumPy");
            hasher.finalize().to_hex().to_string()
        };
        assert_str_col(&works, "content_hash", &[&expected_hash]);

        // Verify works_keys table
        let keys = read_parquet(&dir.path().join("works_keys/shard_0000.parquet"));
        assert_eq!(keys.num_rows(), 1);
        assert_str_col(&keys, "work_id", &["W2741809807"]);
        assert_str_col(&keys, "doi_norm", &["10.1038/s41586-020-2649-2"]);
        assert_str_col(&keys, "pmid", &["32939066"]);
        assert_str_col(&keys, "pmcid", &["PMC7759461"]);

        // Citations: 2 referenced_works → 2 rows
        let cit = read_parquet(&dir.path().join("citations/shard_0000.parquet"));
        assert_eq!(cit.num_rows(), 2);
        assert_str_col(&cit, "referenced_work_id", &["W100", "W200"]);

        // Counts by year: 2 entries → 2 rows
        let cby = read_parquet(&dir.path().join("work_counts_by_year/shard_0000.parquet"));
        assert_eq!(cby.num_rows(), 2);
        assert_i32_col(&cby, "year", &[2023, 2022]);
        assert_i32_col(&cby, "cited_by_count", &[1500, 1200]);

        // Authorships: 1 entry
        let auth = read_parquet(&dir.path().join("work_authorships/shard_0000.parquet"));
        assert_eq!(auth.num_rows(), 1);
        assert_str_col(&auth, "author_position", &["first"]);
        assert_str_col(&auth, "author_display_name", &["Charles Harris"]);

        // Topics: 1 entry
        let topics = read_parquet(&dir.path().join("work_topics/shard_0000.parquet"));
        assert_eq!(topics.num_rows(), 1);
        assert_str_col(&topics, "display_name", &["Scientific Computing"]);

        // Locations: 1 entry
        let locs = read_parquet(&dir.path().join("work_locations/shard_0000.parquet"));
        assert_eq!(locs.num_rows(), 1);
        assert_str_col(&locs, "source_display_name", &["Nature"]);

        // Keywords: 1 entry
        let kw = read_parquet(&dir.path().join("work_keywords/shard_0000.parquet"));
        assert_eq!(kw.num_rows(), 1);
        assert_str_col(&kw, "display_name", &["NumPy"]);

        // Mesh: 1 entry
        let mesh = read_parquet(&dir.path().join("work_mesh/shard_0000.parquet"));
        assert_eq!(mesh.num_rows(), 1);
        assert_str_col(&mesh, "descriptor_ui", &["D000123"]);

        // SDGs: 1 entry
        let sdgs = read_parquet(&dir.path().join("work_sdgs/shard_0000.parquet"));
        assert_eq!(sdgs.num_rows(), 1);
        assert_str_col(&sdgs, "display_name", &["Industry and Innovation"]);

        // Funders: 1 entry
        let funders = read_parquet(&dir.path().join("work_funders/shard_0000.parquet"));
        assert_eq!(funders.num_rows(), 1);
        assert_str_col(&funders, "display_name", &["NSF"]);

        // Awards: 1 award entry
        let awards = read_parquet(&dir.path().join("work_awards/shard_0000.parquet"));
        assert_eq!(awards.num_rows(), 1);
        assert_str_col(&awards, "id", &["https://openalex.org/G1234567"]);
        assert_str_col(&awards, "funder_award_id", &["1234567"]);
        assert_str_col(&awards, "funder_display_name", &["NSF"]);
    }

    #[test]
    fn parse_work_no_filter() {
        let line = r#"{"id":"https://openalex.org/W1","doi":"https://doi.org/10.1234/test"}"#;
        let filter = Filter::default();
        assert!(matches!(parse_work(line, &filter), ParseResult::Matched(_)));
    }

    #[test]
    fn parse_work_invalid_json() {
        assert!(matches!(
            parse_work("not json", &Filter::default()),
            ParseResult::ParseError
        ));
    }

    #[test]
    fn filter_empty_matches_all() {
        let filter = Filter::default();
        let line = r#"{"id":"https://openalex.org/W1"}"#;
        assert!(matches!(parse_work(line, &filter), ParseResult::Matched(_)));
    }

    #[test]
    fn filter_domain_match() {
        let mut filter = Filter::default();
        filter.domains.insert("Health Sciences".to_string());
        let line = r#"{"id":"https://openalex.org/W1","topics":[{"id":"https://openalex.org/T123","domain":{"display_name":"Health Sciences"}}]}"#;
        assert!(matches!(parse_work(line, &filter), ParseResult::Matched(_)));
    }

    #[test]
    fn filter_domain_no_match() {
        let mut filter = Filter::default();
        filter.domains.insert("Health Sciences".to_string());
        let line = r#"{"id":"https://openalex.org/W1","topics":[{"id":"https://openalex.org/T123","domain":{"display_name":"Physical Sciences"}}]}"#;
        assert!(matches!(parse_work(line, &filter), ParseResult::Filtered));
    }

    #[test]
    fn filter_topic_id_match() {
        let mut filter = Filter::default();
        filter.topic_ids.insert("T11162".to_string());
        let line =
            r#"{"id":"https://openalex.org/W1","topics":[{"id":"https://openalex.org/T11162"}]}"#;
        assert!(matches!(parse_work(line, &filter), ParseResult::Matched(_)));
    }

    /// Rows with `institution_ids: [null]` should parse successfully (not be rejected).
    #[test]
    fn null_institution_ids_preserves_row() {
        let line = r#"{"id":"https://openalex.org/W1","authorships":[{"author_position":"first","author":{"id":"https://openalex.org/A1","display_name":"Alice"},"institutions":[],"affiliations":[{"raw_affiliation_string":"MIT","institution_ids":[null]}]}]}"#;
        let filter = Filter::default();
        match parse_work(line, &filter) {
            ParseResult::Matched(row) => {
                assert_eq!(row.authorships.len(), 1);
                let af = &row.authorships[0].affiliations[0];
                assert_eq!(af.institution_ids.len(), 1);
                assert!(af.institution_ids[0].is_none());
            }
            other => panic!("expected Matched, got {other:?}"),
        }
    }

    /// Null elements in institution_ids are filtered out in the accumulator.
    #[test]
    fn null_institution_ids_end_to_end() {
        let dir = tempfile::TempDir::new().unwrap();
        let ctx = crate::provider::RunContext {
            output_dir: dir.path().to_path_buf(),
            zstd_level: 3,
            concurrency: 1,
        };

        let mut pipeline = ShardPipeline::create(0, &ctx, None, 0u16).unwrap();

        let input = r#"{"id":"https://openalex.org/W99","authorships":[{"author_position":"first","author":{"id":"https://openalex.org/A1","display_name":"Bob"},"institutions":[],"affiliations":[{"raw_affiliation_string":"MIT","institution_ids":[null,"https://openalex.org/I1"]}]}]}
"#;
        let mut reader = std::io::BufReader::new(input.as_bytes());
        let counter: ByteCounter = Arc::new(AtomicU64::new(0));
        let pb = ProgressBar::hidden();
        let filter = Filter::default();

        let result =
            process_works_lines(&mut reader, &counter, &mut pipeline, &filter, &pb).unwrap();
        assert_eq!(result.rows_written, 1);
        assert_eq!(result.parse_failures, 0);
        assert_eq!(result.null_institution_ids, 1);

        pipeline.finalize().unwrap();

        // The works row should exist
        let works = read_parquet(&dir.path().join("works/shard_0000.parquet"));
        assert_eq!(works.num_rows(), 1);
        assert_str_col(&works, "work_id", &["W99"]);

        // The authorship row should exist
        let auth = read_parquet(&dir.path().join("work_authorships/shard_0000.parquet"));
        assert_eq!(auth.num_rows(), 1);
        assert_str_col(&auth, "author_position", &["first"]);
    }
}
