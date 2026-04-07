//! Parquet file writer for parsed data.
//!
//! Each write function takes a slice of parsed structs and writes a single Parquet file.
//! Output uses Zstd compression, row group size ~128MB for efficient CH ingestion.

use std::path::Path;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::build::oa_json::{OaAuthor, OaCitation, OaCountByYear, OaCrosswalk, OaTopic, OaWork};
use crate::parse::mesh::{MeshEntry, MeshTerm};
use crate::parse::xml::{Author, Chemical, Grant, MeshHeading, Paper};

/// Parse "YYYY-MM-DD" to days since epoch for Arrow Date32.
/// Returns None for invalid dates or values outside CH Date32 range [-25567, 120530].
fn date_str_to_days(s: &str) -> Option<i32> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let y: i32 = parts[0].parse().ok()?;
    let m: u32 = parts[1].parse().ok()?;
    let d: u32 = parts[2].parse().ok()?;

    // Reject invalid dates (month-aware day limit, no leap-year pedantry)
    let max_day = match m {
        2 => 29,
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => return None,
    };
    if d < 1 || d > max_day {
        return None;
    }

    // Days from 1970-01-01 using the same algorithm as chrono
    let m_adj = if m > 2 { m - 3 } else { m + 9 };
    let y_adj = if m <= 2 { y - 1 } else { y };
    let era = y_adj.div_euclid(400);
    let yoe = y_adj.rem_euclid(400) as u32;
    let doy = (153 * m_adj + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe as i32 - 719468;

    // CH Date32 range: 1900-01-03 (-25567) to 2299-12-31 (120530)
    if (-25567..=120530).contains(&days) {
        Some(days)
    } else {
        None
    }
}

/// Safe i32 → u32 conversion: negative values become None (null in Parquet).
fn i32_to_u32(v: i32) -> Option<u32> {
    u32::try_from(v).ok()
}

/// Safe i16 → u16 conversion: negative values become None.
fn i16_to_u16(v: i16) -> Option<u16> {
    u16::try_from(v).ok()
}

/// Safe i64 → u64 conversion: negative values become 0 (should never happen for IDs).
fn i64_to_u64(v: i64) -> u64 {
    u64::try_from(v).unwrap_or(0)
}

fn writer_props() -> WriterProperties {
    WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_size(1_000_000)
        .build()
}

/// Write batch to Parquet via .tmp → rename for crash safety.
fn write_batch(path: &Path, batch: &RecordBatch) -> Result<usize, Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("parquet.tmp");
    let file = std::fs::File::create(&tmp)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(writer_props()))?;
    let n = batch.num_rows();
    writer.write(batch)?;
    writer.close()?;
    std::fs::rename(&tmp, path)?;
    Ok(n)
}

// ── OA Parquet writers ──

pub fn write_oa_works(works: &[OaWork], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if works.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("work_id_int", DataType::UInt64, false),
        Field::new("tier", DataType::Utf8, false),
        Field::new("pmid", DataType::UInt32, true),
        Field::new("doi", DataType::Utf8, true),
        Field::new("title", DataType::Utf8, false),
        Field::new("abstract", DataType::Utf8, true),
        Field::new("pub_year", DataType::UInt16, true),
        Field::new("pub_date", DataType::Date32, true),
        Field::new("type", DataType::Utf8, true),
        Field::new("cited_by_count", DataType::UInt32, true),
        Field::new("host_venue", DataType::Utf8, true),
        Field::new("oa_status", DataType::Utf8, true),
        Field::new("oa_url", DataType::Utf8, true),
        Field::new("is_retracted", DataType::Boolean, false),
        Field::new("updated_date", DataType::Date32, true),
        Field::new("language", DataType::Utf8, true),
        Field::new("fwci", DataType::Float32, true),
        Field::new("citation_normalized_percentile", DataType::Float32, true),
        Field::new("cited_by_percentile_year_min", DataType::UInt16, true),
        Field::new("cited_by_percentile_year_max", DataType::UInt16, true),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from_iter_values(works.iter().map(|w| &*w.work_id))),
        Arc::new(UInt64Array::from_iter_values(works.iter().map(|w| i64_to_u64(w.work_id_int)))),
        Arc::new(StringArray::from_iter_values(works.iter().map(|w| w.tier.as_str()))),
        Arc::new(UInt32Array::from_iter(works.iter().map(|w| w.pmid.and_then(i32_to_u32)))),
        Arc::new(StringArray::from_iter(works.iter().map(|w| w.doi.as_deref()))),
        Arc::new(StringArray::from_iter_values(works.iter().map(|w| w.title.as_str()))),
        Arc::new(StringArray::from_iter(works.iter().map(|w| w.abstract_text.as_deref()))),
        Arc::new(UInt16Array::from_iter(works.iter().map(|w| w.pub_year.and_then(i16_to_u16)))),
        Arc::new(Date32Array::from_iter(works.iter().map(|w| w.pub_date.as_deref().and_then(date_str_to_days)))),
        Arc::new(StringArray::from_iter(works.iter().map(|w| w.work_type.as_deref()))),
        Arc::new(UInt32Array::from_iter(works.iter().map(|w| w.cited_by_count.and_then(i32_to_u32)))),
        Arc::new(StringArray::from_iter(works.iter().map(|w| w.host_venue.as_deref()))),
        Arc::new(StringArray::from_iter(works.iter().map(|w| w.oa_status.as_deref()))),
        Arc::new(StringArray::from_iter(works.iter().map(|w| w.oa_url.as_deref()))),
        Arc::new(BooleanArray::from_iter(works.iter().map(|w| Some(w.is_retracted)))),
        Arc::new(Date32Array::from_iter(works.iter().map(|w| w.updated_date.as_deref().and_then(date_str_to_days)))),
        Arc::new(StringArray::from_iter(works.iter().map(|w| w.language.as_deref()))),
        Arc::new(Float32Array::from_iter(works.iter().map(|w| w.fwci))),
        Arc::new(Float32Array::from_iter(works.iter().map(|w| w.citation_normalized_percentile))),
        Arc::new(UInt16Array::from_iter(works.iter().map(|w| w.cited_by_percentile_year_min.and_then(i16_to_u16)))),
        Arc::new(UInt16Array::from_iter(works.iter().map(|w| w.cited_by_percentile_year_max.and_then(i16_to_u16)))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_oa_work_authors(authors: &[OaAuthor], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if authors.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("author_position", DataType::UInt16, false),
        Field::new("display_name", DataType::Utf8, true),
        Field::new("orcid", DataType::Utf8, true),
        Field::new("institution_name", DataType::Utf8, true),
        Field::new("institution_ror", DataType::Utf8, true),
        Field::new("raw_affiliation", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from_iter_values(authors.iter().map(|a| &*a.work_id))),
        Arc::new(UInt16Array::from_iter_values(authors.iter().map(|a| i16_to_u16(a.author_position).unwrap_or(0)))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.display_name.as_deref()))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.orcid.as_deref()))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.institution_name.as_deref()))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.institution_ror.as_deref()))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.raw_affiliation.as_deref()))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_oa_work_topics(topics: &[OaTopic], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if topics.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("topic_id", DataType::Utf8, false),
        Field::new("topic_name", DataType::Utf8, true),
        Field::new("subfield", DataType::Utf8, true),
        Field::new("field", DataType::Utf8, true),
        Field::new("domain", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
        Field::new("is_primary", DataType::Boolean, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from_iter_values(topics.iter().map(|t| &*t.work_id))),
        Arc::new(StringArray::from_iter_values(topics.iter().map(|t| t.topic_id.as_str()))),
        Arc::new(StringArray::from_iter(topics.iter().map(|t| t.topic_name.as_deref()))),
        Arc::new(StringArray::from_iter(topics.iter().map(|t| t.subfield.as_deref()))),
        Arc::new(StringArray::from_iter(topics.iter().map(|t| t.field.as_deref()))),
        Arc::new(StringArray::from_iter(topics.iter().map(|t| t.domain.as_deref()))),
        Arc::new(Float32Array::from_iter(topics.iter().map(|t| t.score))),
        Arc::new(BooleanArray::from_iter(topics.iter().map(|t| Some(t.is_primary)))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_oa_work_citations(citations: &[OaCitation], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if citations.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("citing_id", DataType::UInt64, false),
        Field::new("cited_id", DataType::UInt64, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(UInt64Array::from_iter_values(citations.iter().map(|c| i64_to_u64(c.citing_id)))),
        Arc::new(UInt64Array::from_iter_values(citations.iter().map(|c| i64_to_u64(c.cited_id)))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_oa_id_crosswalk(crosswalk: &[OaCrosswalk], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if crosswalk.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("pmid", DataType::UInt32, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from_iter_values(crosswalk.iter().map(|c| &*c.work_id))),
        Arc::new(UInt32Array::from_iter_values(crosswalk.iter().map(|c| i32_to_u32(c.pmid).unwrap_or(0)))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_oa_counts_by_year(counts: &[OaCountByYear], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if counts.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("year", DataType::UInt16, false),
        Field::new("cited_by_count", DataType::UInt32, false),
        Field::new("updated_date", DataType::Date32, true),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from_iter_values(counts.iter().map(|c| &*c.work_id))),
        // year validated > 0 in parser
        Arc::new(UInt16Array::from_iter_values(counts.iter().map(|c| i16_to_u16(c.year).unwrap()))),
        Arc::new(UInt32Array::from_iter_values(counts.iter().map(|c| i32_to_u32(c.cited_by_count).unwrap_or(0)))),
        Arc::new(Date32Array::from_iter(counts.iter().map(|c| c.updated_date.as_deref().and_then(date_str_to_days)))),
    ])?;

    write_batch(path, &batch)
}

// ── PubMed Parquet writers ──

pub fn write_pm_papers(papers: &[Paper], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if papers.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("pmid", DataType::UInt32, false),
        Field::new("doi", DataType::Utf8, true),
        Field::new("pmc_id", DataType::Utf8, true),
        Field::new("title", DataType::Utf8, true),
        Field::new("abstract", DataType::Utf8, true),
        Field::new("pub_year", DataType::UInt16, true),
        Field::new("pub_date", DataType::Date32, true),
        Field::new("journal_title", DataType::Utf8, true),
        Field::new("journal_issn", DataType::Utf8, true),
        Field::new("journal_abbr", DataType::Utf8, true),
        Field::new("volume", DataType::Utf8, true),
        Field::new("issue", DataType::Utf8, true),
        Field::new("pages", DataType::Utf8, true),
        Field::new("language", DataType::Utf8, true),
        Field::new("pub_type", DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))), false),
        Field::new("country", DataType::Utf8, true),
        Field::new("medline_status", DataType::Utf8, true),
        Field::new("created_date", DataType::Date32, true),
        Field::new("revised_date", DataType::Date32, true),
        Field::new("indexed_date", DataType::Date32, true),
    ]));

    // Build pub_type list array
    let mut list_builder = ListBuilder::new(StringBuilder::new());
    for p in papers {
        for pt in &p.pub_type {
            list_builder.values().append_value(pt);
        }
        list_builder.append(true);
    }
    let pub_type_array = list_builder.finish();

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(UInt32Array::from_iter_values(papers.iter().map(|p| i32_to_u32(p.pmid).unwrap_or(0)))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.doi.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.pmc_id.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.title.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.r#abstract.as_deref()))),
        Arc::new(UInt16Array::from_iter(papers.iter().map(|p| p.pub_year.and_then(i16_to_u16)))),
        Arc::new(Date32Array::from_iter(papers.iter().map(|p| p.pub_date.as_deref().and_then(date_str_to_days)))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.journal_title.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.journal_issn.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.journal_abbr.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.volume.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.issue.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.pages.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.language.as_deref()))),
        Arc::new(pub_type_array),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.country.as_deref()))),
        Arc::new(StringArray::from_iter(papers.iter().map(|p| p.medline_status.as_deref()))),
        Arc::new(Date32Array::from_iter(papers.iter().map(|p| p.created_date.as_deref().and_then(date_str_to_days)))),
        Arc::new(Date32Array::from_iter(papers.iter().map(|p| p.revised_date.as_deref().and_then(date_str_to_days)))),
        Arc::new(Date32Array::from_iter(papers.iter().map(|p| p.indexed_date.as_deref().and_then(date_str_to_days)))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_pm_authors(authors: &[Author], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if authors.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("pmid", DataType::UInt32, false),
        Field::new("author_position", DataType::UInt16, false),
        Field::new("last_name", DataType::Utf8, true),
        Field::new("fore_name", DataType::Utf8, true),
        Field::new("initials", DataType::Utf8, true),
        Field::new("orcid", DataType::Utf8, true),
        Field::new("affiliation", DataType::Utf8, true),
        Field::new("is_collective", DataType::Boolean, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(UInt32Array::from_iter_values(authors.iter().map(|a| i32_to_u32(a.pmid).unwrap_or(0)))),
        Arc::new(UInt16Array::from_iter_values(authors.iter().map(|a| i16_to_u16(a.author_position).unwrap_or(0)))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.last_name.as_deref()))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.fore_name.as_deref()))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.initials.as_deref()))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.orcid.as_deref()))),
        Arc::new(StringArray::from_iter(authors.iter().map(|a| a.affiliation.as_deref()))),
        Arc::new(BooleanArray::from_iter(authors.iter().map(|a| Some(a.is_collective)))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_pm_mesh_headings(headings: &[MeshHeading], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if headings.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("pmid", DataType::UInt32, false),
        Field::new("descriptor_ui", DataType::Utf8, false),
        Field::new("descriptor_name", DataType::Utf8, false),
        Field::new("qualifier_ui", DataType::Utf8, true),
        Field::new("qualifier_name", DataType::Utf8, true),
        Field::new("is_major_topic", DataType::Boolean, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(UInt32Array::from_iter_values(headings.iter().map(|m| i32_to_u32(m.pmid).unwrap_or(0)))),
        Arc::new(StringArray::from_iter_values(headings.iter().map(|m| m.descriptor_ui.as_str()))),
        Arc::new(StringArray::from_iter_values(headings.iter().map(|m| m.descriptor_name.as_str()))),
        Arc::new(StringArray::from_iter(headings.iter().map(|m| m.qualifier_ui.as_deref()))),
        Arc::new(StringArray::from_iter(headings.iter().map(|m| m.qualifier_name.as_deref()))),
        Arc::new(BooleanArray::from_iter(headings.iter().map(|m| Some(m.is_major_topic)))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_pm_grants(grants: &[Grant], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if grants.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("pmid", DataType::UInt32, false),
        Field::new("grant_id", DataType::Utf8, true),
        Field::new("acronym", DataType::Utf8, true),
        Field::new("agency", DataType::Utf8, true),
        Field::new("country", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(UInt32Array::from_iter_values(grants.iter().map(|g| i32_to_u32(g.pmid).unwrap_or(0)))),
        Arc::new(StringArray::from_iter(grants.iter().map(|g| g.grant_id.as_deref()))),
        Arc::new(StringArray::from_iter(grants.iter().map(|g| g.acronym.as_deref()))),
        Arc::new(StringArray::from_iter(grants.iter().map(|g| g.agency.as_deref()))),
        Arc::new(StringArray::from_iter(grants.iter().map(|g| g.country.as_deref()))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_pm_chemicals(chemicals: &[Chemical], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if chemicals.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("pmid", DataType::UInt32, false),
        Field::new("registry_number", DataType::Utf8, true),
        Field::new("substance_ui", DataType::Utf8, true),
        Field::new("substance_name", DataType::Utf8, true),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(UInt32Array::from_iter_values(chemicals.iter().map(|c| i32_to_u32(c.pmid).unwrap_or(0)))),
        Arc::new(StringArray::from_iter(chemicals.iter().map(|c| c.registry_number.as_deref()))),
        Arc::new(StringArray::from_iter(chemicals.iter().map(|c| c.substance_ui.as_deref()))),
        Arc::new(StringArray::from_iter(chemicals.iter().map(|c| c.substance_name.as_deref()))),
    ])?;

    write_batch(path, &batch)
}

// ── MeSH Parquet writer ──

pub fn write_mesh_tree(entries: &[MeshEntry], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if entries.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("descriptor_ui", DataType::Utf8, false),
        Field::new("descriptor_name", DataType::Utf8, false),
        Field::new("tree_number", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from_iter_values(entries.iter().map(|e| e.descriptor_ui.as_str()))),
        Arc::new(StringArray::from_iter_values(entries.iter().map(|e| e.descriptor_name.as_str()))),
        Arc::new(StringArray::from_iter_values(entries.iter().map(|e| e.tree_number.as_str()))),
    ])?;

    write_batch(path, &batch)
}

pub fn write_mesh_terms(terms: &[MeshTerm], path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    if terms.is_empty() {
        return Ok(0);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("descriptor_ui", DataType::Utf8, false),
        Field::new("descriptor_name", DataType::Utf8, false),
        Field::new("term", DataType::Utf8, false),
        Field::new("is_preferred", DataType::Boolean, false),
    ]));

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(StringArray::from_iter_values(terms.iter().map(|t| t.descriptor_ui.as_str()))),
        Arc::new(StringArray::from_iter_values(terms.iter().map(|t| t.descriptor_name.as_str()))),
        Arc::new(StringArray::from_iter_values(terms.iter().map(|t| t.term.as_str()))),
        Arc::new(BooleanArray::from(terms.iter().map(|t| t.is_preferred).collect::<Vec<_>>())),
    ])?;

    write_batch(path, &batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_date_str_to_days() {
        assert_eq!(date_str_to_days("1970-01-01"), Some(0));
        assert_eq!(date_str_to_days("2024-06-15"), Some(19889));
        assert!(date_str_to_days("not-a-date").is_none());
        assert!(date_str_to_days("2024").is_none());
        // Invalid month/day rejected
        assert!(date_str_to_days("2024-00-15").is_none());
        assert!(date_str_to_days("2024-13-01").is_none());
        assert!(date_str_to_days("2024-06-00").is_none());
        assert!(date_str_to_days("2024-06-32").is_none());
    }

    #[test]
    fn test_write_oa_works_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.parquet");
        assert_eq!(write_oa_works(&[], &path).unwrap(), 0);
    }

    #[test]
    fn test_write_oa_works_new_fields_roundtrip() {
        use crate::build::oa_json::{OaWork, Tier};
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("works.parquet");
        let works = vec![OaWork {
            work_id: Arc::from("W1"),
            work_id_int: 1,
            tier: Tier::T1,
            pmid: Some(99999),
            doi: None,
            title: "Test".to_string(),
            abstract_text: Some("Abstract".to_string()),
            pub_year: Some(2024),
            pub_date: Some("2024-06-15".to_string()),
            work_type: Some("article".to_string()),
            cited_by_count: Some(10),
            host_venue: None,
            oa_status: None,
            oa_url: None,
            is_retracted: false,
            updated_date: Some("2024-07-01".to_string()),
            language: Some("en".to_string()),
            fwci: Some(8.02),
            citation_normalized_percentile: Some(0.984),
            cited_by_percentile_year_min: Some(98),
            cited_by_percentile_year_max: Some(100),
        }];
        let n = write_oa_works(&works, &path).unwrap();
        assert_eq!(n, 1);

        // Read back and verify schema
        let file = std::fs::File::open(&path).unwrap();
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReader::try_new(file, 1024).unwrap();
        let schema = reader.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(names.contains(&"language"));
        assert!(names.contains(&"fwci"));
        assert!(names.contains(&"citation_normalized_percentile"));
        assert!(names.contains(&"cited_by_percentile_year_min"));
        assert!(names.contains(&"cited_by_percentile_year_max"));
        assert!(!names.contains(&"is_paratext")); // removed
    }

    #[test]
    fn test_write_oa_counts_by_year_roundtrip() {
        use crate::build::oa_json::OaCountByYear;
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("counts.parquet");
        let counts = vec![
            OaCountByYear {
                work_id: Arc::from("W1"),
                year: 2024,
                cited_by_count: 5,
                updated_date: Some("2024-07-01".to_string()),
            },
            OaCountByYear {
                work_id: Arc::from("W1"),
                year: 2023,
                cited_by_count: 3,
                updated_date: Some("2024-07-01".to_string()),
            },
        ];
        let n = write_oa_counts_by_year(&counts, &path).unwrap();
        assert_eq!(n, 2);

        let file = std::fs::File::open(&path).unwrap();
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReader::try_new(file, 1024).unwrap();
        let schema = reader.schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, vec!["work_id", "year", "cited_by_count", "updated_date"]);
    }

    #[test]
    fn test_write_mesh_tree_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mesh.parquet");
        let entries = vec![
            MeshEntry {
                descriptor_ui: "D000001".to_string(),
                descriptor_name: "Calcimycin".to_string(),
                tree_number: "D03.633".to_string(),
            },
        ];
        let n = write_mesh_tree(&entries, &path).unwrap();
        assert_eq!(n, 1);
        assert!(path.exists());
    }
}
