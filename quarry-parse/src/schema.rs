//! Arrow schema definitions — single source of truth.
//!
//! Used by both quarry-parse (RecordBatch creation) and quarry-build (Parquet writing).

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};

/// PubMed papers table schema.
pub fn papers_schema() -> Schema {
    Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("doi", DataType::Utf8, true),
        Field::new("pmc_id", DataType::Utf8, true),
        Field::new("title", DataType::Utf8, true),
        Field::new("abstract", DataType::Utf8, true),
        Field::new("pub_year", DataType::Int16, true),
        Field::new("pub_date", DataType::Utf8, true),
        Field::new("journal_title", DataType::Utf8, true),
        Field::new("journal_issn", DataType::Utf8, true),
        Field::new("journal_abbr", DataType::Utf8, true),
        Field::new("volume", DataType::Utf8, true),
        Field::new("issue", DataType::Utf8, true),
        Field::new("pages", DataType::Utf8, true),
        Field::new("language", DataType::Utf8, true),
        Field::new(
            "pub_type",
            DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
            true,
        ),
        Field::new("country", DataType::Utf8, true),
        Field::new("medline_status", DataType::Utf8, true),
        Field::new("created_date", DataType::Utf8, true),
        Field::new("revised_date", DataType::Utf8, true),
        Field::new("indexed_date", DataType::Utf8, true),
    ])
}

/// PubMed authors table schema.
pub fn authors_schema() -> Schema {
    Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("author_position", DataType::Int16, false),
        Field::new("last_name", DataType::Utf8, true),
        Field::new("fore_name", DataType::Utf8, true),
        Field::new("initials", DataType::Utf8, true),
        Field::new("orcid", DataType::Utf8, true),
        Field::new("affiliation", DataType::Utf8, true),
        Field::new("is_collective", DataType::Boolean, false),
    ])
}

/// PubMed MeSH headings table schema.
pub fn mesh_headings_schema() -> Schema {
    Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("descriptor_ui", DataType::Utf8, false),
        Field::new("descriptor_name", DataType::Utf8, false),
        Field::new("qualifier_ui", DataType::Utf8, true),
        Field::new("qualifier_name", DataType::Utf8, true),
        Field::new("is_major_topic", DataType::Boolean, false),
    ])
}

/// PubMed grants table schema.
pub fn grants_schema() -> Schema {
    Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("grant_id", DataType::Utf8, true),
        Field::new("acronym", DataType::Utf8, true),
        Field::new("agency", DataType::Utf8, true),
        Field::new("country", DataType::Utf8, true),
    ])
}

/// PubMed chemicals table schema.
pub fn chemicals_schema() -> Schema {
    Schema::new(vec![
        Field::new("pmid", DataType::Int32, false),
        Field::new("registry_number", DataType::Utf8, true),
        Field::new("substance_ui", DataType::Utf8, true),
        Field::new("substance_name", DataType::Utf8, true),
    ])
}

/// MeSH descriptor lookup table schema.
pub fn mesh_descriptors_schema() -> Schema {
    Schema::new(vec![
        Field::new("descriptor_ui", DataType::Utf8, false),
        Field::new("descriptor_name", DataType::Utf8, false),
        Field::new("tree_number", DataType::Utf8, false),
    ])
}

// ── OpenAlex schemas (DDL_V2 1:1 correspondence) ──

/// OpenAlex works table schema.
pub fn works_schema() -> Schema {
    Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("work_id_int", DataType::Int64, false),
        Field::new("tier", DataType::Utf8, false),
        Field::new("pmid", DataType::Int32, true),
        Field::new("doi", DataType::Utf8, true),
        Field::new("title", DataType::Utf8, false),
        Field::new("abstract", DataType::Utf8, true),
        Field::new("pub_year", DataType::Int16, true),
        Field::new("pub_date", DataType::Utf8, true),
        Field::new("type", DataType::Utf8, true),
        Field::new("cited_by_count", DataType::Int32, true),
        Field::new("host_venue", DataType::Utf8, true),
        Field::new("oa_status", DataType::Utf8, true),
        Field::new("oa_url", DataType::Utf8, true),
        Field::new("is_retracted", DataType::Boolean, false),
        Field::new("updated_date", DataType::Utf8, true),
    ])
}

/// OpenAlex work authors table schema.
pub fn work_authors_schema() -> Schema {
    Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("author_position", DataType::Int16, false),
        Field::new("display_name", DataType::Utf8, true),
        Field::new("orcid", DataType::Utf8, true),
        Field::new("institution_name", DataType::Utf8, true),
        Field::new("institution_ror", DataType::Utf8, true),
        Field::new("raw_affiliation", DataType::Utf8, true),
    ])
}

/// OpenAlex work topics table schema.
pub fn work_topics_schema() -> Schema {
    Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("topic_id", DataType::Utf8, false),
        Field::new("topic_name", DataType::Utf8, true),
        Field::new("subfield", DataType::Utf8, true),
        Field::new("field", DataType::Utf8, true),
        Field::new("domain", DataType::Utf8, true),
        Field::new("score", DataType::Float32, true),
    ])
}

/// OpenAlex work citations table schema.
pub fn work_citations_schema() -> Schema {
    Schema::new(vec![
        Field::new("citing_id", DataType::Int64, false),
        Field::new("cited_id", DataType::Int64, false),
    ])
}

/// OpenAlex ID crosswalk table schema (work_id ↔ PMID).
pub fn id_crosswalk_schema() -> Schema {
    Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("pmid", DataType::Int32, false),
    ])
}

/// OpenAlex work MeSH table schema.
pub fn work_mesh_schema() -> Schema {
    Schema::new(vec![
        Field::new("work_id", DataType::Utf8, false),
        Field::new("descriptor_ui", DataType::Utf8, false),
        Field::new("descriptor_name", DataType::Utf8, false),
        Field::new("qualifier_ui", DataType::Utf8, true),
        Field::new("qualifier_name", DataType::Utf8, true),
        Field::new("is_major_topic", DataType::Boolean, false),
    ])
}
