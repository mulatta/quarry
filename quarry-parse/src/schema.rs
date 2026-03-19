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
