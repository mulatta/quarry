//! Arrow schema definitions for OpenAlex dataset tables.
//!
//! 12 output tables forming a star schema around the `works` fact table.
//! All column names are plain (no `oa_` prefix) — the table name provides namespace.

use std::sync::{Arc, LazyLock};

use arrow::datatypes::{DataType, Field, Schema};

fn list_utf8() -> DataType {
    DataType::List(Arc::new(Field::new("item", DataType::Utf8, true)))
}

fn list_list_utf8() -> DataType {
    DataType::List(Arc::new(Field::new("item", list_utf8(), true)))
}

/// works.parquet — 82-column fact table
pub fn works() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            // -- identity (6)
            Field::new("work_id", DataType::Utf8, true),
            Field::new("doi", DataType::Utf8, true),
            Field::new("doi_norm", DataType::Utf8, true),
            Field::new("title", DataType::Utf8, true),
            Field::new("display_name", DataType::Utf8, true),
            Field::new("abstract_text", DataType::Utf8, true),
            // -- dates (4)
            Field::new("publication_date", DataType::Utf8, true),
            Field::new("publication_year", DataType::Int32, true),
            Field::new("created_date", DataType::Utf8, true),
            Field::new("updated_date", DataType::Utf8, true),
            // -- classification (3)
            Field::new("language", DataType::Utf8, true),
            Field::new("work_type", DataType::Utf8, true),
            Field::new("type_crossref", DataType::Utf8, true),
            // -- counts (5)
            Field::new("cited_by_count", DataType::Int32, true),
            Field::new("referenced_works_count", DataType::Int32, true),
            Field::new("countries_distinct_count", DataType::Int32, true),
            Field::new("institutions_distinct_count", DataType::Int32, true),
            Field::new("locations_count", DataType::Int32, true),
            // -- metrics (1)
            Field::new("fwci", DataType::Float64, true),
            // -- flags (2)
            Field::new("is_retracted", DataType::Boolean, true),
            Field::new("is_paratext", DataType::Boolean, true),
            // -- open_access (4)
            Field::new("is_oa", DataType::Boolean, true),
            Field::new("oa_status", DataType::Utf8, true),
            Field::new("oa_url", DataType::Utf8, true),
            Field::new("any_repository_has_fulltext", DataType::Boolean, true),
            // -- citation percentiles (5)
            Field::new("citation_normalized_percentile", DataType::Float64, true),
            Field::new("is_in_top_1_pct", DataType::Boolean, true),
            Field::new("is_in_top_10_pct", DataType::Boolean, true),
            Field::new("cited_by_percentile_min", DataType::Float64, true),
            Field::new("cited_by_percentile_max", DataType::Float64, true),
            // -- has_content (2)
            Field::new("has_content_pdf", DataType::Boolean, true),
            Field::new("has_content_grobid_xml", DataType::Boolean, true),
            // -- biblio (4)
            Field::new("biblio_volume", DataType::Utf8, true),
            Field::new("biblio_issue", DataType::Utf8, true),
            Field::new("biblio_first_page", DataType::Utf8, true),
            Field::new("biblio_last_page", DataType::Utf8, true),
            // -- apc_list (4)
            Field::new("apc_list_value", DataType::Int32, true),
            Field::new("apc_list_currency", DataType::Utf8, true),
            Field::new("apc_list_value_usd", DataType::Int32, true),
            Field::new("apc_list_provenance", DataType::Utf8, true),
            // -- apc_paid (4)
            Field::new("apc_paid_value", DataType::Int32, true),
            Field::new("apc_paid_currency", DataType::Utf8, true),
            Field::new("apc_paid_value_usd", DataType::Int32, true),
            Field::new("apc_paid_provenance", DataType::Utf8, true),
            // -- primary_topic flattened (9)
            Field::new("primary_topic_id", DataType::Utf8, true),
            Field::new("primary_topic_display_name", DataType::Utf8, true),
            Field::new("primary_topic_score", DataType::Float64, true),
            Field::new("primary_topic_subfield_id", DataType::Utf8, true),
            Field::new("primary_topic_subfield_display_name", DataType::Utf8, true),
            Field::new("primary_topic_field_id", DataType::Utf8, true),
            Field::new("primary_topic_field_display_name", DataType::Utf8, true),
            Field::new("primary_topic_domain_id", DataType::Utf8, true),
            Field::new("primary_topic_domain_display_name", DataType::Utf8, true),
            // -- primary_location flattened (11)
            Field::new("primary_location_is_oa", DataType::Boolean, true),
            Field::new("primary_location_landing_page_url", DataType::Utf8, true),
            Field::new("primary_location_pdf_url", DataType::Utf8, true),
            Field::new("primary_location_license", DataType::Utf8, true),
            Field::new("primary_location_version", DataType::Utf8, true),
            Field::new("primary_location_source_id", DataType::Utf8, true),
            Field::new("primary_location_source_display_name", DataType::Utf8, true),
            Field::new("primary_location_source_issn_l", DataType::Utf8, true),
            Field::new("primary_location_source_issn", list_utf8(), true),
            Field::new(
                "primary_location_source_host_organization",
                DataType::Utf8,
                true,
            ),
            Field::new("primary_location_source_type", DataType::Utf8, true),
            // -- best_oa_location flattened (11)
            Field::new("best_oa_location_is_oa", DataType::Boolean, true),
            Field::new("best_oa_location_landing_page_url", DataType::Utf8, true),
            Field::new("best_oa_location_pdf_url", DataType::Utf8, true),
            Field::new("best_oa_location_license", DataType::Utf8, true),
            Field::new("best_oa_location_version", DataType::Utf8, true),
            Field::new("best_oa_location_source_id", DataType::Utf8, true),
            Field::new("best_oa_location_source_display_name", DataType::Utf8, true),
            Field::new("best_oa_location_source_issn_l", DataType::Utf8, true),
            Field::new("best_oa_location_source_issn", list_utf8(), true),
            Field::new(
                "best_oa_location_source_host_organization",
                DataType::Utf8,
                true,
            ),
            Field::new("best_oa_location_source_type", DataType::Utf8, true),
            // -- external ids (3)
            Field::new("ids_mag", DataType::Utf8, true),
            Field::new("ids_pmid", DataType::Utf8, true),
            Field::new("ids_pmcid", DataType::Utf8, true),
            // -- simple list columns (4)
            Field::new("corresponding_author_ids", list_utf8(), true),
            Field::new("corresponding_institution_ids", list_utf8(), true),
            Field::new("indexed_in", list_utf8(), true),
            Field::new("related_works", list_utf8(), true),
            // -- shard metadata (1)
            Field::new("shard_updated_date", DataType::Utf8, true),
        ]))
    });
    &SCHEMA
}

/// works_keys.parquet — narrow lookup for Phase 3 key matching
pub fn works_keys() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("doi_norm", DataType::Utf8, true),
            Field::new("pmid", DataType::Utf8, true),
            Field::new("pmcid", DataType::Utf8, true),
        ]))
    });
    &SCHEMA
}

/// citations.parquet — referenced_works exploded
pub fn citations() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("referenced_work_id", DataType::Utf8, true),
        ]))
    });
    &SCHEMA
}

/// work_authorships.parquet — one row per authorship entry
pub fn work_authorships() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("author_position", DataType::Utf8, true),
            Field::new("author_id", DataType::Utf8, true),
            Field::new("author_display_name", DataType::Utf8, true),
            Field::new("author_orcid", DataType::Utf8, true),
            Field::new("raw_author_name", DataType::Utf8, true),
            Field::new("is_corresponding", DataType::Boolean, true),
            Field::new("countries", list_utf8(), true),
            Field::new("raw_affiliation_strings", list_utf8(), true),
            // institutions as parallel lists
            Field::new("institution_ids", list_utf8(), true),
            Field::new("institution_display_names", list_utf8(), true),
            Field::new("institution_rors", list_utf8(), true),
            Field::new("institution_country_codes", list_utf8(), true),
            Field::new("institution_types", list_utf8(), true),
            Field::new("institution_lineages", list_list_utf8(), true),
            // affiliations as parallel lists
            Field::new("affiliation_raw_strings", list_utf8(), true),
            Field::new("affiliation_institution_ids", list_list_utf8(), true),
        ]))
    });
    &SCHEMA
}

/// work_topics.parquet — one row per topic assignment
pub fn work_topics() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("topic_id", DataType::Utf8, true),
            Field::new("display_name", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
            Field::new("subfield_id", DataType::Utf8, true),
            Field::new("subfield_display_name", DataType::Utf8, true),
            Field::new("field_id", DataType::Utf8, true),
            Field::new("field_display_name", DataType::Utf8, true),
            Field::new("domain_id", DataType::Utf8, true),
            Field::new("domain_display_name", DataType::Utf8, true),
        ]))
    });
    &SCHEMA
}

/// work_keywords.parquet — one row per keyword
pub fn work_keywords() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("keyword_id", DataType::Utf8, true),
            Field::new("display_name", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
        ]))
    });
    &SCHEMA
}

/// work_mesh.parquet — one row per MeSH descriptor
pub fn work_mesh() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("descriptor_ui", DataType::Utf8, true),
            Field::new("descriptor_name", DataType::Utf8, true),
            Field::new("qualifier_ui", DataType::Utf8, true),
            Field::new("qualifier_name", DataType::Utf8, true),
            Field::new("is_major_topic", DataType::Boolean, true),
        ]))
    });
    &SCHEMA
}

/// work_locations.parquet — one row per location
pub fn work_locations() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("is_oa", DataType::Boolean, true),
            Field::new("landing_page_url", DataType::Utf8, true),
            Field::new("pdf_url", DataType::Utf8, true),
            Field::new("license", DataType::Utf8, true),
            Field::new("version", DataType::Utf8, true),
            Field::new("source_id", DataType::Utf8, true),
            Field::new("source_display_name", DataType::Utf8, true),
            Field::new("source_issn_l", DataType::Utf8, true),
            Field::new("source_issn", list_utf8(), true),
            Field::new("source_host_organization", DataType::Utf8, true),
            Field::new("source_type", DataType::Utf8, true),
        ]))
    });
    &SCHEMA
}

/// work_funders.parquet — one row per funder
pub fn work_funders() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("funder_id", DataType::Utf8, true),
            Field::new("display_name", DataType::Utf8, true),
            Field::new("ror", DataType::Utf8, true),
        ]))
    });
    &SCHEMA
}

/// work_awards.parquet — one row per grant/award
pub fn work_awards() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("award_id", DataType::Utf8, true),
            Field::new("display_name", DataType::Utf8, true),
            Field::new("funder_award_id", DataType::Utf8, true),
            Field::new("funder_id", DataType::Utf8, true),
            Field::new("funder_display_name", DataType::Utf8, true),
            Field::new("doi", DataType::Utf8, true),
        ]))
    });
    &SCHEMA
}

/// work_sdgs.parquet — one row per SDG
pub fn work_sdgs() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("sdg_id", DataType::Utf8, true),
            Field::new("display_name", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
        ]))
    });
    &SCHEMA
}

/// work_counts_by_year.parquet — one row per (work, year)
pub fn work_counts_by_year() -> &'static Arc<Schema> {
    static SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
        Arc::new(Schema::new(vec![
            Field::new("work_id", DataType::Utf8, true),
            Field::new("year", DataType::Int32, true),
            Field::new("cited_by_count", DataType::Int32, true),
        ]))
    });
    &SCHEMA
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn works_has_83_cols() {
        assert_eq!(works().fields().len(), 83);
    }

    #[test]
    fn works_keys_has_4_cols() {
        assert_eq!(works_keys().fields().len(), 4);
    }

    #[test]
    fn citations_has_2_cols() {
        assert_eq!(citations().fields().len(), 2);
    }

    #[test]
    fn work_authorships_has_17_cols() {
        assert_eq!(work_authorships().fields().len(), 17);
    }

    #[test]
    fn work_topics_has_10_cols() {
        assert_eq!(work_topics().fields().len(), 10);
    }

    #[test]
    fn work_keywords_has_4_cols() {
        assert_eq!(work_keywords().fields().len(), 4);
    }

    #[test]
    fn work_mesh_has_6_cols() {
        assert_eq!(work_mesh().fields().len(), 6);
    }

    #[test]
    fn work_locations_has_12_cols() {
        assert_eq!(work_locations().fields().len(), 12);
    }

    #[test]
    fn work_funders_has_4_cols() {
        assert_eq!(work_funders().fields().len(), 4);
    }

    #[test]
    fn work_awards_has_7_cols() {
        assert_eq!(work_awards().fields().len(), 7);
    }

    #[test]
    fn work_sdgs_has_4_cols() {
        assert_eq!(work_sdgs().fields().len(), 4);
    }

    #[test]
    fn work_counts_by_year_has_3_cols() {
        assert_eq!(work_counts_by_year().fields().len(), 3);
    }

    #[test]
    fn works_list_fields() {
        let schema = works();
        for name in [
            "corresponding_author_ids",
            "corresponding_institution_ids",
            "indexed_in",
            "related_works",
            "primary_location_source_issn",
            "best_oa_location_source_issn",
        ] {
            let field = schema.field_with_name(name).unwrap();
            assert!(
                matches!(field.data_type(), DataType::List(_)),
                "{name} should be List"
            );
        }
    }

    #[test]
    fn authorships_list_list_fields() {
        let schema = work_authorships();
        for name in ["institution_lineages", "affiliation_institution_ids"] {
            let field = schema.field_with_name(name).unwrap();
            match field.data_type() {
                DataType::List(inner) => assert!(
                    matches!(inner.data_type(), DataType::List(_)),
                    "{name} should be List<List<Utf8>>"
                ),
                _ => panic!("{name} should be List<List<Utf8>>"),
            }
        }
    }
}
