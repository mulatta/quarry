//! Serde structs for OpenAlex Work object deserialization.

use std::collections::HashMap;

use serde::Deserialize;
use sonic_rs::JsonValueTrait;

// ============================================================
// Serde structs — full OA Work object deserialization
// ============================================================

#[derive(Debug, Deserialize)]
pub struct WorkRow {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub doi: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub publication_date: Option<String>,
    #[serde(default)]
    pub publication_year: Option<i32>,
    #[serde(default)]
    pub created_date: Option<String>,
    #[serde(default)]
    pub updated_date: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(rename = "type", default)]
    pub work_type: Option<String>,
    #[serde(default)]
    pub cited_by_count: Option<i32>,
    #[serde(default)]
    pub referenced_works_count: Option<i32>,
    #[serde(default)]
    pub countries_distinct_count: Option<i32>,
    #[serde(default)]
    pub institutions_distinct_count: Option<i32>,
    #[serde(default)]
    pub locations_count: Option<i32>,
    #[serde(default)]
    pub fwci: Option<f64>,
    #[serde(default)]
    pub is_retracted: Option<bool>,
    #[serde(default)]
    pub is_paratext: Option<bool>,
    #[serde(default)]
    pub open_access: Option<OpenAccess>,
    #[serde(default)]
    pub citation_normalized_percentile: Option<CitationPercentile>,
    #[serde(default)]
    pub cited_by_percentile_year: Option<PercentileYear>,
    #[serde(default)]
    pub has_fulltext: Option<bool>,
    #[serde(default)]
    pub has_content: Option<HasContent>,
    #[serde(default)]
    pub biblio: Option<Biblio>,
    #[serde(default)]
    pub apc_list: Option<Apc>,
    #[serde(default)]
    pub apc_paid: Option<Apc>,
    #[serde(default)]
    pub primary_topic: Option<Topic>,
    #[serde(default)]
    pub topics: Vec<Topic>,
    #[serde(default)]
    pub primary_location: Option<Location>,
    #[serde(default)]
    pub best_oa_location: Option<Location>,
    #[serde(default)]
    pub locations: Vec<Location>,
    #[serde(default)]
    pub authorships: Vec<Authorship>,
    #[serde(default)]
    pub corresponding_author_ids: Vec<String>,
    #[serde(default)]
    pub corresponding_institution_ids: Vec<String>,
    #[serde(default)]
    pub ids: Option<WorkIds>,
    #[serde(default)]
    pub keywords: Vec<Keyword>,
    #[serde(default)]
    pub mesh: Vec<MeshTerm>,
    #[serde(default)]
    pub sustainable_development_goals: Vec<Sdg>,
    #[serde(default)]
    pub funders: Vec<Funder>,
    #[serde(default)]
    pub awards: Vec<Award>,
    #[serde(default)]
    pub counts_by_year: Vec<CountByYear>,
    #[serde(default)]
    pub referenced_works: Vec<String>,
    #[serde(default)]
    pub related_works: Vec<String>,
    #[serde(default)]
    pub indexed_in: Vec<String>,
    #[serde(default)]
    pub abstract_inverted_index: Option<HashMap<String, Vec<u32>>>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAccess {
    #[serde(default)]
    pub is_oa: Option<bool>,
    #[serde(default)]
    pub oa_status: Option<String>,
    #[serde(default)]
    pub oa_url: Option<String>,
    #[serde(default)]
    pub any_repository_has_fulltext: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct HasContent {
    #[serde(default)]
    pub pdf: Option<bool>,
    #[serde(default)]
    pub grobid_xml: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct CitationPercentile {
    #[serde(default)]
    pub value: Option<f64>,
    #[serde(default)]
    pub is_in_top_1_percent: Option<bool>,
    #[serde(default)]
    pub is_in_top_10_percent: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct PercentileYear {
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct Biblio {
    #[serde(default)]
    pub volume: Option<String>,
    #[serde(default)]
    pub issue: Option<String>,
    #[serde(default)]
    pub first_page: Option<String>,
    #[serde(default)]
    pub last_page: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Apc {
    #[serde(default, deserialize_with = "number_as_i32")]
    pub value: Option<i32>,
    #[serde(default)]
    pub currency: Option<String>,
    #[serde(default, deserialize_with = "number_as_i32")]
    pub value_usd: Option<i32>,
    #[serde(default)]
    pub provenance: Option<String>,
}

/// Deserialize a JSON number (int or float like `2500.0`) as `Option<i32>`.
fn number_as_i32<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<i32>, D::Error> {
    let v: Option<sonic_rs::Value> = Option::deserialize(d)?;
    Ok(v.and_then(|v| {
        if v.is_i64() {
            Some(v.as_i64().unwrap() as i32)
        } else if v.is_f64() {
            Some(v.as_f64().unwrap() as i32)
        } else {
            None
        }
    }))
}

#[derive(Debug, Deserialize)]
pub struct Topic {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub subfield: Option<TopicClassification>,
    #[serde(default)]
    pub field: Option<TopicClassification>,
    #[serde(default)]
    pub domain: Option<TopicClassification>,
}

#[derive(Debug, Deserialize)]
pub struct TopicClassification {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Location {
    #[serde(default)]
    pub is_oa: Option<bool>,
    #[serde(default)]
    pub landing_page_url: Option<String>,
    #[serde(default)]
    pub pdf_url: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub source: Option<Source>,
}

#[derive(Debug, Deserialize)]
pub struct Source {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub issn_l: Option<String>,
    #[serde(default)]
    pub issn: Option<Vec<String>>,
    #[serde(default)]
    pub host_organization: Option<String>,
    #[serde(rename = "type", default)]
    pub source_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Authorship {
    #[serde(default)]
    pub author_position: Option<String>,
    #[serde(default)]
    pub author: Option<AuthorRef>,
    #[serde(default)]
    pub institutions: Vec<InstitutionRef>,
    #[serde(default)]
    pub is_corresponding: Option<bool>,
    #[serde(default)]
    pub raw_author_name: Option<String>,
    #[serde(default)]
    pub countries: Vec<String>,
    #[serde(default)]
    pub raw_affiliation_strings: Vec<String>,
    #[serde(default)]
    pub affiliations: Vec<Affiliation>,
}

#[derive(Debug, Deserialize)]
pub struct AuthorRef {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub orcid: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct InstitutionRef {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub ror: Option<String>,
    #[serde(default)]
    pub country_code: Option<String>,
    #[serde(rename = "type", default)]
    pub institution_type: Option<String>,
    #[serde(default)]
    pub lineage: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Affiliation {
    #[serde(default)]
    pub raw_affiliation_string: Option<String>,
    #[serde(default)]
    pub institution_ids: Vec<Option<String>>,
}

#[derive(Debug, Deserialize)]
pub struct WorkIds {
    #[serde(default, deserialize_with = "mag_to_string")]
    pub mag: Option<String>,
    #[serde(default)]
    pub pmid: Option<String>,
    #[serde(default)]
    pub pmcid: Option<String>,
}

fn mag_to_string<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<String>, D::Error> {
    let v: Option<sonic_rs::Value> = Option::deserialize(d)?;
    Ok(v.and_then(|v| {
        if v.is_str() {
            Some(v.as_str().unwrap().to_string())
        } else if v.is_number() {
            Some(v.to_string())
        } else {
            None
        }
    }))
}

#[derive(Debug, Deserialize)]
pub struct Keyword {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct MeshTerm {
    #[serde(default)]
    pub descriptor_ui: Option<String>,
    #[serde(default)]
    pub descriptor_name: Option<String>,
    #[serde(default)]
    pub qualifier_ui: Option<String>,
    #[serde(default)]
    pub qualifier_name: Option<String>,
    #[serde(default)]
    pub is_major_topic: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct Funder {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub ror: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Award {
    /// OpenAlex award/grant ID (e.g. "https://openalex.org/G1466802593")
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub funder_award_id: Option<String>,
    /// Funder OpenAlex ID (e.g. "https://openalex.org/F4320306076")
    #[serde(default, alias = "funder")]
    pub funder_id: Option<String>,
    #[serde(default)]
    pub funder_display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Sdg {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct CountByYear {
    #[serde(default)]
    pub year: Option<i32>,
    #[serde(default)]
    pub cited_by_count: Option<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_minimal_work() {
        let json = r#"{}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        assert!(w.id.is_none());
        assert!(w.title.is_none());
        assert!(w.publication_year.is_none());
        assert!(w.authorships.is_empty());
        assert!(w.topics.is_empty());
        assert!(w.keywords.is_empty());
    }

    #[test]
    fn deserialize_core_fields() {
        let json = r#"{
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1234/test",
            "title": "Test Paper",
            "publication_year": 2024,
            "type": "article",
            "language": "en",
            "cited_by_count": 42,
            "is_retracted": false
        }"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        assert_eq!(w.id.as_deref(), Some("https://openalex.org/W123"));
        assert_eq!(w.work_type.as_deref(), Some("article"));
        assert_eq!(w.publication_year, Some(2024));
        assert_eq!(w.cited_by_count, Some(42));
        assert_eq!(w.is_retracted, Some(false));
    }

    #[test]
    fn deserialize_open_access() {
        let json = r#"{
            "open_access": {
                "is_oa": true,
                "oa_status": "gold",
                "oa_url": "https://example.com/paper.pdf",
                "any_repository_has_fulltext": true
            }
        }"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        let oa = w.open_access.unwrap();
        assert_eq!(oa.is_oa, Some(true));
        assert_eq!(oa.oa_status.as_deref(), Some("gold"));
    }

    #[test]
    fn deserialize_apc_float_value() {
        let json = r#"{"apc_list": {"value": 2500.0, "currency": "USD", "value_usd": 2500}}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        let apc = w.apc_list.unwrap();
        assert_eq!(apc.value, Some(2500));
        assert_eq!(apc.currency.as_deref(), Some("USD"));
    }

    #[test]
    fn deserialize_apc_int_value() {
        let json = r#"{"apc_list": {"value": 1500, "value_usd": 1500}}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        assert_eq!(w.apc_list.unwrap().value, Some(1500));
    }

    #[test]
    fn deserialize_apc_null_value() {
        let json = r#"{"apc_list": {"value": null}}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        assert_eq!(w.apc_list.unwrap().value, None);
    }

    #[test]
    fn deserialize_mag_as_number() {
        let json = r#"{"ids": {"mag": 12345678}}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        assert_eq!(w.ids.unwrap().mag.as_deref(), Some("12345678"));
    }

    #[test]
    fn deserialize_mag_as_string() {
        let json = r#"{"ids": {"mag": "12345678"}}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        assert_eq!(w.ids.unwrap().mag.as_deref(), Some("12345678"));
    }

    #[test]
    fn deserialize_abstract_inverted_index() {
        let json = r#"{"abstract_inverted_index": {"hello": [0], "world": [1]}}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        let idx = w.abstract_inverted_index.unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx["hello"], vec![0u32]);
    }

    #[test]
    fn deserialize_authorships() {
        let json = r#"{"authorships": [{
            "author_position": "first",
            "author": {"id": "https://openalex.org/A1", "display_name": "Alice"},
            "institutions": [{"id": "https://openalex.org/I1", "type": "education"}],
            "is_corresponding": true,
            "countries": ["US"]
        }]}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        assert_eq!(w.authorships.len(), 1);
        let a = &w.authorships[0];
        assert_eq!(a.author_position.as_deref(), Some("first"));
        assert_eq!(a.is_corresponding, Some(true));
        assert_eq!(
            a.institutions[0].institution_type.as_deref(),
            Some("education")
        );
    }

    #[test]
    fn deserialize_topic_hierarchy() {
        let json = r#"{"primary_topic": {
            "id": "https://openalex.org/T1",
            "display_name": "ML",
            "score": 0.95,
            "subfield": {"id": "SF1", "display_name": "AI"},
            "field": {"id": "F1", "display_name": "CS"},
            "domain": {"id": "D1", "display_name": "Physical Sciences"}
        }}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        let t = w.primary_topic.unwrap();
        assert_eq!(t.score, Some(0.95));
        assert_eq!(t.subfield.unwrap().display_name.as_deref(), Some("AI"));
    }

    #[test]
    fn deserialize_location_with_source() {
        let json = r#"{"primary_location": {
            "is_oa": true,
            "landing_page_url": "https://example.com",
            "source": {
                "id": "https://openalex.org/S1",
                "display_name": "Nature",
                "issn": ["1234-5678"],
                "type": "journal"
            }
        }}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        let loc = w.primary_location.unwrap();
        assert_eq!(loc.is_oa, Some(true));
        let src = loc.source.unwrap();
        assert_eq!(src.source_type.as_deref(), Some("journal"));
        assert_eq!(src.issn.unwrap(), vec!["1234-5678"]);
    }

    #[test]
    fn deserialize_award_with_funder_alias() {
        let json = r#"{"awards": [{"id": "G1", "funder": "F1", "funder_award_id": "R01"}]}"#;
        let w: WorkRow = sonic_rs::from_str(json).unwrap();
        let award = &w.awards[0];
        assert_eq!(award.funder_id.as_deref(), Some("F1"));
        assert_eq!(award.funder_award_id.as_deref(), Some("R01"));
    }
}
