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
