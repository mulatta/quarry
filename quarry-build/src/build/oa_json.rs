//! OpenAlex JSONL line parser.
//!
//! Parses a single newline-delimited JSON line from the OpenAlex works dump
//! into intermediate Rust structs. Transformation logic mirrors
//! `quarry/assets/load.py:494-530`.

use std::collections::HashSet;

use serde::Deserialize;

use quarry_core::abstract_recon;

// ── URL prefixes (stripped during transformation) ──

const OA_PREFIX: &str = "https://openalex.org/";
const OA_W_PREFIX: &str = "https://openalex.org/W";
const PMID_PREFIX: &str = "https://pubmed.ncbi.nlm.nih.gov/";

// ── Public intermediate structs ──

/// Tier classification for an OpenAlex work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    T1, // has PMID
    T2, // domain in t2_domains
    T3, // everything else
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::T1 => "t1",
            Tier::T2 => "t2",
            Tier::T3 => "t3",
        }
    }
}

pub struct OaWork {
    pub work_id: String,
    pub work_id_int: i64,
    pub tier: Tier,
    pub pmid: Option<i32>,
    pub doi: Option<String>,
    pub title: String,
    pub abstract_text: Option<String>,
    pub pub_year: Option<i16>,
    pub pub_date: Option<String>,
    pub work_type: Option<String>,
    pub cited_by_count: Option<i32>,
    pub host_venue: Option<String>,
    pub oa_status: Option<String>,
    pub oa_url: Option<String>,
    pub is_retracted: bool,
    pub updated_date: Option<String>,
    pub pm_journal_abbr: Option<String>,
    pub pm_country: Option<String>,
    pub pm_medline_status: Option<String>,
    pub pm_pub_type: Vec<String>,
    pub pm_created_date: Option<String>,
    pub pm_revised_date: Option<String>,
    pub pm_indexed_date: Option<String>,
}

pub struct OaAuthor {
    pub work_id: String,
    pub author_position: i16,
    pub display_name: Option<String>,
    pub orcid: Option<String>,
    pub institution_name: Option<String>,
    pub institution_ror: Option<String>,
    pub raw_affiliation: Option<String>,
}

pub struct OaTopic {
    pub work_id: String,
    pub topic_id: String,
    pub topic_name: Option<String>,
    pub subfield: Option<String>,
    pub field: Option<String>,
    pub domain: Option<String>,
    pub score: Option<f32>,
}

pub struct OaCitation {
    pub citing_id: i64,
    pub cited_id: i64,
}

pub struct OaCrosswalk {
    pub work_id: String,
    pub pmid: i32,
}

/// Result of parsing a single JSON line.
pub struct OaLineResult {
    pub work: OaWork,
    pub authors: Vec<OaAuthor>,
    pub topics: Vec<OaTopic>,
    pub citations: Vec<OaCitation>,
    pub crosswalk: Option<OaCrosswalk>,
}

// ── Serde JSON structs ──

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawWork {
    id: String,
    ids: Option<RawIds>,
    title: Option<String>,
    abstract_inverted_index: Option<sonic_rs::Value>,
    publication_year: Option<i16>,
    publication_date: Option<String>,
    #[serde(rename = "type")]
    work_type: Option<String>,
    cited_by_count: Option<i32>,
    is_retracted: Option<bool>,
    primary_location: Option<RawPrimaryLocation>,
    open_access: Option<RawOpenAccess>,
    primary_topic: Option<RawPrimaryTopic>,
    authorships: Option<Vec<RawAuthorship>>,
    topics: Option<Vec<RawTopic>>,
    referenced_works: Option<Vec<String>>,
    updated_date: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawIds {
    pmid: Option<String>,
    doi: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawPrimaryLocation {
    source: Option<RawDisplayName>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawOpenAccess {
    oa_status: Option<String>,
    oa_url: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawPrimaryTopic {
    domain: Option<RawDisplayName>,
}

/// Reusable struct for nested objects with only `display_name`.
#[derive(Deserialize, Default)]
#[serde(default)]
struct RawDisplayName {
    display_name: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawAuthorship {
    author: Option<RawAuthor>,
    institutions: Option<Vec<RawInstitution>>,
    raw_affiliation_strings: Option<Vec<String>>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawAuthor {
    display_name: Option<String>,
    orcid: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawInstitution {
    display_name: Option<String>,
    ror: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawTopic {
    id: Option<String>,
    display_name: Option<String>,
    subfield: Option<RawDisplayName>,
    field: Option<RawDisplayName>,
    domain: Option<RawDisplayName>,
    score: Option<f32>,
}

// ── Parsing ──

/// Parse a single OA JSON line into structured data.
///
/// Mirrors the SQL transformations in `load.py:494-530` (works),
/// `537-551` (authors), `558-572` (topics), `579-586` (citations).
pub fn parse_line(
    line: &str,
    t2_domains: &HashSet<String>,
) -> Result<OaLineResult, Box<dyn std::error::Error>> {
    let raw: RawWork = sonic_rs::from_str(line)?;

    if raw.id.is_empty() {
        return Err("work missing id".into());
    }

    // work_id: strip "https://openalex.org/" prefix
    let work_id = raw.id.strip_prefix(OA_PREFIX).unwrap_or(&raw.id).to_string();

    // work_id_int: strip "https://openalex.org/W" and parse as i64
    let work_id_int: i64 = raw
        .id
        .strip_prefix(OA_W_PREFIX)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("cannot parse work_id_int from: {}", raw.id))?;

    // PMID: strip URL prefix, parse as i32
    let pmid: Option<i32> = raw
        .ids
        .as_ref()
        .and_then(|ids| ids.pmid.as_ref())
        .and_then(|url| url.strip_prefix(PMID_PREFIX))
        .and_then(|s| s.parse().ok());

    let doi = raw.ids.and_then(|ids| ids.doi);

    // Domain for tier classification
    let domain_name: Option<&str> = raw
        .primary_topic
        .as_ref()
        .and_then(|pt| pt.domain.as_ref())
        .and_then(|d| d.display_name.as_deref());

    let tier = if pmid.is_some() {
        Tier::T1
    } else if domain_name.is_some_and(|d| t2_domains.contains(d)) {
        Tier::T2
    } else {
        Tier::T3
    };

    // Abstract reconstruction — T1/T2 only, skip for T3
    let abstract_text = if tier <= Tier::T2 {
        raw.abstract_inverted_index
            .as_ref()
            .and_then(|v| {
                let json_str = sonic_rs::to_string(v).ok()?;
                let text = abstract_recon::reconstruct_one(&json_str);
                if text.is_empty() { None } else { Some(text) }
            })
    } else {
        None
    };

    let title = raw.title.unwrap_or_default();
    let host_venue = raw
        .primary_location
        .and_then(|loc| loc.source)
        .and_then(|src| src.display_name);
    let (oa_status, oa_url) = match raw.open_access {
        Some(oa) => (oa.oa_status, oa.oa_url),
        None => (None, None),
    };

    let work = OaWork {
        work_id: work_id.clone(),
        work_id_int,
        tier,
        pmid,
        doi,
        title,
        abstract_text,
        pub_year: raw.publication_year,
        pub_date: raw.publication_date,
        work_type: raw.work_type,
        cited_by_count: raw.cited_by_count,
        host_venue,
        oa_status,
        oa_url,
        is_retracted: raw.is_retracted.unwrap_or(false),
        updated_date: raw.updated_date,
        pm_journal_abbr: None,
        pm_country: None,
        pm_medline_status: None,
        pm_pub_type: Vec::new(),
        pm_created_date: None,
        pm_revised_date: None,
        pm_indexed_date: None,
    };

    // Authors — T1/T2 only (mirrors load.py:537-551)
    let authors = if tier <= Tier::T2 {
        raw.authorships
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(i, a)| {
                let (display_name, orcid) = match a.author {
                    Some(au) => (au.display_name, au.orcid),
                    None => (None, None),
                };
                let (inst_name, inst_ror) = a
                    .institutions
                    .and_then(|mut v| if v.is_empty() { None } else { Some(v.remove(0)) })
                    .map(|inst| (inst.display_name, inst.ror))
                    .unwrap_or((None, None));

                OaAuthor {
                    work_id: work_id.clone(),
                    author_position: i16::try_from(i + 1).unwrap_or(i16::MAX),
                    display_name,
                    orcid,
                    institution_name: inst_name,
                    institution_ror: inst_ror,
                    raw_affiliation: a
                        .raw_affiliation_strings
                        .and_then(|v| v.into_iter().next()),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // Topics — T1/T2 only (mirrors load.py:558-572)
    let topics = if tier <= Tier::T2 {
        raw.topics
            .unwrap_or_default()
            .into_iter()
            .filter_map(|t| {
                let topic_id = t
                    .id
                    .as_deref()
                    .map(|s| s.strip_prefix(OA_PREFIX).unwrap_or(s).to_string())?;
                Some(OaTopic {
                    work_id: work_id.clone(),
                    topic_id,
                    topic_name: t.display_name,
                    subfield: t.subfield.and_then(|s| s.display_name),
                    field: t.field.and_then(|f| f.display_name),
                    domain: t.domain.and_then(|d| d.display_name),
                    score: t.score,
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    // Citations — all tiers (mirrors load.py:579-586)
    let citations: Vec<OaCitation> = raw
        .referenced_works
        .unwrap_or_default()
        .iter()
        .filter_map(|ref_url| {
            let cited_id: i64 = ref_url.strip_prefix(OA_W_PREFIX)?.parse().ok()?;
            Some(OaCitation {
                citing_id: work_id_int,
                cited_id,
            })
        })
        .collect();

    // Crosswalk — only if PMID exists
    let crosswalk = pmid.map(|p| OaCrosswalk {
        work_id: work_id.clone(),
        pmid: p,
    });

    Ok(OaLineResult {
        work,
        authors,
        topics,
        citations,
        crosswalk,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t2_set() -> HashSet<String> {
        ["Health Sciences", "Life Sciences"]
            .into_iter()
            .map(String::from)
            .collect()
    }

    #[test]
    fn test_t1_work_with_pmid() {
        let json = r#"{
            "id": "https://openalex.org/W12345",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/99999"},
            "title": "Test Paper",
            "abstract_inverted_index": {"hello": [0], "world": [1]},
            "publication_year": 2024,
            "publication_date": "2024-06-15",
            "type": "article",
            "cited_by_count": 10,
            "is_retracted": false,
            "primary_location": {"source": {"display_name": "Nature"}},
            "open_access": {"oa_status": "gold", "oa_url": "https://example.com"},
            "primary_topic": {"domain": {"display_name": "Health Sciences"}},
            "authorships": [
                {"author": {"display_name": "Alice", "orcid": "https://orcid.org/0000-0001-2345-6789"},
                 "institutions": [{"display_name": "MIT", "ror": "https://ror.org/042nb2s44"}],
                 "raw_affiliation_strings": ["MIT, Cambridge"]}
            ],
            "topics": [
                {"id": "https://openalex.org/T001", "display_name": "ML",
                 "subfield": {"display_name": "AI"}, "field": {"display_name": "CS"},
                 "domain": {"display_name": "Physical Sciences"}, "score": 0.95}
            ],
            "referenced_works": ["https://openalex.org/W111", "https://openalex.org/W222"],
            "updated_date": "2024-07-01"
        }"#;

        let result = parse_line(json, &t2_set()).unwrap();
        let w = &result.work;
        assert_eq!(w.work_id, "W12345");
        assert_eq!(w.work_id_int, 12345);
        assert_eq!(w.tier, Tier::T1);
        assert_eq!(w.pmid, Some(99999));
        assert_eq!(w.title, "Test Paper");
        assert_eq!(w.abstract_text.as_deref(), Some("hello world"));
        assert_eq!(w.host_venue.as_deref(), Some("Nature"));
        assert_eq!(w.oa_status.as_deref(), Some("gold"));

        assert_eq!(result.authors.len(), 1);
        assert_eq!(result.authors[0].display_name.as_deref(), Some("Alice"));
        assert_eq!(result.authors[0].author_position, 1);
        assert_eq!(
            result.authors[0].institution_name.as_deref(),
            Some("MIT")
        );

        assert_eq!(result.topics.len(), 1);
        assert_eq!(result.topics[0].topic_id, "T001");

        assert_eq!(result.citations.len(), 2);
        assert_eq!(result.citations[0].citing_id, 12345);
        assert_eq!(result.citations[0].cited_id, 111);
        assert_eq!(result.citations[1].cited_id, 222);

        assert!(result.crosswalk.is_some());
        assert_eq!(result.crosswalk.unwrap().pmid, 99999);
    }

    #[test]
    fn test_t3_skips_authors_topics_abstract() {
        let json = r#"{
            "id": "https://openalex.org/W99999",
            "title": "T3 Paper",
            "abstract_inverted_index": {"should": [0], "skip": [1]},
            "primary_topic": {"domain": {"display_name": "Social Sciences"}},
            "authorships": [{"author": {"display_name": "Bob"}}],
            "topics": [{"id": "https://openalex.org/T002", "display_name": "Soc"}],
            "referenced_works": ["https://openalex.org/W333"]
        }"#;

        let result = parse_line(json, &t2_set()).unwrap();
        assert_eq!(result.work.tier, Tier::T3);
        assert!(result.work.abstract_text.is_none()); // skipped
        assert!(result.authors.is_empty()); // skipped
        assert!(result.topics.is_empty()); // skipped
        assert_eq!(result.citations.len(), 1); // kept
    }

    #[test]
    fn test_t2_domain_match() {
        let json = r#"{
            "id": "https://openalex.org/W55555",
            "title": "T2",
            "primary_topic": {"domain": {"display_name": "Life Sciences"}}
        }"#;

        let result = parse_line(json, &t2_set()).unwrap();
        assert_eq!(result.work.tier, Tier::T2);
    }

}
