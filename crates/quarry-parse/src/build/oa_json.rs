//! OpenAlex JSONL line parser.
//!
//! Parses a single newline-delimited JSON line from the OpenAlex works dump
//! into intermediate Rust structs. Transformation logic mirrors
//! `quarry/assets/load.py:494-530`.

use std::sync::Arc;

use serde::Deserialize;

use quarry_core::{abstract_recon, normalize};

// ── URL prefixes (stripped during transformation) ──

const OA_PREFIX: &str = "https://openalex.org/";
const OA_W_PREFIX: &str = "https://openalex.org/W";
const PMID_PREFIX: &str = "https://pubmed.ncbi.nlm.nih.gov/";

// ── Public intermediate structs ──

/// Tier classification for an OpenAlex work.
///
/// T1: pmid + abstract — full enrichment, embedding-ready
/// T2: abstract only (no pmid) — embedding-ready, OA metadata only
/// T3: pmid only (no abstract) — PubMed metadata serving (MeSH, grants, authors)
/// T4: neither pmid nor abstract — citation graph node only
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    T1,
    T2,
    T3,
    T4,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::T1 => "t1",
            Tier::T2 => "t2",
            Tier::T3 => "t3",
            Tier::T4 => "t4",
        }
    }
}

pub struct OaWork {
    pub work_id: Arc<str>,
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
    pub language: Option<String>,
    pub fwci: Option<f32>,
    pub citation_normalized_percentile: Option<f32>,
    pub cited_by_percentile_year_min: Option<i16>,
    pub cited_by_percentile_year_max: Option<i16>,
}

pub struct OaAuthor {
    pub work_id: Arc<str>,
    pub author_id: Option<String>,
    pub author_position: i16,
    pub display_name: Option<String>,
    pub orcid: Option<String>,
    pub institution_name: Option<String>,
    pub institution_ror: Option<String>,
    pub raw_affiliation: Option<String>,
}

pub struct OaTopic {
    pub work_id: Arc<str>,
    pub topic_id: String,
    pub topic_name: Option<String>,
    pub subfield: Option<String>,
    pub field: Option<String>,
    pub domain: Option<String>,
    pub score: Option<f32>,
    pub is_primary: bool,
}

pub struct OaCitation {
    pub citing_id: i64,
    pub cited_id: i64,
}

pub struct OaCrosswalk {
    pub work_id: Arc<str>,
    pub pmid: i32,
}

pub struct OaCountByYear {
    pub work_id: Arc<str>,
    pub year: i16,
    pub cited_by_count: i32,
    pub updated_date: Option<String>,
}

/// Result of parsing a single JSON line.
pub struct OaLineResult {
    pub work: OaWork,
    pub authors: Vec<OaAuthor>,
    pub topics: Vec<OaTopic>,
    pub citations: Vec<OaCitation>,
    pub crosswalk: Option<OaCrosswalk>,
    pub counts_by_year: Vec<OaCountByYear>,
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
    is_paratext: Option<bool>,
    language: Option<String>,
    fwci: Option<f32>,
    citation_normalized_percentile: Option<RawCitationPercentile>,
    cited_by_percentile_year: Option<RawCitedByPercentileYear>,
    counts_by_year: Option<Vec<RawCountByYear>>,
    primary_location: Option<RawPrimaryLocation>,
    open_access: Option<RawOpenAccess>,
    authorships: Option<Vec<RawAuthorship>>,
    topics: Option<Vec<RawTopic>>,
    referenced_works: Option<Vec<String>>,
    updated_date: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawCitationPercentile {
    value: Option<f32>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawCitedByPercentileYear {
    min: Option<i16>,
    max: Option<i16>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct RawCountByYear {
    year: Option<i16>,
    cited_by_count: Option<i32>,
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
    id: Option<String>,
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
/// Returns `Ok(None)` for intentionally skipped records (paratext).
/// Returns `Err` for actual parse failures.
pub fn parse_line(
    line: &str,
) -> Result<Option<OaLineResult>, Box<dyn std::error::Error>> {
    let raw: RawWork = sonic_rs::from_str(line)?;

    if raw.id.is_empty() {
        return Err("work missing id".into());
    }

    // Skip paratext (ToC, errata, ads) — no scholarly value
    if raw.is_paratext.unwrap_or(false) {
        return Ok(None);
    }

    // work_id: strip "https://openalex.org/" prefix
    let work_id: Arc<str> = Arc::from(raw.id.strip_prefix(OA_PREFIX).unwrap_or(&raw.id));

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

    // Abstract reconstruction + normalization
    // HTML strip + whitespace collapse here so downstream (CH/PG/embeddings) gets clean text.
    // Must happen before tier classification — empty/HTML-only inverted indices
    // reconstruct to None and should not count as "has abstract".
    let abstract_text = raw
        .abstract_inverted_index
        .as_ref()
        .and_then(|v| {
            let json_str = sonic_rs::to_string(v).ok()?;
            let text = abstract_recon::reconstruct_one(&json_str);
            if text.is_empty() {
                None
            } else {
                let normalized = normalize::normalize_one(&text);
                if normalized.is_empty() { None } else { Some(normalized) }
            }
        });

    // Tier classification: pmid × abstract presence (post-reconstruction)
    let tier = match (pmid.is_some(), abstract_text.is_some()) {
        (true, true) => Tier::T1,   // pmid + abstract: full enrichment
        (false, true) => Tier::T2,  // abstract only: embedding-ready
        (true, false) => Tier::T3,  // pmid only: PubMed metadata
        (false, false) => Tier::T4, // neither: graph node only
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

    let citation_normalized_percentile = raw
        .citation_normalized_percentile
        .and_then(|c| c.value);
    let (cited_by_pct_min, cited_by_pct_max) = match raw.cited_by_percentile_year {
        Some(p) => (p.min, p.max),
        None => (None, None),
    };

    let updated_date = raw.updated_date;

    let work = OaWork {
        work_id: Arc::clone(&work_id),
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
        updated_date: updated_date.clone(),
        language: raw.language,
        fwci: raw.fwci,
        citation_normalized_percentile,
        cited_by_percentile_year_min: cited_by_pct_min,
        cited_by_percentile_year_max: cited_by_pct_max,
    };

    // Authors — T1/T2/T3 (skip T4 only)
    let authors = if tier <= Tier::T3 {
        raw.authorships
            .unwrap_or_default()
            .into_iter()
            .enumerate()
            .map(|(i, a)| {
                let (author_id, display_name, orcid) = match a.author {
                    Some(au) => (au.id, au.display_name, au.orcid),
                    None => (None, None, None),
                };
                let (inst_name, inst_ror) = a
                    .institutions
                    .and_then(|mut v| if v.is_empty() { None } else { Some(v.remove(0)) })
                    .map(|inst| (inst.display_name, inst.ror))
                    .unwrap_or((None, None));

                OaAuthor {
                    work_id: Arc::clone(&work_id),
                    author_id,
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

    // Topics — T1/T2/T3 (skip T4 only). First topic = is_primary.
    let topics = if tier <= Tier::T3 {
        let mut is_first = true;
        raw.topics
            .unwrap_or_default()
            .into_iter()
            .filter_map(|t| {
                let topic_id = t
                    .id
                    .as_deref()
                    .map(|s| s.strip_prefix(OA_PREFIX).unwrap_or(s).to_string())?;
                let primary = is_first;
                is_first = false;
                Some(OaTopic {
                    work_id: Arc::clone(&work_id),
                    topic_id,
                    topic_name: t.display_name,
                    subfield: t.subfield.and_then(|s| s.display_name),
                    field: t.field.and_then(|f| f.display_name),
                    domain: t.domain.and_then(|d| d.display_name),
                    score: t.score,
                    is_primary: primary,
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
        work_id: Arc::clone(&work_id),
        pmid: p,
    });


    // Counts by year — T1/T2/T3 only (T4 has no metadata value)
    let counts_by_year: Vec<OaCountByYear> = if tier <= Tier::T3 {
        raw.counts_by_year
            .unwrap_or_default()
            .into_iter()
            .filter_map(|c| {
                let year = c.year?;
                if year <= 0 {
                    return None;
                }
                Some(OaCountByYear {
                    work_id: Arc::clone(&work_id),
                    year,
                    cited_by_count: c.cited_by_count.unwrap_or(0),
                    updated_date: updated_date.clone(),
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(Some(OaLineResult {
        work,
        authors,
        topics,
        citations,
        crosswalk,
        counts_by_year,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t1_pmid_and_abstract() {
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
            "language": "en",
            "fwci": 8.02,
            "citation_normalized_percentile": {"value": 0.984, "is_in_top_1_percent": false, "is_in_top_10_percent": true},
            "cited_by_percentile_year": {"min": 98, "max": 100},
            "primary_location": {"source": {"display_name": "Nature"}},
            "open_access": {"oa_status": "gold", "oa_url": "https://example.com"},
            "authorships": [
                {"author": {"id": "https://openalex.org/A5001", "display_name": "Alice", "orcid": "https://orcid.org/0000-0001-2345-6789"},
                 "institutions": [{"display_name": "MIT", "ror": "https://ror.org/042nb2s44"}],
                 "raw_affiliation_strings": ["MIT, Cambridge"]}
            ],
            "topics": [
                {"id": "https://openalex.org/T001", "display_name": "ML",
                 "subfield": {"display_name": "AI"}, "field": {"display_name": "CS"},
                 "domain": {"display_name": "Physical Sciences"}, "score": 0.95},
                {"id": "https://openalex.org/T002", "display_name": "DL",
                 "subfield": {"display_name": "AI"}, "field": {"display_name": "CS"},
                 "domain": {"display_name": "Physical Sciences"}, "score": 0.80}
            ],
            "referenced_works": ["https://openalex.org/W111", "https://openalex.org/W222"],
            "counts_by_year": [
                {"year": 2024, "cited_by_count": 5},
                {"year": 2023, "cited_by_count": 3}
            ],
            "updated_date": "2024-07-01"
        }"#;

        let result = parse_line(json).unwrap().unwrap();
        let w = &result.work;
        assert_eq!(&*w.work_id, "W12345");
        assert_eq!(w.work_id_int, 12345);
        assert_eq!(w.tier, Tier::T1);
        assert_eq!(w.pmid, Some(99999));
        assert_eq!(w.title, "Test Paper");
        assert_eq!(w.abstract_text.as_deref(), Some("hello world"));
        assert_eq!(w.host_venue.as_deref(), Some("Nature"));
        assert_eq!(w.oa_status.as_deref(), Some("gold"));
        assert_eq!(w.language.as_deref(), Some("en"));
        assert!((w.fwci.unwrap() - 8.02).abs() < 0.01);
        assert!((w.citation_normalized_percentile.unwrap() - 0.984).abs() < 0.001);
        assert_eq!(w.cited_by_percentile_year_min, Some(98));
        assert_eq!(w.cited_by_percentile_year_max, Some(100));

        assert_eq!(result.counts_by_year.len(), 2);
        assert_eq!(result.counts_by_year[0].year, 2024);
        assert_eq!(result.counts_by_year[0].cited_by_count, 5);
        assert_eq!(&*result.counts_by_year[1].work_id, "W12345");

        assert_eq!(result.authors.len(), 1);
        assert_eq!(result.authors[0].author_id.as_deref(), Some("https://openalex.org/A5001"));
        assert_eq!(result.authors[0].display_name.as_deref(), Some("Alice"));
        assert_eq!(result.authors[0].author_position, 1);
        assert_eq!(
            result.authors[0].institution_name.as_deref(),
            Some("MIT")
        );

        assert_eq!(result.topics.len(), 2);
        assert_eq!(result.topics[0].topic_id, "T001");
        assert!(result.topics[0].is_primary);
        assert!(!result.topics[1].is_primary);

        assert_eq!(result.citations.len(), 2);
        assert!(result.crosswalk.is_some());
        assert_eq!(result.crosswalk.unwrap().pmid, 99999);
    }

    #[test]
    fn test_abstract_html_stripped() {
        let json = r#"{
            "id": "https://openalex.org/W11111",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/88888"},
            "title": "HTML test",
            "abstract_inverted_index": {"the": [0], "<em>important</em>": [1], "result": [2], "(p": [3], "<": [4], "0.05)": [5]}
        }"#;

        let result = parse_line(json).unwrap().unwrap();
        let abs = result.work.abstract_text.as_deref().unwrap();
        // HTML tags stripped, math inequality preserved
        assert_eq!(abs, "the important result (p < 0.05)");
    }

    #[test]
    fn test_abstract_whitespace_collapsed() {
        let json = r#"{
            "id": "https://openalex.org/W22222",
            "title": "Whitespace test",
            "abstract_inverted_index": {"hello": [0], "world": [2]}
        }"#;

        let result = parse_line(json).unwrap().unwrap();
        let abs = result.work.abstract_text.as_deref().unwrap();
        // Gap at position 1 produces extra space, collapsed to single space
        assert!(!abs.contains("  "), "should not have double spaces: {abs}");
    }

    #[test]
    fn test_t2_abstract_no_pmid() {
        let json = r#"{
            "id": "https://openalex.org/W55555",
            "title": "T2 Paper",
            "abstract_inverted_index": {"some": [0], "text": [1]},
            "topics": [
                {"id": "https://openalex.org/T003", "display_name": "Bio", "score": 0.9}
            ]
        }"#;

        let result = parse_line(json).unwrap().unwrap();
        assert_eq!(result.work.tier, Tier::T2);
        assert_eq!(result.work.abstract_text.as_deref(), Some("some text"));
        assert!(result.work.pmid.is_none());
        assert_eq!(result.topics.len(), 1);
        assert!(result.topics[0].is_primary);
    }

    #[test]
    fn test_t3_pmid_no_abstract() {
        let json = r#"{
            "id": "https://openalex.org/W66666",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/22222"},
            "title": "T3 Paper",
            "authorships": [{"author": {"display_name": "Bob"}}],
            "topics": [{"id": "https://openalex.org/T002", "display_name": "Soc"}],
            "referenced_works": ["https://openalex.org/W333"]
        }"#;

        let result = parse_line(json).unwrap().unwrap();
        assert_eq!(result.work.tier, Tier::T3);
        assert!(result.work.abstract_text.is_none());
        assert_eq!(result.work.pmid, Some(22222));
        assert_eq!(result.authors.len(), 1); // T3 parses authors
        assert_eq!(result.topics.len(), 1);  // T3 parses topics
        assert_eq!(result.citations.len(), 1);
    }

    #[test]
    fn test_t4_no_pmid_no_abstract() {
        let json = r#"{
            "id": "https://openalex.org/W99999",
            "title": "T4 Paper",
            "authorships": [{"author": {"display_name": "Carol"}}],
            "topics": [{"id": "https://openalex.org/T004", "display_name": "Math"}],
            "referenced_works": ["https://openalex.org/W333"]
        }"#;

        let result = parse_line(json).unwrap().unwrap();
        assert_eq!(result.work.tier, Tier::T4);
        assert!(result.work.abstract_text.is_none());
        assert!(result.authors.is_empty());  // T4 skips authors
        assert!(result.topics.is_empty());   // T4 skips topics
        assert_eq!(result.citations.len(), 1); // T4 keeps citations
    }

    #[test]
    fn test_empty_id_returns_error() {
        let json = r#"{"id": "", "title": "Bad"}"#;
        assert!(parse_line(json).is_err());
    }

    #[test]
    fn test_invalid_work_id_int() {
        let json = r#"{"id": "https://openalex.org/Wabc"}"#;
        assert!(parse_line(json).is_err());
    }

    #[test]
    fn test_malformed_pmid_url() {
        let json = r#"{
            "id": "https://openalex.org/W77777",
            "ids": {"pmid": "urn:pmid:12345"},
            "title": "No PMID"
        }"#;
        let result = parse_line(json).unwrap().unwrap();
        assert!(result.work.pmid.is_none());
        assert!(result.crosswalk.is_none());
    }

    #[test]
    fn test_topic_without_id_filtered() {
        let json = r#"{
            "id": "https://openalex.org/W88888",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/11111"},
            "title": "Topic test",
            "abstract_inverted_index": {"x": [0]},
            "topics": [
                {"display_name": "No ID topic", "score": 0.5},
                {"id": "https://openalex.org/T999", "display_name": "Has ID", "score": 0.8}
            ]
        }"#;
        let result = parse_line(json).unwrap().unwrap();
        // Topic without id is filtered out, but is_primary tracks original position
        assert_eq!(result.topics.len(), 1);
        assert_eq!(result.topics[0].topic_id, "T999");
        // First valid topic gets is_primary (first raw topic had no id, skipped)
        assert!(result.topics[0].is_primary);
    }

    #[test]
    fn test_invalid_referenced_work_filtered() {
        let json = r#"{
            "id": "https://openalex.org/W44444",
            "title": "Citation test",
            "referenced_works": [
                "https://openalex.org/W111",
                "https://example.com/not-oa",
                "https://openalex.org/W222"
            ]
        }"#;
        let result = parse_line(json).unwrap().unwrap();
        assert_eq!(result.citations.len(), 2);
    }

    #[test]
    fn test_empty_abstract_inverted_index_no_hash() {
        let json = r#"{
            "id": "https://openalex.org/W33333",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/44444"},
            "title": "Empty abstract map",
            "abstract_inverted_index": {}
        }"#;
        let result = parse_line(json).unwrap().unwrap();
        // Empty inverted index reconstructs to "" → filtered to None
        assert!(result.work.abstract_text.is_none());
    }

    #[test]
    fn test_html_only_abstract_no_hash() {
        let json = r#"{
            "id": "https://openalex.org/W44444",
            "title": "HTML-only abstract",
            "abstract_inverted_index": {"<br/>": [0]}
        }"#;
        let result = parse_line(json).unwrap().unwrap();
        // HTML tag stripped → empty string → None
        assert!(result.work.abstract_text.is_none());
    }

    #[test]
    fn test_empty_title_with_abstract_has_hash() {
        let json = r#"{
            "id": "https://openalex.org/W55556",
            "title": "",
            "abstract_inverted_index": {"some": [0], "text": [1]}
        }"#;
        let result = parse_line(json).unwrap().unwrap();
        assert_eq!(result.work.title, "");
        assert_eq!(result.work.abstract_text.as_deref(), Some("some text"));
    }

    #[test]
    fn test_paratext_skipped() {
        let json = r#"{
            "id": "https://openalex.org/W10001",
            "title": "Table of Contents",
            "is_paratext": true
        }"#;
        // Paratext returns Ok(None), not Err
        assert!(parse_line(json).unwrap().is_none());
    }

    #[test]
    fn test_new_fields_null_when_missing() {
        let json = r#"{
            "id": "https://openalex.org/W10002",
            "title": "Minimal work"
        }"#;
        let result = parse_line(json).unwrap().unwrap();
        let w = &result.work;
        assert!(w.language.is_none());
        assert!(w.fwci.is_none());
        assert!(w.citation_normalized_percentile.is_none());
        assert!(w.cited_by_percentile_year_min.is_none());
        assert!(w.cited_by_percentile_year_max.is_none());
    }

    #[test]
    fn test_counts_by_year_empty_when_missing() {
        let json = r#"{
            "id": "https://openalex.org/W10004",
            "title": "No counts"
        }"#;
        let result = parse_line(json).unwrap().unwrap();
        assert!(result.counts_by_year.is_empty());
    }

    #[test]
    fn test_counts_by_year_skips_null_year() {
        let json = r#"{
            "id": "https://openalex.org/W10005",
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/11111"},
            "title": "Partial counts",
            "abstract_inverted_index": {"x": [0]},
            "counts_by_year": [
                {"year": 2024, "cited_by_count": 3},
                {"cited_by_count": 1},
                {"year": 2022}
            ]
        }"#;
        let result = parse_line(json).unwrap().unwrap();
        // null year entry skipped, null cited_by_count defaults to 0
        assert_eq!(result.counts_by_year.len(), 2);
        assert_eq!(result.counts_by_year[0].cited_by_count, 3);
        assert_eq!(result.counts_by_year[1].year, 2022);
        assert_eq!(result.counts_by_year[1].cited_by_count, 0);
    }

    #[test]
    fn test_citation_percentile_flatten() {
        let json = r#"{
            "id": "https://openalex.org/W10003",
            "title": "Percentile test",
            "citation_normalized_percentile": {"value": 0.75, "is_in_top_1_percent": false, "is_in_top_10_percent": true},
            "cited_by_percentile_year": {"min": 50, "max": 60}
        }"#;
        let w = &parse_line(json).unwrap().unwrap().work;
        assert!((w.citation_normalized_percentile.unwrap() - 0.75).abs() < 0.001);
        assert_eq!(w.cited_by_percentile_year_min, Some(50));
        assert_eq!(w.cited_by_percentile_year_max, Some(60));
    }
}
