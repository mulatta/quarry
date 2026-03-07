//! JSON-to-Arrow record batch accumulators for OpenAlex works.
//!
//! 13-table star schema: 1 fact table (works, 84 cols) + 12 fan-out tables.
//! All complex array fields are exploded into separate tables — zero JSON columns.

mod fanout;
pub mod model;
mod works;

use std::collections::HashMap;

pub use fanout::{
    AuthorshipsAccumulator, AwardsAccumulator, CitationsAccumulator, CountsByYearAccumulator,
    FundersAccumulator, KeywordsAccumulator, LocationsAccumulator, MeshAccumulator,
    SdgsAccumulator, TopicsAccumulator, WorksKeysAccumulator,
};
pub use model::WorkRow;
pub use works::WorksAccumulator;

use model::Topic;

// ============================================================
// Helpers
// ============================================================

/// Decode abstract_inverted_index to plain text.
pub fn decode_inverted_index(index: &HashMap<String, Vec<u32>>) -> String {
    if index.is_empty() {
        return String::new();
    }
    let max_pos = index.values().flatten().copied().max().unwrap_or(0) as usize;
    let mut words: Vec<&str> = vec![""; max_pos + 1];
    for (word, positions) in index {
        for &pos in positions {
            let p = pos as usize;
            if p < words.len() {
                words[p] = word;
            }
        }
    }
    let total_len: usize = words.iter().map(|w| w.len()).sum::<usize>() + words.len();
    let mut result = String::with_capacity(total_len);
    for (i, w) in words.iter().enumerate() {
        if i > 0 {
            result.push(' ');
        }
        result.push_str(w);
    }
    result
}

/// Strip inline HTML/XML tags from text (e.g. `<italic>`, `<jats:p>`).
///
/// Simple `<...>` removal without regex dependency. Handles self-closing tags
/// and attributes. Does NOT decode HTML entities (OpenAlex data doesn't use them).
pub fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut inside_tag = false;
    for c in s.chars() {
        if c == '<' {
            inside_tag = true;
        } else if c == '>' && inside_tag {
            inside_tag = false;
        } else if !inside_tag {
            out.push(c);
        }
    }
    out
}

/// Extract bare PMID from OpenAlex URL format.
pub(crate) fn extract_pmid(url: &str) -> Option<&str> {
    url.strip_prefix("https://pubmed.ncbi.nlm.nih.gov/")
        .map(|s| s.trim_end_matches('/'))
        .filter(|s| !s.is_empty())
}

/// Extract bare PMCID from OpenAlex URL format.
pub(crate) fn extract_pmcid(url: &str) -> Option<&str> {
    url.strip_prefix("https://www.ncbi.nlm.nih.gov/pmc/articles/")
        .map(|s| s.trim_end_matches('/'))
        .filter(|s| !s.is_empty())
}

/// Strip OpenAlex URL prefix from ID.
pub fn strip_oa_prefix(url: &str) -> &str {
    url.strip_prefix("https://openalex.org/").unwrap_or(url)
}

// ============================================================
// Domain/Topic Filter
// ============================================================

/// Row-level filter for domain, topic, and language matching.
///
/// A work matches if it passes ALL active filter dimensions:
///   - language: row.language must be in the set (if non-empty)
///   - domain/topic: ANY topic belongs to a listed domain OR has a listed topic ID (if non-empty)
///
/// Empty filter matches everything.
#[derive(Debug, Clone, Default)]
pub struct Filter {
    pub domains: rustc_hash::FxHashSet<String>,
    pub topic_ids: rustc_hash::FxHashSet<String>,
    pub languages: rustc_hash::FxHashSet<String>,
    pub work_types: rustc_hash::FxHashSet<String>,
    pub require_abstract: bool,
}

impl Filter {
    pub fn is_empty(&self) -> bool {
        self.domains.is_empty()
            && self.topic_ids.is_empty()
            && self.languages.is_empty()
            && self.work_types.is_empty()
            && !self.require_abstract
    }

    /// Returns true if work passes all active filter dimensions.
    pub fn matches(&self, row: &WorkRow) -> bool {
        if self.is_empty() {
            return true;
        }

        // Language filter
        if !self.languages.is_empty() {
            match &row.language {
                Some(lang) if self.languages.contains(lang.as_str()) => {}
                _ => return false,
            }
        }

        // Work type filter
        if !self.work_types.is_empty() {
            match &row.work_type {
                Some(wt) if self.work_types.contains(wt.as_str()) => {}
                _ => return false,
            }
        }

        // Abstract requirement
        if self.require_abstract
            && row
                .abstract_inverted_index
                .as_ref()
                .is_none_or(|idx| idx.is_empty())
        {
            return false;
        }

        // Domain/topic filter
        if !self.domains.is_empty() || !self.topic_ids.is_empty() {
            let mut topic_match = false;
            if let Some(topic) = &row.primary_topic
                && self.matches_topic(topic)
            {
                topic_match = true;
            }
            if !topic_match {
                for topic in &row.topics {
                    if self.matches_topic(topic) {
                        topic_match = true;
                        break;
                    }
                }
            }
            if !topic_match {
                return false;
            }
        }

        true
    }

    fn matches_topic(&self, topic: &Topic) -> bool {
        if let Some(domain) = &topic.domain
            && let Some(name) = &domain.display_name
            && self.domains.contains(name.as_str())
        {
            return true;
        }
        if let Some(id) = &topic.id {
            let short_id = strip_oa_prefix(id);
            if self.topic_ids.contains(short_id) {
                return true;
            }
        }
        false
    }
}

// ============================================================
// Tests for helpers and filter (mod-level)
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_inverted_index_basic() {
        let mut idx = HashMap::new();
        idx.insert("Hello".to_string(), vec![0]);
        idx.insert("world".to_string(), vec![1]);
        assert_eq!(decode_inverted_index(&idx), "Hello world");
    }

    #[test]
    fn decode_inverted_index_repeated_word() {
        let mut idx = HashMap::new();
        idx.insert("the".to_string(), vec![0, 2]);
        idx.insert("cat".to_string(), vec![1]);
        idx.insert("hat".to_string(), vec![3]);
        assert_eq!(decode_inverted_index(&idx), "the cat the hat");
    }

    #[test]
    fn decode_inverted_index_empty() {
        assert_eq!(decode_inverted_index(&HashMap::new()), "");
    }

    #[test]
    fn extract_pmid_valid() {
        assert_eq!(
            extract_pmid("https://pubmed.ncbi.nlm.nih.gov/14907713"),
            Some("14907713")
        );
    }

    #[test]
    fn extract_pmid_trailing_slash() {
        assert_eq!(
            extract_pmid("https://pubmed.ncbi.nlm.nih.gov/14907713/"),
            Some("14907713")
        );
    }

    #[test]
    fn extract_pmid_no_match() {
        assert_eq!(extract_pmid("https://example.com/14907713"), None);
    }

    #[test]
    fn extract_pmcid_valid() {
        assert_eq!(
            extract_pmcid("https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123456"),
            Some("PMC123456")
        );
    }

    #[test]
    fn strip_oa_prefix_works() {
        assert_eq!(
            strip_oa_prefix("https://openalex.org/W2741809807"),
            "W2741809807"
        );
    }

    #[test]
    fn strip_oa_prefix_no_prefix() {
        assert_eq!(strip_oa_prefix("W2741809807"), "W2741809807");
    }

    #[test]
    fn strip_html_tags_inline() {
        assert_eq!(
            strip_html_tags("Oxidative stress <italic>in vivo</italic> leads to damage"),
            "Oxidative stress in vivo leads to damage"
        );
    }

    #[test]
    fn strip_html_tags_jats() {
        assert_eq!(
            strip_html_tags(
                "<jats:p>Results showed <jats:bold>significant</jats:bold> improvement.</jats:p>"
            ),
            "Results showed significant improvement."
        );
    }

    #[test]
    fn strip_html_tags_no_tags() {
        assert_eq!(strip_html_tags("plain text"), "plain text");
    }

    #[test]
    fn strip_html_tags_empty() {
        assert_eq!(strip_html_tags(""), "");
    }

    #[test]
    fn strip_html_tags_self_closing() {
        assert_eq!(strip_html_tags("line1<br/>line2"), "line1line2");
    }
}
