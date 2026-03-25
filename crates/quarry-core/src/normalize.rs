//! Batch HTML strip + whitespace normalization.
//!
//! Replaces Python `normalize_text()` in `etl/embeddings.py`.
//! Strips HTML tags (preserving math inequalities), collapses whitespace.

use rayon::prelude::*;
use regex::Regex;
use std::sync::LazyLock;

// Same pattern as Python _HTML_TAG in etl/embeddings.py:
// Matches HTML tags like <p>, </div>, <br/>, <a href="...">, etc.
// Does NOT match bare < or > used in math (e.g., "p < 0.05").
static HTML_TAG: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"</?\w[\w.-]*(?:\s+\w[\w.-]*(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]*))?)*\s*/?>"#,
    )
    .unwrap()
});

static MULTI_SP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// Strip HTML tags, collapse whitespace, trim.
pub fn normalize_one(text: &str) -> String {
    let text = HTML_TAG.replace_all(text, " ");
    let text = MULTI_SP.replace_all(&text, " ");
    text.trim().to_string()
}

/// Batch normalize texts in parallel.
pub fn normalize_texts(texts: Vec<String>) -> Vec<String> {
    // Force lazy init before parallel iteration
    LazyLock::force(&HTML_TAG);
    LazyLock::force(&MULTI_SP);
    texts.par_iter().map(|t| normalize_one(t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html() {
        assert_eq!(normalize_one("<p>Hello</p>"), "Hello");
        assert_eq!(normalize_one("<br/>world"), "world");
        assert_eq!(
            normalize_one(r#"<a href="x">link</a> text"#),
            "link text"
        );
    }

    #[test]
    fn test_collapse_whitespace() {
        assert_eq!(normalize_one("hello   world"), "hello world");
        assert_eq!(normalize_one("  leading  trailing  "), "leading trailing");
        assert_eq!(normalize_one("tab\there\nnewline"), "tab here newline");
    }

    #[test]
    fn test_preserve_math() {
        // Bare < > in math should NOT be stripped
        assert_eq!(normalize_one("p < 0.05"), "p < 0.05");
        assert_eq!(normalize_one("n > 18"), "n > 18");
    }

    #[test]
    fn test_batch() {
        let texts = vec![
            "<p>Hello</p>  world".to_string(),
            "no  <b>html</b>  here".to_string(),
        ];
        let results = normalize_texts(texts);
        assert_eq!(results, vec!["Hello world", "no html here"]);
    }
}
