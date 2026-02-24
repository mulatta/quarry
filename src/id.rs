//! DOI normalization.

/// Normalize a raw DOI string for consistent matching.
///
/// 1. Strip `https://doi.org/` (and variants) prefix
/// 2. Trim whitespace
/// 3. Lowercase
///
/// Returns `None` if the result would be empty.
pub fn normalize_doi(raw: &str) -> Option<Box<str>> {
    let stripped = raw
        .strip_prefix("https://doi.org/")
        .or_else(|| raw.strip_prefix("http://doi.org/"))
        .or_else(|| raw.strip_prefix("https://dx.doi.org/"))
        .or_else(|| raw.strip_prefix("http://dx.doi.org/"))
        .unwrap_or(raw)
        .trim();

    if stripped.is_empty() {
        return None;
    }

    Some(stripped.to_ascii_lowercase().into_boxed_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_doi_plain() {
        assert_eq!(
            normalize_doi("10.1234/abc").unwrap().as_ref(),
            "10.1234/abc"
        );
    }

    #[test]
    fn normalize_doi_https_prefix() {
        assert_eq!(
            normalize_doi("https://doi.org/10.1234/ABC")
                .unwrap()
                .as_ref(),
            "10.1234/abc"
        );
    }

    #[test]
    fn normalize_doi_http_prefix() {
        assert_eq!(
            normalize_doi("http://doi.org/10.1234/ABC")
                .unwrap()
                .as_ref(),
            "10.1234/abc"
        );
    }

    #[test]
    fn normalize_doi_dx_prefix() {
        assert_eq!(
            normalize_doi("https://dx.doi.org/10.1234/ABC")
                .unwrap()
                .as_ref(),
            "10.1234/abc"
        );
    }

    #[test]
    fn normalize_doi_uppercase() {
        assert_eq!(
            normalize_doi("10.1234/ABC.DEF").unwrap().as_ref(),
            "10.1234/abc.def"
        );
    }

    #[test]
    fn normalize_doi_whitespace() {
        assert_eq!(
            normalize_doi("  10.1234/abc  ").unwrap().as_ref(),
            "10.1234/abc"
        );
    }

    #[test]
    fn normalize_doi_empty() {
        assert!(normalize_doi("").is_none());
        assert!(normalize_doi("  ").is_none());
        assert!(normalize_doi("https://doi.org/").is_none());
    }
}
