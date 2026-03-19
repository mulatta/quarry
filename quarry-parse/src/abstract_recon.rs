//! Batch reconstruct OpenAlex abstract_inverted_index → plaintext.
//!
//! OA stores abstracts as `{"word": [pos0, pos1], ...}` — inverted index format.
//! Reconstruct by collecting all (position, word) pairs, sorting by position,
//! and joining with spaces.

use rayon::prelude::*;
use std::collections::HashMap;

/// Reconstruct a single abstract from its JSON inverted-index string.
///
/// Returns empty string on parse failure or empty input.
fn reconstruct_one(json_str: &str) -> String {
    let map: HashMap<&str, Vec<usize>> = match sonic_rs::from_str(json_str) {
        Ok(m) => m,
        Err(_) => return String::new(),
    };

    if map.is_empty() {
        return String::new();
    }

    let total_positions: usize = map.values().map(Vec::len).sum();
    let mut pairs: Vec<(usize, &str)> = Vec::with_capacity(total_positions);
    for (word, positions) in &map {
        for &pos in positions {
            pairs.push((pos, word));
        }
    }
    pairs.sort_unstable_by_key(|&(pos, _)| pos);
    pairs.iter().map(|(_, w)| *w).collect::<Vec<_>>().join(" ")
}

/// Batch reconstruct abstracts from JSON inverted-index strings.
///
/// Uses rayon for parallel processing. Invalid JSON entries produce empty strings.
pub fn reconstruct_abstracts(jsons: Vec<String>) -> Vec<String> {
    jsons.par_iter().map(|s| reconstruct_one(s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_reconstruction() {
        let json = r#"{"The": [0], "quick": [1], "brown": [2], "fox": [3]}"#;
        assert_eq!(reconstruct_one(json), "The quick brown fox");
    }

    #[test]
    fn test_multi_position() {
        let json = r#"{"the": [0, 4], "cat": [1], "sat": [2], "on": [3], "mat": [5]}"#;
        assert_eq!(reconstruct_one(json), "the cat sat on the mat");
    }

    #[test]
    fn test_empty_input() {
        assert_eq!(reconstruct_one("{}"), "");
        assert_eq!(reconstruct_one(""), "");
        assert_eq!(reconstruct_one("null"), "");
    }

    #[test]
    fn test_batch() {
        let jsons = vec![
            r#"{"hello": [0], "world": [1]}"#.to_string(),
            r#"{"foo": [0], "bar": [1]}"#.to_string(),
        ];
        let results = reconstruct_abstracts(jsons);
        assert_eq!(results, vec!["hello world", "foo bar"]);
    }
}
