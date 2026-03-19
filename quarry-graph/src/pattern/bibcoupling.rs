//! Bibliographic coupling: papers that share references with a given paper.
//!
//! Two papers A and B are bibliographically coupled when they both cite paper C.
//! For paper A: find all C that A cites (forward neighbors), then for each C
//! find all B that also cite C (reverse neighbors). Count shared references.

use std::collections::HashMap;

use crate::graph::Graph;

/// Find papers bibliographically coupled with `pmid`, returning
/// (pmid, shared_count) where shared_count >= min_shared, sorted descending.
/// limit=0 for unlimited.
pub fn compute(graph: &Graph, pmid: i32, min_shared: usize, limit: usize) -> Vec<(i32, usize)> {
    let Some(idx) = graph.resolve(pmid) else {
        return vec![];
    };

    let references = graph.fwd_neighbors(idx);

    let mut coupled: HashMap<u32, usize> = HashMap::new();
    for &reference in references {
        for &citer in graph.rev_neighbors(reference) {
            if citer != idx {
                *coupled.entry(citer).or_insert(0) += 1;
            }
        }
    }

    let mut result: Vec<(i32, usize)> = coupled
        .into_iter()
        .filter(|&(_, count)| count >= min_shared)
        .map(|(idx, count)| (graph.pmid_of(idx), count))
        .collect();
    result.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    if limit > 0 {
        result.truncate(limit);
    }
    result
}
