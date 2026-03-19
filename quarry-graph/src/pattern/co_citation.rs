//! Co-citation: papers that are co-cited with a given paper.
//!
//! Two papers A and B are co-cited when a third paper C cites both A and B.
//! For paper A: find all C that cite A (reverse neighbors), then for each C
//! find all B that C also cites (forward neighbors). Count shared citers.

use std::collections::HashMap;

use crate::graph::Graph;

/// Find papers co-cited with `pmid`, returning (pmid, shared_count) where
/// shared_count >= min_shared, sorted by count descending. limit=0 for unlimited.
pub fn compute(graph: &Graph, pmid: i32, min_shared: usize, limit: usize) -> Vec<(i32, usize)> {
    let Some(idx) = graph.resolve(pmid) else {
        return vec![];
    };

    let citers = graph.rev_neighbors(idx);

    let mut co_cited: HashMap<u32, usize> = HashMap::new();
    for &citer in citers {
        for &cited in graph.fwd_neighbors(citer) {
            if cited != idx {
                *co_cited.entry(cited).or_insert(0) += 1;
            }
        }
    }

    let mut result: Vec<(i32, usize)> = co_cited
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
