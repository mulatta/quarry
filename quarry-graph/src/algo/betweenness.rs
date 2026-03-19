//! Brandes betweenness centrality — subgraph only.
//!
//! O(V·E) per source, suitable for subgraphs up to ~10K nodes.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::graph::Graph;

/// Betweenness centrality on the induced subgraph of given PMIDs.
/// If `normalized` is true, scores are divided by (n-1)(n-2) for directed graphs.
pub fn compute(graph: &Graph, pmids: &[i32], normalized: bool) -> Vec<(i32, f64)> {
    let mut idx_set = HashSet::new();
    let mut local_nodes = Vec::new();
    for &pmid in pmids {
        if let Some(idx) = graph.resolve(pmid)
            && idx_set.insert(idx)
        {
            local_nodes.push(idx);
        }
    }

    let n = local_nodes.len();
    if n == 0 {
        return vec![];
    }

    let global_to_local: HashMap<u32, usize> = local_nodes
        .iter()
        .enumerate()
        .map(|(li, &gi)| (gi, li))
        .collect();

    // Build local adjacency
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for (li, &gi) in local_nodes.iter().enumerate() {
        for &nb in graph.fwd_neighbors(gi) {
            if let Some(&nb_local) = global_to_local.get(&nb) {
                adj[li].push(nb_local);
            }
        }
    }

    // Brandes algorithm
    let mut cb = vec![0.0f64; n];

    for s in 0..n {
        let mut stack: Vec<usize> = Vec::new();
        let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
        let mut sigma = vec![0.0f64; n];
        sigma[s] = 1.0;
        let mut dist: Vec<i64> = vec![-1; n];
        dist[s] = 0;

        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &w in &adj[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0f64; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != s {
                cb[w] += delta[w];
            }
        }
    }

    // Optional normalization for directed graph: 1/((n-1)(n-2))
    if normalized && n > 2 {
        let norm = 1.0 / ((n - 1) as f64 * (n - 2) as f64);
        for v in &mut cb {
            *v *= norm;
        }
    }

    let mut result: Vec<(i32, f64)> = local_nodes
        .iter()
        .zip(cb.iter())
        .map(|(&gi, &score)| (graph.pmid_of(gi), score))
        .collect();
    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}
