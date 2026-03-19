//! Pull-based PageRank power iteration with rayon parallelism.
//!
//! Uses reverse CSR: for each node, pull rank contributions from in-neighbors.
//! This avoids atomic writes and scales linearly with cores.

use rayon::prelude::*;

use crate::graph::Graph;

fn f64_cmp_desc(a: &(i64, f64), b: &(i64, f64)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
}

/// Full-graph PageRank. Returns all (pmid, score) sorted by score descending.
pub fn compute(graph: &Graph, alpha: f64, max_iter: usize, tol: f64) -> Vec<(i64, f64)> {
    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    let init = 1.0 / n as f64;
    let mut rank = vec![init; n];
    let mut new_rank = vec![0.0f64; n];

    let fwd_indptr = graph.fwd_indptr();
    let out_degree: Vec<f64> = (0..n)
        .map(|i| (fwd_indptr[i + 1] - fwd_indptr[i]) as f64)
        .collect();

    let rev_indptr = graph.rev_indptr();
    let rev_indices = graph.rev_indices();
    let teleport = (1.0 - alpha) / n as f64;

    for _iter in 0..max_iter {
        let dangling_sum: f64 = rank
            .par_iter()
            .enumerate()
            .filter(|&(i, _)| out_degree[i] == 0.0)
            .map(|(_, &r)| r)
            .sum();
        let dangling_contrib = alpha * dangling_sum / n as f64;

        new_rank
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, nr)| {
                let start = rev_indptr[i] as usize;
                let end = rev_indptr[i + 1] as usize;
                let mut sum = 0.0;
                for &j in &rev_indices[start..end] {
                    let j = j as usize;
                    let od = out_degree[j];
                    if od > 0.0 {
                        sum += rank[j] / od;
                    }
                }
                *nr = teleport + alpha * sum + dangling_contrib;
            });

        let diff: f64 = rank
            .par_iter()
            .zip(new_rank.par_iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        std::mem::swap(&mut rank, &mut new_rank);

        if diff < tol {
            break;
        }
    }

    let mut result: Vec<(i64, f64)> = rank
        .into_iter()
        .enumerate()
        .map(|(i, score)| (graph.id_of(i as u32), score))
        .collect();
    result.par_sort_unstable_by(f64_cmp_desc);
    result
}

/// PageRank on an induced subgraph. All parameters exposed.
pub fn subgraph(
    graph: &Graph,
    ids: &[i64],
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<(i64, f64)> {
    let mut idx_set = std::collections::HashSet::new();
    let mut local_nodes = Vec::new();
    for &id in ids {
        if let Some(idx) = graph.resolve(id)
            && idx_set.insert(idx)
        {
            local_nodes.push(idx);
        }
    }

    let n = local_nodes.len();
    if n == 0 {
        return vec![];
    }

    let global_to_local: std::collections::HashMap<u32, usize> = local_nodes
        .iter()
        .enumerate()
        .map(|(li, &gi)| (gi, li))
        .collect();

    let mut in_neighbors: Vec<Vec<usize>> = vec![vec![]; n];
    let mut out_degree: Vec<f64> = vec![0.0; n];

    for (li, &gi) in local_nodes.iter().enumerate() {
        for &nb in graph.fwd_neighbors(gi) {
            if let Some(&nb_local) = global_to_local.get(&nb) {
                in_neighbors[nb_local].push(li);
                out_degree[li] += 1.0;
            }
        }
    }

    let init = 1.0 / n as f64;
    let mut rank = vec![init; n];
    let mut new_rank = vec![0.0f64; n];
    let teleport = (1.0 - alpha) / n as f64;

    for _ in 0..max_iter {
        let dangling_sum: f64 = (0..n)
            .filter(|&i| out_degree[i] == 0.0)
            .map(|i| rank[i])
            .sum();
        let dangling_contrib = alpha * dangling_sum / n as f64;

        for i in 0..n {
            let mut sum = 0.0;
            for &j in &in_neighbors[i] {
                if out_degree[j] > 0.0 {
                    sum += rank[j] / out_degree[j];
                }
            }
            new_rank[i] = teleport + alpha * sum + dangling_contrib;
        }

        let diff: f64 = rank
            .iter()
            .zip(new_rank.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        std::mem::swap(&mut rank, &mut new_rank);
        if diff < tol {
            break;
        }
    }

    let mut result: Vec<(i64, f64)> = local_nodes
        .iter()
        .zip(rank.iter())
        .map(|(&gi, &score)| (graph.id_of(gi), score))
        .collect();
    result.sort_unstable_by(f64_cmp_desc);
    result
}
