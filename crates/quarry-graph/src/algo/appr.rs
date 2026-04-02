//! Push-based Approximate Personalized PageRank (Andersen et al. 2006).
//!
//! Local computation on full CSR graph. Only visits nodes reachable from seed
//! with significant score. Time complexity O(1/ε), independent of graph size.

use std::collections::{HashMap, VecDeque};

use crate::graph::Graph;

fn f64_cmp_desc(a: &(i64, f64), b: &(i64, f64)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
}

/// Push-based APPR on full CSR graph (bidirectional edges).
///
/// Returns (node_id, score) pairs sorted by score descending.
/// Only nodes with score > 0 are included (sparse result).
///
/// `top_k`: if Some(k), return only the top k results.
pub fn compute(
    graph: &Graph,
    seed: i64,
    alpha: f64,
    epsilon: f64,
    top_k: Option<usize>,
) -> Vec<(i64, f64)> {
    let Some(seed_idx) = graph.resolve(seed) else {
        return vec![];
    };

    let mut residual: HashMap<u32, f64> = HashMap::new();
    let mut estimate: HashMap<u32, f64> = HashMap::new();
    let mut queue: VecDeque<u32> = VecDeque::new();

    residual.insert(seed_idx, 1.0);
    queue.push_back(seed_idx);

    while let Some(v) = queue.pop_front() {
        let r_v = match residual.get(&v) {
            Some(&r) if r > 0.0 => r,
            _ => continue,
        };

        let degree = total_degree(graph, v);
        if degree == 0 {
            // dangling node: absorb all residual as estimate
            *estimate.entry(v).or_insert(0.0) += r_v;
            residual.insert(v, 0.0);
            continue;
        }

        // check push condition: residual[v] / degree(v) > ε
        if r_v / degree as f64 <= epsilon {
            continue;
        }

        // retain α fraction as PPR score
        *estimate.entry(v).or_insert(0.0) += alpha * r_v;

        // push (1-α) fraction to neighbors
        let push_mass = (1.0 - alpha) * r_v / degree as f64;
        residual.insert(v, 0.0);

        for &nb in graph.fwd_neighbors(v) {
            let r_nb = residual.entry(nb).or_insert(0.0);
            let was_pushable = *r_nb / total_degree(graph, nb) as f64 > epsilon;
            *r_nb += push_mass;
            if !was_pushable && *r_nb / total_degree(graph, nb) as f64 > epsilon {
                queue.push_back(nb);
            }
        }
        for &nb in graph.rev_neighbors(v) {
            let r_nb = residual.entry(nb).or_insert(0.0);
            let was_pushable = *r_nb / total_degree(graph, nb) as f64 > epsilon;
            *r_nb += push_mass;
            if !was_pushable && *r_nb / total_degree(graph, nb) as f64 > epsilon {
                queue.push_back(nb);
            }
        }
    }

    let mut result: Vec<(i64, f64)> = estimate
        .into_iter()
        .map(|(idx, score)| (graph.id_of(idx), score))
        .collect();
    result.sort_unstable_by(f64_cmp_desc);

    if let Some(k) = top_k {
        result.truncate(k);
    }

    result
}

/// Total degree (forward + reverse neighbors).
fn total_degree(graph: &Graph, idx: u32) -> usize {
    graph.fwd_neighbors(idx).len() + graph.rev_neighbors(idx).len()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Linear chain: all reachable nodes get scores, further nodes get less.
    ///
    /// Graph: 0 → 1 → 2 → 3 (IDs 100..103), bidirectional (reverse edges exist).
    /// Seed=100 (node 0, chain endpoint). Node 3 is furthest from seed.
    ///
    /// With bidirectional edges and low α, interior nodes can accumulate more
    /// mass than the seed (they receive from both directions). We verify:
    /// - all nodes have positive scores
    /// - endpoint furthest from seed (node 3) has lowest score
    #[test]
    fn linear_chain_all_nodes_scored() {
        let dir = tempfile::tempdir().unwrap();
        let graph_dir = dir.path();
        write_test_graph(graph_dir, 4, &[100, 101, 102, 103], &[(0, 1), (1, 2), (2, 3)]);

        let graph = Graph::open(graph_dir).unwrap();
        let result = compute(&graph, 100, 0.15, 1e-8, None);

        assert_eq!(result.len(), 4, "all nodes should be reached");

        let scores: HashMap<i64, f64> = result.into_iter().collect();
        let s0 = scores.get(&100).copied().unwrap_or(0.0);
        let s3 = scores.get(&103).copied().unwrap_or(0.0);

        assert!(s0 > 0.0, "seed should have positive score");
        assert!(s3 > 0.0, "furthest node should have positive score");
        assert!(s0 > s3, "seed should score higher than furthest node: {s0} > {s3}");
    }

    #[test]
    fn top_k_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let graph_dir = dir.path();
        write_test_graph(graph_dir, 4, &[100, 101, 102, 103], &[(0, 1), (1, 2), (2, 3)]);

        let graph = Graph::open(graph_dir).unwrap();
        let result = compute(&graph, 100, 0.15, 1e-8, Some(2));

        assert!(result.len() <= 2, "top_k=2 should return at most 2 results");
    }

    #[test]
    fn hub_dilution() {
        // Hub node 0 connects to many leaves. Seed is node 1 (leaf).
        // Hub should score lower than direct neighbor of seed.
        //
        // Graph: 0→1, 0→2, 0→3, 0→4, 1→5
        // IDs:  100..105
        // Seed: 101 (node 1)
        // Expected: node 5 (direct neighbor via 1→5) scores higher than
        //           nodes 2,3,4 (only reachable via hub 0)
        let dir = tempfile::tempdir().unwrap();
        let graph_dir = dir.path();
        write_test_graph(
            graph_dir,
            6,
            &[100, 101, 102, 103, 104, 105],
            &[(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)],
        );

        let graph = Graph::open(graph_dir).unwrap();
        let result = compute(&graph, 101, 0.15, 1e-8, None);

        let scores: HashMap<i64, f64> = result.into_iter().collect();
        let s5 = scores.get(&105).copied().unwrap_or(0.0);
        let s2 = scores.get(&102).copied().unwrap_or(0.0);

        assert!(
            s5 > s2,
            "direct neighbor of seed ({s5}) should score higher than hub leaf ({s2})"
        );
    }

    #[test]
    fn unknown_seed_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let graph_dir = dir.path();
        write_test_graph(graph_dir, 2, &[100, 101], &[(0, 1)]);

        let graph = Graph::open(graph_dir).unwrap();
        let result = compute(&graph, 999, 0.15, 1e-6, None);
        assert!(result.is_empty());
    }

    /// Write a minimal CSR graph to disk for testing.
    pub(crate) fn write_test_graph(dir: &std::path::Path, n: usize, ids: &[i64], edges: &[(u32, u32)]) {
        // meta.json
        let total_edges = edges.len();
        std::fs::write(
            dir.join("meta.json"),
            format!(r#"{{"num_nodes":{n},"num_edges":{total_edges}}}"#),
        )
        .unwrap();

        // id_map.bin (text, one id per line)
        let id_map: String = ids.iter().map(|id| format!("{id}\n")).collect();
        std::fs::write(dir.join("id_map.bin"), id_map).unwrap();

        // Build forward CSR
        let fwd_dir = dir.join("forward");
        std::fs::create_dir_all(&fwd_dir).unwrap();
        let mut fwd_adj: Vec<Vec<u32>> = vec![vec![]; n];
        for &(src, dst) in edges {
            fwd_adj[src as usize].push(dst);
        }
        write_csr(&fwd_dir, n, &fwd_adj);

        // Build reverse CSR
        let rev_dir = dir.join("reverse");
        std::fs::create_dir_all(&rev_dir).unwrap();
        let mut rev_adj: Vec<Vec<u32>> = vec![vec![]; n];
        for &(src, dst) in edges {
            rev_adj[dst as usize].push(src);
        }
        write_csr(&rev_dir, n, &rev_adj);
    }

    fn write_csr(dir: &std::path::Path, n: usize, adj: &[Vec<u32>]) {
        use std::io::Write;

        let mut indptr: Vec<u64> = Vec::with_capacity(n + 1);
        let mut indices: Vec<u32> = Vec::new();
        indptr.push(0);
        for neighbors in adj {
            indices.extend_from_slice(neighbors);
            indptr.push(indices.len() as u64);
        }

        let mut f = std::fs::File::create(dir.join("indptr.bin")).unwrap();
        for &v in &indptr {
            f.write_all(&v.to_ne_bytes()).unwrap();
        }

        let mut f = std::fs::File::create(dir.join("indices.bin")).unwrap();
        for &v in &indices {
            f.write_all(&v.to_ne_bytes()).unwrap();
        }
    }
}
