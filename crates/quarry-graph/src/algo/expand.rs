//! `expand`: APPR + AA-weighted coupling/cocitation fusion.
//!
//! Core entry point for subgraph mining. Two modes:
//! - **fused**: wRRF merges all candidates into a single ranked list.
//!   Better for focused exploration (specific follow-up experiments).
//! - **separated**: APPR fills structural slots, coupling/cocitation fills
//!   guaranteed lateral slots. Better for broad surveys (review writing).

use std::collections::{HashMap, HashSet};

use crate::algo::appr;
use crate::graph::Graph;

/// Expansion mode.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// wRRF: all candidates compete in a single pool.
    Fused,
    /// APPR top-N + lateral top-M (guaranteed lateral slots).
    Separated,
}

/// Parameters for expand operation.
pub struct ExpandParams {
    pub alpha: f64,
    pub epsilon: f64,
    pub mode: Mode,
    pub weights: [f64; 3],
    pub rrf_k: usize,
    pub lateral_pct: usize,
    pub limit: usize,
}

/// Result of expand operation.
pub struct ExpandResult {
    /// (work_id, score) sorted descending.
    pub papers: Vec<(i64, f64)>,
    pub stats: ExpandStats,
}

pub struct ExpandStats {
    pub appr_candidates: usize,
    pub coupling_candidates: usize,
    pub cocitation_candidates: usize,
    pub elapsed_ms: u64,
}

/// Run subgraph expansion.
pub fn compute(graph: &Graph, seed: i64, params: &ExpandParams) -> ExpandResult {
    let ExpandParams { alpha, epsilon, mode, weights, rrf_k, lateral_pct, limit } = *params;
    let start = std::time::Instant::now();

    let Some(seed_idx) = graph.resolve(seed) else {
        return empty_result();
    };

    // 1. APPR
    let appr_results = appr::compute(graph, seed, alpha, epsilon, None);
    let appr_count = appr_results.len();

    // 2. AA-weighted cosine-normalized bibliographic coupling
    let coupling_results = coupling_aa(graph, seed_idx);
    let coupling_count = coupling_results.len();

    // 3. AA-weighted cosine-normalized co-citation
    let cocitation_results = cocitation_aa(graph, seed_idx);
    let cocitation_count = cocitation_results.len();

    // 4. Fuse based on mode
    let papers = match mode {
        Mode::Fused => {
            let fused = wrrf_fuse(
                &appr_results,
                &coupling_results,
                &cocitation_results,
                weights,
                rrf_k,
                seed,
            );
            let mut v: Vec<(i64, f64)> = fused.into_iter().collect();
            v.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            v.truncate(limit);
            v
        }
        Mode::Separated => separated_fuse(
            &appr_results,
            &coupling_results,
            &cocitation_results,
            seed,
            seed_idx,
            graph,
            lateral_pct,
            limit,
        ),
    };

    let elapsed_ms = start.elapsed().as_millis() as u64;

    ExpandResult {
        papers,
        stats: ExpandStats {
            appr_candidates: appr_count,
            coupling_candidates: coupling_count,
            cocitation_candidates: cocitation_count,
            elapsed_ms,
        },
    }
}

fn empty_result() -> ExpandResult {
    ExpandResult {
        papers: vec![],
        stats: ExpandStats {
            appr_candidates: 0,
            coupling_candidates: 0,
            cocitation_candidates: 0,
            elapsed_ms: 0,
        },
    }
}

#[allow(clippy::too_many_arguments)]
/// Separated fusion: APPR fills structural slots, coupling/cocitation fills lateral slots.
///
/// Lateral = papers NOT in seed's direct references or citers AND not in APPR top.
/// This guarantees lateral discovery even when APPR dominates.
fn separated_fuse(
    appr: &[(i64, f64)],
    coupling: &[(i64, f64)],
    cocitation: &[(i64, f64)],
    seed: i64,
    seed_idx: u32,
    graph: &Graph,
    lateral_pct: usize,
    limit: usize,
) -> Vec<(i64, f64)> {
    let lateral_slots = limit * lateral_pct / 100;
    let structural_slots = limit - lateral_slots;

    // Seed's direct neighbors (refs + citers)
    let refs: HashSet<u32> = graph.fwd_neighbors(seed_idx).iter().copied().collect();
    let citers: HashSet<u32> = graph.rev_neighbors(seed_idx).iter().copied().collect();

    // Structural: APPR top-N (excluding seed)
    let mut structural: Vec<(i64, f64)> = Vec::with_capacity(structural_slots);
    let mut structural_ids: HashSet<i64> = HashSet::new();
    for &(id, score) in appr {
        if id == seed {
            continue;
        }
        structural.push((id, score));
        structural_ids.insert(id);
        if structural.len() >= structural_slots {
            break;
        }
    }

    // Lateral: coupling/cocitation candidates NOT in refs, citers, or structural top
    let mut lateral_scores: HashMap<i64, f64> = HashMap::new();
    for &(id, score) in coupling.iter().chain(cocitation.iter()) {
        if id == seed || structural_ids.contains(&id) {
            continue;
        }
        if let Some(idx) = graph.resolve(id)
            && (refs.contains(&idx) || citers.contains(&idx))
        {
            continue;
        }
        *lateral_scores.entry(id).or_insert(0.0) += score;
    }

    let mut laterals: Vec<(i64, f64)> = lateral_scores.into_iter().collect();
    laterals.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    laterals.truncate(lateral_slots);

    structural.extend(laterals);
    structural
}

/// Adamic-Adar weighted bibliographic coupling with cosine normalization.
///
/// For seed's references, weight each by 1/log(indegree). Papers sharing
/// rare references get higher scores than papers sharing popular ones.
fn coupling_aa(graph: &Graph, seed_idx: u32) -> Vec<(i64, f64)> {
    let references = graph.fwd_neighbors(seed_idx);
    if references.is_empty() {
        return vec![];
    }

    // Precompute AA weight for each reference: 1/log(indegree)
    let ref_weights: Vec<(u32, f64)> = references
        .iter()
        .map(|&r| {
            let indegree = graph.rev_neighbors(r).len();
            let w = if indegree > 1 {
                1.0 / (indegree as f64).ln()
            } else {
                1.0 // single citation — maximum informativeness
            };
            (r, w)
        })
        .collect();

    // Seed self-weight for cosine normalization
    let seed_self: f64 = ref_weights.iter().map(|&(_, w)| w).sum();

    // Accumulate AA coupling scores for each neighbor
    let mut scores: HashMap<u32, f64> = HashMap::new();
    for &(reference, weight) in &ref_weights {
        for &citer in graph.rev_neighbors(reference) {
            if citer != seed_idx {
                *scores.entry(citer).or_insert(0.0) += weight;
            }
        }
    }

    // Cosine normalization: score / sqrt(seed_self * other_self)
    let mut result: Vec<(i64, f64)> = scores
        .into_iter()
        .map(|(idx, raw)| {
            let other_self = compute_self_weight_coupling(graph, idx);
            let norm = if seed_self > 0.0 && other_self > 0.0 {
                raw / (seed_self * other_self).sqrt()
            } else {
                0.0
            };
            (graph.id_of(idx), norm)
        })
        .filter(|&(_, s)| s > 0.0)
        .collect();
    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Compute self-weight for coupling cosine: Σ 1/log(indegree(r)) for all references of node.
fn compute_self_weight_coupling(graph: &Graph, idx: u32) -> f64 {
    graph
        .fwd_neighbors(idx)
        .iter()
        .map(|&r| {
            let indegree = graph.rev_neighbors(r).len();
            if indegree > 1 {
                1.0 / (indegree as f64).ln()
            } else {
                1.0
            }
        })
        .sum()
}

/// Adamic-Adar weighted co-citation with cosine normalization.
///
/// For seed's citers, weight each by 1/log(outdegree). Papers co-cited
/// by niche citers get higher scores than those co-cited by review papers.
fn cocitation_aa(graph: &Graph, seed_idx: u32) -> Vec<(i64, f64)> {
    let citers = graph.rev_neighbors(seed_idx);
    if citers.is_empty() {
        return vec![];
    }

    // Precompute AA weight for each citer: 1/log(outdegree)
    let citer_weights: Vec<(u32, f64)> = citers
        .iter()
        .map(|&c| {
            let outdegree = graph.fwd_neighbors(c).len();
            let w = if outdegree > 1 {
                1.0 / (outdegree as f64).ln()
            } else {
                1.0
            };
            (c, w)
        })
        .collect();

    // Seed self-weight for cosine normalization
    let seed_self: f64 = citer_weights.iter().map(|&(_, w)| w).sum();

    // Accumulate AA co-citation scores
    let mut scores: HashMap<u32, f64> = HashMap::new();
    for &(citer, weight) in &citer_weights {
        for &cited in graph.fwd_neighbors(citer) {
            if cited != seed_idx {
                *scores.entry(cited).or_insert(0.0) += weight;
            }
        }
    }

    // Cosine normalization
    let mut result: Vec<(i64, f64)> = scores
        .into_iter()
        .map(|(idx, raw)| {
            let other_self = compute_self_weight_cocitation(graph, idx);
            let norm = if seed_self > 0.0 && other_self > 0.0 {
                raw / (seed_self * other_self).sqrt()
            } else {
                0.0
            };
            (graph.id_of(idx), norm)
        })
        .filter(|&(_, s)| s > 0.0)
        .collect();
    result.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Compute self-weight for cocitation cosine: Σ 1/log(outdegree(c)) for all citers of node.
fn compute_self_weight_cocitation(graph: &Graph, idx: u32) -> f64 {
    graph
        .rev_neighbors(idx)
        .iter()
        .map(|&c| {
            let outdegree = graph.fwd_neighbors(c).len();
            if outdegree > 1 {
                1.0 / (outdegree as f64).ln()
            } else {
                1.0
            }
        })
        .sum()
}

/// Weighted Reciprocal Rank Fusion.
///
/// Combines three ranked lists into one. Papers missing from a signal
/// contribute 0 for that signal. Seed is excluded from output.
fn wrrf_fuse(
    appr: &[(i64, f64)],
    coupling: &[(i64, f64)],
    cocitation: &[(i64, f64)],
    weights: [f64; 3],
    k: usize,
    seed: i64,
) -> HashMap<i64, f64> {
    // Build rank maps (1-based)
    let appr_rank: HashMap<i64, usize> = appr
        .iter()
        .enumerate()
        .map(|(i, &(id, _))| (id, i + 1))
        .collect();
    let coupling_rank: HashMap<i64, usize> = coupling
        .iter()
        .enumerate()
        .map(|(i, &(id, _))| (id, i + 1))
        .collect();
    let cocitation_rank: HashMap<i64, usize> = cocitation
        .iter()
        .enumerate()
        .map(|(i, &(id, _))| (id, i + 1))
        .collect();

    // Collect all candidate IDs (excluding seed)
    let mut all_ids: HashMap<i64, f64> = HashMap::new();
    for &(id, _) in appr.iter().chain(coupling.iter()).chain(cocitation.iter()) {
        if id != seed {
            all_ids.entry(id).or_insert(0.0);
        }
    }

    // Compute fused scores
    let k_f = k as f64;
    for (&id, score) in all_ids.iter_mut() {
        if let Some(&rank) = appr_rank.get(&id) {
            *score += weights[0] / (k_f + rank as f64);
        }
        if let Some(&rank) = coupling_rank.get(&id) {
            *score += weights[1] / (k_f + rank as f64);
        }
        if let Some(&rank) = cocitation_rank.get(&id) {
            *score += weights[2] / (k_f + rank as f64);
        }
    }

    all_ids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(mode: Mode, limit: usize) -> ExpandParams {
        ExpandParams {
            alpha: 0.15,
            epsilon: 1e-8,
            mode,
            weights: [0.5, 0.25, 0.25],
            rrf_k: 60,
            lateral_pct: 25,
            limit,
        }
    }

    fn write_test_graph(dir: &std::path::Path, n: usize, ids: &[i64], edges: &[(u32, u32)]) {
        std::fs::write(
            dir.join("meta.json"),
            format!(r#"{{"num_nodes":{n},"num_edges":{}}}"#, edges.len()),
        )
        .unwrap();

        let id_map: String = ids.iter().map(|id| format!("{id}\n")).collect();
        std::fs::write(dir.join("id_map.bin"), id_map).unwrap();

        let fwd_dir = dir.join("forward");
        std::fs::create_dir_all(&fwd_dir).unwrap();
        let mut fwd_adj: Vec<Vec<u32>> = vec![vec![]; n];
        for &(src, dst) in edges {
            fwd_adj[src as usize].push(dst);
        }
        write_csr(&fwd_dir, n, &fwd_adj);

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

    /// Graph for expand tests:
    ///
    ///   0 (seed) → 2 (shared ref, niche, indegree=2)
    ///   0 → 3 (shared ref, popular, indegree=4)
    ///   1 → 2
    ///   1 → 3
    ///   4 → 2
    ///   4 → 3
    ///   5 → 3
    ///   5 → 0 (citer of seed)
    ///   6 → 0 (citer of seed)
    ///   6 → 1 (co-cited with seed by node 6)
    ///
    /// IDs: 100..106
    /// Seed = 100 (node 0)
    ///
    /// Coupling: node 1 and 4 share refs with seed (nodes 2,3)
    ///   Node 1: shares ref 2 (niche) + ref 3 (popular) → AA higher for ref 2
    ///   Node 4: shares ref 2 (niche) + ref 3 (popular) → same as node 1
    ///   Node 5: shares ref 3 only (popular) → lower AA score
    ///
    /// Co-citation: nodes 5,6 cite seed
    ///   Node 6 also cites node 1 → node 1 is co-cited with seed
    fn test_expand_graph() -> (tempfile::TempDir, Graph) {
        let dir = tempfile::tempdir().unwrap();
        write_test_graph(
            dir.path(),
            7,
            &[100, 101, 102, 103, 104, 105, 106],
            &[
                (0, 2), // seed → ref_niche
                (0, 3), // seed → ref_popular
                (1, 2), // node1 → ref_niche
                (1, 3), // node1 → ref_popular
                (4, 2), // node4 → ref_niche
                (4, 3), // node4 → ref_popular
                (5, 3), // node5 → ref_popular only
                (5, 0), // node5 cites seed
                (6, 0), // node6 cites seed
                (6, 1), // node6 cites node1 (co-cited with seed)
            ],
        );
        let graph = Graph::open(dir.path()).unwrap();
        (dir, graph)
    }

    #[test]
    fn fused_excludes_seed() {
        let (_dir, graph) = test_expand_graph();
        let result = compute(&graph, 100, &params(Mode::Fused, 100));
        assert!(
            result.papers.iter().all(|&(id, _)| id != 100),
            "seed should not appear in results"
        );
    }

    #[test]
    fn fused_returns_candidates() {
        let (_dir, graph) = test_expand_graph();
        let result = compute(&graph, 100, &params(Mode::Fused, 100));
        assert!(!result.papers.is_empty(), "expand should return candidates");
        assert!(result.stats.appr_candidates > 0);
    }

    #[test]
    fn fused_limit_works() {
        let (_dir, graph) = test_expand_graph();
        let result = compute(&graph, 100, &params(Mode::Fused, 2));
        assert!(result.papers.len() <= 2);
    }

    #[test]
    fn fused_sorted_descending() {
        let (_dir, graph) = test_expand_graph();
        let result = compute(&graph, 100, &params(Mode::Fused, 100));
        for w in result.papers.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "results should be sorted descending: {} >= {}",
                w[0].1,
                w[1].1
            );
        }
    }

    #[test]
    fn separated_excludes_seed() {
        let (_dir, graph) = test_expand_graph();
        let result = compute(&graph, 100, &params(Mode::Separated, 100));
        assert!(
            result.papers.iter().all(|&(id, _)| id != 100),
            "seed should not appear in separated results"
        );
    }

    #[test]
    fn separated_returns_candidates() {
        let (_dir, graph) = test_expand_graph();
        let result = compute(&graph, 100, &params(Mode::Separated, 100));
        assert!(!result.papers.is_empty(), "separated should return candidates");
    }

    #[test]
    fn aa_coupling_prefers_niche_references() {
        let (_dir, graph) = test_expand_graph();
        // Node 1 shares both niche ref (2, indegree=3) and popular ref (3, indegree=5)
        // Node 5 shares only popular ref (3)
        // Node 1 should score higher than node 5 in coupling
        let coupling = coupling_aa(&graph, 0);
        let scores: HashMap<i64, f64> = coupling.into_iter().collect();
        let s1 = scores.get(&101).copied().unwrap_or(0.0);
        let s5 = scores.get(&105).copied().unwrap_or(0.0);
        assert!(
            s1 > s5,
            "node sharing niche+popular refs ({s1}) should score higher than popular-only ({s5})"
        );
    }

    #[test]
    fn cocitation_finds_co_cited_papers() {
        let (_dir, graph) = test_expand_graph();
        // Node 6 cites both seed (0) and node 1 → node 1 is co-cited with seed
        let cocite = cocitation_aa(&graph, 0);
        let scores: HashMap<i64, f64> = cocite.into_iter().collect();
        assert!(
            scores.contains_key(&101),
            "node 1 should be co-cited with seed (via citer node 6)"
        );
    }

    #[test]
    fn wrrf_fuse_combines_signals() {
        let appr = vec![(1, 0.5), (2, 0.3), (3, 0.1)];
        let coupling = vec![(2, 10.0), (3, 5.0), (4, 1.0)];
        let cocitation = vec![(3, 8.0), (1, 3.0)];

        let result = wrrf_fuse(&appr, &coupling, &cocitation, [0.5, 0.25, 0.25], 60, 999);

        // node 3 appears in all 3 signals → should have score > 0
        assert!(result[&3] > 0.0, "node in all 3 signals should have positive score");
        // node 4 appears only in coupling
        assert!(result[&4] > 0.0, "lateral candidate should be included");
        // seed (999) should not be in results
        assert!(!result.contains_key(&999));
    }

    #[test]
    fn wrrf_multi_signal_beats_single() {
        let appr = vec![(1, 0.5), (2, 0.3)];
        let coupling = vec![(1, 10.0), (2, 5.0)];
        let cocitation = vec![(1, 8.0)]; // node 1 in all 3, node 2 in 2

        let result = wrrf_fuse(&appr, &coupling, &cocitation, [0.5, 0.25, 0.25], 60, 999);

        assert!(
            result[&1] > result[&2],
            "node in 3 signals ({}) should beat node in 2 signals ({})",
            result[&1],
            result[&2]
        );
    }

    #[test]
    fn fused_unknown_seed_returns_empty() {
        let (_dir, graph) = test_expand_graph();
        let result = compute(&graph, 999, &params(Mode::Fused, 100));
        assert!(result.papers.is_empty());
    }

    #[test]
    fn separated_unknown_seed_returns_empty() {
        let (_dir, graph) = test_expand_graph();
        let result = compute(&graph, 999, &params(Mode::Separated, 100));
        assert!(result.papers.is_empty());
    }
}
