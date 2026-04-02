//! Heat Kernel PageRank (Kloster & Gleich 2014).
//!
//! Level-based push on full CSR graph. Uses Poisson decay instead of
//! geometric teleportation — diffusion time `t` controls local/global range.
//!
//! Key difference from APPR:
//! - APPR: geometric decay α^k, seed dominance ~15-20%
//! - HKPR: Poisson decay e^{-t}·t^k/k!, seed dominance ~0.7% at t=3
//!
//! Same O(1/ε) complexity, different score distribution shape.

use std::collections::HashMap;

use crate::graph::Graph;

fn f64_cmp_desc(a: &(i64, f64), b: &(i64, f64)) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
}

/// Level-based push HKPR on full CSR graph (bidirectional edges).
///
/// Returns (node_id, score) pairs sorted by score descending.
///
/// `t`: diffusion time. Small t = local (like high α in APPR), large t = global.
/// `epsilon`: push threshold per level.
/// `top_k`: if Some(k), return only the top k results.
pub fn compute(
    graph: &Graph,
    seed: i64,
    t: f64,
    epsilon: f64,
    top_k: Option<usize>,
) -> Vec<(i64, f64)> {
    let Some(seed_idx) = graph.resolve(seed) else {
        return vec![];
    };

    // Precompute Poisson weights ψ_k = e^{-t} · t^k / k!
    let max_k = max_level(t, epsilon);
    let psi = poisson_weights(t, max_k);

    let mut estimate: HashMap<u32, f64> = HashMap::new();

    // Level 0: seed has all mass
    let mut cur: HashMap<u32, f64> = HashMap::new();
    cur.insert(seed_idx, 1.0);

    for &psi_k in psi.iter().take(max_k + 1) {
        if cur.is_empty() {
            break;
        }
        let mut next: HashMap<u32, f64> = HashMap::new();

        for (&v, &r_v) in &cur {
            if r_v <= 0.0 {
                continue;
            }

            // Absorb: contribution of k-step walks ending at v
            *estimate.entry(v).or_insert(0.0) += psi_k * r_v;

            let degree = total_degree(graph, v);
            if degree == 0 {
                continue; // dangling: absorbed, no propagation
            }

            // Push to level k+1 if significant
            if r_v / degree as f64 > epsilon {
                let push_per_nb = r_v / degree as f64;
                for &nb in graph.fwd_neighbors(v) {
                    *next.entry(nb).or_insert(0.0) += push_per_nb;
                }
                for &nb in graph.rev_neighbors(v) {
                    *next.entry(nb).or_insert(0.0) += push_per_nb;
                }
            }
            // Unpushed: absorbed at this level, mass lost for future levels (approximation)
        }

        cur = next;
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

/// Poisson weights ψ_k = e^{-t} · t^k / k! for k = 0..=max_k.
/// Computed iteratively to avoid overflow: ψ_{k+1} = ψ_k · t / (k+1).
fn poisson_weights(t: f64, max_k: usize) -> Vec<f64> {
    let mut weights = Vec::with_capacity(max_k + 1);
    let mut w = (-t).exp(); // ψ_0 = e^{-t}
    weights.push(w);
    for k in 1..=max_k {
        w *= t / k as f64;
        weights.push(w);
    }
    weights
}

/// Determine max level K where Poisson weight is still meaningful.
/// Stop when ψ_k < ε * 1e-4 and k > t (past the Poisson peak).
fn max_level(t: f64, epsilon: f64) -> usize {
    let threshold = epsilon * 1e-4;
    let mut w = (-t).exp();
    let mut k = 0;
    loop {
        if w < threshold && k > t as usize {
            return k;
        }
        k += 1;
        w *= t / k as f64;
        if k > 500 {
            return 500; // safety cap
        }
    }
}

/// Total degree (forward + reverse neighbors).
fn total_degree(graph: &Graph, idx: u32) -> usize {
    graph.fwd_neighbors(idx).len() + graph.rev_neighbors(idx).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_chain_all_nodes_scored() {
        let dir = tempfile::tempdir().unwrap();
        crate::algo::appr::tests::write_test_graph(
            dir.path(),
            4,
            &[100, 101, 102, 103],
            &[(0, 1), (1, 2), (2, 3)],
        );

        let graph = Graph::open(dir.path()).unwrap();
        let result = compute(&graph, 100, 3.0, 1e-8, None);

        assert_eq!(result.len(), 4, "all nodes should be reached");
        let scores: HashMap<i64, f64> = result.into_iter().collect();
        assert!(scores[&100] > 0.0);
        assert!(scores[&103] > 0.0);
    }

    #[test]
    fn unknown_seed_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        crate::algo::appr::tests::write_test_graph(dir.path(), 2, &[100, 101], &[(0, 1)]);

        let graph = Graph::open(dir.path()).unwrap();
        let result = compute(&graph, 999, 3.0, 1e-6, None);
        assert!(result.is_empty());
    }

    #[test]
    fn seed_dominance_lower_than_appr() {
        // HKPR should give seed a smaller fraction than APPR
        let dir = tempfile::tempdir().unwrap();
        crate::algo::appr::tests::write_test_graph(
            dir.path(),
            4,
            &[100, 101, 102, 103],
            &[(0, 1), (1, 2), (2, 3)],
        );

        let graph = Graph::open(dir.path()).unwrap();

        let appr = crate::algo::appr::compute(&graph, 100, 0.15, 1e-8, None);
        let appr_total: f64 = appr.iter().map(|&(_, s)| s).sum();
        let appr_seed_frac = appr.iter().find(|&&(id, _)| id == 100).unwrap().1 / appr_total;

        let hkpr = compute(&graph, 100, 3.0, 1e-8, None);
        let hkpr_total: f64 = hkpr.iter().map(|&(_, s)| s).sum();
        let hkpr_seed_frac = hkpr.iter().find(|&&(id, _)| id == 100).unwrap().1 / hkpr_total;

        assert!(
            hkpr_seed_frac < appr_seed_frac,
            "HKPR seed fraction ({hkpr_seed_frac:.4}) should be less than APPR ({appr_seed_frac:.4})"
        );
    }

    #[test]
    fn poisson_weights_sum_to_one() {
        let t = 3.0;
        let weights = poisson_weights(t, 50);
        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Poisson weights should sum to ~1.0, got {sum}"
        );
    }
}
