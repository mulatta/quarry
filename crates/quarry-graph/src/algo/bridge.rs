// No PyO3 binding yet — entire module appears unused to the compiler.
#![allow(dead_code)]

//! Bridge discovery: find papers connecting two or more seeds.
//!
//! 7 independent bridge types, each answering a different question:
//! 1. common_refs (LCA) — shared intellectual foundation
//! 2. common_citers (Common Descendant) — synthesis papers
//! 3. coupling_bridges — combined methodology usage
//! 4. cocitation_bridges — shared community attention
//! 5. path_bridges — narrative chain through literature
//! 6. ppr_bridges — structural centrality to all seeds
//! 7. steiner_bridges — minimal reading path (k ≥ 3)
//!
//! See docs/design/11-bridge.md for full design rationale.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::graph::Graph;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which bridge types to compute.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BridgeType {
    CommonRefs,
    CommonCiters,
    Coupling,
    Cocitation,
    Path,
    Ppr,
    Steiner,
}

/// Parameters for bridge computation.
pub struct BridgeParams {
    /// Which types to compute. Empty = all applicable.
    pub types: Vec<BridgeType>,
    /// Max results per type.
    pub limit: usize,
    /// Type 5: number of shortest paths (Yen's). Default 1.
    pub k_paths: usize,
    /// Type 6: APPR alpha.
    pub alpha: f64,
    /// Type 6: APPR epsilon.
    pub epsilon: f64,
}

impl Default for BridgeParams {
    fn default() -> Self {
        Self {
            types: vec![],
            limit: 100,
            k_paths: 1,
            alpha: 0.15,
            epsilon: 1e-6,
        }
    }
}

/// A paper in common_refs or common_citers.
#[derive(Clone, Debug)]
pub struct CommonEntry {
    pub work_id: i64,
    pub aa_weight: f64,
    /// How many seeds share this node (for k > 2).
    pub seed_count: usize,
}

/// A paper scored by per-seed overlap product (Types 3-4, 6).
#[derive(Clone, Debug)]
pub struct ScoredEntry {
    pub work_id: i64,
    pub per_seed_scores: Vec<f64>,
    pub score: f64,
}

/// A node on shortest path(s) between seeds (Type 5).
#[derive(Clone, Debug)]
pub struct PathEntry {
    pub work_id: i64,
    /// Distance from each seed (index matches seed order).
    pub hop_from: Vec<usize>,
    /// Number of distinct k-shortest paths passing through this node.
    pub path_count: usize,
}

/// Aggregate statistics.
#[derive(Clone, Debug, Default)]
pub struct BridgeStats {
    pub per_seed_ref_count: Vec<usize>,
    pub per_seed_citer_count: Vec<usize>,
    pub overlap_refs: usize,
    pub overlap_citers: usize,
    pub shortest_path_length: Option<usize>,
    pub elapsed_ms: u64,
}

/// Full bridge result — one vec per type, empty if not requested.
#[derive(Clone, Debug)]
pub struct BridgeResult {
    pub seeds: Vec<i64>,
    pub stats: BridgeStats,
    pub common_refs: Vec<CommonEntry>,
    pub common_citers: Vec<CommonEntry>,
    pub coupling_bridges: Vec<ScoredEntry>,
    pub cocitation_bridges: Vec<ScoredEntry>,
    pub path_bridges: Vec<PathEntry>,
    pub ppr_bridges: Vec<ScoredEntry>,
    pub steiner_bridges: Vec<i64>,
}

impl BridgeResult {
    fn empty(seeds: &[i64]) -> Self {
        Self {
            seeds: seeds.to_vec(),
            stats: BridgeStats::default(),
            common_refs: vec![],
            common_citers: vec![],
            coupling_bridges: vec![],
            cocitation_bridges: vec![],
            path_bridges: vec![],
            ppr_bridges: vec![],
            steiner_bridges: vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Adamic-Adar weight: 1/log(degree). degree ≤ 1 → 1.0.
#[inline]
fn aa_weight(degree: usize) -> f64 {
    if degree <= 1 {
        1.0
    } else {
        1.0 / (degree as f64).ln()
    }
}

/// Pre-computed neighbor sets for each seed.
struct SeedNeighbors {
    /// Forward neighbors (references) per seed.
    refs: Vec<HashSet<u32>>,
    /// Reverse neighbors (citers) per seed.
    citers: Vec<HashSet<u32>>,
}

impl SeedNeighbors {
    fn new(graph: &Graph, seed_indices: &[u32]) -> Self {
        let refs = seed_indices
            .iter()
            .map(|&idx| graph.fwd_neighbors(idx).iter().copied().collect())
            .collect();
        let citers = seed_indices
            .iter()
            .map(|&idx| graph.rev_neighbors(idx).iter().copied().collect())
            .collect();
        Self { refs, citers }
    }
}

/// Determine which types to compute.
fn effective_types(params: &BridgeParams, k: usize) -> Vec<BridgeType> {
    if params.types.is_empty() {
        let mut all = vec![
            BridgeType::CommonRefs,
            BridgeType::CommonCiters,
            BridgeType::Coupling,
            BridgeType::Cocitation,
            BridgeType::Path,
            BridgeType::Ppr,
        ];
        if k >= 3 {
            all.push(BridgeType::Steiner);
        }
        all
    } else {
        params
            .types
            .iter()
            .copied()
            .filter(|&ty| ty != BridgeType::Steiner || k >= 3)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Compute bridge papers between seeds.
///
/// Seeds must be valid work_ids present in the graph.
/// Returns per-type results; types not requested are empty vecs.
pub fn compute(graph: &Graph, seeds: &[i64], params: &BridgeParams) -> BridgeResult {
    assert!(seeds.len() >= 2, "bridge requires at least 2 seeds");

    let start = Instant::now();
    let seed_indices: Vec<u32> = seeds
        .iter()
        .map(|&id| {
            graph
                .resolve(id)
                .unwrap_or_else(|| panic!("seed {id} not found in graph"))
        })
        .collect();
    let seed_set: HashSet<u32> = seed_indices.iter().copied().collect();
    let neighbors = SeedNeighbors::new(graph, &seed_indices);
    let types = effective_types(params, seeds.len());

    let mut result = BridgeResult::empty(seeds);

    // Stats: neighbor counts
    result.stats.per_seed_ref_count = neighbors.refs.iter().map(|s| s.len()).collect();
    result.stats.per_seed_citer_count = neighbors.citers.iter().map(|s| s.len()).collect();

    for &ty in &types {
        match ty {
            BridgeType::CommonRefs => {
                let (entries, overlap) =
                    common_refs(graph, &seed_set, &neighbors, params.limit);
                result.stats.overlap_refs = overlap;
                result.common_refs = entries;
            }
            BridgeType::CommonCiters => {
                let (entries, overlap) =
                    common_citers(graph, &seed_set, &neighbors, params.limit);
                result.stats.overlap_citers = overlap;
                result.common_citers = entries;
            }
            // Types 3-7: not yet implemented
            _ => {}
        }
    }

    result.stats.elapsed_ms = start.elapsed().as_millis() as u64;
    result
}

// ---------------------------------------------------------------------------
// Type 1: common_refs (LCA)
// ---------------------------------------------------------------------------

/// Papers cited by all (or multiple) seeds.
///
/// For k=2: strict intersection. For k>2: count how many seeds cite each ref.
/// Weighted by AA: 1/log(indegree) — rare references score higher.
fn common_refs(
    graph: &Graph,
    seed_set: &HashSet<u32>,
    neighbors: &SeedNeighbors,
    limit: usize,
) -> (Vec<CommonEntry>, usize) {
    // Count how many seeds cite each reference
    let mut ref_counts: HashMap<u32, usize> = HashMap::new();
    for refs in &neighbors.refs {
        for &r in refs {
            // Exclude seeds themselves from results
            if !seed_set.contains(&r) {
                *ref_counts.entry(r).or_insert(0) += 1;
            }
        }
    }

    // Keep only refs cited by ≥ 2 seeds
    let mut entries: Vec<CommonEntry> = ref_counts
        .into_iter()
        .filter(|&(_, count)| count >= 2)
        .map(|(idx, count)| {
            let indegree = graph.rev_neighbors(idx).len();
            CommonEntry {
                work_id: graph.id_of(idx),
                aa_weight: aa_weight(indegree),
                seed_count: count,
            }
        })
        .collect();

    let overlap = entries.len();

    // Sort by seed_count desc, then aa_weight desc
    entries.sort_unstable_by(|a, b| {
        b.seed_count
            .cmp(&a.seed_count)
            .then_with(|| b.aa_weight.partial_cmp(&a.aa_weight).unwrap_or(std::cmp::Ordering::Equal))
    });
    entries.truncate(limit);

    (entries, overlap)
}

// ---------------------------------------------------------------------------
// Type 2: common_citers (Common Descendant)
// ---------------------------------------------------------------------------

/// Papers that cite all (or multiple) seeds.
///
/// Weighted by AA: 1/log(outdegree) — niche citers score higher.
fn common_citers(
    graph: &Graph,
    seed_set: &HashSet<u32>,
    neighbors: &SeedNeighbors,
    limit: usize,
) -> (Vec<CommonEntry>, usize) {
    // Count how many seeds each citer references
    let mut citer_counts: HashMap<u32, usize> = HashMap::new();
    for citers in &neighbors.citers {
        for &c in citers {
            if !seed_set.contains(&c) {
                *citer_counts.entry(c).or_insert(0) += 1;
            }
        }
    }

    // Keep only citers that reference ≥ 2 seeds
    let mut entries: Vec<CommonEntry> = citer_counts
        .into_iter()
        .filter(|&(_, count)| count >= 2)
        .map(|(idx, count)| {
            let outdegree = graph.fwd_neighbors(idx).len();
            CommonEntry {
                work_id: graph.id_of(idx),
                aa_weight: aa_weight(outdegree),
                seed_count: count,
            }
        })
        .collect();

    let overlap = entries.len();

    entries.sort_unstable_by(|a, b| {
        b.seed_count
            .cmp(&a.seed_count)
            .then_with(|| b.aa_weight.partial_cmp(&a.aa_weight).unwrap_or(std::cmp::Ordering::Equal))
    });
    entries.truncate(limit);

    (entries, overlap)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    /// Build a small test graph from CSV edges.
    ///
    /// ```text
    ///   10 → 20, 30          (seed A cites 20, 30)
    ///   11 → 20, 30, 40      (seed B cites 20, 30, 40)
    ///   12 → 30, 40, 50      (seed C cites 30, 40, 50)
    ///   60 → 10, 11          (citer of both A and B)
    ///   61 → 10, 11, 12      (citer of all three)
    ///   62 → 11, 12          (citer of B and C)
    /// ```
    fn test_graph() -> Graph {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("bridge_test.csv");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "citing,cited").unwrap();
            // Seed A (10) references
            writeln!(f, "10,20").unwrap();
            writeln!(f, "10,30").unwrap();
            // Seed B (11) references
            writeln!(f, "11,20").unwrap();
            writeln!(f, "11,30").unwrap();
            writeln!(f, "11,40").unwrap();
            // Seed C (12) references
            writeln!(f, "12,30").unwrap();
            writeln!(f, "12,40").unwrap();
            writeln!(f, "12,50").unwrap();
            // Citers
            writeln!(f, "60,10").unwrap();
            writeln!(f, "60,11").unwrap();
            writeln!(f, "61,10").unwrap();
            writeln!(f, "61,11").unwrap();
            writeln!(f, "61,12").unwrap();
            writeln!(f, "62,11").unwrap();
            writeln!(f, "62,12").unwrap();
        }
        let graph_dir = dir.path().join("graph");
        crate::build::build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        std::mem::forget(dir); // keep files alive for mmap
        Graph::open(&graph_dir).unwrap()
    }

    // -- Type 1: common_refs --

    #[test]
    fn common_refs_two_seeds() {
        let g = test_graph();
        let seeds = [10, 11]; // A and B share refs 20, 30
        let params = BridgeParams {
            types: vec![BridgeType::CommonRefs],
            limit: 100,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        let ids: HashSet<i64> = result.common_refs.iter().map(|e| e.work_id).collect();
        assert!(ids.contains(&20), "20 is shared ref of A and B");
        assert!(ids.contains(&30), "30 is shared ref of A and B");
        assert!(!ids.contains(&40), "40 is only ref of B");

        for e in &result.common_refs {
            assert_eq!(e.seed_count, 2);
            assert!(e.aa_weight > 0.0);
        }
        assert_eq!(result.stats.overlap_refs, 2);
    }

    #[test]
    fn common_refs_three_seeds_ranking() {
        let g = test_graph();
        let seeds = [10, 11, 12]; // A, B, C
        // Shared refs: 30 by all 3, 20 by A+B, 40 by B+C
        let params = BridgeParams {
            types: vec![BridgeType::CommonRefs],
            limit: 100,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        assert!(!result.common_refs.is_empty());

        // Node 30 is cited by all 3 seeds — should be first (seed_count=3)
        let first = &result.common_refs[0];
        assert_eq!(first.work_id, 30);
        assert_eq!(first.seed_count, 3);

        // Nodes 20, 40 have seed_count=2
        let rest: Vec<_> = result.common_refs.iter().skip(1).collect();
        for e in &rest {
            assert_eq!(e.seed_count, 2);
        }
    }

    // -- Type 2: common_citers --

    #[test]
    fn common_citers_two_seeds() {
        let g = test_graph();
        let seeds = [10, 11]; // citers of both: 60, 61
        let params = BridgeParams {
            types: vec![BridgeType::CommonCiters],
            limit: 100,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        let ids: HashSet<i64> = result.common_citers.iter().map(|e| e.work_id).collect();
        assert!(ids.contains(&60), "60 cites both A and B");
        assert!(ids.contains(&61), "61 cites both A and B");
        assert!(!ids.contains(&62), "62 cites only B and C");

        for e in &result.common_citers {
            assert_eq!(e.seed_count, 2);
            assert!(e.aa_weight > 0.0);
        }
        assert_eq!(result.stats.overlap_citers, 2);
    }

    #[test]
    fn common_citers_three_seeds_ranking() {
        let g = test_graph();
        let seeds = [10, 11, 12];
        // 61 cites all 3 → seed_count=3
        // 60 cites A+B → seed_count=2
        // 62 cites B+C → seed_count=2
        let params = BridgeParams {
            types: vec![BridgeType::CommonCiters],
            limit: 100,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        let first = &result.common_citers[0];
        assert_eq!(first.work_id, 61);
        assert_eq!(first.seed_count, 3);
    }

    // -- Seeds excluded --

    #[test]
    fn seeds_excluded_from_results() {
        let g = test_graph();
        let seeds = [10, 11];
        let params = BridgeParams::default();
        let result = compute(&g, &seeds, &params);

        let seed_set: HashSet<i64> = seeds.iter().copied().collect();
        for e in &result.common_refs {
            assert!(!seed_set.contains(&e.work_id), "seed in common_refs");
        }
        for e in &result.common_citers {
            assert!(!seed_set.contains(&e.work_id), "seed in common_citers");
        }
    }

    // -- Stats --

    #[test]
    fn stats_populated() {
        let g = test_graph();
        let seeds = [10, 11];
        let params = BridgeParams::default();
        let result = compute(&g, &seeds, &params);

        assert_eq!(result.stats.per_seed_ref_count.len(), 2);
        assert_eq!(result.stats.per_seed_citer_count.len(), 2);
        // Seed A (10) has 2 refs, seed B (11) has 3 refs
        assert_eq!(result.stats.per_seed_ref_count[0], 2);
        assert_eq!(result.stats.per_seed_ref_count[1], 3);
    }

    // -- AA weight --

    #[test]
    fn aa_weight_values() {
        assert_eq!(aa_weight(0), 1.0);
        assert_eq!(aa_weight(1), 1.0);
        // degree=e (~2.718) → 1/ln(e) = 1.0... but e is not integer
        // degree=10 → 1/ln(10) ≈ 0.434
        let w10 = aa_weight(10);
        assert!((w10 - 1.0 / 10_f64.ln()).abs() < 1e-10);
    }

    // -- effective_types --

    #[test]
    fn effective_types_default_k2() {
        let params = BridgeParams::default();
        let types = effective_types(&params, 2);
        assert!(!types.contains(&BridgeType::Steiner));
        assert!(types.contains(&BridgeType::CommonRefs));
        assert!(types.contains(&BridgeType::Ppr));
    }

    #[test]
    fn effective_types_default_k3() {
        let params = BridgeParams::default();
        let types = effective_types(&params, 3);
        assert!(types.contains(&BridgeType::Steiner));
    }

    #[test]
    fn effective_types_steiner_filtered_k2() {
        let params = BridgeParams {
            types: vec![BridgeType::Steiner, BridgeType::CommonRefs],
            ..Default::default()
        };
        let types = effective_types(&params, 2);
        assert!(!types.contains(&BridgeType::Steiner));
        assert!(types.contains(&BridgeType::CommonRefs));
    }

    // -- Limit --

    #[test]
    fn limit_truncates() {
        let g = test_graph();
        let seeds = [10, 11]; // 2 common refs (20, 30)
        let params = BridgeParams {
            types: vec![BridgeType::CommonRefs],
            limit: 1,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);
        assert_eq!(result.common_refs.len(), 1);
        // overlap_refs still reports full count before truncation
        assert_eq!(result.stats.overlap_refs, 2);
    }

    // -- Edge case: seeds citing each other --

    #[test]
    fn seeds_citing_each_other_excluded() {
        // Graph where seed A cites seed B
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("mutual.csv");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "citing,cited").unwrap();
            writeln!(f, "1,2").unwrap(); // seed A cites seed B
            writeln!(f, "1,3").unwrap(); // shared ref
            writeln!(f, "2,3").unwrap(); // shared ref
            writeln!(f, "4,1").unwrap(); // shared citer
            writeln!(f, "4,2").unwrap(); // shared citer
        }
        let graph_dir = dir.path().join("graph");
        crate::build::build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        std::mem::forget(dir);
        let g = Graph::open(&graph_dir).unwrap();

        let seeds = [1, 2];
        let params = BridgeParams::default();
        let result = compute(&g, &seeds, &params);

        let ref_ids: HashSet<i64> = result.common_refs.iter().map(|e| e.work_id).collect();
        // Seed B (2) is in refs of seed A, but should be excluded
        assert!(!ref_ids.contains(&1));
        assert!(!ref_ids.contains(&2));
        // Only node 3 is a valid common ref
        assert!(ref_ids.contains(&3));

        let citer_ids: HashSet<i64> = result.common_citers.iter().map(|e| e.work_id).collect();
        assert!(!citer_ids.contains(&1));
        assert!(!citer_ids.contains(&2));
    }

    // -- Integration tests (require real CSR graph) --

    #[test]
    #[ignore] // requires QUARRY_CSR_DIR
    fn integration_close_pair() {
        let csr_dir = std::env::var("QUARRY_CSR_DIR")
            .unwrap_or_else(|_| "/data/quarry/csr".to_string());
        let g = Graph::open(std::path::Path::new(&csr_dir)).unwrap();

        // KOBLAN + SICKLE (close pair, ~17.9% overlap)
        let seeds = [3119813698_i64, 3165885441];
        let params = BridgeParams {
            types: vec![BridgeType::CommonRefs, BridgeType::CommonCiters],
            limit: 50,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        assert!(!result.common_refs.is_empty(), "close pair should share refs");
        assert!(!result.common_citers.is_empty(), "close pair should have common citers");
        assert!(result.stats.elapsed_ms < 5000);
    }
}
