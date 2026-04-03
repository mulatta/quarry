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
    /// Types 3-4: skip neighbors with degree above this threshold.
    /// High-degree nodes have low AA weight and dominate traversal cost.
    pub max_neighbor_degree: usize,
}

impl Default for BridgeParams {
    fn default() -> Self {
        Self {
            types: vec![],
            limit: 100,
            k_paths: 1,
            alpha: 0.15,
            epsilon: 1e-6,
            max_neighbor_degree: 10_000,
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

/// A paper scored by per-seed overlap product.
#[derive(Clone, Debug)]
pub struct ScoredEntry {
    pub work_id: i64,
    pub per_seed_scores: Vec<f64>,
    pub score: f64,
}

/// A node on shortest path(s) between seeds.
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
            BridgeType::Coupling => {
                result.coupling_bridges = coupling_bridges(
                    graph,
                    &seed_indices,
                    &seed_set,
                    &neighbors,
                    params.limit,
                    params.max_neighbor_degree,
                );
            }
            BridgeType::Cocitation => {
                result.cocitation_bridges = cocitation_bridges(
                    graph,
                    &seed_indices,
                    &seed_set,
                    &neighbors,
                    params.limit,
                    params.max_neighbor_degree,
                );
            }
            // Types 5-7: not yet implemented
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
// Type 3: coupling_bridges (Cross-seed Bibliographic Coupling)
// ---------------------------------------------------------------------------

/// Papers whose references overlap with multiple seeds' references.
///
/// For each seed, accumulate AA-weighted overlap scores for papers that
/// cite the same references. Then keep only candidates with overlap > 0
/// for ALL seeds, scored by the product of per-seed overlaps.
fn coupling_bridges(
    graph: &Graph,
    seed_indices: &[u32],
    seed_set: &HashSet<u32>,
    neighbors: &SeedNeighbors,
    limit: usize,
    max_neighbor_degree: usize,
) -> Vec<ScoredEntry> {
    let k = seed_indices.len();

    // Per-seed overlap: for each candidate C, how much do C's refs overlap
    // with this seed's refs?
    let mut per_seed: Vec<HashMap<u32, f64>> = Vec::with_capacity(k);

    for (i, &_seed_idx) in seed_indices.iter().enumerate() {
        let mut overlap: HashMap<u32, f64> = HashMap::new();
        for &ref_idx in &neighbors.refs[i] {
            let indegree = graph.rev_neighbors(ref_idx).len();
            if indegree > max_neighbor_degree {
                continue; // skip hub refs — low AA, high traversal cost
            }
            let w = aa_weight(indegree);
            // All papers that cite this ref are coupling candidates
            for &citer in graph.rev_neighbors(ref_idx) {
                if !seed_set.contains(&citer) {
                    *overlap.entry(citer).or_insert(0.0) += w;
                }
            }
        }
        per_seed.push(overlap);
    }

    // Intersect: keep candidates present in ALL seeds' overlap maps
    score_cross_seed(&per_seed, graph, limit)
}

// ---------------------------------------------------------------------------
// Type 4: cocitation_bridges (Cross-seed Co-citation)
// ---------------------------------------------------------------------------

/// Papers that both seeds' downstream communities commonly cite.
///
/// For each seed, look at what its citers cite (2-hop forward from citers).
/// Papers cited by citers of multiple seeds are cocitation bridges.
fn cocitation_bridges(
    graph: &Graph,
    seed_indices: &[u32],
    seed_set: &HashSet<u32>,
    neighbors: &SeedNeighbors,
    limit: usize,
    max_neighbor_degree: usize,
) -> Vec<ScoredEntry> {
    let k = seed_indices.len();

    let mut per_seed: Vec<HashMap<u32, f64>> = Vec::with_capacity(k);

    for (i, &_seed_idx) in seed_indices.iter().enumerate() {
        let mut overlap: HashMap<u32, f64> = HashMap::new();
        for &citer_idx in &neighbors.citers[i] {
            let outdegree = graph.fwd_neighbors(citer_idx).len();
            if outdegree > max_neighbor_degree {
                continue; // skip review papers — low AA, high traversal cost
            }
            let w = aa_weight(outdegree);
            // All papers cited by this citer are cocitation candidates
            for &cited in graph.fwd_neighbors(citer_idx) {
                if !seed_set.contains(&cited) {
                    *overlap.entry(cited).or_insert(0.0) += w;
                }
            }
        }
        per_seed.push(overlap);
    }

    score_cross_seed(&per_seed, graph, limit)
}

/// Score candidates that appear in ALL per-seed overlap maps.
/// score = product of per-seed overlaps.
fn score_cross_seed(
    per_seed: &[HashMap<u32, f64>],
    graph: &Graph,
    limit: usize,
) -> Vec<ScoredEntry> {
    let k = per_seed.len();
    if k == 0 {
        return vec![];
    }

    // Start from the smallest map to minimize iteration
    let smallest = per_seed
        .iter()
        .enumerate()
        .min_by_key(|(_, m)| m.len())
        .map(|(i, _)| i)
        .unwrap();

    let mut entries: Vec<ScoredEntry> = per_seed[smallest]
        .iter()
        .filter_map(|(&idx, _)| {
            let mut scores = Vec::with_capacity(k);
            for map in per_seed {
                match map.get(&idx) {
                    Some(&s) => scores.push(s),
                    None => return None, // not in all seeds
                }
            }
            let product: f64 = scores.iter().product();
            Some(ScoredEntry {
                work_id: graph.id_of(idx),
                per_seed_scores: scores,
                score: product,
            })
        })
        .collect();

    entries.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(limit);
    entries
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
    ///   Seeds:
    ///     10 → 20, 30          (seed A cites 20, 30)
    ///     11 → 20, 30, 40      (seed B cites 20, 30, 40)
    ///     12 → 30, 40, 50      (seed C cites 30, 40, 50)
    ///   Citers of seeds:
    ///     60 → 10, 11, 80      (cites A+B, also cites 80)
    ///     61 → 10, 11, 12, 80  (cites all three + 80)
    ///     62 → 11, 12, 90      (cites B+C, also cites 90)
    ///   Coupling candidates (cite refs of multiple seeds):
    ///     70 → 20, 40          (cites ref 20 from A, ref 40 from B)
    ///     71 → 20, 50          (cites ref 20 from A, ref 50 from C)
    ///     72 → 30              (cites only shared ref — not a bridge)
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
            // Citers of seeds (+ cocitation targets 80, 90)
            writeln!(f, "60,10").unwrap();
            writeln!(f, "60,11").unwrap();
            writeln!(f, "60,80").unwrap();
            writeln!(f, "61,10").unwrap();
            writeln!(f, "61,11").unwrap();
            writeln!(f, "61,12").unwrap();
            writeln!(f, "61,80").unwrap();
            writeln!(f, "62,11").unwrap();
            writeln!(f, "62,12").unwrap();
            writeln!(f, "62,90").unwrap();
            // Coupling candidates
            writeln!(f, "70,20").unwrap(); // cites A's ref 20
            writeln!(f, "70,40").unwrap(); // cites B's ref 40
            writeln!(f, "71,20").unwrap(); // cites A's ref 20
            writeln!(f, "71,50").unwrap(); // cites C's ref 50
            writeln!(f, "72,30").unwrap(); // cites only shared ref 30
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

    // -- Type 3: coupling_bridges --

    #[test]
    fn coupling_two_seeds() {
        let g = test_graph();
        // Seeds A(10) refs: {20,30}, B(11) refs: {20,30,40}
        // Node 70 cites 20 (A's ref) and 40 (B-only ref) → coupling bridge
        // Node 72 cites only 30 (shared ref) → overlap with A and B, but
        //   only through shared ref, so it IS a coupling bridge
        // Node 71 cites 20 (A's ref) and 50 (C-only ref) → no overlap with B
        let seeds = [10, 11];
        let params = BridgeParams {
            types: vec![BridgeType::Coupling],
            limit: 100,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        let ids: HashSet<i64> = result.coupling_bridges.iter().map(|e| e.work_id).collect();
        // 70 cites ref 20 (A+B share) and ref 40 (B-only) → overlap_A > 0 (via 20), overlap_B > 0 (via 20,40)
        assert!(ids.contains(&70), "70 couples A and B refs");
        // 71 cites ref 20 (A+B) and ref 50 (C-only) → overlap_A > 0 (via 20), overlap_B > 0 (via 20)
        assert!(ids.contains(&71), "71 also has overlap with both A and B via ref 20");
        // Seeds should not appear
        assert!(!ids.contains(&10));
        assert!(!ids.contains(&11));

        // Score is product of per-seed overlaps
        for e in &result.coupling_bridges {
            let product: f64 = e.per_seed_scores.iter().product();
            assert!(
                (e.score - product).abs() < 1e-10,
                "score should be product of per_seed_scores"
            );
            assert_eq!(e.per_seed_scores.len(), 2);
        }
    }

    #[test]
    fn coupling_three_seeds() {
        let g = test_graph();
        // Seeds A(10), B(11), C(12)
        // A refs: {20,30}, B refs: {20,30,40}, C refs: {30,40,50}
        // Node 70 cites 20,40 → overlap_A via 20, overlap_B via 20+40, overlap_C via 40 → all > 0
        // Node 71 cites 20,50 → overlap_A via 20, overlap_B via 20, overlap_C via 50 → all > 0
        let seeds = [10, 11, 12];
        let params = BridgeParams {
            types: vec![BridgeType::Coupling],
            limit: 100,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        let ids: HashSet<i64> = result.coupling_bridges.iter().map(|e| e.work_id).collect();
        assert!(ids.contains(&70), "70 has overlap with all 3 seeds");
        assert!(ids.contains(&71), "71 has overlap with all 3 seeds");

        for e in &result.coupling_bridges {
            assert_eq!(e.per_seed_scores.len(), 3);
        }
    }

    // -- Type 4: cocitation_bridges --

    #[test]
    fn cocitation_two_seeds() {
        let g = test_graph();
        // Seeds A(10), B(11)
        // Citers of A: {60,61}, Citers of B: {60,61,62}
        // Citer 60 cites: 10,11,80 → cocitation candidates from A's perspective: {11,80}
        // Citer 61 cites: 10,11,12,80 → cocitation candidates from A's perspective: {11,12,80}
        // So overlap_A[80] > 0 (from citers 60,61)
        // Citer 60 cites: 10,11,80 → from B's perspective: {10,80}
        // Citer 61 cites: 10,11,12,80 → from B's perspective: {10,12,80}
        // Citer 62 cites: 11,12,90 → from B's perspective: {12,90}
        // So overlap_B[80] > 0 (from citers 60,61)
        // → 80 appears in both → cocitation bridge
        // 90 only appears in B's perspective (via citer 62) → not a bridge
        let seeds = [10, 11];
        let params = BridgeParams {
            types: vec![BridgeType::Cocitation],
            limit: 100,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        let ids: HashSet<i64> = result.cocitation_bridges.iter().map(|e| e.work_id).collect();
        assert!(ids.contains(&80), "80 is cited by citers of both A and B");
        assert!(!ids.contains(&90), "90 is only cited by B's citers");
        // Seeds excluded
        assert!(!ids.contains(&10));
        assert!(!ids.contains(&11));

        for e in &result.cocitation_bridges {
            let product: f64 = e.per_seed_scores.iter().product();
            assert!(
                (e.score - product).abs() < 1e-10,
                "score should be product"
            );
        }
    }

    #[test]
    fn cocitation_three_seeds() {
        let g = test_graph();
        // A citers: {60,61}, B citers: {60,61,62}, C citers: {61,62}
        // 80 is cited by 60,61 → overlap_A(80)>0, overlap_B(80)>0, overlap_C(80)>0 (via 61)
        // → cocitation bridge for all 3
        let seeds = [10, 11, 12];
        let params = BridgeParams {
            types: vec![BridgeType::Cocitation],
            limit: 100,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        let ids: HashSet<i64> = result.cocitation_bridges.iter().map(|e| e.work_id).collect();
        assert!(ids.contains(&80), "80 is cocited by all 3 seed communities");

        for e in &result.cocitation_bridges {
            assert_eq!(e.per_seed_scores.len(), 3);
        }
    }

    #[test]
    fn coupling_max_degree_prune() {
        let g = test_graph();
        let seeds = [10, 11];
        // With max_neighbor_degree=1, all refs with indegree > 1 are skipped
        // In our graph, most refs have indegree > 1 (cited by seeds + others)
        let params = BridgeParams {
            types: vec![BridgeType::Coupling],
            limit: 100,
            max_neighbor_degree: 1,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);
        // With aggressive pruning, fewer or no results
        // This just verifies it doesn't panic
        for e in &result.coupling_bridges {
            assert!(e.score > 0.0);
        }
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
