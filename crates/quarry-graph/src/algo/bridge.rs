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

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use rayon::prelude::*;

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
    /// Type 5: number of shortest paths (Yen's). Default 1. (Phase 2)
    #[allow(dead_code)]
    pub k_paths: usize,
    /// Type 6: APPR alpha.
    pub alpha: f64,
    /// Type 6: APPR epsilon.
    pub epsilon: f64,
    /// Types 3-4: skip neighbors with degree above this threshold.
    /// High-degree nodes have low AA weight and dominate traversal cost.
    pub max_neighbor_degree: usize,
    /// Type 5: max BFS depth for path search. Default 5.
    pub max_path_depth: usize,
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
            max_path_depth: 5,
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
    /// Pairwise shortest path lengths: Vec<(seed_i_idx, seed_j_idx, sp)>.
    /// Always computed for k ≥ 2.
    pub pairwise_sp: Vec<(usize, usize, Option<usize>)>,
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

    // Pairwise shortest path — min of both directions (directed graph is asymmetric).
    // Pairs are independent: parallelise with rayon.
    let k = seed_indices.len();
    let pairs: Vec<(usize, usize)> = (0..k)
        .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
        .collect();
    result.stats.pairwise_sp = pairs
        .par_iter()
        .map(|&(i, j)| {
            let sp_ij = shortest_path_length(
                graph,
                seed_indices[i],
                seed_indices[j],
                params.max_path_depth,
            );
            let sp_ji = shortest_path_length(
                graph,
                seed_indices[j],
                seed_indices[i],
                params.max_path_depth,
            );
            let sp = match (sp_ij, sp_ji) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a @ Some(_), None) | (None, a @ Some(_)) => a,
                (None, None) => None,
            };
            (i, j, sp)
        })
        .collect();

    // Compute bridge types. Coupling and cocitation are the most expensive
    // (O(degree² × seeds)), so run them in parallel via rayon::join.
    // The remaining types are comparatively cheap and run sequentially.
    let type_set: HashSet<BridgeType> = types.iter().copied().collect();

    if type_set.contains(&BridgeType::CommonRefs) {
        let (entries, overlap) = common_refs(graph, &seed_set, &neighbors, params.limit);
        result.stats.overlap_refs = overlap;
        result.common_refs = entries;
    }
    if type_set.contains(&BridgeType::CommonCiters) {
        let (entries, overlap) = common_citers(graph, &seed_set, &neighbors, params.limit);
        result.stats.overlap_citers = overlap;
        result.common_citers = entries;
    }

    // Coupling ‖ cocitation — both are O(degree² × seeds), independent reads only.
    let want_coupling = type_set.contains(&BridgeType::Coupling);
    let want_cocitation = type_set.contains(&BridgeType::Cocitation);
    if want_coupling || want_cocitation {
        let (coup, cocit) = rayon::join(
            || {
                if want_coupling {
                    Some(coupling_bridges(
                        graph,
                        &seed_indices,
                        &seed_set,
                        &neighbors,
                        params.limit,
                        params.max_neighbor_degree,
                    ))
                } else {
                    None
                }
            },
            || {
                if want_cocitation {
                    Some(cocitation_bridges(
                        graph,
                        &seed_indices,
                        &seed_set,
                        &neighbors,
                        params.limit,
                        params.max_neighbor_degree,
                    ))
                } else {
                    None
                }
            },
        );
        if let Some(c) = coup {
            result.coupling_bridges = c;
        }
        if let Some(c) = cocit {
            result.cocitation_bridges = c;
        }
    }

    if type_set.contains(&BridgeType::Path) {
        let (entries, sp_len) = path_bridges(
            graph,
            &seed_indices,
            &seed_set,
            params.limit,
            params.max_path_depth,
        );
        result.stats.shortest_path_length = sp_len;
        result.path_bridges = entries;
    }
    if type_set.contains(&BridgeType::Ppr) {
        result.ppr_bridges = ppr_bridges(
            graph,
            &seed_indices,
            &seed_set,
            params.alpha,
            params.epsilon,
            params.limit,
        );
    }
    if type_set.contains(&BridgeType::Steiner) {
        result.steiner_bridges = steiner_bridges(
            graph,
            &seed_indices,
            &seed_set,
            params.max_path_depth,
        );
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

    // Sort by seed_count desc, aa_weight desc, work_id asc (deterministic tiebreaker)
    entries.sort_unstable_by(|a, b| {
        b.seed_count
            .cmp(&a.seed_count)
            .then_with(|| b.aa_weight.partial_cmp(&a.aa_weight).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.work_id.cmp(&b.work_id))
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
            .then_with(|| a.work_id.cmp(&b.work_id))
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
            .then_with(|| a.work_id.cmp(&b.work_id))
    });
    entries.truncate(limit);
    entries
}

// ---------------------------------------------------------------------------
// Type 5: path_bridges (Directed BFS)
// ---------------------------------------------------------------------------

/// Directed path bridges between seeds.
///
/// Forward BFS from src (following references → older papers),
/// Reverse BFS from dst (following citers → newer papers).
/// Bridge nodes = intersection: papers on directed citation chain src→...→v→...→dst.
///
/// Why directed, not undirected: citation graphs are small-world — undirected BFS
/// always finds trivial sp=2 paths through hubs, producing 100% overlap with
/// Type 1-2 (common_refs/citers). Directed BFS forces longer, semantically
/// meaningful chains (sp=3-8) with unique results.
/// See docs/design/11-bridge.md "Architecture Decision: Why Directed" for details.
fn path_bridges(
    graph: &Graph,
    seed_indices: &[u32],
    seed_set: &HashSet<u32>,
    limit: usize,
    max_depth: usize,
) -> (Vec<PathEntry>, Option<usize>) {
    if seed_indices.len() == 2 {
        let (e_fwd, sp_fwd) = path_bridges_pair(
            graph, seed_indices[0], seed_indices[1],
            seed_set, limit, max_depth,
        );
        let (e_rev, sp_rev) = path_bridges_pair(
            graph, seed_indices[1], seed_indices[0],
            seed_set, limit, max_depth,
        );
        return match (sp_fwd, sp_rev) {
            (Some(a), Some(b)) if a <= b => (e_fwd, sp_fwd),
            (Some(_), Some(_)) => (e_rev, sp_rev),
            (Some(_), None) => (e_fwd, sp_fwd),
            (None, Some(_)) => (e_rev, sp_rev),
            (None, None) => (vec![], None),
        };
    }

    // k > 2: pairwise, merge results.
    // Step 1 — parallel: each pair computes fwd+rev BFS independently.
    let n = seed_indices.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    let pair_results: Vec<(Vec<PathEntry>, Option<usize>)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let (e_fwd, sp_fwd) = path_bridges_pair(
                graph,
                seed_indices[i],
                seed_indices[j],
                seed_set,
                0,
                max_depth,
            );
            let (e_rev, sp_rev) = path_bridges_pair(
                graph,
                seed_indices[j],
                seed_indices[i],
                seed_set,
                0,
                max_depth,
            );
            match (sp_fwd, sp_rev) {
                (Some(a), Some(b)) if a <= b => (e_fwd, sp_fwd),
                (Some(_), Some(_)) => (e_rev, sp_rev),
                (Some(_), None) => (e_fwd, sp_fwd),
                (None, Some(_)) => (e_rev, sp_rev),
                (None, None) => (vec![], None),
            }
        })
        .collect();

    // Step 2 — sequential merge: deterministic because pair_results
    // preserves input order (rayon IndexedParallelIterator guarantee).
    let mut merged: HashMap<u32, (Vec<usize>, usize)> = HashMap::new();
    let mut min_sp_len: Option<usize> = None;

    for (entries, sp_len) in pair_results {
        if let Some(d) = sp_len {
            min_sp_len = Some(min_sp_len.map_or(d, |m: usize| m.min(d)));
        }
        for e in entries {
            let idx = graph.resolve(e.work_id).unwrap();
            match merged.entry(idx) {
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    let (existing_hop, count) = o.get_mut();
                    *count += e.path_count;
                    if e.hop_from < *existing_hop {
                        *existing_hop = e.hop_from;
                    }
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert((e.hop_from, e.path_count));
                }
            }
        }
    }

    let mut result: Vec<PathEntry> = merged
        .into_iter()
        .map(|(idx, (hop_from, path_count))| PathEntry {
            work_id: graph.id_of(idx),
            hop_from,
            path_count,
        })
        .collect();

    result.sort_unstable_by(|a, b| {
        b.path_count
            .cmp(&a.path_count)
            .then_with(|| a.work_id.cmp(&b.work_id))
    });
    if limit > 0 {
        result.truncate(limit);
    }

    (result, min_sp_len)
}

/// Directed path bridges for a single pair.
fn path_bridges_pair(
    graph: &Graph,
    src: u32,
    dst: u32,
    seed_set: &HashSet<u32>,
    limit: usize,
    max_depth: usize,
) -> (Vec<PathEntry>, Option<usize>) {
    const MAX_VISITED: usize = 500_000;

    // Forward BFS from src: papers src cites (directly/transitively)
    let (dist_fwd, count_fwd) = directed_bfs(graph, src, max_depth, MAX_VISITED, Direction::Forward);

    // Check if dst is reachable via forward edges
    let total_dist = dist_fwd.get(&dst).copied();

    // Reverse BFS from dst: papers that cite dst (directly/transitively)
    let (dist_rev, count_rev) = directed_bfs(graph, dst, max_depth, MAX_VISITED, Direction::Reverse);

    // If direct forward path exists, use it
    if let Some(td) = total_dist {
        return collect_bridge_entries(
            &dist_fwd, &count_fwd, &dist_rev, &count_rev,
            td, seed_set, graph, limit,
        );
    }

    // No direct forward path. Find bridge nodes reachable from both sides:
    // dist_fwd[v] exists AND dist_rev[v] exists → v is on a directed chain
    // Total "cost" = dist_fwd[v] + dist_rev[v] (not a single shortest path,
    // but the shortest chain src→...→v→...→dst)
    let mut best_total = usize::MAX;
    for (&v, &d_fwd) in &dist_fwd {
        if let Some(&d_rev) = dist_rev.get(&v) {
            let total = d_fwd + d_rev;
            if total < best_total {
                best_total = total;
            }
        }
    }

    if best_total == usize::MAX {
        return (vec![], None);
    }

    collect_bridge_entries(
        &dist_fwd, &count_fwd, &dist_rev, &count_rev,
        best_total, seed_set, graph, limit,
    )
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Reverse,
}

/// Single-direction BFS (forward or reverse only).
fn directed_bfs(
    graph: &Graph,
    start: u32,
    max_depth: usize,
    max_visited: usize,
    direction: Direction,
) -> (HashMap<u32, usize>, HashMap<u32, usize>) {
    let mut dist: HashMap<u32, usize> = HashMap::new();
    let mut count: HashMap<u32, usize> = HashMap::new();
    dist.insert(start, 0);
    count.insert(start, 1);
    let mut queue = VecDeque::new();
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        let d = dist[&node];
        if d >= max_depth {
            continue;
        }
        let c = count[&node];

        let neighbors: &[u32] = match direction {
            Direction::Forward => graph.fwd_neighbors(node),
            Direction::Reverse => graph.rev_neighbors(node),
        };

        for &nb in neighbors {
            if dist.len() >= max_visited {
                break;
            }
            match dist.entry(nb) {
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(d + 1);
                    count.insert(nb, c);
                    queue.push_back(nb);
                }
                std::collections::hash_map::Entry::Occupied(e) => {
                    if *e.get() == d + 1 {
                        *count.entry(nb).or_insert(0) += c;
                    }
                }
            }
        }
    }

    (dist, count)
}

/// Collect bridge nodes from bidirectional distance/count maps.
#[allow(clippy::too_many_arguments)]
fn collect_bridge_entries(
    dist_src: &HashMap<u32, usize>,
    count_src: &HashMap<u32, usize>,
    dist_dst: &HashMap<u32, usize>,
    count_dst: &HashMap<u32, usize>,
    total_dist: usize,
    seed_set: &HashSet<u32>,
    graph: &Graph,
    limit: usize,
) -> (Vec<PathEntry>, Option<usize>) {
    let mut entries: Vec<PathEntry> = dist_src
        .iter()
        .filter_map(|(&v, &d_src)| {
            if seed_set.contains(&v) {
                return None;
            }
            let d_dst = *dist_dst.get(&v)?;
            if d_src + d_dst != total_dist {
                return None;
            }
            let pc_src = count_src.get(&v).copied().unwrap_or(1);
            let pc_dst = count_dst.get(&v).copied().unwrap_or(1);
            Some(PathEntry {
                work_id: graph.id_of(v),
                hop_from: vec![d_src, d_dst],
                path_count: pc_src * pc_dst,
            })
        })
        .collect();

    entries.sort_unstable_by(|a, b| {
        b.path_count
            .cmp(&a.path_count)
            .then_with(|| a.work_id.cmp(&b.work_id))
    });
    if limit > 0 {
        entries.truncate(limit);
    }

    (entries, Some(total_dist))
}

// ---------------------------------------------------------------------------
// Type 6: ppr_bridges (PPR Product)
// ---------------------------------------------------------------------------

/// PPR product bridges: geometric mean of per-seed APPR scores.
///
/// For each seed, run APPR independently. For nodes scored by ALL seeds,
/// compute score = (∏ π_i(v))^(1/k). This finds papers structurally
/// central to all seeds' citation neighborhoods in the global graph.
fn ppr_bridges(
    graph: &Graph,
    seed_indices: &[u32],
    seed_set: &HashSet<u32>,
    alpha: f64,
    epsilon: f64,
    limit: usize,
) -> Vec<ScoredEntry> {
    let k = seed_indices.len();

    // Run APPR for each seed, collecting internal index → score
    let per_seed: Vec<HashMap<u32, f64>> = seed_indices
        .iter()
        .map(|&idx| {
            let seed_id = graph.id_of(idx);
            let results = crate::algo::appr::compute(graph, seed_id, alpha, epsilon, None);
            results
                .into_iter()
                .filter_map(|(id, score)| {
                    let node_idx = graph.resolve(id)?;
                    Some((node_idx, score))
                })
                .collect()
        })
        .collect();

    // Find nodes scored by ALL seeds, compute geometric mean
    // Start from the smallest map to minimize iteration
    let smallest = per_seed
        .iter()
        .enumerate()
        .min_by_key(|(_, m)| m.len())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut entries: Vec<ScoredEntry> = per_seed[smallest]
        .iter()
        .filter_map(|(&idx, _)| {
            if seed_set.contains(&idx) {
                return None; // exclude seeds
            }
            let mut scores = Vec::with_capacity(k);
            for map in &per_seed {
                match map.get(&idx) {
                    Some(&s) if s > 0.0 => scores.push(s),
                    _ => return None, // not scored by all seeds
                }
            }
            let product: f64 = scores.iter().product();
            let geo_mean = product.powf(1.0 / k as f64);
            Some(ScoredEntry {
                work_id: graph.id_of(idx),
                per_seed_scores: scores,
                score: geo_mean,
            })
        })
        .collect();

    entries.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.work_id.cmp(&b.work_id))
    });
    entries.truncate(limit);
    entries
}

// ---------------------------------------------------------------------------
// Type 7: steiner_bridges (Steiner Tree — k ≥ 3)
// ---------------------------------------------------------------------------

/// Steiner tree bridges: internal nodes of the minimum Steiner tree
/// connecting all seeds. Uses KMB heuristic (2-approximation).
///
/// Algorithm:
/// 1. All-pairs directed BFS among k seeds → distance matrix
/// 2. Prim's MST on the complete seed graph (∞ = unreachable)
/// 3. Replace each MST edge with actual shortest path nodes
/// 4. Prune non-seed leaves
/// 5. Return internal non-seed nodes
fn steiner_bridges(
    graph: &Graph,
    seed_indices: &[u32],
    seed_set: &HashSet<u32>,
    max_depth: usize,
) -> Vec<i64> {
    let k = seed_indices.len();
    if k < 3 {
        return vec![];
    }

    const MAX_VISITED: usize = 500_000;

    // Step 1: All-pairs directed BFS (parallelised per seed).
    // Forward and reverse BFS are independent — run each set in parallel.
    let bfs_results: Vec<(HashMap<u32, usize>, HashMap<u32, u32>)> = seed_indices
        .par_iter()
        .map(|&idx| directed_bfs_with_parent(graph, idx, max_depth, MAX_VISITED, Direction::Forward))
        .collect();

    let bfs_rev_results: Vec<(HashMap<u32, usize>, HashMap<u32, u32>)> = seed_indices
        .par_iter()
        .map(|&idx| directed_bfs_with_parent(graph, idx, max_depth, MAX_VISITED, Direction::Reverse))
        .collect();

    // Build distance matrix: dist[i][j] = shortest directed path from seed_i to seed_j
    // Try both: fwd from i reaching j, OR fwd from i meeting rev from j
    let mut dist_matrix: Vec<Vec<usize>> = vec![vec![usize::MAX; k]; k];
    for i in 0..k {
        dist_matrix[i][i] = 0;
        for j in 0..k {
            if i == j {
                continue;
            }
            // Direct forward: fwd BFS from i reaches j
            if let Some(&d) = bfs_results[i].0.get(&seed_indices[j]) {
                dist_matrix[i][j] = dist_matrix[i][j].min(d);
            }
            // Bidirectional: fwd from i + rev from j meet at some node
            let (ref dist_fwd, _) = bfs_results[i];
            let (ref dist_rev, _) = bfs_rev_results[j];
            for (&v, &d_fwd) in dist_fwd {
                if let Some(&d_rev) = dist_rev.get(&v) {
                    dist_matrix[i][j] = dist_matrix[i][j].min(d_fwd + d_rev);
                }
            }
        }
    }

    // Step 2: Prim's MST on seed complete graph
    // Use minimum of dist[i][j] and dist[j][i] as undirected edge weight
    let mst_edges = prim_mst(&dist_matrix);

    // Step 3: Replace MST edges with actual path nodes
    let mut tree_nodes: HashSet<u32> = HashSet::new();
    for &(i, j) in &mst_edges {
        // Find the best meeting point and collect all nodes on the path
        let nodes = reconstruct_steiner_path(
            &bfs_results[i],
            &bfs_rev_results[j],
            &bfs_results[j],
            &bfs_rev_results[i],
            seed_indices[i],
            seed_indices[j],
        );
        tree_nodes.extend(nodes);
    }

    // Step 4-5: Return non-seed internal nodes
    let mut result: Vec<i64> = tree_nodes
        .into_iter()
        .filter(|n| !seed_set.contains(n))
        .map(|n| graph.id_of(n))
        .collect();
    result.sort_unstable();
    result
}

/// Shortest path length between two nodes.
/// Uses forward BFS from src + reverse BFS from dst, same approach as path_bridges.
/// Returns None if no path within max_depth.
fn shortest_path_length(graph: &Graph, src: u32, dst: u32, max_depth: usize) -> Option<usize> {
    if src == dst {
        return Some(0);
    }
    const MAX_VISITED: usize = 200_000;
    let (dist_fwd, _) = directed_bfs(graph, src, max_depth, MAX_VISITED, Direction::Forward);
    // Direct forward path
    if let Some(&d) = dist_fwd.get(&dst) {
        return Some(d);
    }
    let (dist_rev, _) = directed_bfs(graph, dst, max_depth, MAX_VISITED, Direction::Reverse);
    // Meeting point
    let mut best = usize::MAX;
    for (&v, &d_fwd) in &dist_fwd {
        if let Some(&d_rev) = dist_rev.get(&v) {
            best = best.min(d_fwd + d_rev);
        }
    }
    if best < usize::MAX { Some(best) } else { None }
}

/// BFS with parent tracking for path reconstruction.
fn directed_bfs_with_parent(
    graph: &Graph,
    start: u32,
    max_depth: usize,
    max_visited: usize,
    direction: Direction,
) -> (HashMap<u32, usize>, HashMap<u32, u32>) {
    let mut dist: HashMap<u32, usize> = HashMap::new();
    let mut parent: HashMap<u32, u32> = HashMap::new();
    dist.insert(start, 0);
    let mut queue = VecDeque::new();
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        let d = dist[&node];
        if d >= max_depth {
            continue;
        }

        let neighbors: &[u32] = match direction {
            Direction::Forward => graph.fwd_neighbors(node),
            Direction::Reverse => graph.rev_neighbors(node),
        };

        for &nb in neighbors {
            if dist.len() >= max_visited {
                break;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(nb) {
                e.insert(d + 1);
                parent.insert(nb, node);
                queue.push_back(nb);
            }
        }
    }

    (dist, parent)
}

/// Prim's MST on a k×k distance matrix. Returns list of (i, j) edges.
/// Unreachable pairs (usize::MAX) are skipped.
fn prim_mst(dist: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let k = dist.len();
    if k == 0 {
        return vec![];
    }

    let mut in_tree = vec![false; k];
    let mut min_edge = vec![(usize::MAX, 0usize); k]; // (cost, from)
    let mut edges = Vec::with_capacity(k - 1);

    in_tree[0] = true;
    for j in 1..k {
        let w = dist[0][j].min(dist[j][0]); // undirected weight
        min_edge[j] = (w, 0);
    }

    for _ in 0..k - 1 {
        // Find cheapest edge to a node not yet in tree
        let mut best = usize::MAX;
        let mut best_node = 0;
        for j in 0..k {
            if !in_tree[j] && min_edge[j].0 < best {
                best = min_edge[j].0;
                best_node = j;
            }
        }

        if best == usize::MAX {
            break; // remaining nodes unreachable
        }

        let from = min_edge[best_node].1;
        in_tree[best_node] = true;
        edges.push((from, best_node));

        // Update min_edge for remaining nodes
        for j in 0..k {
            if !in_tree[j] {
                let w = dist[best_node][j].min(dist[j][best_node]);
                if w < min_edge[j].0 {
                    min_edge[j] = (w, best_node);
                }
            }
        }
    }

    edges
}

/// Reconstruct path nodes between two seeds using BFS results.
/// Tries both directions (fwd_i→j and fwd_j→i) and picks the shorter one.
fn reconstruct_steiner_path(
    bfs_fwd_i: &(HashMap<u32, usize>, HashMap<u32, u32>),
    bfs_rev_j: &(HashMap<u32, usize>, HashMap<u32, u32>),
    bfs_fwd_j: &(HashMap<u32, usize>, HashMap<u32, u32>),
    bfs_rev_i: &(HashMap<u32, usize>, HashMap<u32, u32>),
    src: u32,
    dst: u32,
) -> Vec<u32> {
    // Try direction 1: fwd from src meeting rev from dst
    let path1 = find_meeting_path(&bfs_fwd_i.0, &bfs_fwd_i.1, &bfs_rev_j.0, &bfs_rev_j.1, src, dst);
    // Try direction 2: fwd from dst meeting rev from src
    let path2 = find_meeting_path(&bfs_fwd_j.0, &bfs_fwd_j.1, &bfs_rev_i.0, &bfs_rev_i.1, dst, src);

    match (path1, path2) {
        (Some(p1), Some(p2)) => {
            if p1.len() <= p2.len() { p1 } else { p2 }
        }
        (Some(p), None) | (None, Some(p)) => p,
        (None, None) => vec![],
    }
}

/// Find the best meeting point between forward and reverse BFS, return path nodes.
fn find_meeting_path(
    dist_fwd: &HashMap<u32, usize>,
    parent_fwd: &HashMap<u32, u32>,
    dist_rev: &HashMap<u32, usize>,
    parent_rev: &HashMap<u32, u32>,
    src: u32,
    dst: u32,
) -> Option<Vec<u32>> {
    // Find meeting point with minimum (total_distance, node_id).
    // Tuple ordering makes ties deterministic regardless of HashMap iteration order.
    let mut best: Option<(usize, u32)> = None;

    for (&v, &d_fwd) in dist_fwd {
        if let Some(&d_rev) = dist_rev.get(&v) {
            let candidate = (d_fwd + d_rev, v);
            if best.is_none_or(|b| candidate < b) {
                best = Some(candidate);
            }
        }
    }

    let (_, meet) = best?;

    // Trace path: src → ... → meet (via parent_fwd)
    let mut nodes = Vec::new();
    let mut cur = meet;
    while cur != src {
        nodes.push(cur);
        cur = *parent_fwd.get(&cur)?;
    }
    nodes.push(src);
    nodes.reverse();

    // Trace path: meet → ... → dst (via parent_rev)
    cur = meet;
    while cur != dst {
        cur = *parent_rev.get(&cur)?;
        nodes.push(cur);
    }

    Some(nodes)
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

    // -- Type 5: path_bridges --

    #[test]
    fn path_bridges_directed() {
        // Directed path test: A→C→D, B→D (so forward from A reaches D, reverse from B reaches D)
        // Bridge node = D (hop 2 from A forward, hop 1 from B reverse)
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("directed.csv");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "citing,cited").unwrap();
            writeln!(f, "1,3").unwrap(); // A(1) → C(3)
            writeln!(f, "3,4").unwrap(); // C(3) → D(4)
            writeln!(f, "2,4").unwrap(); // B(2) → D(4) — B cites D
            writeln!(f, "5,3").unwrap(); // E(5) → C(3) — so C has a citer
        }
        let graph_dir = dir.path().join("graph");
        crate::build::build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        std::mem::forget(dir);
        let g = Graph::open(&graph_dir).unwrap();

        // Forward from A(1): 1→3→4 (depth 2)
        // Reverse from B(2): 2→4 means 4 is cited by 2, but reverse BFS
        //   from 2 follows rev_neighbors(2) = papers that cite 2 = none
        //   Actually: reverse BFS from dst follows papers that cite dst.
        //   rev_neighbors(2) = {} (nobody cites 2)
        //   So we need dst to be cited by something.
        // Let's use seeds [1, 5]:
        //   Forward from 1: 1→3→4
        //   Reverse from 5: rev_neighbors(5) = {} (nobody cites 5)
        // That won't work either. Need:
        //   Forward from src reaches some node V, reverse from dst also reaches V.
        //   Reverse BFS from dst = papers that cite dst (transitively).
        // seeds [3, 2]: forward from 3 = {4}, reverse from 2: rev(2) = {}. No.
        // seeds [1, 4]: forward from 1 = {3, 4}. If dst=4, fwd reaches 4 directly.
        //   sp = 2 (1→3→4), bridge = 3
        let seeds = [1, 4];
        let params = BridgeParams {
            types: vec![BridgeType::Path],
            limit: 100,
            max_path_depth: 5,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);

        assert!(result.stats.shortest_path_length.is_some());
        let sp = result.stats.shortest_path_length.unwrap();
        assert_eq!(sp, 2, "1→3→4 = 2 hops");

        // Bridge = node 3 (intermediate on 1→3→4)
        assert!(!result.path_bridges.is_empty(), "should find bridge node 3");
        let bridge_ids: HashSet<i64> = result.path_bridges.iter().map(|e| e.work_id).collect();
        assert!(bridge_ids.contains(&3), "node 3 is on path 1→3→4");
        assert!(!bridge_ids.contains(&1), "seed excluded");
        assert!(!bridge_ids.contains(&4), "seed excluded");

        for e in &result.path_bridges {
            assert_eq!(e.hop_from[0] + e.hop_from[1], sp);
        }
    }

    #[test]
    fn path_bridges_unreachable() {
        // Disconnected nodes
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("disconn.csv");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "citing,cited").unwrap();
            writeln!(f, "1,2").unwrap();
            writeln!(f, "3,4").unwrap(); // separate component
        }
        let graph_dir = dir.path().join("graph");
        crate::build::build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        std::mem::forget(dir);
        let g = Graph::open(&graph_dir).unwrap();

        let seeds = [1, 4]; // different components
        let params = BridgeParams {
            types: vec![BridgeType::Path],
            max_path_depth: 10,
            ..Default::default()
        };
        let result = compute(&g, &seeds, &params);
        assert!(result.path_bridges.is_empty());
        assert!(result.stats.shortest_path_length.is_none());
    }

    #[test]
    fn pairwise_sp_asymmetric() {
        // A(1) → B(2) → C(3), no reverse edges.
        // sp(1→2) = 1, sp(2→1) = None → min = 1
        // sp(1→3) = 2, sp(3→1) = None → min = 2
        // Regardless of seed order, pairwise_sp should pick the shorter direction.
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("asym.csv");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "citing,cited").unwrap();
            writeln!(f, "1,2").unwrap();
            writeln!(f, "2,3").unwrap();
        }
        let graph_dir = dir.path().join("graph");
        crate::build::build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        std::mem::forget(dir);
        let g = Graph::open(&graph_dir).unwrap();

        // seeds = [2, 1] — "wrong" order where src=2→dst=1 has no forward path
        let params = BridgeParams {
            types: vec![BridgeType::Path],
            max_path_depth: 10,
            ..Default::default()
        };
        let result = compute(&g, &[2, 1], &params);
        // pairwise_sp should be 1 (from 1→2 direction)
        assert_eq!(result.stats.pairwise_sp.len(), 1);
        let (_, _, sp) = result.stats.pairwise_sp[0];
        assert_eq!(sp, Some(1), "min(sp(2→1)=None, sp(1→2)=1) = 1");

        // path_bridges should find the path via the shorter direction
        assert_eq!(
            result.stats.shortest_path_length,
            Some(1),
            "path bridges sp should also be 1"
        );

        // Reversed seed order should give same result
        let result2 = compute(&g, &[1, 2], &params);
        assert_eq!(result2.stats.pairwise_sp[0].2, Some(1));
        assert_eq!(result2.stats.shortest_path_length, Some(1));
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
