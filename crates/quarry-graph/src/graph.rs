//! Graph PyClass: mmap-backed CSR with query methods.
//!
//! Node IDs are i64 (OpenAlex work_id_int). Internal indices are u32.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use pyo3::prelude::*;

use crate::algo;
use crate::pattern;

/// Validate direction string, returning an error for invalid values.
fn validate_direction(direction: &str) -> PyResult<()> {
    match direction {
        "forward" | "reverse" | "both" => Ok(()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "invalid direction '{}': must be 'forward', 'reverse', or 'both'",
            direction
        ))),
    }
}

/// Memory-mapped CSR arrays for one direction.
struct CsrArrays {
    _indptr_mmap: Mmap,
    _indices_mmap: Mmap,
    num_nodes: usize,
}

impl CsrArrays {
    fn open(dir: &Path, num_nodes: usize) -> PyResult<Self> {
        let indptr_file = File::open(dir.join("indptr.bin"))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let indices_file = File::open(dir.join("indices.bin"))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let indptr_mmap = unsafe { Mmap::map(&indptr_file) }
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let indices_mmap = unsafe { Mmap::map(&indices_file) }
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Validate sizes — indptr is u32 (format 2)
        let expected_indptr_bytes = (num_nodes + 1) * 4; // u32
        if indptr_mmap.len() < expected_indptr_bytes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "indptr.bin too small: {} < {}",
                indptr_mmap.len(),
                expected_indptr_bytes
            )));
        }

        // Validate alignment (mmap is page-aligned, but assert for safety)
        assert!(
            indptr_mmap.as_ptr().align_offset(std::mem::align_of::<u32>()) == 0,
            "indptr mmap not aligned to u32"
        );
        assert!(
            indices_mmap.as_ptr().align_offset(std::mem::align_of::<u32>()) == 0,
            "indices mmap not aligned to u32"
        );

        Ok(Self {
            _indptr_mmap: indptr_mmap,
            _indices_mmap: indices_mmap,
            num_nodes,
        })
    }

    #[cfg(test)]
    fn open_raw(dir: &Path, num_nodes: usize) -> Result<Self, String> {
        let indptr_file = File::open(dir.join("indptr.bin")).map_err(|e| e.to_string())?;
        let indices_file = File::open(dir.join("indices.bin")).map_err(|e| e.to_string())?;
        let indptr_mmap = unsafe { Mmap::map(&indptr_file) }.map_err(|e| e.to_string())?;
        let indices_mmap = unsafe { Mmap::map(&indices_file) }.map_err(|e| e.to_string())?;

        let expected_indptr_bytes = (num_nodes + 1) * 4; // u32
        if indptr_mmap.len() < expected_indptr_bytes {
            return Err(format!(
                "indptr.bin too small: {} < {}",
                indptr_mmap.len(),
                expected_indptr_bytes
            ));
        }

        assert!(
            indptr_mmap.as_ptr().align_offset(std::mem::align_of::<u32>()) == 0,
            "indptr mmap not aligned to u32"
        );
        assert!(
            indices_mmap.as_ptr().align_offset(std::mem::align_of::<u32>()) == 0,
            "indices mmap not aligned to u32"
        );

        Ok(Self {
            _indptr_mmap: indptr_mmap,
            _indices_mmap: indices_mmap,
            num_nodes,
        })
    }

    fn indptr(&self) -> &[u32] {
        // SAFETY: alignment validated in open/open_raw, size validated, mmap lifetime matches self
        unsafe {
            std::slice::from_raw_parts(
                self._indptr_mmap.as_ptr() as *const u32,
                self.num_nodes + 1,
            )
        }
    }

    fn indices(&self) -> &[u32] {
        let len = self._indices_mmap.len() / 4;
        // SAFETY: alignment validated in open/open_raw, mmap lifetime matches self
        unsafe { std::slice::from_raw_parts(self._indices_mmap.as_ptr() as *const u32, len) }
    }

    fn neighbors(&self, idx: u32) -> &[u32] {
        let indptr = self.indptr();
        let start = indptr[idx as usize] as usize;
        let end = indptr[idx as usize + 1] as usize;
        &self.indices()[start..end]
    }

    fn degree(&self, idx: u32) -> usize {
        let indptr = self.indptr();
        (indptr[idx as usize + 1] - indptr[idx as usize]) as usize
    }
}

/// CSR citation graph with mmap-backed storage and Rust-native traversal.
#[pyclass]
pub struct Graph {
    fwd: CsrArrays,
    rev: CsrArrays,
    /// node index → external ID (i64), mmap-backed binary array
    _id_map_mmap: Mmap,
    num_nodes: usize,
    num_edges: usize,
}

impl Graph {
    /// Sorted i64 ID array, zero-copy from mmap.
    fn idx_to_id(&self) -> &[i64] {
        // SAFETY: id_map.bin is binary i64 array written by build.rs,
        // alignment validated on load, lifetime tied to self._id_map_mmap
        unsafe {
            std::slice::from_raw_parts(
                self._id_map_mmap.as_ptr() as *const i64,
                self.num_nodes,
            )
        }
    }

    /// Resolve external ID to node index via binary search on sorted idx_to_id.
    pub(crate) fn resolve(&self, id: i64) -> Option<u32> {
        self.idx_to_id()
            .binary_search(&id)
            .ok()
            .map(|i| i as u32)
    }

    fn csr_validated(&self, direction: &str) -> &CsrArrays {
        if direction == "reverse" {
            &self.rev
        } else {
            &self.fwd
        }
    }

    pub(crate) fn fwd_neighbors(&self, idx: u32) -> &[u32] {
        self.fwd.neighbors(idx)
    }

    pub(crate) fn rev_neighbors(&self, idx: u32) -> &[u32] {
        self.rev.neighbors(idx)
    }

    pub(crate) fn fwd_indptr(&self) -> &[u32] {
        self.fwd.indptr()
    }

    pub(crate) fn fwd_indices(&self) -> &[u32] {
        self.fwd.indices()
    }

    pub(crate) fn node_count(&self) -> usize {
        self.num_nodes
    }

    pub(crate) fn id_of(&self, idx: u32) -> i64 {
        self.idx_to_id()[idx as usize]
    }

    /// Pure-Rust constructor for tests.
    #[cfg(test)]
    pub(crate) fn open(graph_dir: &Path) -> Result<Self, String> {
        let meta_str =
            std::fs::read_to_string(graph_dir.join("meta.json")).map_err(|e| e.to_string())?;
        let meta: serde_json::Value =
            serde_json::from_str(&meta_str).map_err(|e| e.to_string())?;
        let num_nodes = meta["num_nodes"].as_u64().ok_or("missing num_nodes")? as usize;
        let num_edges = meta["num_edges"].as_u64().ok_or("missing num_edges")? as usize;

        let id_map_file =
            File::open(graph_dir.join("id_map.bin")).map_err(|e| e.to_string())?;
        let id_map_mmap =
            unsafe { Mmap::map(&id_map_file) }.map_err(|e| e.to_string())?;

        let expected_id_map_bytes = num_nodes * 8; // i64
        if id_map_mmap.len() < expected_id_map_bytes {
            return Err(format!(
                "id_map.bin too small: {} < {}",
                id_map_mmap.len(),
                expected_id_map_bytes
            ));
        }
        assert!(
            id_map_mmap.as_ptr().align_offset(std::mem::align_of::<i64>()) == 0,
            "id_map mmap not aligned to i64"
        );

        let fwd = CsrArrays::open_raw(&graph_dir.join("forward"), num_nodes)?;
        let rev = CsrArrays::open_raw(&graph_dir.join("reverse"), num_nodes)?;

        Ok(Self {
            fwd,
            rev,
            _id_map_mmap: id_map_mmap,
            num_nodes,
            num_edges,
        })
    }
}

#[pymethods]
impl Graph {
    #[new]
    fn new(graph_dir: &str) -> PyResult<Self> {
        let dir = Path::new(graph_dir);

        let meta_str = std::fs::read_to_string(dir.join("meta.json"))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let num_nodes = meta["num_nodes"]
            .as_u64()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing num_nodes"))?
            as usize;
        let num_edges = meta["num_edges"]
            .as_u64()
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing num_edges"))?
            as usize;
        let format = meta["format"].as_u64().unwrap_or(1);
        if format != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported CSR format {format}, expected 2 (rebuild with latest quarry-graph)"
            )));
        }

        // id_map.bin — binary i64 array, mmap zero-copy
        let id_map_file = File::open(dir.join("id_map.bin"))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let id_map_mmap = unsafe { Mmap::map(&id_map_file) }
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let expected_id_map_bytes = num_nodes * 8; // i64
        if id_map_mmap.len() < expected_id_map_bytes {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "id_map.bin too small: {} < {}",
                id_map_mmap.len(),
                expected_id_map_bytes
            )));
        }
        assert!(
            id_map_mmap.as_ptr().align_offset(std::mem::align_of::<i64>()) == 0,
            "id_map mmap not aligned to i64"
        );

        let fwd = CsrArrays::open(&dir.join("forward"), num_nodes)?;
        let rev = CsrArrays::open(&dir.join("reverse"), num_nodes)?;

        Ok(Self {
            fwd,
            rev,
            _id_map_mmap: id_map_mmap,
            num_nodes,
            num_edges,
        })
    }

    // -- Properties --

    #[getter]
    fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Check if a node ID exists in the graph.
    fn has_node(&self, id: i64) -> bool {
        self.resolve(id).is_some()
    }

    // -- Query methods --

    /// Direct neighbors. direction: "forward" or "reverse".
    fn neighbors(&self, id: i64, direction: &str) -> PyResult<Vec<i64>> {
        validate_direction(direction)?;
        let Some(idx) = self.resolve(id) else {
            return Ok(vec![]);
        };
        Ok(self
            .csr_validated(direction)
            .neighbors(idx)
            .iter()
            .map(|&i| self.idx_to_id()[i as usize])
            .collect())
    }

    /// (out_degree, in_degree).
    fn degree(&self, id: i64) -> (usize, usize) {
        let Some(idx) = self.resolve(id) else {
            return (0, 0);
        };
        (self.fwd.degree(idx), self.rev.degree(idx))
    }

    /// BFS k-hop neighborhood. direction: "forward", "reverse", or "both".
    fn k_hop(
        &self,
        py: Python<'_>,
        id: i64,
        k: usize,
        direction: &str,
        max_nodes: usize,
    ) -> PyResult<Vec<i64>> {
        validate_direction(direction)?;
        let Some(start) = self.resolve(id) else {
            return Ok(vec![]);
        };

        let directions: Vec<&str> = if direction == "both" {
            vec!["forward", "reverse"]
        } else {
            vec![direction]
        };

        py.allow_threads(|| {
            let mut visited = HashSet::new();
            visited.insert(start);
            let mut frontier = vec![start];

            for _ in 0..k {
                let mut next_frontier = Vec::new();
                for &node in &frontier {
                    for d in &directions {
                        for &nb in self.csr_validated(d).neighbors(node) {
                            if visited.insert(nb) {
                                next_frontier.push(nb);
                                if visited.len() >= max_nodes {
                                    return Ok(visited
                                        .iter()
                                        .map(|&i| self.idx_to_id()[i as usize])
                                        .collect());
                                }
                            }
                        }
                    }
                }
                if next_frontier.is_empty() {
                    break;
                }
                frontier = next_frontier;
            }

            Ok(visited
                .iter()
                .map(|&i| self.idx_to_id()[i as usize])
                .collect())
        })
    }

    /// Bidirectional BFS shortest path.
    fn shortest_path(
        &self,
        py: Python<'_>,
        src: i64,
        dst: i64,
        max_depth: usize,
    ) -> PyResult<Option<Vec<i64>>> {
        let src_idx = match self.resolve(src) {
            Some(i) => i,
            None => return Ok(None),
        };
        let dst_idx = match self.resolve(dst) {
            Some(i) => i,
            None => return Ok(None),
        };

        if src_idx == dst_idx {
            return Ok(Some(vec![src]));
        }

        py.allow_threads(|| {
            let mut fwd_parent: HashMap<u32, Option<u32>> = HashMap::new();
            let mut rev_parent: HashMap<u32, Option<u32>> = HashMap::new();
            fwd_parent.insert(src_idx, None);
            rev_parent.insert(dst_idx, None);
            let mut fwd_frontier: VecDeque<u32> = VecDeque::from([src_idx]);
            let mut rev_frontier: VecDeque<u32> = VecDeque::from([dst_idx]);

            for _ in 0..max_depth {
                if !fwd_frontier.is_empty() {
                    let mut next_fwd = VecDeque::new();
                    while let Some(node) = fwd_frontier.pop_front() {
                        for &nb in self.fwd.neighbors(node) {
                            if rev_parent.contains_key(&nb) {
                                return Ok(Some(self.reconstruct_path(
                                    &fwd_parent,
                                    &rev_parent,
                                    node,
                                    nb,
                                )));
                            }
                            if let std::collections::hash_map::Entry::Vacant(e) = fwd_parent.entry(nb) {
                                e.insert(Some(node));
                                next_fwd.push_back(nb);
                            }
                        }
                    }
                    fwd_frontier = next_fwd;
                }

                if !rev_frontier.is_empty() {
                    let mut next_rev = VecDeque::new();
                    while let Some(node) = rev_frontier.pop_front() {
                        for &nb in self.rev.neighbors(node) {
                            if fwd_parent.contains_key(&nb) {
                                return Ok(Some(self.reconstruct_path(
                                    &fwd_parent,
                                    &rev_parent,
                                    nb,
                                    node,
                                )));
                            }
                            if let std::collections::hash_map::Entry::Vacant(e) = rev_parent.entry(nb) {
                                e.insert(Some(node));
                                next_rev.push_back(nb);
                            }
                        }
                    }
                    rev_frontier = next_rev;
                }

                if fwd_frontier.is_empty() && rev_frontier.is_empty() {
                    break;
                }
            }

            Ok(None)
        })
    }

    // -- Full-graph algorithms --

    /// Push-based Approximate Personalized PageRank (Andersen et al. 2006).
    /// Local computation — only visits nodes near the seed. O(1/ε).
    #[pyo3(signature = (seed, alpha=0.15, epsilon=1e-6, top_k=None))]
    fn appr(
        &self,
        py: Python<'_>,
        seed: i64,
        alpha: f64,
        epsilon: f64,
        top_k: Option<usize>,
    ) -> PyResult<Vec<(i64, f64)>> {
        py.allow_threads(|| Ok(algo::appr::compute(self, seed, alpha, epsilon, top_k)))
    }

    /// Heat Kernel PageRank (Kloster & Gleich 2014).
    /// Level-based push with Poisson decay. t controls diffusion range.
    #[pyo3(signature = (seed, t=3.0, epsilon=1e-6, top_k=None))]
    fn hkpr(
        &self,
        py: Python<'_>,
        seed: i64,
        t: f64,
        epsilon: f64,
        top_k: Option<usize>,
    ) -> PyResult<Vec<(i64, f64)>> {
        py.allow_threads(|| Ok(algo::hkpr::compute(self, seed, t, epsilon, top_k)))
    }

    /// Subgraph expansion: APPR + AA coupling + AA cocitation.
    /// mode: "fused" (wRRF, focused) or "separated" (APPR + lateral slots, broad).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (seed, alpha=0.15, epsilon=1e-6, mode="fused", weights=(0.7, 0.15, 0.15), rrf_k=60, lateral_pct=25, limit=200))]
    fn expand(
        &self,
        py: Python<'_>,
        seed: i64,
        alpha: f64,
        epsilon: f64,
        mode: &str,
        weights: (f64, f64, f64),
        rrf_k: usize,
        lateral_pct: usize,
        limit: usize,
    ) -> PyResult<(Vec<HashMap<String, PyObject>>, HashMap<String, u64>)> {
        let m = match mode {
            "fused" => algo::expand::Mode::Fused,
            "separated" => algo::expand::Mode::Separated,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("invalid mode '{}': must be 'fused' or 'separated'", mode),
            )),
        };
        let params = algo::expand::ExpandParams {
            alpha, epsilon, mode: m,
            weights: [weights.0, weights.1, weights.2],
            rrf_k, lateral_pct, limit,
        };
        let result = py.allow_threads(|| algo::expand::compute(self, seed, &params));

        // Convert ExpandPaper structs to Python dicts
        let papers: Vec<HashMap<String, PyObject>> = result
            .papers
            .iter()
            .map(|p| {
                let mut d: HashMap<String, PyObject> = HashMap::new();
                d.insert("work_id".into(), p.work_id.into_pyobject(py).unwrap().into_any().unbind());
                d.insert("fused_score".into(), p.fused_score.into_pyobject(py).unwrap().into_any().unbind());
                d.insert("appr_score".into(), p.appr_score.into_pyobject(py).unwrap().into_any().unbind());

                // Bridges: list of dicts
                let bridges: Vec<HashMap<String, PyObject>> = p.bridges.iter().map(|b| {
                    let mut bd: HashMap<String, PyObject> = HashMap::new();
                    bd.insert("work_id".into(), b.work_id.into_pyobject(py).unwrap().into_any().unbind());
                    bd.insert("type".into(), match b.bridge_type {
                        algo::expand::BridgeType::SharedRef => "shared_ref",
                        algo::expand::BridgeType::SharedCiter => "shared_citer",
                    }.into_pyobject(py).unwrap().into_any().unbind());
                    bd.insert("weight".into(), b.weight.into_pyobject(py).unwrap().into_any().unbind());
                    bd
                }).collect();
                d.insert("bridges".into(), bridges.into_pyobject(py).unwrap().into_any().unbind());
                d
            })
            .collect();

        let mut stats = HashMap::new();
        stats.insert("appr_candidates".into(), result.stats.appr_candidates as u64);
        stats.insert("coupling_candidates".into(), result.stats.coupling_candidates as u64);
        stats.insert("cocitation_candidates".into(), result.stats.cocitation_candidates as u64);
        stats.insert("elapsed_ms".into(), result.stats.elapsed_ms);

        Ok((papers, stats))
    }

    /// Multi-seed bridge discovery (Type 1-4).
    ///
    /// Returns (per_type_results, stats) where per_type_results is a dict
    /// with keys: common_refs, common_citers, coupling_bridges, cocitation_bridges.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    #[pyo3(signature = (seeds, types=None, limit=100, max_neighbor_degree=10_000, max_path_depth=5, alpha=0.15, epsilon=1e-6))]
    fn bridge(
        &self,
        py: Python<'_>,
        seeds: Vec<i64>,
        types: Option<Vec<String>>,
        limit: usize,
        max_neighbor_degree: usize,
        max_path_depth: usize,
        alpha: f64,
        epsilon: f64,
    ) -> PyResult<(HashMap<String, PyObject>, HashMap<String, PyObject>)> {
        if seeds.len() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bridge requires at least 2 seeds",
            ));
        }

        let bridge_types = match types {
            Some(ts) => {
                let mut parsed = Vec::with_capacity(ts.len());
                for t in &ts {
                    parsed.push(match t.as_str() {
                        "common_refs" => algo::bridge::BridgeType::CommonRefs,
                        "common_citers" => algo::bridge::BridgeType::CommonCiters,
                        "coupling" => algo::bridge::BridgeType::Coupling,
                        "cocitation" => algo::bridge::BridgeType::Cocitation,
                        "path" => algo::bridge::BridgeType::Path,
                        "ppr" => algo::bridge::BridgeType::Ppr,
                        "steiner" => algo::bridge::BridgeType::Steiner,
                        _ => return Err(pyo3::exceptions::PyValueError::new_err(
                            format!("unknown bridge type '{t}'"),
                        )),
                    });
                }
                parsed
            }
            None => vec![],
        };

        let params = algo::bridge::BridgeParams {
            types: bridge_types,
            limit,
            max_neighbor_degree,
            max_path_depth,
            alpha,
            epsilon,
            ..Default::default()
        };

        let result = py.allow_threads(|| algo::bridge::compute(self, &seeds, &params));

        // Convert to Python dicts
        let common_refs: Vec<HashMap<String, PyObject>> = result.common_refs.iter().map(|e| {
            let mut d = HashMap::new();
            d.insert("work_id".into(), e.work_id.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("aa_weight".into(), e.aa_weight.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("seed_count".into(), e.seed_count.into_pyobject(py).unwrap().into_any().unbind());
            d
        }).collect();

        let common_citers: Vec<HashMap<String, PyObject>> = result.common_citers.iter().map(|e| {
            let mut d = HashMap::new();
            d.insert("work_id".into(), e.work_id.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("aa_weight".into(), e.aa_weight.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("seed_count".into(), e.seed_count.into_pyobject(py).unwrap().into_any().unbind());
            d
        }).collect();

        let coupling: Vec<HashMap<String, PyObject>> = result.coupling_bridges.iter().map(|e| {
            let mut d = HashMap::new();
            d.insert("work_id".into(), e.work_id.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("per_seed_scores".into(), e.per_seed_scores.clone().into_pyobject(py).unwrap().into_any().unbind());
            d.insert("score".into(), e.score.into_pyobject(py).unwrap().into_any().unbind());
            d
        }).collect();

        let cocitation: Vec<HashMap<String, PyObject>> = result.cocitation_bridges.iter().map(|e| {
            let mut d = HashMap::new();
            d.insert("work_id".into(), e.work_id.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("per_seed_scores".into(), e.per_seed_scores.clone().into_pyobject(py).unwrap().into_any().unbind());
            d.insert("score".into(), e.score.into_pyobject(py).unwrap().into_any().unbind());
            d
        }).collect();

        let path: Vec<HashMap<String, PyObject>> = result.path_bridges.iter().map(|e| {
            let mut d = HashMap::new();
            d.insert("work_id".into(), e.work_id.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("hop_from".into(), e.hop_from.clone().into_pyobject(py).unwrap().into_any().unbind());
            d.insert("path_count".into(), e.path_count.into_pyobject(py).unwrap().into_any().unbind());
            d
        }).collect();

        let ppr: Vec<HashMap<String, PyObject>> = result.ppr_bridges.iter().map(|e| {
            let mut d = HashMap::new();
            d.insert("work_id".into(), e.work_id.into_pyobject(py).unwrap().into_any().unbind());
            d.insert("per_seed_scores".into(), e.per_seed_scores.clone().into_pyobject(py).unwrap().into_any().unbind());
            d.insert("score".into(), e.score.into_pyobject(py).unwrap().into_any().unbind());
            d
        }).collect();

        let mut results = HashMap::new();
        results.insert("common_refs".into(), common_refs.into_pyobject(py).unwrap().into_any().unbind());
        results.insert("common_citers".into(), common_citers.into_pyobject(py).unwrap().into_any().unbind());
        results.insert("coupling_bridges".into(), coupling.into_pyobject(py).unwrap().into_any().unbind());
        results.insert("cocitation_bridges".into(), cocitation.into_pyobject(py).unwrap().into_any().unbind());
        results.insert("path_bridges".into(), path.into_pyobject(py).unwrap().into_any().unbind());
        results.insert("ppr_bridges".into(), ppr.into_pyobject(py).unwrap().into_any().unbind());
        results.insert("steiner_bridges".into(), result.steiner_bridges.into_pyobject(py).unwrap().into_any().unbind());

        // Stats
        let mut stats = HashMap::new();
        stats.insert("seeds".into(), result.seeds.into_pyobject(py).unwrap().into_any().unbind());
        stats.insert("per_seed_ref_count".into(), result.stats.per_seed_ref_count.into_pyobject(py).unwrap().into_any().unbind());
        stats.insert("per_seed_citer_count".into(), result.stats.per_seed_citer_count.into_pyobject(py).unwrap().into_any().unbind());
        stats.insert("overlap_refs".into(), (result.stats.overlap_refs as u64).into_pyobject(py).unwrap().into_any().unbind());
        stats.insert("overlap_citers".into(), (result.stats.overlap_citers as u64).into_pyobject(py).unwrap().into_any().unbind());
        stats.insert("shortest_path_length".into(), result.stats.shortest_path_length.into_pyobject(py).unwrap().into_any().unbind());
        // Pairwise sp as list of [i, j, sp_or_None]
        let pairwise: Vec<(usize, usize, Option<usize>)> = result.stats.pairwise_sp;
        stats.insert("pairwise_sp".into(), pairwise.into_pyobject(py).unwrap().into_any().unbind());
        stats.insert("elapsed_ms".into(), result.stats.elapsed_ms.into_pyobject(py).unwrap().into_any().unbind());

        Ok((results, stats))
    }

    /// Weakly connected components via atomic union-find.
    fn wcc(&self, py: Python<'_>) -> PyResult<Vec<(i64, u32)>> {
        py.allow_threads(|| Ok(algo::wcc::compute(self)))
    }

    // -- Pattern queries --

    /// Papers co-cited with id. limit=0 for unlimited.
    fn co_citation(
        &self,
        py: Python<'_>,
        id: i64,
        min_shared: usize,
        limit: usize,
    ) -> PyResult<Vec<(i64, usize)>> {
        py.allow_threads(|| Ok(pattern::co_citation::compute(self, id, min_shared, limit)))
    }

    /// Papers bibliographically coupled with id. limit=0 for unlimited.
    fn bibliographic_coupling(
        &self,
        py: Python<'_>,
        id: i64,
        min_shared: usize,
        limit: usize,
    ) -> PyResult<Vec<(i64, usize)>> {
        py.allow_threads(|| Ok(pattern::bibcoupling::compute(self, id, min_shared, limit)))
    }

    /// Nodes on shortest paths between src and dst.
    fn bridge_nodes(
        &self,
        py: Python<'_>,
        src: i64,
        dst: i64,
        max_depth: usize,
    ) -> PyResult<Vec<i64>> {
        py.allow_threads(|| Ok(pattern::bridge::compute(self, src, dst, max_depth)))
    }

    // -- Subgraph methods --

    /// Brandes betweenness centrality on induced subgraph.
    fn subgraph_betweenness(
        &self,
        py: Python<'_>,
        ids: Vec<i64>,
        normalized: bool,
    ) -> PyResult<Vec<(i64, f64)>> {
        py.allow_threads(|| Ok(algo::betweenness::compute(self, &ids, normalized)))
    }

    /// Weakly connected components of induced subgraph.
    fn subgraph_components(
        &self,
        py: Python<'_>,
        ids: Vec<i64>,
    ) -> PyResult<Vec<Vec<i64>>> {
        py.allow_threads(|| Ok(algo::wcc::subgraph_components(self, &ids)))
    }

    /// Edges within the induced subgraph.
    fn subgraph_edges(&self, py: Python<'_>, ids: Vec<i64>) -> PyResult<Vec<(i64, i64)>> {
        py.allow_threads(|| {
            let set: HashSet<u32> = ids.iter().filter_map(|&p| self.resolve(p)).collect();
            let mut edges = Vec::new();
            for &idx in &set {
                for &nb in self.fwd.neighbors(idx) {
                    if set.contains(&nb) {
                        edges.push((
                            self.idx_to_id()[idx as usize],
                            self.idx_to_id()[nb as usize],
                        ));
                    }
                }
            }
            Ok(edges)
        })
    }
}

impl Graph {
    fn reconstruct_path(
        &self,
        fwd_parent: &HashMap<u32, Option<u32>>,
        rev_parent: &HashMap<u32, Option<u32>>,
        fwd_meet: u32,
        rev_meet: u32,
    ) -> Vec<i64> {
        let mut fwd_path = Vec::new();
        let mut node = Some(fwd_meet);
        while let Some(n) = node {
            fwd_path.push(n);
            node = *fwd_parent.get(&n).unwrap_or(&None);
        }
        fwd_path.reverse();

        let mut node = Some(rev_meet);
        while let Some(n) = node {
            fwd_path.push(n);
            node = *rev_parent.get(&n).unwrap_or(&None);
        }

        fwd_path
            .iter()
            .map(|&i| self.idx_to_id()[i as usize])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a test graph: 1→2, 1→3, 2→3, 3→4, 4→5
    fn test_graph() -> Graph {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("test.csv");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "citing,cited").unwrap();
            writeln!(f, "1,2").unwrap();
            writeln!(f, "1,3").unwrap();
            writeln!(f, "2,3").unwrap();
            writeln!(f, "3,4").unwrap();
            writeln!(f, "4,5").unwrap();
        }
        let graph_dir = dir.path().join("graph");
        crate::build::build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        // Leak the tempdir so files remain valid for mmap
        std::mem::forget(dir);
        Graph::open(&graph_dir).unwrap()
    }

    /// Test graph with large i64 IDs (OpenAlex-scale)
    fn test_graph_large_ids() -> Graph {
        let dir = tempfile::tempdir().unwrap();
        let csv_path = dir.path().join("large.csv");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&csv_path).unwrap();
            writeln!(f, "src,dst").unwrap();
            writeln!(f, "2741809807,3141592653").unwrap();
            writeln!(f, "2741809807,2718281828").unwrap();
            writeln!(f, "3141592653,2718281828").unwrap();
        }
        let graph_dir = dir.path().join("graph");
        crate::build::build_from_csv_raw(&csv_path, &graph_dir).unwrap();
        std::mem::forget(dir);
        Graph::open(&graph_dir).unwrap()
    }

    // -- Properties --

    #[test]
    fn test_properties() {
        let g = test_graph();
        assert_eq!(g.num_nodes, 5);
        assert_eq!(g.num_edges, 5);
    }

    #[test]
    fn test_has_node() {
        let g = test_graph();
        assert!(g.has_node(1));
        assert!(g.has_node(5));
        assert!(!g.has_node(0));
        assert!(!g.has_node(999));
    }

    #[test]
    fn test_large_id_graph() {
        let g = test_graph_large_ids();
        assert_eq!(g.num_nodes, 3);
        assert_eq!(g.num_edges, 3);
        assert!(g.has_node(2741809807));
        assert!(g.has_node(3141592653));
        assert!(g.has_node(2718281828));
        assert!(!g.has_node(1));
    }

    #[test]
    fn test_large_id_neighbors() {
        let g = test_graph_large_ids();
        // 2741809807 → 3141592653, 2741809807 → 2718281828
        let mut fwd = g.neighbors(2741809807, "forward").unwrap();
        fwd.sort();
        assert_eq!(fwd, vec![2718281828, 3141592653]);
    }

    // -- Query: neighbors --

    #[test]
    fn test_neighbors_forward() {
        let g = test_graph();
        let mut fwd = g.neighbors(1, "forward").unwrap();
        fwd.sort();
        assert_eq!(fwd, vec![2, 3]);
    }

    #[test]
    fn test_neighbors_reverse() {
        let g = test_graph();
        let mut rev = g.neighbors(3, "reverse").unwrap();
        rev.sort();
        assert_eq!(rev, vec![1, 2]);
    }

    #[test]
    fn test_neighbors_missing_node() {
        let g = test_graph();
        assert_eq!(g.neighbors(999, "forward").unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn test_neighbors_invalid_direction() {
        let g = test_graph();
        assert!(g.neighbors(1, "typo").is_err());
    }

    // -- Query: degree --

    #[test]
    fn test_degree() {
        let g = test_graph();
        assert_eq!(g.degree(1), (2, 0)); // 1 cites 2,3; nobody cites 1
        assert_eq!(g.degree(3), (1, 2)); // 3 cites 4; 1,2 cite 3
        assert_eq!(g.degree(5), (0, 1)); // leaf
        assert_eq!(g.degree(999), (0, 0)); // missing
    }

    // -- Query: k_hop --

    #[test]
    fn test_k_hop_forward() {
        let g = test_graph();
        let mut hop1 = g.k_hop_inner(1, 1, "forward", 100);
        hop1.sort();
        assert_eq!(hop1, vec![1, 2, 3]);

        let mut hop2 = g.k_hop_inner(1, 2, "forward", 100);
        hop2.sort();
        assert_eq!(hop2, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_k_hop_reverse() {
        let g = test_graph();
        let mut rev = g.k_hop_inner(5, 1, "reverse", 100);
        rev.sort();
        assert_eq!(rev, vec![4, 5]);
    }

    #[test]
    fn test_k_hop_both() {
        let g = test_graph();
        let both = g.k_hop_inner(3, 1, "both", 100);
        // forward: 4, reverse: 1, 2 → total with 3: {1,2,3,4}
        assert_eq!(both.len(), 4);
    }

    #[test]
    fn test_k_hop_max_nodes() {
        let g = test_graph();
        let result = g.k_hop_inner(1, 10, "forward", 3);
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_k_hop_missing_node() {
        let g = test_graph();
        assert_eq!(g.k_hop_inner(999, 1, "forward", 100), Vec::<i64>::new());
    }

    // -- Query: shortest_path --

    #[test]
    fn test_shortest_path() {
        let g = test_graph();
        let path = g.shortest_path_inner(1, 5, 10).unwrap();
        assert_eq!(*path.first().unwrap(), 1);
        assert_eq!(*path.last().unwrap(), 5);
        assert!(path.len() <= 5);
    }

    #[test]
    fn test_shortest_path_same_node() {
        let g = test_graph();
        assert_eq!(g.shortest_path_inner(3, 3, 10), Some(vec![3]));
    }

    #[test]
    fn test_shortest_path_unreachable() {
        let g = test_graph();
        assert!(g.shortest_path_inner(5, 1, 10).is_none());
    }

    #[test]
    fn test_shortest_path_missing_node() {
        let g = test_graph();
        assert!(g.shortest_path_inner(999, 1, 10).is_none());
        assert!(g.shortest_path_inner(1, 999, 10).is_none());
    }

    // -- Pattern: co_citation --

    #[test]
    fn test_co_citation() {
        let g = test_graph();
        // Node 3 is cited by 1 and 2. Citer 1 also cites 2.
        let result = pattern::co_citation::compute(&g, 3, 1, 0);
        assert!(result.iter().any(|&(id, _)| id == 2));
    }

    #[test]
    fn test_co_citation_min_shared_filter() {
        let g = test_graph();
        // With min_shared=10, should return nothing for this small graph
        let result = pattern::co_citation::compute(&g, 3, 10, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_co_citation_limit() {
        let g = test_graph();
        let result = pattern::co_citation::compute(&g, 3, 1, 1);
        assert!(result.len() <= 1);
    }

    #[test]
    fn test_co_citation_missing_node() {
        let g = test_graph();
        assert!(pattern::co_citation::compute(&g, 999, 1, 0).is_empty());
    }

    // -- Pattern: bibliographic coupling --

    #[test]
    fn test_bibliographic_coupling() {
        let g = test_graph();
        // Node 1 cites 2,3. Node 2 cites 3. Both cite 3 → coupled with count 1.
        let result = pattern::bibcoupling::compute(&g, 1, 1, 0);
        assert!(result.iter().any(|&(id, _)| id == 2));
    }

    #[test]
    fn test_bibliographic_coupling_limit() {
        let g = test_graph();
        let result = pattern::bibcoupling::compute(&g, 1, 1, 1);
        assert!(result.len() <= 1);
    }

    // -- Pattern: bridge_nodes --

    #[test]
    fn test_bridge_nodes() {
        let g = test_graph();
        let mut bridges = pattern::bridge::compute(&g, 1, 5, 10);
        bridges.sort();
        // 1→3→4→5 is shortest, so bridges are {1,3,4,5}
        assert!(bridges.contains(&1));
        assert!(bridges.contains(&5));
        assert!(bridges.len() >= 3);
    }

    #[test]
    fn test_bridge_nodes_same() {
        let g = test_graph();
        assert_eq!(pattern::bridge::compute(&g, 3, 3, 10), vec![3]);
    }

    #[test]
    fn test_bridge_nodes_unreachable() {
        let g = test_graph();
        assert!(pattern::bridge::compute(&g, 5, 1, 10).is_empty());
    }

    // -- Subgraph --

    #[test]
    fn test_subgraph_edges() {
        let g = test_graph();
        let mut edges = g.subgraph_edges_inner(vec![1, 2, 3]);
        edges.sort();
        assert_eq!(edges, vec![(1, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn test_subgraph_edges_empty() {
        let g = test_graph();
        assert!(g.subgraph_edges_inner(vec![]).is_empty());
    }

    #[test]
    fn test_subgraph_components() {
        let g = test_graph();
        let comps = algo::wcc::subgraph_components(&g, &[1, 2, 3, 5]);
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn test_appr() {
        let g = test_graph();
        let result = algo::appr::compute(&g, 1, 0.15, 1e-8, None);
        assert!(!result.is_empty());
        assert!(result.iter().all(|&(_, s)| s > 0.0));
        // seed should have highest score
        assert_eq!(result[0].0, 1);
    }

    #[test]
    fn test_subgraph_betweenness() {
        let g = test_graph();
        let bc = algo::betweenness::compute(&g, &[1, 2, 3, 4, 5], false);
        assert_eq!(bc.len(), 5);
        // Node 3 should have highest betweenness
        assert_eq!(bc[0].0, 3);
    }

    #[test]
    fn test_subgraph_betweenness_normalized() {
        let g = test_graph();
        let bc = algo::betweenness::compute(&g, &[1, 2, 3, 4, 5], true);
        assert_eq!(bc.len(), 5);
        // Normalized values should be <= 1.0
        assert!(bc.iter().all(|&(_, s)| s <= 1.0));
    }

    // -- Full-graph --

    #[test]
    fn test_wcc() {
        let g = test_graph();
        let comps = algo::wcc::compute(&g);
        let roots: HashSet<u32> = comps.iter().map(|&(_, r)| r).collect();
        assert_eq!(roots.len(), 1);
    }

    #[test]
    fn test_appr_top_k() {
        let g = test_graph();
        let result = algo::appr::compute(&g, 1, 0.15, 1e-8, Some(2));
        assert!(result.len() <= 2);
    }

    // -- Helpers for tests (avoid PyO3 `py` parameter) --
    impl Graph {
        fn k_hop_inner(&self, id: i64, k: usize, direction: &str, max_nodes: usize) -> Vec<i64> {
            let Some(start) = self.resolve(id) else {
                return vec![];
            };
            let directions: Vec<&str> = if direction == "both" {
                vec!["forward", "reverse"]
            } else {
                vec![direction]
            };
            let mut visited = HashSet::new();
            visited.insert(start);
            let mut frontier = vec![start];
            for _ in 0..k {
                let mut next = Vec::new();
                for &node in &frontier {
                    for d in &directions {
                        for &nb in self.csr_validated(d).neighbors(node) {
                            if visited.insert(nb) {
                                next.push(nb);
                                if visited.len() >= max_nodes {
                                    return visited
                                        .iter()
                                        .map(|&i| self.idx_to_id()[i as usize])
                                        .collect();
                                }
                            }
                        }
                    }
                }
                if next.is_empty() {
                    break;
                }
                frontier = next;
            }
            visited
                .iter()
                .map(|&i| self.idx_to_id()[i as usize])
                .collect()
        }

        fn shortest_path_inner(&self, src: i64, dst: i64, max_depth: usize) -> Option<Vec<i64>> {
            let src_idx = self.resolve(src)?;
            let dst_idx = self.resolve(dst)?;
            if src_idx == dst_idx {
                return Some(vec![src]);
            }
            let mut fwd_parent: HashMap<u32, Option<u32>> = HashMap::new();
            let mut rev_parent: HashMap<u32, Option<u32>> = HashMap::new();
            fwd_parent.insert(src_idx, None);
            rev_parent.insert(dst_idx, None);
            let mut fwd_frontier = VecDeque::from([src_idx]);
            let mut rev_frontier = VecDeque::from([dst_idx]);
            for _ in 0..max_depth {
                if !fwd_frontier.is_empty() {
                    let mut next = VecDeque::new();
                    while let Some(node) = fwd_frontier.pop_front() {
                        for &nb in self.fwd.neighbors(node) {
                            if rev_parent.contains_key(&nb) {
                                return Some(self.reconstruct_path(
                                    &fwd_parent,
                                    &rev_parent,
                                    node,
                                    nb,
                                ));
                            }
                            if let std::collections::hash_map::Entry::Vacant(e) = fwd_parent.entry(nb) {
                                e.insert(Some(node));
                                next.push_back(nb);
                            }
                        }
                    }
                    fwd_frontier = next;
                }
                if !rev_frontier.is_empty() {
                    let mut next = VecDeque::new();
                    while let Some(node) = rev_frontier.pop_front() {
                        for &nb in self.rev.neighbors(node) {
                            if fwd_parent.contains_key(&nb) {
                                return Some(self.reconstruct_path(
                                    &fwd_parent, &rev_parent, nb, node,
                                ));
                            }
                            if let std::collections::hash_map::Entry::Vacant(e) = rev_parent.entry(nb) {
                                e.insert(Some(node));
                                next.push_back(nb);
                            }
                        }
                    }
                    rev_frontier = next;
                }
                if fwd_frontier.is_empty() && rev_frontier.is_empty() {
                    break;
                }
            }
            None
        }

        fn subgraph_edges_inner(&self, ids: Vec<i64>) -> Vec<(i64, i64)> {
            let set: HashSet<u32> = ids.iter().filter_map(|&p| self.resolve(p)).collect();
            let mut edges = Vec::new();
            for &idx in &set {
                for &nb in self.fwd.neighbors(idx) {
                    if set.contains(&nb) {
                        edges.push((
                            self.idx_to_id()[idx as usize],
                            self.idx_to_id()[nb as usize],
                        ));
                    }
                }
            }
            edges
        }
    }
}
