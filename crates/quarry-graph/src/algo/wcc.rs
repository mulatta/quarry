//! Weakly connected components via atomic union-find.
//!
//! Parallel edge processing with CAS-based union.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};

use rayon::prelude::*;

use crate::graph::Graph;

/// Full-graph WCC. Returns (pmid, component_id) for all nodes.
pub fn compute(graph: &Graph) -> Vec<(i64, u32)> {
    let n = graph.node_count();
    if n == 0 {
        return vec![];
    }

    // Initialize parent array: each node is its own root
    let parent: Vec<AtomicU32> = (0..n).map(|i| AtomicU32::new(i as u32)).collect();

    // Process all forward edges in parallel
    let fwd_indptr = graph.fwd_indptr();
    let fwd_indices = graph.fwd_indices();

    (0..n).into_par_iter().for_each(|i| {
        let start = fwd_indptr[i] as usize;
        let end = fwd_indptr[i + 1] as usize;
        for &j in &fwd_indices[start..end] {
            union(&parent, i as u32, j);
        }
    });

    // Collect results with path-compressed root
    (0..n as u32)
        .map(|i| (graph.id_of(i), find(&parent, i)))
        .collect()
}

/// WCC on an induced subgraph.
pub fn subgraph_components(graph: &Graph, ids: &[i64]) -> Vec<Vec<i64>> {
    let mut idx_set = HashSet::new();
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

    let global_to_local: HashMap<u32, usize> = local_nodes
        .iter()
        .enumerate()
        .map(|(li, &gi)| (gi, li))
        .collect();

    // Simple sequential union-find for subgraphs (typically small)
    let mut uf_parent: Vec<usize> = (0..n).collect();

    for (li, &gi) in local_nodes.iter().enumerate() {
        for &nb in graph.fwd_neighbors(gi) {
            if let Some(&nb_local) = global_to_local.get(&nb) {
                seq_union(&mut uf_parent, li, nb_local);
            }
        }
    }

    // Group by component
    let mut components: HashMap<usize, Vec<i64>> = HashMap::new();
    for (i, &node) in local_nodes.iter().enumerate() {
        let root = seq_find(&mut uf_parent, i);
        components
            .entry(root)
            .or_default()
            .push(graph.id_of(node));
    }

    components.into_values().collect()
}

// -- Atomic union-find --

fn find(parent: &[AtomicU32], mut x: u32) -> u32 {
    loop {
        let p = parent[x as usize].load(Ordering::Relaxed);
        if p == x {
            return x;
        }
        let gp = parent[p as usize].load(Ordering::Relaxed);
        // Path splitting
        let _ = parent[x as usize].compare_exchange_weak(p, gp, Ordering::Relaxed, Ordering::Relaxed);
        x = gp;
    }
}

fn union(parent: &[AtomicU32], a: u32, b: u32) {
    let mut ra = find(parent, a);
    let mut rb = find(parent, b);
    while ra != rb {
        if ra > rb {
            std::mem::swap(&mut ra, &mut rb);
        }
        // Try to make rb point to ra (smaller root wins)
        match parent[rb as usize].compare_exchange_weak(rb, ra, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return,
            Err(_) => {
                ra = find(parent, a);
                rb = find(parent, b);
            }
        }
    }
}

// -- Sequential union-find for subgraphs --

fn seq_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path halving
        x = parent[x];
    }
    x
}

fn seq_union(parent: &mut [usize], a: usize, b: usize) {
    let ra = seq_find(parent, a);
    let rb = seq_find(parent, b);
    if ra != rb {
        if ra < rb {
            parent[rb] = ra;
        } else {
            parent[ra] = rb;
        }
    }
}
