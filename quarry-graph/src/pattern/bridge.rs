//! Bridge nodes: nodes on shortest paths between two papers.
//!
//! Bidirectional BFS from src and dst, then collect nodes where
//! dist_from_src[v] + dist_to_dst[v] == shortest_distance.

use std::collections::{HashMap, VecDeque};

use crate::graph::Graph;

/// Find all nodes that lie on a shortest path between src and dst.
pub fn compute(graph: &Graph, src: i64, dst: i64, max_depth: usize) -> Vec<i64> {
    let Some(src_idx) = graph.resolve(src) else {
        return vec![];
    };
    let Some(dst_idx) = graph.resolve(dst) else {
        return vec![];
    };

    if src_idx == dst_idx {
        return vec![src];
    }

    // BFS from src (forward direction)
    let dist_from_src = bfs_distances(graph, src_idx, max_depth, "forward");

    // Check if dst is reachable
    let Some(&total_dist) = dist_from_src.get(&dst_idx) else {
        return vec![];
    };

    // BFS from dst (reverse direction — follow edges backwards)
    let dist_to_dst = bfs_distances(graph, dst_idx, max_depth, "reverse");

    // Collect bridge nodes: dist_from_src[v] + dist_to_dst[v] == total_dist
    let mut bridges: Vec<i64> = dist_from_src
        .iter()
        .filter_map(|(&v, &d_src)| {
            dist_to_dst.get(&v).and_then(|&d_dst| {
                if d_src + d_dst == total_dist {
                    Some(graph.id_of(v))
                } else {
                    None
                }
            })
        })
        .collect();
    bridges.sort_unstable();
    bridges
}

fn bfs_distances(
    graph: &Graph,
    start: u32,
    max_depth: usize,
    direction: &str,
) -> HashMap<u32, usize> {
    let mut dist = HashMap::new();
    dist.insert(start, 0usize);
    let mut queue = VecDeque::new();
    queue.push_back(start);

    while let Some(node) = queue.pop_front() {
        let d = dist[&node];
        if d >= max_depth {
            continue;
        }

        let neighbors = if direction == "forward" {
            graph.fwd_neighbors(node)
        } else {
            graph.rev_neighbors(node)
        };

        for &nb in neighbors {
            if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(nb) {
                e.insert(d + 1);
                queue.push_back(nb);
            }
        }
    }

    dist
}
