# Graph Algorithms

## Current Implementation

All algorithms run on a custom Rust CSR (Compressed Sparse Row) mmap structure.
34GB on disk, zero-copy access, bidirectional (forward + reverse).

| Algorithm | Type | Description | Complexity |
|-----------|------|-------------|------------|
| APPR (Andersen et al. 2006) | Local ranking | Push-based approximate personalized PageRank | O(1/ε) |
| HKPR (Kloster-Gleich 2014) | Local ranking | Heat kernel diffusion, Poisson decay | O(1/ε) |
| Expand | Candidate mining | wRRF fusion of APPR + coupling + cocitation | APPR + AA |
| WCC | Components | Parallel atomic union-find | O(V + E) |
| Betweenness | Centrality | Brandes on induced subgraph only (~10K nodes) | O(V·E) |
| Co-Citation | Pattern | Papers co-cited by same citers (Adamic-Adar) | O(shared citers) |
| Bibliographic Coupling | Pattern | Papers sharing references (Adamic-Adar) | O(shared refs) |
| Bridge Nodes | Pattern | Bidirectional BFS shortest-path intermediaries | O(max_depth) |
| Multi-Seed Bridge | Discovery | Cross-seed common_refs/citers/coupling/cocitation (Type 1-4) | O(Σ deg) |

### Library Replaceability

APPR, HKPR, Expand, Co-Citation, Bibliographic Coupling, and Multi-Seed Bridge
are **citation-graph domain-specific** — no existing library provides these.
They must remain as custom Rust implementations on the CSR structure.

WCC, Betweenness, and Bridge Nodes are generic graph algorithms replaceable by
libraries, but the current Rust implementations are already optimized for the CSR layout.

### Multi-Seed Bridge (algo/bridge.rs)

Dedicated multi-seed bridge discovery — structurally different from single-seed
patterns above. See [11-bridge.md](11-bridge.md) for full design.

| Type | Status | Description |
|------|--------|-------------|
| common_refs (Type 1) | Done | `refs(A) ∩ refs(B)`, AA-weighted |
| common_citers (Type 2) | Done | `citers(A) ∩ citers(B)`, AA-weighted |
| coupling_bridges (Type 3) | Done | Cross-seed bibliographic coupling, product scoring |
| cocitation_bridges (Type 4) | Done | Cross-seed co-citation, product scoring |
| path_bridges (Type 5) | Planned | BFS shortest path / Yen's k-paths |
| ppr_bridges (Type 6) | Planned | Per-seed APPR geometric mean |
| steiner_bridges (Type 7) | Planned | KMB heuristic (k ≥ 3 only) |

**Note**: Type 1-4 are Rust-only (`#![allow(dead_code)]`). No Python binding or
CLI yet — `lib.rs` does not expose `algo::bridge`. See 11-bridge.md Step 6.

## Future Algorithm Candidates

### High Priority

| Algorithm | Use Case | Implementation Path |
|-----------|----------|---------------------|
| Global PageRank | Paper importance ranking, quality signal | Rust on CSR (iterative, parallel) or NetworKit |
| Louvain/Leiden | Research community detection, cluster-based filtering | NetworKit PLM (parallel, ~60-120s at scale) |
| Label Propagation | Fast community detection (warm-start from Louvain) | NetworKit PLP |
| Adamic-Adar Link Prediction | Related paper recommendation beyond expand | NetworKit (already used in expand AA-weighting) |

### Medium Priority

| Algorithm | Use Case | Implementation Path |
|-----------|----------|---------------------|
| Closeness Centrality (approx) | Interdisciplinary hub paper identification | NetworKit TopCloseness |
| Katz Centrality | Influence propagation through indirect citations | NetworKit |
| Triangle Count | Local clustering coefficient, density measure | Rust on CSR or graph crate |
| Graph Diameter (approx) | Network characterization metric | NetworKit |
| Effective Diameter | 90th percentile distance, small-world property | NetworKit |

### Lower Priority

| Algorithm | Use Case | Implementation Path |
|-----------|----------|---------------------|
| Sparsification | Reduce graph for visualization (LocalDegree, Simmelian) | NetworKit (7 variants) |
| Strongly Connected Components | Citation cycle detection | NetworKit |
| Core Decomposition | k-core extraction, dense subgraph mining | NetworKit |
| Biconnected Components | Articulation points — bridge papers between fields | NetworKit |
| MaxFlow (Edmonds-Karp) | Knowledge transfer flow analysis | NetworKit |

## Library Landscape

### Scale: 183M nodes, 3.86B edges

| | Rust CSR (current) | graph crate (neo4j-labs) | NetworKit | Neo4j GDS |
|--|:-:|:-:|:-:|:-:|
| Algorithms | 8 custom | 6 | 100+ | 400+ |
| Memory | 34GB (mmap) | ~33GB (in-mem) | 44-80GB | 64-128GB (server) |
| mmap support | Yes | No | No | N/A |
| CSR native | Yes | Yes | No (conversion needed) | N/A (internal) |
| Parallelism | Rayon | Rayon | OpenMP | Internal |
| Python API | PyO3 bindings | None | Cython | Cypher/REST |
| Deployment | Embedded | Embedded | Embedded | Separate server |

### graph crate (Rust, neo4j-labs)

6 algorithms: PageRank, WCC, Afforest, SSSP, Triangle Count, DSS.
CSR-native but no mmap. Useful for PageRank and triangle counting.
Limited algorithm set, sparse documentation (45%).

### NetworKit (C++/Python)

100+ algorithms across centrality, community detection, distance, components,
flow, matching, spanning, link prediction, sparsification.
Proven at billion-edge scale (uk-2007 ~3.3B edges, Twitter ~1.47B edges).
OpenMP parallel, near-linear core scaling.
No CSR direct load — requires edge-list or METIS conversion.
Memory overhead ~25-30% vs CSR due to per-node vector structure.

### Neo4j GDS

400+ algorithms, Cypher query language, distributed clustering.
Requires separate server process, 64-128GB dedicated RAM.
Best for ad-hoc exploratory queries, overkill for batch pipeline.

## Recommended Architecture

```
Core algorithms (latency-critical, domain-specific):
  → Rust CSR mmap (APPR, HKPR, Expand, patterns)

Batch analytics (community, global centrality, link prediction):
  → NetworKit via Python (when needed)
  → Load from CSR: export edge list → nk.readGraph()
  → Or: Rust PyO3 bridge to pass adjacency data

Future scaling:
  → If distributed needed: Neo4j GDS or Apache Giraph
  → If more Rust algorithms: graph crate for PageRank/TriangleCount
```
