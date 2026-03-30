# Layer 1: Structure (Citation Graph Topology)

> Signal: structural proximity to seed in citation network.

## Data Source

- CSR graph: 181M nodes, 3B edges (mmap, quarry-graph Rust)
- Forward (outgoing references) + Reverse (incoming citations)

## L0 Atomic Operations

### PPR (Personalized PageRank)

- **Input**: seed node, candidate set, α (teleport probability)
- **Output**: {node_id: ppr_score} for each candidate
- **Property**: random walk with restart at seed — score decays with graph distance
- **Hub handling**: restart brings walker back to seed, suppressing hub scores
- **Dangling nodes**: 50.5% of nodes have zero out-degree → walk restarts at seed (reinforces locality)
- **Implementation**: modify existing `subgraph_pagerank` in Rust with `restart_node` parameter
- **Complexity**: O(E/ε) with push-based approximation
- **Open**: optimal α for citation networks (literature suggests 0.15-0.25, vs 0.85 for web)

### Bibliographic Coupling

- **Input**: seed node, candidate set
- **Output**: {node_id: shared_reference_count}
- **Property**: papers sharing references with seed — citation-path-independent
- **Strength**: discovers related papers even without direct/indirect citation link
- **Implementation**: existing `bibcoupling.rs`
- **Complexity**: O(d_seed × d_candidate) per pair

### Co-citation

- **Input**: seed node, candidate set
- **Output**: {node_id: co_citation_count}
- **Property**: papers cited alongside seed by third papers
- **Strength**: captures community perception ("these papers go together")
- **Implementation**: existing `co_citation.rs`
- **Complexity**: O(d_seed × d_candidate) per pair

### Adamic-Adar (future, Phase 4)

- **Input**: seed, candidate
- **Output**: score = Σ 1/log(degree(common_neighbor))
- **Property**: common neighbors weighted by their rarity — niche connections > hub connections
- **Use**: missing edge prediction / link prediction
- **Status**: not implemented

## L1 Aggregation

How to combine PPR, coupling, co-citation into one Structure score:

**Option A**: Mini-RRF within layer
```
structure_rank(paper) = 1/(k+rank_ppr) + 1/(k+rank_coupling) + 1/(k+rank_cocite)
```

**Option B**: PPR as primary, coupling/cocite as additive boost
```
structure_score(paper) = ppr_score + λ·norm(coupling) + μ·norm(cocite)
```

**Decision**: TBD. Prototype both, evaluate on test cases.

## Known Issues

- 75% of works have no reference data (zero out-degree)
- OA citation graph is all-discipline, not bio-specific → non-bio hubs
- Self-citations (1.07M) — minor but could inflate seed neighbor scores
- CSR is static snapshot — new citations not reflected until rebuild
