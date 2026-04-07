# Bridge Discovery

> Status: Design confirmed. Implementation pending.

## Problem

Given two or more seed papers from different subfields, find papers that
**connect** them. Single-seed expand finds papers *related to* one seed;
bridge finds papers *between* seeds.

Bridge discovery is the primary multi-seed use case for niche/interdisciplinary
ideation.

## Design Principle: Separate Bridge Types

Different algorithms find structurally different bridges. Mixing them into
one ranked list loses information. Each bridge type answers a distinct question
and is computed independently.

This follows quarry's philosophy: "data pipeline — returns complete data,
not opinions" (consistent with AD-2, AD-3).

## 7 Bridge Types

### Type 1: common_refs (LCA)

**Definition**: `refs(A) ∩ refs(B)` — papers cited by all seeds.

**Meaning**: Shared intellectual foundation.

**Question answered**: "What theoretical/methodological basis do these
papers share?"

**Example**: Gaudelli 2017 (ABE) — cited by both progeria BE and sickle cell BE.

**Computation**:

```
fwd_neighbors(seed_A) ∩ fwd_neighbors(seed_B)
Each shared ref weighted by AA: 1/log(indegree)
```

**Cost**: O(deg(A) + deg(B)), < 1ms.

**k > 2**: Full intersection (all seeds cite) + pairwise intersections
with `cited_by_seeds` count.

**Phase 3**: Embedding filters "core foundation" vs "pro-forma citation".

______________________________________________________________________

### Type 2: common_citers (Common Descendant)

**Definition**: `citers(A) ∩ citers(B)` — papers that cite all seeds.

**Meaning**: Synthesis/comparison/review papers that already combine
both fields.

**Question answered**: "Who has already looked at these topics together?"

**Example**: "In vivo base editing for monogenic diseases" (review) — cites
both progeria BE and sickle cell BE.

**Computation**:

```
rev_neighbors(seed_A) ∩ rev_neighbors(seed_B)
Each shared citer weighted by AA: 1/log(outdegree)
```

**Cost**: O(deg(A) + deg(B)), < 1ms.

**k > 2**: Full intersection + pairwise, with `cites_seeds` count.

**Phase 3**: Embedding verifies whether the citer actually compares/synthesizes
vs merely lists both in a generic review.

______________________________________________________________________

### Type 3: coupling_bridges (Cross-seed Bibliographic Coupling)

**Definition**: Paper C where `refs(C) ∩ refs(A) > 0` AND
`refs(C) ∩ refs(B) > 0`.

**Meaning**: A third paper that combines methodologies/concepts from both
seeds' reference worlds.

**Question answered**: "What research actually uses methods from both fields?"

**Example**: A paper citing both in-vivo delivery methods (A's world) and
HSC therapy protocols (B's world).

**Computation**:

```
For each ref r in refs(A):
    For each citer C of r:
        overlap_A(C) += AA_weight(r)
Filter to candidates where overlap_A > 0

For each candidate C:
    overlap_B(C) = Σ AA_weight(r) for r ∈ refs(C) ∩ refs(B)

score(C) = overlap_A(C) × overlap_B(C)
Return candidates where score > 0
```

**Cost**: O(Σ indegree(ref)), ~100ms.

**k > 2**: `score = Π overlap_i` over all seeds.

**Phase 3**: Embedding midpoint candidates validated by coupling score.

**Origin**: Kessler (1963) bibliographic coupling + Uzzi (2013) atypical
combinations.

______________________________________________________________________

### Type 4: cocitation_bridges (Cross-seed Co-citation)

**Definition**: Paper C where `citers(A)` and `citers(B)` communities both
reference C.

**Meaning**: A paper that both seeds' downstream communities consider important.

**Question answered**: "What do researchers in both fields commonly read?"

**Computation**:

```
For each citer c of A:
    For each paper C cited by c:
        overlap_A(C) += AA_weight(c)

For each candidate C:
    overlap_B(C) = Σ AA_weight(c) for c ∈ citers(B) that cite C

score(C) = overlap_A(C) × overlap_B(C)
```

**Cost**: O(Σ outdegree(citer)), ~100ms.

**Distinction from coupling**: Coupling = "author of C combined both worlds".
Co-citation = "both communities pay attention to C".

**Phase 3**: Embedding distinguishes "genuinely important to both" vs
"high-citation hub coincidence".

**Origin**: Small (1973) co-citation analysis.

______________________________________________________________________

### Type 5: path_bridges (Shortest Path / K-Shortest Paths)

**Definition**: Intermediate nodes on the shortest path(s) between seeds
in the undirected citation graph.

**Meaning**: Stepping-stone papers — "read these in order to get from A to B".

**Question answered**: "How does the literature connect these two topics?"

**Example**: progeria BE → ABE method → CRISPR review → gene therapy →
sickle cell BE. The intermediate papers form a narrative chain.

**Computation**:

```
k=1: BFS on undirected view (fwd + rev), trace path
k>1: Yen's algorithm for k-shortest simple paths
     Nodes appearing in multiple paths = restricted betweenness bridges
```

**Cost**: k=1: O(V+E) worst case, typically a few hops. k>1: O(k × V²) on
reachable subgraph.

**Parameters**: `--k-paths N` (default 1). When k > 1, each node gets a
`path_count` field (number of distinct shortest paths passing through it).

**k > 2 seeds**: Pairwise shortest paths. For unified multi-seed paths,
see Type 7 (Steiner tree).

**Phase 3**: When no citation path exists (far pair), embedding provides
an "implicit path" — semantic stepping stones.

______________________________________________________________________

### Type 6: ppr_bridges (PPR Product)

**Definition**: `score(v) = geometric_mean(APPR_1(v), ..., APPR_k(v))`.

**Meaning**: Papers that are structurally central to all seeds' citation
neighborhoods in the global graph.

**Question answered**: "What papers are important in both fields' networks?"

**Distinction from Types 1-4**: Types 1-4 are local (1-2 hop neighborhood).
Type 5 is global but deterministic (path). Type 6 is global and probabilistic
(random walk).

**Computation**:

```
For each seed_i:
    π_i = APPR(graph, seed_i, α, ε)

For each node v scored by ALL seeds:
    score(v) = (Π π_i(v))^(1/k)

Exclude seeds. Sort descending. Truncate to limit.
```

**Cost**: O(k/ε), ~1s per seed.

**k > 2**: Natural extension — geometric mean of k scores.

**Phase 3**: Embedding midpoint ANN may subsume this for far pairs. PPR
product remains valuable for close/medium pairs where citation structure
is informative.

**Origin**: Haveliwala (2002) personalized PageRank; Lofgren et al. (2016)
PPR product ≈ meeting probability of independent random walks.

______________________________________________________________________

### Type 7: steiner_bridges (Steiner Tree — k ≥ 3)

**Definition**: Non-seed internal nodes of the minimum Steiner tree
connecting all seeds.

**Meaning**: The minimal set of papers needed to connect all seeds. "Read
only these to understand how all topics relate."

**Question answered**: "What is the most efficient reading path through
all these topics?"

**Distinction from path_bridges**: Pairwise paths may have redundant
overlaps. Steiner tree finds the optimal shared structure.

```
path_bridges(A,B) + path_bridges(A,C) + path_bridges(B,C)
  = 3 independent paths, may share segments redundantly

steiner_bridges(A,B,C)
  = 1 optimal tree, exploits shared segments

    A ──→ X ──→ Y ──→ B       (path A-B)
    A ──→ X ──→ Z ──→ C       (path A-C, shares A→X)

    Steiner tree: A → X → Y → B
                       └→ Z → C
    Steiner nodes: X, Y, Z
```

**Computation**: KMB heuristic (Kou-Markowsky-Berman 1981):

```
1. All-pairs shortest path among k seeds → distance matrix
2. Build complete graph on seeds, edge weight = shortest path distance
3. MST of complete graph (Prim/Kruskal)
4. Replace each MST edge with actual shortest path
5. Prune non-seed leaves
6. Internal non-seed nodes = Steiner bridges
```

**Cost**: O(k² × V) for all-pairs BFS among k seeds.

**k = 2**: Identical to shortest path (Type 5). Skipped; use path_bridges.

**k ≥ 3**: Distinct from pairwise paths. Activated automatically.

**Phase 3**: Embedding-based "semantic Steiner" — find papers that minimize
semantic distance to all seeds, even without citation paths.

**Origin**: Steiner tree problem (Karp 1972, NP-hard). KMB heuristic gives
2-approximation in polynomial time.

______________________________________________________________________

## Summary Table

| Type | Scope | Cost | k=2 | k≥3 | Question |
|------|-------|------|-----|-----|----------|
| common_refs | local 1-hop | O(d) | ✅ | ✅ | Shared foundation? |
| common_citers | local 1-hop | O(d) | ✅ | ✅ | Who synthesized both? |
| coupling | local 2-hop | O(Σd) | ✅ | ✅ | Who combined both methods? |
| cocitation | local 2-hop | O(Σd) | ✅ | ✅ | What do both communities read? |
| path | global | O(V) | ✅ | pairwise | How does literature connect them? |
| ppr | global | O(k/ε) | ✅ | ✅ | What's central to both networks? |
| steiner | global | O(k²V) | skip | ✅ | Minimal reading path through all? |

## Architecture Decision: Why Not APPR-based Multi-Seed

Evaluated and rejected (Phase 2a experiment):

| Approach | Problem |
|----------|---------|
| Shared teleport APPR (field mode) | = union of single-seed expands, no new information |
| APPR intersection (common mode) | Score > 0 filter too permissive, 72% noise in results |
| APPR intersection + coupling (bridge mode) | Far pair overlap = 0, cannot find bridges |

Root cause: APPR is designed for "neighborhood exploration around a seed",
not "connection between seeds". Citation topology bridge discovery requires
algorithms that directly examine inter-seed structure (shared refs, shared
citers, paths), not diffusion-based scoring.

APPR remains the core algorithm for single-seed `expand`. Bridge uses
dedicated algorithms per type.

## CLI Design

```bash
# All types (steiner only if k ≥ 3)
quarry bridge W1234 W5678

# Specific type
quarry bridge W1234 W5678 --type common_refs
quarry bridge W1234 W5678 --type path --k-paths 5

# 3+ seeds (enables steiner)
quarry bridge W1234 W5678 W9012

# Per-type limit
quarry bridge W1234 W5678 --limit 50

# Output
quarry bridge W1234 W5678 --format json > bridges.json
quarry bridge W1234 W5678 --format table
```

## Output Schema

```json
{
  "seeds": [1234, 5678],
  "stats": {
    "per_seed_ref_count": [45, 62],
    "per_seed_citer_count": [120, 89],
    "overlap_refs": 15,
    "overlap_citers": 8,
    "shortest_path_length": 4,
    "elapsed_ms": 230
  },
  "common_refs": [
    {"work_id": 111, "aa_weight": 0.85, "cited_by_seeds": 2}
  ],
  "common_citers": [
    {"work_id": 222, "aa_weight": 0.72, "cites_seeds": 2}
  ],
  "coupling_bridges": [
    {"work_id": 333, "per_seed_overlap": [0.6, 0.4], "score": 0.24}
  ],
  "cocitation_bridges": [
    {"work_id": 444, "per_seed_overlap": [0.5, 0.3], "score": 0.15}
  ],
  "path_bridges": [
    {"work_id": 555, "hop_from_a": 2, "hop_from_b": 2, "path_count": 1}
  ],
  "ppr_bridges": [
    {"work_id": 666, "per_seed_scores": [0.003, 0.001], "score": 0.0017}
  ],
  "steiner_bridges": []
}
```

## Phase 3 Extension Points

```
Citation candidates (Phase 2)        Semantic candidates (Phase 3)
  common_refs                          embedding midpoint ANN
  common_citers                        abstract similarity search
  coupling_bridges
  cocitation_bridges
  path_bridges
  ppr_bridges
  steiner_bridges
         │                                      │
         ▼                                      ▼
    ┌──────── Candidate Pool (union) ─────────┐
    └────────────────┬────────────────────────┘
                     ▼
    ┌──────── Scoring ────────────────────────┐
    │  citation: type-specific scores          │
    │  semantic: embedding similarity (Phase 3)│
    └────────────────┬────────────────────────┘
                     ▼
              per-type ranked bridges
```

Key insight: For far pairs (overlap < 1%), citation-only bridges are sparse.
Embedding candidates fill the gap. The candidate/scoring separation means
Phase 3 adds a new candidate source without changing existing scoring.

Distance classification (free — computed as byproduct of bridge):

- overlap > 10%: citation bridges sufficient
- overlap 1-10%: citation bridges + caution
- overlap < 1%: citation bridges sparse, recommend embedding (Phase 3)

## Implementation Plan

| Step | Scope | Cost |
|------|-------|------|
| 1 | Rust `bridge()` + common_refs/common_citers | Low |
| 2 | coupling_bridges + cocitation_bridges | Low (refactor from expand) |
| 3 | path_bridges (BFS shortest path, k-paths via Yen) | Medium |
| 4 | ppr_bridges (per-seed APPR product) | Low (reuse APPR) |
| 5 | steiner_bridges (KMB heuristic) | Medium |
| 6 | Python binding + CLI | Low |
| 7 | Quality evaluation (abstract review, 3 seed pairs) | Eval |

## References

- Kessler (1963) — Bibliographic coupling
- Small (1973) — Co-citation analysis
- Karp (1972) — Steiner tree (NP-hardness)
- Kou, Markowsky, Berman (1981) — Steiner tree 2-approximation
- Freeman (1977) — Betweenness centrality
- Haveliwala (2002) — Personalized PageRank
- Uzzi et al. (2013) — Atypical combinations and scientific impact
- Granovetter (1973) — Strength of weak ties
- Lofgren et al. (2016) — PPR product as random walk meeting probability
- Yen (1971) — K-shortest simple paths algorithm
