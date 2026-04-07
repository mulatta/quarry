# System Layers

> Status: Design. Documents the full system architecture across CLI,
> MCP, Agent, and Core layers.

## Layer Overview

```
┌───────────────────────────────────────────────────────────┐
│  User Interface                                           │
│  CLI (human) / Claude Code SKILL / MCP Client (agent)     │
└──────────────────┬────────────────────────────────────────┘
                   │
┌──────────────────▼────────────────────────────────────────┐
│  Agent Layer (LLM judgment)                               │
│  BAML: type-safe LLM function definitions                 │
│  DSPy: pipeline optimization (Phase 2)                    │
│  SKILL: workflow orchestration                            │
└──────────────────┬────────────────────────────────────────┘
                   │ MCP tool calls
┌──────────────────▼────────────────────────────────────────┐
│  quarry MCP Server (atomic tools, no LLM)                 │
└──────────────────┬────────────────────────────────────────┘
                   │ Rust FFI + PG + LanceDB
┌──────────────────▼────────────────────────────────────────┐
│  quarry Core (Rust + Python, no LLM)                      │
└───────────────────────────────────────────────────────────┘
```

### Layer Principles

| Layer | Principle | Contains | Does NOT contain |
|-------|-----------|----------|-----------------|
| Core | Computation only, no opinions | Graph algorithms, DB queries, scoring | Seed selection, filtering, judgment |
| MCP | Atomic tools, composable | search, expand, bridge, diagnose | Tool composition, judgment |
| Agent | Intent → tool composition | Seed selection, iterative refinement, quality judgment | Algorithm implementation |
| CLI | Human-friendly wrapper | MCP tool wrappers with formatting | Agent logic |

This follows AD-2: "data pipeline, not opinions". Core and MCP
produce complete data. Agent and SKILL make judgment calls.

## CLI Subcommands

```
quarry search     — hybrid search (BM25 + ANN + reranker)
quarry vsearch    — vector similarity search (embedding ANN)
quarry info       — single/multi paper metadata lookup (AD-8)
quarry expand     — single-seed APPR + coupling/cocitation subgraph
quarry bridge     — multi-seed bridge discovery (7 types)
quarry sql        — read-only PG query (development only, see AD-8)
quarry mcp-server — start MCP server
```

Note: `similar` renamed to `vsearch` — clarifies that this is
vector search, not content similarity.

Note: `sql` is a development escape hatch; not exposed in Server
API (Phase 3). `info` is the safe, parameterized alternative.

## MCP Tools

### Current

| Tool | Category | Description |
|------|----------|-------------|
| search_papers | Search | Hybrid BM25 + ANN + RRF |
| similar_papers | Search | Embedding ANN similarity |
| get_paper | Lookup | Single paper by work_id/PMID/DOI |
| query_metadata | Lookup | Read-only PG SQL |
| expand_citations | Graph | N-hop citation expansion |
| find_path | Graph | Citation chain shortest path |
| get_subgraph | Graph | Session subgraph + metrics |
| mesh_explore | Browse | MeSH hierarchy navigation |

### Planned additions

| Tool | Category | Description | Rationale |
|------|----------|-------------|-----------|
| bridge | Analysis | Multi-seed bridge (7 types) | Agent needs bridge access |
| diagnose_seeds | Analysis | Seed quality assessment | Agent needs pre-check before bridge |
| expand | Graph | APPR-based expand (= CLI expand) | Current expand_citations is N-hop only, mismatches CLI |

### CLI ↔ MCP alignment

| CLI | MCP (current) | MCP (planned) | Gap |
|-----|---------------|---------------|-----|
| search | search_papers | — | Aligned |
| vsearch | similar_papers | — | Aligned (rename) |
| info | get_paper | — | **Add CLI wrapper** (AD-8) |
| sql | query_metadata | — | Dev-only; **not exposed in Server API** (AD-8) |
| expand | expand_citations (**mismatch**) | expand (**APPR**) | Fix: add APPR expand, keep N-hop as expand_citations |
| bridge | ❌ | bridge | Add |
| — | find_path | — | CLI not needed (available via bridge Type 5) |
| — | get_subgraph | — | CLI not needed (analysis tool) |
| — | mesh_explore | — | CLI not needed (browse tool) |
| — | ❌ | diagnose_seeds | Add |

## Agent Layer

### Where LLM judgment is needed

Bridge discovery has three scenarios requiring judgment:

| Scenario | Judgment needed | Input | Output |
|----------|----------------|-------|--------|
| Seed present, low quality | "This is a hub, find a more specific paper" | expand results | Better seed |
| No seed, intent only | "Which search result best represents this topic?" | search results | Seed selection |
| Single seed | "Which lateral paper opens an interesting cross-domain direction?" | expand laterals | 2nd seed |

These judgments are **not** in quarry Core or MCP — they require
understanding paper content and user intent.

### BAML: Type-safe LLM functions

Individual LLM calls with structured input/output:

```
ClassifyPaper(title, abstract) -> PaperType
  enum PaperType { Original, Review, Preprint, Meta }

AssessSeedQuality(paper, graph_stats) -> SeedAssessment
  class SeedAssessment {
    quality: Good | Hub | LowCitation | TooOld
    reason: string
    suggestion: string?   // "Try a more specific paper on X"
  }

JudgeRelevance(bridge_paper, seeds) -> float
  // 0.0 = noise, 1.0 = highly relevant bridge

SelectSeed(papers, intent) -> WorkId
  // Pick best seed from candidates given user intent

SuggestNextSeed(bridge_results, seeds) -> WorkId?
  // Iterative: pick 3rd seed from bridge results
```

BAML provides: type safety, structured parsing, retry on parse
failure, test harness for LLM functions.

### DSPy: Pipeline optimization (Phase 2)

After BAML functions are stable and eval dataset exists:

```python
class BridgeDiscovery(dspy.Module):
    def __init__(self):
        self.search = dspy.Predict("intent -> query")
        self.select_seed = dspy.ChainOfThought("papers, intent -> seed")
        self.evaluate = dspy.Predict("bridge_results, seeds -> relevance_score")

    def forward(self, intent):
        # search → select seeds → bridge → evaluate → refine
        ...

# Optimize with human-rated bridge results
optimizer = MIPROv2(metric=bridge_relevance_metric)
optimized = optimizer.compile(BridgeDiscovery(), trainset=eval_data)
```

DSPy requires: metric definition + labeled eval data. Deferred until
bridge usage accumulates quality ratings.

### SKILL: Workflow orchestration

Claude Code SKILL wrapping the above into user-facing workflows:

```
Skill(bridge-discovery)
  Input: user intent (natural language) or seed paper(s)
  Steps:
    1. If no seeds: search → SelectSeed (BAML)
    2. If seeds: diagnose_seeds (MCP) → AssessSeedQuality (BAML)
    3. If quality warning: expand → SelectSeed for better seed
    4. bridge (MCP) → results
    5. JudgeRelevance (BAML) on top results
    6. Optional: SuggestNextSeed → bridge with 3rd seed
  Output: ranked bridge papers with relevance scores
```

### Layer boundary: what goes where

| Function | SKILL | BAML | DSPy | MCP | Core |
|----------|-------|------|------|-----|------|
| "Find bridges between progeria and protein eng" | ✅ orchestrate | | | | |
| "Is this paper a review?" | | ✅ classify | | | |
| "Optimize seed selection prompts" | | | ✅ optimize | | |
| bridge(seeds) → results | | | | ✅ tool | |
| APPR + geometric mean | | | | | ✅ compute |
| "Which search result is best seed?" | | ✅ judge | | | |
| search("progeria") → papers | | | | ✅ tool | |

## Implementation Roadmap

| Phase | Work | Layer |
|-------|------|-------|
| **Now** | MCP: add bridge, diagnose_seeds, APPR expand | MCP |
| **Now** | CLI: rename similar → vsearch | CLI |
| **Next** | BAML: ClassifyPaper, AssessSeedQuality | Agent |
| **Next** | SKILL: bridge-discovery workflow | Agent |
| **Later** | DSPy: pipeline optimization with eval data | Agent |
| **Later** | Phase 3 embedding integration | Core + MCP |

## References

- AD-2: Quality signals as metadata only (no opinions in pipeline)
- AD-3: No --sort/--filter in CLI (downstream processing)
- 11-bridge.md: Bridge types, seed quality criteria
- 12-graph-algorithms.md: Algorithm catalog
