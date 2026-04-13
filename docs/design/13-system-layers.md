# System Layers

> Status: Design. Documents the full system architecture across CLI,
> MCP, Agent, and Core layers.

## Layer Overview

```
┌───────────────────────────────────────────────────────────┐
│  User Interface                                           │
│  CLI (human) / .claude/agents/ (interactive) / Web (TBD)  │
└──────────────────┬────────────────────────────────────────┘
                   │
┌──────────────────▼────────────────────────────────────────┐
│  Agent Layer (LLM judgment)                               │
│  Python orchestrator: DAG, resume, cost gate              │
│  Pydantic schemas + provider-agnostic LLM (httpx)         │
│  DSPy: prompt optimization (deferred, 200+ eval)          │
└──────────────────┬────────────────────────────────────────┘
                   │ QuarryClient (local import or HTTP)
┌──────────────────▼────────────────────────────────────────┐
│  quarry Service (data + compute, no LLM)                  │
│  Phase 2: Python server (Starlette, unix socket)          │
│  Phase 3: Rust server (axum), protocol TBD                │
└──────────────────┬────────────────────────────────────────┘
                   │ Rust FFI + PG + LanceDB
┌──────────────────▼────────────────────────────────────────┐
│  quarry Core (Rust graph + Python enrichment)             │
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
quarry mesh       — MeSH-based paper discovery + tree browsing (AD-9)
quarry expand     — single-seed APPR + coupling/cocitation subgraph
quarry shrink     — minimum covering set from top venues (AD-10)
quarry bridge     — multi-seed bridge discovery (7 types)
quarry sql        — read-only PG query (development only, see AD-8)
quarry mcp-server — start MCP server
```

Note: `similar` renamed to `vsearch` — clarifies that this is
vector search, not content similarity.

Note: `sql` is a development escape hatch; not exposed in Server
API (Phase 3). `info` is the safe, parameterized alternative.

Note: `mesh` provides curated vocabulary-based discovery, complementary
to keyword search. See AD-9 for ontology strategy.

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

### Planned additions (deferred — CLI-first, MCP after CLI validation)

MCP additions are deferred until all CLI commands are validated via
dogfood sessions. Rationale: CLI is the primary interface for research
workflows; MCP duplication adds maintenance burden without proven need.
See AD-14 for agent architecture that uses CLI via Bash tool.

| Tool | Category | Description | Status |
|------|----------|-------------|--------|
| bridge | Analysis | Multi-seed bridge (7 types) | **Deferred** — CLI bridge complete and validated |
| diagnose_seeds | Analysis | Seed quality assessment | **Deferred** — agent handles via expand + info |
| expand | Graph | APPR-based expand (= CLI expand) | **Deferred** — CLI expand complete with --min-citations |

### CLI ↔ MCP alignment

| CLI | MCP (current) | MCP (planned) | Gap |
|-----|---------------|---------------|-----|
| search | search_papers | — | Aligned |
| vsearch | similar_papers | — | Aligned (rename) |
| info | get_paper | — | **Add CLI wrapper** (AD-8) |
| mesh | mesh_explore | — | Aligned (AD-9). CLI uses mesh_lookup for synonym search |
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

### LLM Functions (Pydantic schemas, provider-agnostic)

Structured LLM calls via OpenAI-compatible tool_use. Pydantic models
define input/output schemas. No framework dependency — BAML dropped
(solves 20% of problem with 100% dependency; Claude tool_use sufficient).

```python
# quarry/agents/schemas.py — Pydantic models
class SeedEvaluation(BaseModel):
    relevance: int = Field(ge=0, le=10)
    keep: bool
    reason: str

class SerendipityScore(BaseModel):
    novelty: Literal[0, 1]
    specificity: Literal[0, 1]
    actionability: Literal[0, 1]
    verdict: Literal["validated", "rejected", "related_finding"]
```

Provider-agnostic: httpx + configurable base_url (Anthropic, OpenRouter,
vLLM, Ollama). Model per function (haiku for bulk eval, opus for report).

### DSPy: Prompt optimization (deferred)

After Pydantic-based LLM functions are stable and eval dataset exists
(200+ labeled examples from seed-recall + agent runs). Pydantic schemas
convert to dspy.Signature. Blocked by eval data accumulation.

### Python Orchestrator (replaces md agent for automation)

Python code controls DAG, resume, cost gate. LLM called only for
judgment. md agent retained for interactive use.

```
quarry/agents/scout.py    — DAG orchestration, resume, cost gate
quarry/agents/llm.py      — provider-agnostic LLM (httpx + OpenAI-compat)
quarry/agents/schemas.py  — Pydantic (structured output)
quarry/agents/state.py    — YAML state + DAG topological sort
```

### Layer boundary: what goes where

| Function | Orchestrator | LLM (Pydantic) | DSPy | QuarryClient | Core |
|----------|-------------|----------------|------|-------------|------|
| DAG execution, resume, cost gate | ✅ Python code | | | | |
| "Is this paper a review?" | | ✅ classify | | | |
| "Optimize seed selection prompts" | | | ✅ optimize | | |
| bridge(seeds) → results | | | | ✅ client | |
| APPR + geometric mean | | | | | ✅ compute |
| "Which search result is best seed?" | | ✅ judge | | | |
| search("progeria") → papers | | | | ✅ client | |

## Implementation Roadmap

| Phase | Work | Layer | Status |
|-------|------|-------|--------|
| **Done** | CLI: bridge (7 types), shrink, mesh, expand (--min-citations), sp fix | CLI | ✅ AD-10,11,12 |
| **Done** | CLI: rename similar → vsearch | CLI | ✅ |
| **Done** | Bridge: rayon parallelization (5-6x), deterministic sort | Core | ✅ |
| **Done** | Agent: research-scout (Claude Code native, CLARIFY+DAG) | Agent | ✅ AD-14,D11-D16 |
| **Done** | Agent: .claude/agents/ with frontmatter + .claude/skills/quarry-research | Agent | ✅ |
| **Done** | Eval: seed-recall framework (sysrev-seed-collection, 40 reviews) | Eval | ✅ data + prep.py |
| **Now** | Eval: run.py + report.py (recall@K per method) | Eval | |
| **Now** | QuarryClient interface (local/remote) | Core | |
| **Now** | Python orchestrator (scout.py, DAG, resume) | Agent | |
| **Next** | LLM functions: Pydantic schemas + provider-agnostic llm.py | Agent | |
| **Next** | Server API (Python, HTTP+JSON, unix socket) | Core | |
| **Later** | DSPy: prompt optimization (after 200+ eval examples) | Agent | Blocked by eval data |
| **Later** | Real-time viz: streaming protocol + concept graph schema (TBD) | Server | Workload-dependent |
| **Later** | Rust server (axum) | Core | When PG enrichment migrated |
| **Later** | MCP: bridge, expand (after server API validated) | MCP | Deferred |

## References

- AD-2: Quality signals as metadata only (no opinions in pipeline)
- AD-3: No --sort/--filter in CLI (downstream processing)
- AD-11: Bridge sp bidirectional minimum
- AD-12: Expand --min-citations post-filter
- AD-14: Research agent architecture (14-research-agent.md)
- 11-bridge.md: Bridge types, seed quality criteria
- 12-graph-algorithms.md: Algorithm catalog
