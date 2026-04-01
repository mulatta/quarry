# Architecture Decisions

> Record of key architecture decisions with context and rationale.

## AD-1: Python CLI + Rust .so (Phase 1-2), Pure Rust CLI (Phase 2+)

**Date**: 2026-04-01
**Status**: Active

### Context

`quarry expand` has two layers:
- **Graph computation** (APPR, coupling, cocitation, wRRF, bridge): Rust, ~1s
- **Metadata enrichment** (PG lookup, JSON assembly, relation tagging): Python, ~50ms

The Python CLI is ~200 lines of thin wrapper around Rust .so (via PyO3).

### Decision

| Phase | Architecture | Rationale |
|-------|-------------|-----------|
| 1b (now) | Python CLI + Rust .so | Working, no performance issue. Avoid delay. |
| 2 | Evaluate pure Rust CLI | If MCP expand duplication or deployment burden triggers |
| 3 | Server-client (Rust server + gRPC) | When embedding service (Python) is added |

### Migration Triggers (Phase 2)

Migrate to pure Rust CLI when **any** of these occur:
- MCP server needs `expand()` → logic duplication becomes real
- Output schema has downstream consumers that need type safety
- Python dependency (pydantic, psycopg, rich) becomes deployment burden

### Phase 3 Embedding Architecture (Pre-decided)

```
Python embedding service (sentence-transformers + LanceDB)
  ↕ gRPC
Rust CLI / Rust server
  ↕ mmap
CSR graph
```

Python stays for embedding inference. Rust stays for graph computation.
gRPC bridges them. This means Phase 2 Rust CLI should include gRPC client
scaffolding for Phase 3 readiness.

### Mitigation: Shared Logic Extraction

To minimize Phase 2 migration cost, extract expand logic into
`quarry/core/expand.py` — a pure function that both CLI and MCP server
call. This function becomes the exact scope of Rust migration.

```python
# quarry/core/expand.py
def run_expand(seed, mode, limit, ...) -> dict:
    """Resolve seed → Rust expand → PG enrich → JSON-ready dict."""
```

### Alternatives Considered

| Alternative | Why not (now) |
|-------------|--------------|
| Pure Rust CLI now | Phase 1b delayed. PyO3 binding still needed for MCP. Maintenance doesn't decrease. |
| Server-client now | Over-engineering for single-user CLI. No embedding service yet. |
| Keep Python forever | maturin .venv issues recurring. Schema safety weak (dict-based). |

## AD-2: Quality Signals as Metadata Only

**Date**: 2026-04-01
**Status**: Active

### Context

Quality signals (cnp, fwci, rcr) measure "field importance" not "relevance
to seed". Evaluation showed fwci reranking promotes off-topic high-impact
papers and demotes seed-relevant niche papers.

### Decision

Quality signals are **exposed in output JSON as metadata**. Not used for
scoring, reranking, or fusion. Consumer decides how to use them.

### Rationale

- quarry is a data pipeline — returns complete data, not opinions
- fwci measures importance, not relevance (verified on 5 seeds)
- cnp and fwci are ρ=0.99 correlated — cnp preferred (0-1 range)
- NULL handling: return null, no imputation (preprints = unmeasured, not average)

See: [03-layer-quality.md](03-layer-quality.md), [07-missing-data.md](07-missing-data.md)

## AD-3: No --sort/--filter in CLI

**Date**: 2026-04-01
**Status**: Active

### Decision

quarry expand outputs complete JSON. No `--sort`, `--filter`, `--rel`
options. Filtering/sorting is done downstream with jq, DuckDB, pandas.

### Rationale

Adding sort/filter options creates a query language — each combination
needs testing and maintenance. This is a solved problem (SQL, jq).
quarry's value is in graph computation, not in reimplementing WHERE clauses.

```bash
# quarry outputs data
quarry expand W4406019873 --format json > results.json

# downstream filters (user's choice of tool)
jq '.papers[] | select(.relation == "lateral")' results.json
duckdb -c "SELECT * FROM read_json('results.json') WHERE year >= 2020"
```
