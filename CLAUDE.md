# quarry: Citation Graph Paper Exploration

Academic paper discovery via citation graph traversal (APPR),
bibliographic coupling, co-citation, and MeSH ontology.

## Key Architecture

| Layer | Tech | Role |
|-------|------|------|
| CLI | Python (typer) | `quarry` commands: search, info, mesh, expand, bridge, shrink |
| Core | Rust (PyO3) + Python | Graph algorithms (CSR mmap), PG enrichment |
| MCP | Python | Atomic tools for agent layer |
| Data | PG + CH + LanceDB + CSR | Metadata, ETL staging, embeddings, citation graph |

Design docs: `docs/design/10-architecture-decisions.md` (AD-1 through AD-10)

## MCP Server

Connect the quarry MCP server to Claude Code or Claude Desktop:

```bash
claude mcp add --transport http quarry http://<host>:8000/mcp
```

Use the `/quarry-mcp` skill for tool reference, exploration sequences, and fallback patterns.

## Paper Exploration

When exploring papers, building reading lists, or tracing technology
lineages, use the `/dogfood` skill. It provides structured UX evaluation
with 4 personas (newbie/reader/bridger/surveyor) and automatically saves
results to `docs/dogfood/`.

Previous session scores and findings: `docs/dogfood/README.md`

## CLI Commands (dev only)

```
quarry search     — hybrid search (BM25 + embedding, building)
quarry vsearch    — vector similarity (embedding, building)
quarry info       — paper metadata lookup (--mesh for MeSH tags)
quarry mesh       — MeSH vocabulary search + hierarchy browsing
quarry expand     — APPR subgraph (--mesh-summary for topic distribution)
quarry bridge     — multi-seed connection discovery (7 types)
quarry shrink     — minimum covering set from top venues (--no-foundation)
quarry sql        — dev-only SQL escape hatch (--schema for column reference)
```

## Data Pipeline (Dagster)

Use `/dagster-expert` skill before modifying assets in `quarry/assets/`.

Pipeline: OA/PubMed/MeSH/iCite → CH (staging) → CH (transform) → PG (serving)

Key tables: works (445M), work_mesh (379M), mesh_lookup (272K), mesh_tree (65K)

## Testing

```bash
just test-py           # pytest: unit + integration (33 tests)
nix build .#quarry-parse  # Rust parser build + tests
```

## Skills Reference

| Skill | When |
|-------|------|
| `/quarry-mcp` | MCP tool reference, exploration sequences |
| `/dogfood` | Paper exploration, MCP UX testing |
| `/dagster-expert` | ELT pipeline, asset definitions |
| `/clickhouse-best-practices` | CH schema/query review |
| `/dignified-python` | Python code standards |
