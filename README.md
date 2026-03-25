# quarry

Academic literature data pipeline: OpenAlex + PubMed + iCite → ClickHouse → PostgreSQL.

## Tier Classification

Works are classified into tiers based on data availability. Tier determines parsing scope and downstream utility.

| Tier | Condition | Count | Parsing Scope | Use Case |
|------|-----------|-------|---------------|----------|
| **T1** | pmid + abstract | ~21M | full (abstract, authors, topics, citations) | Embedding, search, PubMed enrichment (MeSH, grants, iCite) |
| **T2** | abstract, no pmid | ~102M | full (abstract, authors, topics, citations) | Embedding, search, OA metadata only |
| **T3** | pmid, no abstract | ~18M | authors, topics, citations (no abstract) | PubMed metadata serving (MeSH, grants, authors) |
| **T4** | neither | ~342M | citations only | Citation graph node |

Classification is determined at parse time (`quarry-parse`) by the presence of `pmid` and `abstract_inverted_index` in the OpenAlex JSON.

### Topic Primary Marker

Each work has up to 3 topics from OpenAlex. The first topic (`topics[0]`) is marked `is_primary = true` in `work_topics`, matching OpenAlex's `primary_topic` field. This preserves the original classification without denormalizing into the works table.

## Pipeline Architecture

```
Download          Parse              Load           Transform        Serve
────────          ─────              ────           ─────────        ─────
oa_sync ────→ oa_parse ──────┐
                             │
pm_sync ────→ pm_parse ──────┼──→ ch_load ──→ ch_transform ──┬──→ pg_load
                             │                               │
mesh_sync ──→ mesh_stage ────┤                               ├──→ paper_embeddings
                             │                               │
icite_sync ──────────────────┘                               └──→ csr_graph
```

### Stages

1. **Download** — `aws s3 sync` (OA), FTP (PubMed, MeSH), figshare (iCite)
2. **Parse** — `quarry-parse` (Rust): gz JSONL / XML → Parquet (Hive partitioned)
3. **Load** — Parquet / CSV → ClickHouse (ReplacingMergeTree for dedup)
4. **Transform** — CH: `OPTIMIZE FINAL` + enriched export tables (OA×PubMed×iCite JOIN)
5. **Serve** — CH export → PG COPY (serving DB), LanceDB embeddings, CSR citation graph

### Key Design Decisions

- **ClickHouse for transform**: ReplacingMergeTree handles dedup as append-only (no DELETE), JOINs for enrichment run in CH, PG receives clean data via COPY only.
- **Parquet intermediate**: File-level reprocessing — 1 input file = 1 Parquet file. Skip-if-exists for incremental runs.
- **CH → PG pipe**: `clickhouse-client --format TabSeparated | psql COPY FROM STDIN`. Zero intermediate files.
- **CH → embeddings pipe**: `clickhouse-client --format ArrowStream | pyarrow.ipc.open_stream()`. Memory-bounded streaming.

## Project Structure

```
crates/
  quarry-parse/   Rust: OA JSONL + PubMed XML → Parquet (CLI binary)
  quarry-core/    Rust: shared utilities (abstract reconstruction)
  quarry-graph/   Rust: CSR citation graph (PyO3 bindings)
quarry/           Python: Dagster assets, MCP server, embeddings
sql/              Schema DDL (PG, CH) and transform SQL
tests/            E2E pipeline tests
nix/              Flake modules (dev shell, services, packages)
```

## Development

```bash
# Enter dev shell (Nix)
nix develop

# Start local PG + CH
nix run .#services

# Run pipeline (Dagster)
dagster dev
```
