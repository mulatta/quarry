# quarry CLI Reference

These are the available commands. Use whichever ones make sense for
your research goal — there is no prescribed order.

## Commands

### quarry search "<query>"

Hybrid search combining keyword matching (BM25) and semantic
vectors, fused with RRF. Returns ranked papers with work_id,
title, year.

Options: `--limit N` (default 20), `--mode hybrid|vector|text`

### quarry vsearch \<work_id>

Vector similarity search — finds papers with similar embeddings
to a given paper. Useful when you have a paper and want more
like it.

Options: `--limit N` (default 20)

### quarry expand <seed>

Build a ranked subgraph around a seed paper using APPR (local
PageRank) + bibliographic coupling + co-citation. Returns papers
classified as foundation, follow-up, mutual, or lateral.

Seed formats: `W<id>`, integer work_id, DOI, `https://doi.org/...`

Options: `--limit N`, `--mode fused|separated`, `--format table|json|detail`,
`--alpha`, `--epsilon`

### quarry shrink \<seed>

Find minimum set of high-quality papers that cover an expand
landscape. Runs expand internally, filters to top venues, selects
papers by greedy weighted citation coverage.

```
quarry shrink W2042789810                    # default: top 5, NCS+
quarry shrink W2042789810 --top 10           # more papers
quarry shrink W2042789810 --venue "Nature,Science,Cell"  # custom venues
```

Options: `--top N` (default 5), `--venue NCS+|<list>` (default NCS+),
`--limit N` (expand limit, default 200), `--format table|json`

### quarry bridge <seed1> <seed2> [seed3...]

Find papers connecting two or more seeds. Returns 7 bridge types
(for 2 seeds; steiner activates at 3+).

Options: `--type <type>` (filter to specific type), `--limit N`,
`--format table|json|detail`, `--max-path-depth N`, `--alpha`, `--epsilon`

Bridge types: common_refs, common_citers, coupling, cocitation,
path, ppr, steiner (k>=3 only)

### quarry info \<work_id> [work_id2 ...]

Lookup metadata for one or more papers. Returns work_id, title,
pub_year, doi, host_venue, cited_by_count, oa_status, rcr,
abstract (truncated).

Options: `--full` (show full abstract), `--mesh` (show MeSH tags),
`--format json`

### quarry mesh "<name>" or \<descriptor_ui>

MeSH-based paper discovery and hierarchy browsing. Finds papers
by curated MeSH vocabulary — precise topic filtering without
keyword noise.

```
quarry mesh "retroelement"        # search by name → top papers
quarry mesh D018626               # search by descriptor UI
quarry mesh D018626 --tree        # show hierarchy (parents/children)
```

Options: `--tree` (show hierarchy), `--limit N` (default 15)

### quarry sql "<SELECT query>"

Execute read-only SQL against PostgreSQL metadata. Development-only
escape hatch for ad-hoc queries. Will not be exposed in Server API.

Options: `--schema <table>` (show column names and types)

Tables: works, papers, work_citations, work_authors, work_topics,
work_mesh, authors, mesh_headings, mesh_tree, id_crosswalk
