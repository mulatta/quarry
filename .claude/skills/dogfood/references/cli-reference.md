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

### quarry bridge <seed1> <seed2> [seed3...]

Find papers connecting two or more seeds. Returns 7 bridge types
(for 2 seeds; steiner activates at 3+).

Options: `--type <type>` (filter to specific type), `--limit N`,
`--format table|json|detail`, `--max-path-depth N`, `--alpha`, `--epsilon`

Bridge types: common_refs, common_citers, coupling, cocitation,
path, ppr, steiner (k>=3 only)

### quarry sql "<SELECT query>"

Execute read-only SQL against PostgreSQL metadata. Useful for
looking up specific papers, checking fields, or ad-hoc analysis.

Tables: works, papers, work_citations, work_authors, work_topics,
work_mesh, authors, mesh_headings, mesh_tree, id_crosswalk
