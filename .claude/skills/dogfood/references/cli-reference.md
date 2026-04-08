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
quarry shrink W2042789810 --no-foundation    # skip foundations (recent trends)
quarry shrink W2042789810 --exclude W2336828812  # exclude specific paper
quarry shrink W2042789810 --venue "Nature,Science,Cell"  # custom venues
```

Options: `--top N` (default 5), `--venue NCS+|<list>` (default NCS+),
`--limit N` (expand limit, default 200), `--no-foundation` (exclude
foundation papers), `--exclude W...,W...` (exclude specific papers),
`--format table|json`

Warns when top-1 paper covers >80% (centralized field).

### quarry bridge <seed1> <seed2> [seed3...]

Find papers connecting two or more seeds. Returns 7 bridge types
(for 2 seeds; steiner activates at 3+).

Header shows pairwise shortest path for each seed pair:
`sp: W...↔W...=1  W...↔W...=∞` (∞ = no directed path within depth).
Use this to assess seed compatibility — many ∞ pairs means seeds
are directionally disconnected (consider expand instead of bridge).

Empty bridge types show a diagnostic reason (e.g., "no steiner tree
(2/3 pairs unreachable)").

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
keyword noise. MeSH covers biomedical/life sciences only; CS,
physics, math topics are absent. Use `sql` with title ILIKE
for non-biomedical fields.

```
quarry mesh "retroelement"        # search by name → top papers
quarry mesh D018626               # search by descriptor UI
quarry mesh D018626 --tree        # show hierarchy (parents/children)
```

Options: `--tree` (show hierarchy), `--limit N` (default 15)

**Seed selection tip**: the top-cited paper in a MeSH descriptor may
be a niche subfield (e.g., fluorescent proteins under "Protein
Engineering"). Use `info --mesh` on candidates to verify they are
field-representative before using as expand seeds. Review papers
or highly-cited recent papers with broad MeSH tags are better seeds.

### quarry sql "<SELECT query>"

Execute read-only SQL against PostgreSQL metadata. Development-only
escape hatch for ad-hoc queries. Will not be exposed in Server API.

Options: `--schema <table>` (show column names and types)

Tables: works, papers, work_citations, work_authors, work_topics,
work_mesh, authors, mesh_headings, mesh_tree, id_crosswalk
