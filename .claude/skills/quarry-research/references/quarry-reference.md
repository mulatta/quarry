# quarry CLI Reference

Agent-focused reference for quarry CLI commands. Use `--format json` (or `-f json`) wherever available for machine-readable output.

## ID Formats

All commands accepting paper IDs support these formats:
- `W2042789810` — OpenAlex work ID
- `2042789810` — work_id_int (numeric only)
- `10.1038/s41586-019-1184-4` — DOI
- `https://doi.org/10.1038/...` — DOI URL
- `31488816` — PMID (8-digit numeric)

## Commands

### mesh

```
quarry mesh <query> [--tree/-t] [--limit/-n 15]
```

Search MeSH vocabulary by descriptor name (partial match) or UI (e.g. `D018626`).

- `--tree`: Show hierarchy (parents + children)
- No JSON output mode — returns formatted text
- **Agent pattern**: Use to discover MeSH descriptors for search terms, then drill with `--tree`

**Error recovery**: If no match → try synonyms (max 2). If still no match → SQL fallback:
```
quarry sql "SELECT descriptor_ui, descriptor_name FROM mesh_lookup WHERE descriptor_name ILIKE '%keyword%' LIMIT 10"
```

### info

```
quarry info <work_id...> [--full] [--mesh] [--format/-f table|json]
```

Lookup metadata for one or more papers.

- `--full`: Include abstract
- `--mesh`: Include MeSH descriptors
- `-f json`: Returns JSON array

**JSON output schema** (per paper):
```json
{
  "work_id": "W2042789810",
  "title": "...",
  "pub_year": 2019,
  "doi": "10.1038/...",
  "pmid": "31488816",
  "pmc_id": "PMC6776048",
  "host_venue": "Nature",
  "cited_by_count": 307,
  "cnp": 45.2,
  "fwci": 12.3,
  "rcr": 8.7,
  "oa_status": "gold",
  "abstract": "..." ,
  "mesh": [{"descriptor_ui": "D018626", "descriptor_name": "...", "is_major": true}]
}
```

### expand

```
quarry expand <seed> [--limit/-n 200] [--mode/-m fused|separated]
    [--format/-f table|json|detail] [--mesh-summary]
    [--alpha 0.15] [--epsilon 1e-6] [--min-citations 0]
```

Expand seed paper into ranked subgraph via APPR + bibliometric fusion.

- `--mode fused`: wRRF fusion, focused exploration (default)
- `--mode separated`: APPR + guaranteed lateral slots, broad survey
- `--mesh-summary`: Append MeSH topic distribution of results
- `--min-citations N`: Filter out papers below N citations (noise reduction)

**JSON output schema**:
```json
{
  "meta": {"graph_nodes": 123456789, "build_date": "2025-01-15"},
  "seed": {"work_id": 123, "title": "...", "doi": "...", "year": 2019},
  "params": {"mode": "fused", "alpha": 0.15, "epsilon": 1e-6, "limit": 200},
  "papers": [
    {
      "rank": 1,
      "work_id": 456,
      "doi": "https://doi.org/...",
      "title": "...",
      "year": 2020,
      "host_venue": "Nature",
      "relation": "mutual|foundation|follow-up|lateral",
      "scores": {"fused": 0.95, "appr": 0.0023},
      "quality": {"cited_by": 150, "cnp": 30.0, "fwci": 8.0, "rcr": 5.5},
      "bridges": [{"work_id": 789, "type": "coupling", "title": "...", "weight": 3}]
    }
  ],
  "stats": {"appr_candidates": 5000, "returned": 200, "elapsed_s": 1.2}
}
```

**Relation types**:
| Relation | Meaning |
|----------|---------|
| `mutual` | Seed cites this AND this cites seed |
| `foundation` | Seed cites this (predecessor) |
| `follow-up` | This cites seed (successor) |
| `lateral` | Connected via shared citations, not direct |

**Agent pattern**: Start with `--limit 100 --mesh-summary` for landscape overview. Use `--min-citations N` if results are noisy (classic/highly-cited seed).

### bridge

```
quarry bridge <seed1> <seed2> [<seed3>...]
    [--type/-t TYPE] [--limit/-n 100]
    [--max-degree 10000] [--max-path-depth 5]
    [--alpha 0.15] [--epsilon 1e-6]
    [--format/-f table|json|detail]
```

Discover bridge papers connecting two or more seeds. Requires ≥2 seeds.

- `--type/-t`: Repeat for multiple types (default: all)
  - `common_refs` — shared intellectual foundation
  - `common_citers` — who already synthesized both
  - `coupling` — who combined both methods
  - `cocitation` — what both communities read
  - `path` — stepping-stone reading path between seeds
  - `ppr` — structurally central to both networks

**JSON output schema**:
```json
{
  "meta": {"graph_nodes": 123456789},
  "seeds": [{"work_id": 123, "title": "...", "doi": "...", "year": 2019}],
  "params": {"types": ["common_refs", "coupling"], "limit": 100},
  "common_refs": [
    {"work_id": 456, "title": "...", "year": 2015, "host_venue": "...",
     "aa_weight": 3.5, "seed_count": 2, "quality": {"cited_by": 200}}
  ],
  "common_citers": [...],
  "coupling_bridges": [
    {"work_id": 789, "title": "...", "per_seed_scores": {"123": 0.8, "456": 0.3},
     "score": 1.1, "quality": {"cited_by": 50}}
  ],
  "cocitation_bridges": [...],
  "path_bridges": [
    {"work_id": 101, "title": "...", "hop_from": 123, "path_count": 2, "quality": {...}}
  ],
  "ppr_bridges": [...],
  "steiner_bridges": [...],
  "stats": {
    "overlap_refs": 15, "overlap_citers": 137,
    "shortest_path_length": 3, "pairwise_sp": {"123_456": 3},
    "elapsed_s": 2.5
  }
}
```

**Key metric**: `stats.pairwise_sp` contains shortest-path distances between seed pairs. Used as serendipity novelty signal (sp < median → predictable, sp > median → novel).

**Agent pattern**: Run with all types first. Use `quarry info <id> --full` on top 3 results for abstract-level assessment.

### shrink

```
quarry shrink <seed> [--top/-n 5] [--venue/-v NCS+]
    [--limit 200] [--no-foundation] [--exclude IDs]
    [--format/-f table|json]
```

Find minimum covering set of top-venue papers from expand landscape.

- `--venue NCS+`: Default preset (Nature, Cell, Science + top specialty journals)
- `--no-foundation`: Exclude foundation papers (surfaces recent trends)
- `--exclude`: Comma-separated work_ids to exclude dominant papers

**JSON output schema**:
```json
{
  "seed": {"work_id": 123, "title": "...", "doi": "...", "year": 2019},
  "params": {"top_n": 5, "venues": ["Nature", "Cell", "Science"]},
  "selected": [
    {
      "rank_in_shrink": 1,
      "rank_in_expand": 15,
      "work_id": 456,
      "title": "...",
      "year": 2020,
      "host_venue": "Nature",
      "doi": "...",
      "relation": "follow-up",
      "scores": {"fused": 0.85, "appr": 0.002},
      "quality": {"cited_by": 300, "cnp": 40.0, "fwci": 10.0, "rcr": 7.0},
      "coverage_count": 45,
      "marginal_count": 45,
      "cumulative_coverage": 0.23
    }
  ],
  "stats": {
    "total_candidates": 200,
    "venue_pool_size": 30,
    "covered": 150,
    "uncovered": 50,
    "coverage": 0.75,
    "centralization_warning": "Top-1 covers 85% of candidates"
  }
}
```

**Agent pattern**: Use after expand to build concise reading lists. If `centralization_warning` present, consider `--no-foundation`.

### sql

```
quarry sql <query>
quarry sql --schema/-s <table>
```

Execute read-only SQL against PG (development/fallback only).

**Available tables**: works, work_citations, work_authors, work_topics, work_mesh, mesh_tree, mesh_lookup, id_crosswalk

**Agent pattern**: Use as fallback when `quarry mesh` fails. Example:
```
quarry sql "SELECT w.work_id, w.title, w.cited_by_count FROM works w JOIN work_mesh wm ON w.work_id = wm.work_id WHERE wm.descriptor_ui = 'D018626' ORDER BY w.cited_by_count DESC LIMIT 10"
```

## Error → Recovery Table

| Error | Max Retries | Recovery Action |
|-------|-------------|-----------------|
| `quarry mesh` no match | 2 synonyms | SQL fallback: `quarry sql "...ILIKE '%term%'"` |
| `quarry expand` seed not in graph | 1 alt seed | Mark "no graph data", try different seed |
| `quarry bridge` all types empty | 0 | Log "disconnected pair", skip |
| `quarry sql` timeout (>120s) | 0 | Note "SQL unavailable" |
| `quarry info` not found | 1 alt ID | Skip paper |
| Any command 3× consecutive fail | — | Stop sub-problem, mark "unable_to_assess" |

## Common Patterns

### Seed Discovery Flow
```
quarry mesh "topic"                          # find MeSH descriptor
quarry mesh D012345 --tree                   # browse hierarchy
quarry info W1234567890 --mesh --full -f json  # verify seed
quarry expand W1234567890 --mesh-summary -n 100 -f json  # map landscape
```

### Cross-domain Connection
```
quarry bridge <seed_sp1> <seed_sp2> -f json  # all bridge types
quarry info <top_bridge_paper> --full -f json  # assess mechanism
```

### Reading List Construction
```
quarry shrink <seed> -n 10 -f json           # top-venue coverage
quarry shrink <seed> -n 10 --no-foundation -f json  # if centralized
```
