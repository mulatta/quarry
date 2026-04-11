---
name: quarry-mcp
description: >-
  Reference for quarry MCP tools: paper exploration via citation graph, MeSH ontology,
  and semantic search. Load this skill when the quarry MCP server is connected and you
  need to explore research literature, find cross-domain connections, build reading lists,
  fetch full text, or understand the available tools and their usage sequences.
  Trigger on: paper discovery, literature exploration, citation graph, MeSH search,
  bridge papers, reading list, full-text fetch, "quarry" tool usage.
---

# quarry MCP

Connect: `claude mcp add --transport http quarry http://<host>:8000/mcp`

## Tools

| Tool              | Use when                                                        |
|-------------------|-----------------------------------------------------------------|
| `mesh_explore`    | You have a biomedical topic and need to find entry-point papers |
| `search_papers`   | You have a natural-language query, no known paper               |
| `get_paper`       | Verify a specific paper, get its identifiers, check MeSH tags   |
| `expand`          | Map citation neighborhood of a known seed paper                 |
| `bridge`          | Find connectors between 2+ papers from different fields         |
| `shrink`          | Build a minimal reading list from high-impact venues            |
| `get_full_text`   | Fetch full text (PMC → Unpaywall → OA URL)                      |
| `similar_papers`  | Semantic similarity — for new/isolated papers not in graph      |
| `expand_citations`| Raw k-hop walk — simpler alternative to expand                  |
| `find_path`       | Shortest citation chain between two specific papers             |
| `get_subgraph`    | Structural metrics (centrality, components) for a paper set     |
| `query_metadata`  | SQL fallback when other tools don't cover the query             |

MCP resources: `quarry://docs/tools` (overview), `quarry://docs/bridge-types` (bridge reference)

## Exploration sequences

**Unknown field** (start from topic keyword):
```
mesh_explore(descriptor_name="your topic")
  → mesh_explore(descriptor_ui=X, direction="papers")
  → get_paper(work_id=..., include_mesh=True)    ← verify seed
  → expand(seed=..., mesh_summary=True)
```

**From a known paper**:
```
get_paper(work_id/pmid/doi, include_mesh=True)
  → expand(seed=...)
  → bridge([seed1, seed2])
```

**Cross-domain discovery**:
```
expand(seed1) + expand(seed2)     ← pick best seed from each domain
  → bridge([seed1, seed2], mesh_summary=True)
  → get_paper(work_id=top_result, include_abstract=True)   ← assess each connector
```

**Reading list**:
```
expand(seed) → shrink(seed, top_n=5)
```

**Full text**:
```
get_paper(pmid=...) → get_full_text(pmid=...)
```

**Fallback chain** (when primary tool fails):
```
mesh_explore miss → query_metadata("SELECT ... FROM mesh_lookup WHERE descriptor_name ILIKE '%term%'")
expand "not in graph" → similar_papers(work_id=...)
bridge all empty → seeds too close (same field) or too distant — try different seeds
```

## References

Load only when needed:

| Reference | When |
|-----------|------|
| `references/serendipity.md` | Assessing bridge paper quality, cross-domain signal detection |
| `references/context-discipline.md` | Managing context across multi-step exploration |
| `references/state-schema.md` | Writing exploration state to disk (multi-session workflows) |
