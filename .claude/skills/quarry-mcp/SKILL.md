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
| `quarry_mesh_explore`    | You have a biomedical topic and need to find entry-point papers |
| `quarry_search_papers`   | You have a natural-language query, no known paper               |
| `quarry_get_paper`       | Verify a specific paper, get its identifiers, check MeSH tags   |
| `quarry_expand`          | Map citation neighborhood of a known seed paper                 |
| `quarry_bridge`          | Find connectors between 2+ papers from different fields         |
| `quarry_shrink`          | Build a minimal reading list from high-impact venues            |
| `quarry_get_full_text`   | Fetch full text (PMC → Unpaywall → OA URL)                      |
| `quarry_similar_papers`  | Semantic similarity — for new/isolated papers not in graph      |
| `quarry_expand_citations`| Raw k-hop walk — simpler alternative to expand                  |
| `quarry_find_path`       | Shortest citation chain between two specific papers             |
| `quarry_get_subgraph`    | Structural metrics (centrality, components) for a paper set     |
| `quarry_query_metadata`  | SQL fallback when other tools don't cover the query             |

MCP resources: `quarry://docs/tools` (overview), `quarry://docs/bridge-types` (bridge reference)

## Exploration sequences

**Unknown field** (start from topic keyword):
```
quarry_mesh_explore(descriptor_name="your topic")
  → quarry_mesh_explore(descriptor_ui=X, direction="papers")
  → quarry_get_paper(work_id=..., include_mesh=True)    ← verify seed
  → quarry_expand(seed=..., mesh_summary=True)
```

**From a known paper**:
```
get_paper(work_id/pmid/doi, include_mesh=True)
  → quarry_expand(seed=...)
  → quarry_bridge([seed1, seed2])
```

**Cross-domain discovery**:
```
quarry_expand(seed1) + quarry_expand(seed2)     ← pick best seed from each domain
  → quarry_bridge([seed1, seed2], mesh_summary=True)
  → quarry_get_paper(work_id=top_result, include_abstract=True)   ← assess each connector
```

**Reading list**:
```
quarry_expand(seed) → quarry_shrink(seed, top_n=5)
```

**Full text**:
```
get_paper(pmid=...) → quarry_get_full_text(pmid=...)
```

**Fallback chain** (when primary tool fails):
```
quarry_mesh_explore miss → quarry_query_metadata("SELECT ... FROM mesh_lookup WHERE descriptor_name ILIKE '%term%'")
expand "not in graph" → quarry_similar_papers(work_id=...)
bridge all empty → seeds too close (same field) or too distant — try different seeds
```

## References

Load only when needed:

| Reference | When |
|-----------|------|
| `references/serendipity.md` | Assessing bridge paper quality, cross-domain signal detection |
| `references/context-discipline.md` | Managing context across multi-step exploration |
| `references/state-schema.md` | Writing exploration state to disk (multi-session workflows) |
