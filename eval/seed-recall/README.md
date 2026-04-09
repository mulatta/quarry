# Seed-Based Recall Evaluation

Evaluate quarry's citation graph traversal (expand/bridge) by measuring
how well it recovers the included studies of medical systematic reviews,
starting from a small set of seed papers.

## Motivation

Quarry's core value is **exploration** — discovering related papers via
citation graph traversal (APPR), not just ranking known candidates by
embedding similarity. Standard IR benchmarks (SCIDOCS, BioASQ) measure
pairwise ranking, which embeddings already solve well. What they don't
measure is the ability to **recover a relevant paper set from a few seeds**.

Systematic reviews provide a natural gold standard: a small set of seed
papers should lead to the full list of included studies. This is exactly
what `expand` and `bridge` are designed to do.

### Prior Work

| Study | Key Finding |
|-------|-------------|
| Sjögårde 2024 (JASIST) | Co-citation + BC + direct citation combined > any single method; citation + text = best |
| SciNCL 2022 (EMNLP) | Graph at training time helps embedding, but no one uses graph at inference |
| BMQExpander 2025 | Ontology relations alone give +18.7% NDCG; LLM adds only +2.8% more |

Gap: **no system has evaluated inference-time citation graph traversal
on systematic review recall.** This evaluation fills that gap.

## Data Source

**sysrev-seed-collection** (SIGIR 2022, MIT License)
Wang et al., "From Little Things Big Things Grow: A Collection with Seed
Studies for Medical Systematic Review Literature Search"

- Repository: https://github.com/ielab/sysrev-seed-collection
- Paper: https://dl.acm.org/doi/10.1145/3477495.3531748

## Files

| File | Tracked | Description |
|------|---------|-------------|
| `overall_collection.jsonl` | yes | 40 systematic reviews with seed/included PMIDs, PubMed queries |
| `seed_snowballing_candidates.tsv` | yes | Forward/backward citation candidates from seed papers |
| `screened_snowballing_candidates.tsv` | yes | Citation candidates from screened papers |
| `LICENSE.sysrev` | yes | MIT License from original repository |
| `prep.py` | yes | PMID → OpenAlex work_id resolver |
| `reviews.jsonl` | no | Generated: resolved reviews ready for evaluation |
| `summary.json` | no | Generated: PMID resolution statistics |
| `results/` | no | Generated: per-method evaluation outputs |

## Data Format

Each line in `overall_collection.jsonl`:

```json
{
  "id": "43",
  "title": "Prevalence of Differentiated Thyroid Cancer ...",
  "seed_studies": ["6504772", "6514426", ...],
  "included_studies": ["17159249", "2302665", ...],
  "query": "(\"Thyroid Neoplasms\"[MeSH] OR ...)",
  "link_to_review": "https://pubmed.ncbi.nlm.nih.gov/..."
}
```

- `seed_studies`: PubMed IDs used as starting points
- `included_studies`: PubMed IDs in the final review (gold standard)

## Dataset Statistics

- Reviews: 40 (31 unique by review link)
- Seeds per review: min=1, median=12, max=88
- Gold per review: min=1, median=12, max=137
- Total unique seed PMIDs: 560
- Total unique gold PMIDs: 1,017
- Domain: 100% biomedical (PubMed/Cochrane)

## Evaluation Plan

### Phase 0: Data Preparation (`just eval-prep`)

Resolve PMIDs to OpenAlex work_ids via PG. Filter reviews to those with
≥2 resolved seeds and ≥5 resolved gold studies. Output `reviews.jsonl`.

Critical metric: **PMID resolution rate** — if coverage is low,
evaluation results won't be meaningful.

### Phase 1: Method Comparison (`just eval-run`)

For each review, run four retrieval methods and collect result sets:

| Method | Command | What It Tests |
|--------|---------|---------------|
| **A. BM25 baseline** | `search "{title}" --mode=text` | Text retrieval only |
| **B. expand (APPR)** | `expand {seed} --limit 200` per seed, union | Citation graph traversal |
| **C. expand + bridge** | B seeds → `bridge` top pairs | Multi-seed connection discovery |
| **D. expand + MeSH** | B → `--mesh-summary` → top descriptors → mesh expansion | Ontology-guided expansion |

Each method produces a set of work_ids. Seeds themselves are excluded from
scoring (they are given, not discovered).

### Phase 2: Metrics (`just eval-report`)

| Metric | Definition | Why |
|--------|------------|-----|
| **Recall@K** | |found ∩ gold| / |gold| at K results | Primary: how much of gold is recovered |
| **Precision@K** | |found ∩ gold| / K | Secondary: noise level |
| **Unique discovery** | Papers found by method X but not by others | Shows complementary value |
| **Method overlap** | Jaccard(method_i, method_j) on found∩gold | Are methods redundant? |
| **Coverage** | |gold in DB| / |gold| | Bounds on achievable recall |

K values: 50, 100, 200.

### Phase 3: Ablation (future)

Once Phase 1-2 establish baselines:

| Ablation | Question |
|----------|----------|
| APPR α parameter sweep | Sensitivity of expand to teleport probability |
| Seed selection strategy | Random 2 seeds vs most-cited 2 vs diverse 2 |
| Bridge type breakdown | Which bridge types contribute unique discoveries |
| MeSH hierarchy depth | Parent only vs parent+sibling vs full subtree |
| Embedding fine-tune (Task 2) | Does citation-aware Jina v5 improve reranking in expand |
| MeSH query expansion (Task 1) | Does MeSH QE improve the BM25 baseline |

### Expected Outcomes

Based on prior work (Sjögårde 2024):

- **B > A**: Graph traversal should recover papers that keyword search misses
- **C > B**: Combining bridge types should find papers in blind spots
- **D > B**: MeSH expansion should recover terminologically distant papers
- **Low overlap between methods**: Each method finds different subsets of gold

If B ≈ A or B < A, this signals quarry's graph traversal does not add
value for this task — an important negative result to know.

## Pipeline Commands

```bash
just eval-prep       # PMID → OpenAlex work_id mapping (requires PG)
just eval-run        # Run expand/bridge on each review
just eval-report     # Compute recall, precision, unique discovery
```
