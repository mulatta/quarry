# Evaluation Plan

> How to measure quality of `quarry expand` results without ground truth labels.

## Challenge

No labeled dataset of "paper X is related to paper Y". Traditional IR metrics
(precision, recall, F1) require relevance judgments that don't exist at scale.

## Evaluation Methods

### 1. Reference Recovery (Intrinsic, Automated)

Use the seed's own reference list as ground truth for `foundation` relation.

```
Given: seed with N references (forward edges)
Test:  does expand(seed) recover these references?

recall@k = |expand_top_k ∩ seed_references| / |seed_references|
```

**Strengths**: Fully automated, no manual labels, reproducible.
**Limitations**: Only evaluates `foundation`. Cannot measure `lateral`
or `follow-up` quality. Biased toward citation-heavy signals.

### 2. Leave-One-Out Recovery (Intrinsic, Automated)

Remove direct edges and test if the algorithm recovers them via indirect paths.

```
For each reference R of seed:
  1. Run expand(seed) but exclude direct 1-hop neighbors
  2. Check if R appears in results (recovered via 2+ hop paths)

recovery_rate = count(recovered) / count(references)
```

This tests whether the algorithm understands _structural importance_
beyond direct citation — a reference recovered via 2-hop must be
structurally central to the seed's neighborhood.

**Strengths**: Tests structural reasoning, not just edge lookup.
**Limitations**: Penalizes papers only connected via direct citation.

### 3. Symmetry Test (Stability)

```
For papers A, B where expand(A) contains B:
  Does expand(B) contain A?

symmetry_rate = count(symmetric_pairs) / count(total_pairs)
```

High symmetry indicates stable, bidirectional relevance.
Low symmetry may indicate directional bias in the algorithm.

### 4. Algorithm Comparison (Relative)

Same seed, different algorithms → compare recall/recovery:

```
Algorithms: {PPR, APPR, HKPR, APPR+coupling+RRF}
Metrics:    {recall@50, recall@200, recovery_rate, symmetry_rate}
Seeds:      {N-glycosylation paper, PET depolymerase paper, 10+ others}

Result: table of metric × algorithm × seed
```

### 5. Expert Judgment (Small Scale)

For 2-3 seed papers, manually label top 50 results:

| Label | Definition |
| ------------------ | -------------------------------------------- |
| **relevant** | Directly related to seed's research question |
| **methodological** | Related tool/technique, not topic |
| **background** | Broad field context, not specific |
| **noise** | Unrelated, should not appear |

Compute: precision@k = relevant / k, noise_rate = noise / k.

**Strengths**: Captures nuance that automated metrics miss.
**Limitations**: Not scalable. Subjective. Requires domain expertise.

### 6. Signal Complementarity Analysis

Measure how much each signal contributes unique papers:

```
PPR_only      = PPR_top200 - coupling_top200 - cocite_top200
coupling_only = coupling_top200 - PPR_top200 - cocite_top200
overlap       = PPR_top200 ∩ coupling_top200 ∩ cocite_top200

If overlap is high → signals are redundant → fusion adds little value
If PPR_only is large → coupling/cocite miss important papers
```

### 7. External Baseline Comparison (OpenAlex / Semantic Scholar)

Compare expand results against production recommendation systems as baselines.

#### 7a. OpenAlex Related Works API

```
GET https://api.openalex.org/works/{id}/related_works
```

Returns ~10 related works per paper. Uses citation-based similarity.
Low volume but freely accessible.

```
For each eval seed:
  oa_related = openalex.related_works(seed_doi)
  expand_top = expand(seed, limit=200)

  # Overlap: how many OA-recommended papers does expand find?
  overlap = oa_related ∩ expand_top
  oa_coverage = |overlap| / |oa_related|

  # Disagreement analysis: papers OA recommends but expand misses
  oa_only = oa_related - expand_top
  # → Are these genuinely related? Or OA noise?

  # Papers expand finds but OA doesn't (expand's added value)
  expand_only = expand_top[:10] - oa_related
```

**Purpose**: OA uses a different algorithm (likely co-citation/coupling based).
If expand finds OA's recommendations AND more, that's evidence of quality.
If expand misses OA's core recommendations, investigate why.

#### 7b. Semantic Scholar Related Papers API

```
GET https://api.semanticscholar.org/graph/v1/paper/{id}/recommendations
```

Returns up to 500 recommended papers. Uses SPECTER embeddings + citation
graph features + LTR reranking — significantly more sophisticated than our
structure-only approach.

```
For each eval seed:
  s2_recs = semantic_scholar.recommendations(seed_doi, limit=500)
  expand_top = expand(seed, limit=200)

  # Tier analysis: how well does expand cover S2's top tiers?
  s2_top10  = s2_recs[:10]
  s2_top50  = s2_recs[:50]
  s2_top200 = s2_recs[:200]

  coverage_10  = |expand_top ∩ s2_top10|  / 10
  coverage_50  = |expand_top ∩ s2_top50|  / 50
  coverage_200 = |expand_top ∩ s2_top200| / 200

  # Rank correlation: for papers in both, do rankings agree?
  common = expand_top ∩ s2_top200
  τ = kendall_tau(expand_ranks[common], s2_ranks[common])
```

**Purpose**: S2 is the strongest freely available baseline. It uses
embeddings (content signal) that we don't have yet in Phase 1.

- **High coverage of S2 top-10**: our structure-only approach captures
  the most important related papers even without embeddings.
- **Low rank correlation**: expected — different signals produce different
  orderings. Not a problem if both orderings are valid.
- **S2-only papers**: likely content-similar but structurally distant.
  These are the papers our Phase 3 (embedding layer) should capture.

#### 7c. Baseline Comparison Protocol

```python
EXTERNAL_BASELINES = {
    "openalex": fetch_oa_related,       # ~10 papers, citation-based
    "semantic_scholar": fetch_s2_recs,   # ~500 papers, embedding+citation+LTR
}

for seed in EVAL_SEEDS:
    expand_results = expand(seed, limit=200)
    for name, fetch_fn in EXTERNAL_BASELINES.items():
        baseline = fetch_fn(seed)
        report(
            seed=seed,
            baseline=name,
            overlap=len(set(expand_results) & set(baseline)),
            baseline_coverage=overlap / len(baseline),
            expand_unique=len(set(expand_results) - set(baseline)),
            baseline_unique=len(set(baseline) - set(expand_results)),
        )
```

**Rate limits**: OA: 10 req/s (polite pool), S2: 100 req/5min.
Cache responses per seed to avoid repeated API calls during sweeps.

**When to run**: After major algorithm changes (not every parameter tweak).
Results provide context for internal metrics — if reference recovery drops
but S2 coverage improves, the change may still be positive.

### 8. Temporal Consistency

```
Run expand(seed) on CSR built from:
  - Q1 2025 snapshot
  - Q2 2025 snapshot (3 months later)

stable_rate = |Q1_top100 ∩ Q2_top100| / 100
```

High stability = algorithm finds enduring relevance.
Low stability = sensitive to citation fluctuations.

## Evaluation Protocol

### Phase 1: Automated (run after each algorithm change)

```python
EVAL_SEEDS = [
    4406019873,  # PET depolymerase (Seo 2025)
    4402354657,  # N-glycosylation IL6 (Hung 2024)
    # ... 10+ diverse papers across fields
]

for seed in EVAL_SEEDS:
    for algo in [ppr, appr, hkpr, appr_rrf]:
        results = algo(seed, limit=200)

        ref_recall_50  = reference_recovery(seed, results[:50])
        ref_recall_200 = reference_recovery(seed, results[:200])
        loo_recovery   = leave_one_out(seed, algo)
        symmetry       = symmetry_test(seed, results[:50], algo)

        log(seed, algo, ref_recall_50, ref_recall_200, loo_recovery, symmetry)
```

### Phase 2: External baselines (after major algorithm changes)

- Run Method 7 (OA + S2 comparison) for all eval seeds
- Cache API responses to avoid rate limits
- Determines whether structure-only approach is competitive

### Phase 3: Expert (once per major algorithm change)

- 2-3 seeds, top 50 results
- Manual relevant/noise labels
- Compare precision across algorithms

### Phase 4: Signal analysis (once per new signal addition)

- Measure overlap/complementarity of PPR, coupling, cocite
- Determine if RRF adds value over single signal
- Baseline: current sweep shows APPR-only ≥ all fusion configs on reference recovery

## Seed Selection Criteria for Evaluation

Diverse set covering:

| Criterion | Reason |
| -------------------------- | --------------------------------------------------------------- |
| Different fields | PET enzymology, cancer biology, crystallography |
| Different citation counts | low (c\<50), medium (c:100-1000), high (c>5000) |
| Different years | recent (2024-2025), established (2015-2020), classic (pre-2010) |
| Different reference counts | few refs (\<20), many refs (>100) |
| Review vs research article | Different citation patterns |

## Success Criteria

| Metric | Minimum | Good |
| ------------------------------- | ------- | ---- |
| recall@200 (reference recovery) | >0.5 | >0.7 |
| LOO recovery rate | >0.3 | >0.5 |
| symmetry rate | >0.5 | >0.7 |
| S2 top-10 coverage | >0.3 | >0.6 |
| OA related coverage | >0.5 | >0.8 |
| noise rate (expert) | \<0.2 | \<0.1 |
| elapsed time | \<5s | \<2s |
