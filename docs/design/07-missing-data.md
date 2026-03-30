# Missing Data Fairness

> How to handle signals with partial coverage without introducing bias.

## The Problem

Different signals have vastly different coverage:

| Signal | Coverage (PubMed) | Coverage (all OA) |
|--------|-------------------|-------------------|
| Citation edges (out) | 74% of iCite | 25% of OA works |
| Citation edges (in) | ~80% | ~28% |
| RCR | 74% | 4% |
| FCR | 84% | 5% |
| MeSH | ~95% of T1 | ~5% of all |
| OA Topics | ~100% | ~100% |
| Embeddings | 0% (in progress) | 0% |
| pub_year | ~100% | ~100% |

When a signal is available for some papers but not others,
using it as a weight or score creates **systematic bias**:
papers WITH the signal get boosted/penalized, papers WITHOUT
are treated as default → unfair comparison.

## Approaches Under Consideration

### A. Imputation (fill missing with field average)

```
if RCR available: use RCR
else: use median RCR for the paper's field/year
```

Pros: all papers get a value, no missing data gap.
Cons: imputed values add noise, may not reflect true quality.

### B. Segregated scoring (only compare within same coverage)

```
papers_with_RCR = [... scored with RCR ...]
papers_without = [... scored without RCR ...]
interleave by primary signal (PPR)
```

Pros: no bias from missing data.
Cons: complex, papers in different pools may not be fairly compared.

### C. Reranking-only for partial signals

```
L2 fusion: uses only ~100% coverage signals (PPR, topics, pub_year)
L3 rerank: uses partial signals (RCR, MeSH) as tiebreaker only
```

Pros: primary ranking unbiased, partial signals only adjust within ties.
Cons: high-quality signal (RCR) is underutilized.

### D. Coverage-weighted contribution

```
RRF weight for signal S = coverage(S) within candidate pool
if pool has 80% RCR coverage → RCR weight = 0.8 in fusion
if pool has 10% RCR coverage → RCR weight = 0.1
```

Pros: signal influence proportional to its reliability in this query.
Cons: weight varies per query, harder to reason about.

## Current Decision

**Undecided.** Prototype with approach C (reranking-only for partial signals),
evaluate bias on test cases, then consider alternatives.

## Test Cases for Evaluation

1. Seed with high RCR in well-covered field (e.g., cancer biology)
   → RCR should help differentiate
2. Seed in poorly-covered field (e.g., niche ecology)
   → RCR coverage low, should not distort results
3. Seed citing both PubMed and non-PubMed papers
   → MeSH available for some neighbors, not others

## Open Questions

- [ ] Which approach to implement first
- [ ] How to measure bias (compare results with/without partial signal)
- [ ] Whether to precompute coverage statistics per field for approach D
