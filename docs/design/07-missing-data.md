# Missing Data Fairness

> How to handle signals with partial coverage without introducing bias.

## Decision: Expose, Don't Impute

**quarry returns NULL for missing data. No imputation, no default values.**

This resolves the previously undecided question (approaches A-D) by choosing
a simpler path: quarry is a data pipeline, not a scoring engine. Missing data
is the consumer's problem.

### Rationale

Evaluated on 5 seed papers (PET enzyme, base editing, PLM, bioinformatics):

| Signal | Coverage in expand top-200 | NULL pattern |
|--------|---------------------------|-------------|
| cnp/fwci | 87-99% | Mostly preprints (bioRxiv, 2025) |
| rcr | 55-85% | Non-PubMed papers (chemistry, CS) |
| cited_by_count | ~100% | Rare |

NULL is not random — it's **systematic**:
- Preprints: no fwci/cnp (not yet indexed by OpenAlex metrics)
- Chemistry journals: no rcr (not in PubMed)
- Very new papers: no citations yet

Imputing these with field averages (approach A) or segregating scores
(approach B) would hide this systematic pattern from the consumer.
Returning NULL preserves the signal that "this paper's quality is unmeasured".

### Coverage by Paper Category (from 1,305 papers across 5 seeds)

| Category | Count | fwci% | cnp% | rcr% |
|----------|-------|-------|------|------|
| PubMed article | 748 | 99% | 99% | 79% |
| PubMed review | 273 | 100% | 100% | 82% |
| Preprint | 105 | 10% | 10% | 8% |
| Chemistry/material | 48 | 100% | 100% | 0% |
| OA other | 109 | 92% | 92% | 0% |

### Previous Approaches (Considered, Not Adopted)

| Approach | Description | Why not |
|----------|-------------|---------|
| A. Imputation | Fill NULL with field median | False precision. Preprint ≠ "average paper" |
| B. Segregated scoring | Separate pools by coverage | Complex, still unfair cross-pool comparison |
| C. Reranking-only | Partial signals as tiebreaker | quarry doesn't rerank — metadata only |
| D. Coverage-weighted | Signal weight ∝ coverage in pool | Unnecessary if signals are metadata only |

All approaches assumed quarry would use quality for scoring.
Since quality is metadata only, the fairness problem dissolves.
