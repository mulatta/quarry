# Layer 3: Quality (Impact Metrics)

> Signal: paper importance / reliability, independent of seed relationship.

## Decision: Metadata Only (No Reranking)

Quality signals are **exposed as metadata in expand output**, not used for
scoring or reranking. Reason: quarry returns complete data, downstream
consumers (jq, SQL, agents) decide how to sort/filter.

This resolves the open question "L2 fusion vs L3 reranking" — neither.
Quality data is informational, not computational.

## Available Signals — Evaluation Results

Evaluated on 5 seed papers across fields (PET enzyme, base editing,
protein language model, bioinformatics). Coverage measured on expand top-200.

### Selected (included in output)

| Signal | Coverage | Source | Why included |
|--------|----------|--------|-------------|
| **cnp** (citation_normalized_percentile) | 87-99% | OpenAlex | 0-1 range, field+year normalized, no additional normalization needed. Primary quality indicator. |
| **fwci** (Field-Weighted Citation Impact) | 87-99% | OpenAlex | Same coverage as cnp, ρ=0.99 correlation with cnp. Included for users who prefer absolute scale. |
| **cited_by_count** | ~100% | OpenAlex | Raw citation count. Biased (power law, not field-normalized) but universally available. Useful as fallback when cnp/fwci are NULL. |
| **rcr** (Relative Citation Ratio) | 55-85% | iCite | PubMed-only. High quality for biomedical papers. NULL for chemistry, CS, preprints. |

### Not Selected (excluded from output)

| Signal | Coverage | Why excluded |
|--------|----------|-------------|
| **apt** (Approximate Potential to Translate) | 7.9% | Too niche — measures clinical translation probability. Only meaningful for translational medicine. Misleading for basic science. |
| **nih_percentile** | 6.4% | NIH-funded papers only. Extreme selection bias. |
| **fcr** (Field Citation Rate) | ~5% global | Baseline for RCR calculation. Not useful standalone — RCR already incorporates it. |
| **Global PageRank** | Not computed | Would require full-graph iteration (~minutes). cited_by_count serves as rough proxy. |

### Key Findings from Evaluation

1. **fwci and cnp are essentially identical** (ρ=0.99 across all seeds).
   cnp preferred because 0-1 range requires no normalization.

1. **Preprints have no quality signals** — fwci/cnp coverage drops to 10%
   for preprints. In ML-heavy seeds (Jiang-2025), 8% of expand results
   are preprints with NULL quality data.

1. **fwci captures landmark papers** — ESM-2 (fwci=672), Yoshida 2016
   (fwci=51) correctly identified as field-defining. But also promotes
   off-topic high-impact papers (e.g., PBAT degradation for PET seed).

1. **fwci demotes seed-relevant niche papers** — recent method papers
   with low citations (fwci\<3) are structurally important but would be
   deprioritized by quality reranking.

1. **Conclusion**: quality signals measure "field importance" not
   "relevance to seed". Using them for reranking would hurt precision.
   Exposing as metadata lets users make informed judgments.

## NULL Handling

Quality signals with NULL are returned as `null` in JSON output.
No imputation, no penalty, no default values. Downstream consumers
decide how to handle missing data.

Rationale: imputing (e.g., fwci=1.0 for NULL) creates false precision.
A preprint with NULL fwci is genuinely unmeasured, not "average".

## Data Sources

- OpenAlex (fwci, cnp, cited_by_count): 58% coverage of all 490M works.
  Includes chemistry, CS, preprints via bioRxiv.
- iCite (rcr): 7.4% of all works (36M PubMed papers). High quality but
  limited to biomedical literature.
