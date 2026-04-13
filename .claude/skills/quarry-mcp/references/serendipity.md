# Serendipity Detection Protocol (3-Layer + Validation)

## Definition

Serendipity = a cross-sub-problem connection that was **not predictable
from the Phase 0 decomposition** (analogues + search terms), that
**suggests a concrete mechanism** (not a generic review), and that
**changes the proposed architecture or reveals a new sub-problem**.

NOT serendipity:

- Connections already implicit in the decomposition (e.g., aaRS connects
  "amino acid recognition" and "codon assignment" — this IS the aaRS's
  known function)
- Generic review papers that mention both sub-problems
- Papers from the same research group/field as the analogue

## Layer 1: Deterministic Flag (hook, after each explorer)

The orchestrator runs `hooks/scripts/serendipity-l1.sh` which does:

1. **Title keyword scan**: case-insensitive substring match of paper
   titles against every OTHER sub-problem's `search` terms.
1. **MeSH cross-check**: if paper has MeSH tags, check against other
   sub-problems' known MeSH descriptors.

Output: `state/sp_{id}/serendipity_flags.yaml` — raw candidates.

## Layer 2: Explorer LLM check (per paper, during exploration)

Explorer runs `quarry info <flagged_paper> --full` and asks:
"Does this paper's mechanism relate to [other sp]?"
YES → flag survives. NO → discarded as false positive.

## Layer 3: Semantic Escape Hatch (explorer, end of each sp)

> Review top 5 papers. Could any mechanism be repurposed for
> another sub-problem in a way keyword matching would miss?

Output: additional candidates with `match_type: semantic_escape`.

## Validation: Orchestrator Full-Context Scoring (Phase 2.5)

Only the orchestrator has the full picture: all sub-problem definitions,
all findings, all bridge results. The orchestrator validates each
candidate using three dimensions:

### Novelty (0 or 1)

Compare the candidate against `sub_problems.yaml`:

- Is the candidate paper's mechanism already listed as an `analogue`
  or `search` term in ANY sub-problem? → 0 (predictable)
- If not listed, use **bridge sp as soft signal**:

```
bridge sp context:
  pair_sp: 4          # sp between the two sub-problems this candidate connects
  session_median: 5   # median sp across all bridge pairs in this session
  session_range: [3, 7]

Interpretation (NOT hard threshold):
  sp < median → fields are closer than average → connection MORE likely predictable → lean toward 0
  sp > median → fields are farther than average → connection LESS likely predictable → lean toward 1
  sp = ∞     → no citation path → if connection is real, strong novelty signal
               BUT consider: graph may be incomplete for new/niche fields
```

The orchestrator makes the final call. sp is evidence, not verdict.

### Specificity (0 or 1)

- Generic review mentioning both topics → 0
- Original research with specific mechanism/data/enzyme → 1
- If uncertain, check: does the abstract contain quantitative results,
  specific protein/gene names, or experimental methods? → 1

### Actionability (0 or 1)

- Does this finding appear in (or change) the proposed architecture? → 1
- Does it create a new sub-problem not in the original decomposition? → 1
- Does it merely confirm what was already planned? → 0

### Threshold

Score = novelty + specificity + actionability.

- ≥ 2: **validated serendipity** → main report section
- 1: **related finding** → report appendix
- 0: **rejected** → not in report

### Output

`state/serendipity_validated.yaml`:

```yaml
validated:
  - paper: {work_id: W2789966123, title: "..."}
    from_sp: 2
    to_sp: 1
    scores: {novelty: 1, specificity: 1, actionability: 1}
    total: 3
    bridge_sp: 4
    session_median_sp: 5
    mechanism: "ClpXP repurposed as processive peptide reader"
    verdict: validated

rejected:
  - paper: {work_id: W2107167859, title: "..."}
    from_sp: 3
    to_sp: 2
    scores: {novelty: 0, specificity: 1, actionability: 0}
    total: 1
    reason: "aaRS connecting SP2-SP3 is implicit in decomposition"
    verdict: related_finding
```
