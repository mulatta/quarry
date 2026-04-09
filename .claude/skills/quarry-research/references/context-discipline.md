# Context Discipline

quarry commands produce verbose output. Retaining everything overflows
the context window. The agent decides what to keep based on the result
distribution — never hard-coded top-N values.

## Expand results

1. Read the `score` column.
1. Find cutoff: score < 30% of rank-1 score.
1. Keep papers above cutoff, plus ALL laterals regardless of score.
1. **Hard cap**: 30 papers max in context (circuit breaker).
1. Write full output to `state/sp_{id}/expand_raw.txt`.

## Bridge results

1. Per bridge type, keep papers with AA > median AA of that type.
1. **Hard cap**: 15 per type in context.
1. Write full output to `state/bridge_{i}_{j}/bridge_raw.txt`.

## Info results

Keep abstract in context. MeSH list and metadata go to state file only.

## Phase summaries

At END of each phase, write 1-paragraph summary to state.
In subsequent phases, reference the summary — not raw output.
