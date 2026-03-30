# Layer 4: Temporal (Citation Dynamics)

> Signal: time-dependent patterns in citation behavior.

## Data Sources

- pub_year / pub_date: OA works_export (~100% coverage)
- citation_count: iCite (40.2M) or OA cited_by_count
- No time-series citation data (only cumulative snapshot)

## L0 Atomic Operations

### Publication Recency

- **Input**: pub_year
- **Output**: decay(current_year - pub_year)
- **Decay function**: TBD (linear, exponential, or step)
- **Use**: opt-in — some queries want recent papers, some want foundational
- **Caveat**: penalizes seminal old papers. Must be user-controlled, not default.

### Citation Velocity (approximate)

- **Input**: cited_by_count, pub_year
- **Output**: velocity ≈ cited_by_count / (current_year - pub_year)
- **Interpretation**: high velocity = rapidly accumulating citations = emerging/hot paper
- **Caveat**: rough approximation — true velocity requires time-series snapshots
- **Improvement**: with quarterly snapshots, compute Δcitations/Δtime

### Sleeping Beauty Detection (future)

- **Input**: time-series citation counts (requires quarterly snapshot accumulation)
- **Output**: "beauty coefficient" — long dormancy followed by sudden citation burst
- **Status**: data insufficient (only cumulative snapshot)

## Role in Pipeline

Temporal is **opt-in and supplementary**, not default.

Default pipeline runs without temporal influence.
User can activate recency or velocity as additional signal in fusion.

## Open Questions

- [ ] Should temporal be a separate layer or a modifier on other layers?
- [ ] Quarterly snapshot accumulation plan for true time-series
- [ ] Decay function design for recency
