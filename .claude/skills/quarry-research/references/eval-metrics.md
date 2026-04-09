# Quality Evaluation Metrics

Measured by the harness after each session, not by the agent itself.

## Automated Metrics (harness computes from state files)

| Metric | Definition | How to measure |
|--------|-----------|---------------|
| **Serendipity novelty** | Fraction of serendipity papers NOT in any single seed's expand top-200 | For each serendipity paper: check all sp expand_raw.txt |
| **MeSH diversity** | Unique major MeSH descriptors in reading list | `quarry info --mesh` on all reading list papers |
| **Temporal span** | Year range of reading list | max_year - min_year |
| **Cross-domain coverage** | Non-empty bridge pairs / total pairs | Count from bridge connection files |
| **Recall vs review** | Reading list overlap with field review paper's references | shrink top-1 is usually a review → extract its refs |

## Manual Metrics (domain expert)

| Metric | Definition |
|--------|-----------|
| **Precision@K** | Fraction of reading list papers actually relevant |
| **Feasibility calibration** | ★ rating agreement with expert assessment |
| **Architecture soundness** | Proposed mechanism doesn't violate known constraints |

## Targets (from S16-agent baseline)

| Metric | S16-agent value | Target |
|--------|----------------|--------|
| Serendipity novelty | 67% (10/15) | ≥50% |
| MeSH diversity | 42 descriptors | ≥30 |
| Temporal span | 47 years | ≥20 |
| Cross-domain coverage | 100% (6/6) | ≥80% |
