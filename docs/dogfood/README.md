# Dogfood Sessions

quarry CLI UX testing sessions. Each session role-plays a researcher
persona and records friction points, scores, and findings.

Scores are 5 dimensions × 1-5 scale = max 25.

## Sessions

| Date | Persona | Topic | Score | Δ | Key Findings |
|------|---------|-------|-------|---|--------------|
| 2026-04-07 | surveyor | [cytidine deaminase](2026-04-07-cytidine-deaminase.md) | 15/25 | — | AD-8: info/sql --schema needed |
| 2026-04-07 | newbie | [retron](2026-04-07-retron.md) | 16/25 | +1 | AD-9: mesh command needed |
| 2026-04-07 | bridger | [FISH → spatial transcriptomics](2026-04-07-spatial-transcriptomics.md) | 21/25 | +5 | AD-10: shrink needed |
| 2026-04-07 | surveyor | [ABE + PTM × CRISPR](2026-04-07-abe-ptm-crispr.md) | 22/25 | +1 | shrink centralization, mesh_lookup |
| 2026-04-08 | reader | [RNA degrader landscape](2026-04-08-rna-degrader-landscape.md) | 24/25 | +2 | reader persona validated, custom venue essential for chem bio |
| 2026-04-08 | bridger | [protein directed evolution](2026-04-08-protein-directed-evolution.md) | 22/25 | -2 | steiner bridge missing in 3-seed, empty type gives no explanation |
| 2026-04-08 | bridger | [protein evolution regression](2026-04-08-protein-evolution-regression.md) | 25/25 | +3 | S6 regression: all 3 friction points resolved, pairwise sp matrix |
| 2026-04-08 | surveyor | [shrink venue comparison](2026-04-08-shrink-venue-comparison.md) | 23/25 | -2 | NCS+ = custom for NCS-concentrated fields; venue preset needed |
| 2026-04-08 | surveyor | [protein engineering](2026-04-08-protein-engineering.md) | 19/25 | -4 | mesh search misses exact descriptors, S1 drill-down loop validated |
| 2026-04-08 | surveyor | [protein eng regression](2026-04-08-protein-eng-regression.md) | 23/25 | +4 | S9 regression: mesh D015202 resolved, NCS+ 48% for enzyme eng |
| 2026-04-08 | bridger | [far-field bridge](2026-04-08-far-field-bridge.md) | 19/25 | -4 | B3: sp=7 QKD×DNA, coupling finds real cross-domain papers despite MeSH gap |
| 2026-04-08 | bridger | [steiner CRISPR apps](2026-04-08-steiner-crispr-apps.md) | 24/25 | +5 | B1: first non-empty steiner! 3/4 hubs pairwise-unique, sp=3/8/∞ |
| 2026-04-08 | surveyor | [prime editing efficiency](2026-04-08-prime-editing-efficiency.md) | 20/25 | -4 | 8 efficiency factor categories mapped; SQL 1m18s bottleneck |
| 2026-04-08 | reader | [Cas9→PE lineage](2026-04-08-cas9-pe-lineage.md) | 14/25 | -6 | R4+S3+R2+B2 chain: classic expand fails, sp bug found, lineage ≠ citation |
| 2026-04-08 | reader | [F1+F2 regression](2026-04-08-f1-f2-regression.md) | 20/25 | +6 | sp fix verified, --min-citations surfaces CBE/ABE/PE from Cas9 seed |
| 2026-04-08 | surveyor | [reverse central dogma](2026-04-08-reverse-central-dogma.md) | 20/25 | 0 | Rqc2p as protein→RNA precedent via MeSH serendipity; SQL 94% of time |

Scenario coverage and priorities: [scenarios.md](scenarios.md)

## Notes

Scores are per-session UX quality snapshots, not controlled benchmarks.
Each session varies by persona, topic, and available features. Do not
interpret the upward trend as pure tool improvement — field characteristics
and persona type also affect scores. For regression testing, re-run an
earlier session's exact scenario with new features.
