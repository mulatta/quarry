# State Directory Schema

```
$HOME/.claude/outputs/research-scout-<slug>/state/
├── sub_problems.yaml          # Phase 0 output
├── sp_1/
│   ├── seeds.yaml             # explorer output
│   ├── findings.yaml          # explorer output
│   ├── serendipity_flags.yaml # L1 hook output
│   ├── expand_raw.txt         # full quarry expand output
│   └── summary.txt            # 1-paragraph landscape summary
├── sp_2/
│   └── ...
├── sp_2a/                     # tree refinement (if triggered)
│   └── ...                    # same structure as sp_*
├── sp_2b/
│   └── ...
├── bridge_1_2/
│   ├── connections.yaml       # bridger output
│   ├── serendipity_flags.yaml # L1 flags from bridge results
│   └── bridge_raw.txt         # full quarry bridge output
├── serendipity_validated.yaml # Phase 2.5: orchestrator-validated entries
├── gaps.yaml                  # synthesizer output: unresolved gaps
└── report.md                  # synthesizer output: final report
```

## sub_problems.yaml

```yaml
sub_problems:
  - id: 1
    name: "short name"
    function: "what it must do"
    analogue: "closest known system"
    search: ["MeSH term 1", "keyword"]
    depends_on: []
    parent_id: null    # null for root, sp id for tree refinements
    status: explored   # pending | explored | refined | unable_to_assess
    feasibility: "★★★☆"
  - id: "2a"           # sub-sp from tree refinement of sp 2
    name: "narrowed aspect"
    function: "specific part of sp 2"
    analogue: "..."
    search: ["..."]
    depends_on: []
    parent_id: 2       # indicates this is a child of sp 2
    status: explored
    feasibility: "★☆☆☆"
```

## seeds.yaml

```yaml
seeds:
  - work_id: W1985387528
    title: "Rqc2p and 60S ribosomal subunits..."
    year: 2015
    cited_by: 307
    why: "protein-directed tRNA selection without mRNA"
    mesh_major: ["Peptide Biosynthesis, Nucleic Acid-Independent"]
```

## connections.yaml

```yaml
connections:
  - pair: [1, 2]
    sp: 4
    overlap_refs: 2
    overlap_citers: 137
    key_paper:
      work_id: W2077686875
      title: "Quality Control Mechanisms During Translation"
    implication: "QC bridges protein reading and codon assignment"
```

## serendipity_validated.yaml

```yaml
sp_distribution:
  median: 5
  min: 3
  max: 7
  values: {1_2: 4, 1_3: 3, 1_4: 7, 2_3: 6, 2_4: 4, 3_4: 3}

validated:
  - paper: {work_id: W2789966123, title: "..."}
    from_sp: 2
    to_sp: 1
    scores: {novelty: 1, specificity: 1, actionability: 1}
    total: 3
    bridge_sp: 4
    mechanism: "..."
    verdict: validated

rejected:
  - paper: {work_id: W2107167859, title: "..."}
    from_sp: 3
    to_sp: 2
    scores: {novelty: 0, specificity: 1, actionability: 0}
    total: 1
    reason: "implicit in decomposition"
    verdict: related_finding
```

## gaps.yaml

```yaml
has_gaps: false
gaps:
  - sp_id: "2b"
    gap_description: "no biological system for non-destructive side chain recognition"
    severity: critical  # critical | major | minor
```
