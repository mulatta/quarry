# State Directory Schema

```
$HOME/.claude/outputs/research-scout-<slug>/state/
├── normalized_query.yaml      # Phase -1 output
├── sub_problems.yaml          # Phase 0 output (DAG)
├── sp_1/
│   ├── seeds.yaml             # explorer output
│   ├── findings.yaml          # explorer output
│   ├── serendipity_flags.yaml # L1 hook output
│   ├── expand_raw.txt         # full expand tool raw output
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
│   └── bridge_raw.txt         # full bridge tool raw output
├── serendipity_validated.yaml # Phase 2.5: orchestrator-validated entries
├── user_decisions.yaml        # HITL checkpoint decisions (all phases)
├── gaps.yaml                  # synthesizer output: unresolved gaps
└── report.md                  # synthesizer output: final report
```

## normalized_query.yaml

```yaml
original: "RNA 자가복제 나노머신이 가능한가?"
system: "RNA-based ribozyme"
mechanism: "self-replication via template-dependent polymerization"
context: "in vitro, synthetic biology"
outcome: "feasibility of autonomous replication cycle"
scope_assumptions:
  - "excludes protein-assisted replication"
  - "focuses on catalytic RNA (ribozymes)"
clarification_asked: false   # true if user was asked clarifying questions
```

## sub_problems.yaml

```yaml
# DAG structure: depends_on defines directed edges (prerequisite relationship).
# No cycles allowed. Topological order determines exploration sequence.
# SPs with no unresolved depends_on may be explored in parallel.
sub_problems:
  - id: 1
    name: "polymerase activity"
    function: "RNA-catalyzed RNA polymerization"
    analogue: "ribozyme polymerase (e.g., tC19Z)"
    search: ["RNA-Directed RNA Polymerase", "ribozyme"]
    depends_on: []          # root node — explored first
    parent_id: null
    depth: 0                # refinement depth (max 2)
    status: explored        # pending | explored | refined | unable_to_assess
    feasibility: "★★★☆"
  - id: 2
    name: "template recognition"
    function: "sequence-specific template binding"
    analogue: "Watson-Crick base pairing in ribozymes"
    search: ["Template, RNA", "base pairing"]
    depends_on: [1]         # needs SP1 results (polymerase context)
    parent_id: null
    depth: 0
    status: explored
    feasibility: "★★☆☆"
  - id: 3
    name: "error correction"
    function: "replication fidelity above error catastrophe threshold"
    analogue: "Qβ replicase fidelity mechanisms"
    search: ["RNA Replicase", "mutation rate"]
    depends_on: [1, 2]      # needs both SP1 and SP2
    parent_id: null
    depth: 0
    status: pending
    feasibility: null
  - id: "2a"                # refinement of SP2 (depth 1)
    name: "internal template binding"
    function: "cis-acting template recognition"
    analogue: "..."
    search: ["..."]
    depends_on: [1]         # inherits parent's dependency
    parent_id: 2
    depth: 1
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

## user_decisions.yaml

Written by the orchestrator (research-scout) after each HITL checkpoint.
Read by the synthesizer when spawned. Append each section as the session
progresses; do not overwrite earlier sections.

```yaml
hitl_0:
  confirmed: true
  adjustments: null             # or "description" if user corrected query

hitl_decompose:
  approved: true
  removed_sps: []               # sp ids user asked to remove before exploring
  notes: null

hitl_1:
  fatal_decisions:
    - issue_id: "critic_fatal_1"
      sp_id: sp_2
      persona: "Mechanistic Skeptic"
      summary: "strand separation requires ~100 kJ/mol, no biological catalyst"
      decision: adjust          # continue | adjust | drop | stop
      note: "assume mineral surface catalysis"
    - issue_id: "critic_fatal_2"
      sp_id: sp_3
      persona: "Prior Art Investigator"
      summary: "Szostak 2012 negative result (PMID 22231509)"
      decision: drop
      note: null
  high_dismissed: []            # issue_ids user explicitly dismissed
  session_action: continue      # continue | stop
  adjusted_sps: [sp_2]
  dropped_sps: [sp_3]
  # If no fatal issues existed:
  # skipped: true

hitl_2:
  focus_sps: null               # null = all SPs; or [sp_1, sp_2]
  skip_serendipity: false
  synthesizer_notes: null       # free text prepended to synthesizer prompt
```

**Synthesizer contract**: read `user_decisions.yaml` before generating
the report. Apply these rules strictly:
- Any sp in `dropped_sps` → exclude from all sections
- Any sp in `adjusted_sps` → note updated hypothesis in feasibility table
- `focus_sps` not null → limit Feasibility Assessment section to listed SPs
- `skip_serendipity: true` → omit Serendipity section entirely
- `synthesizer_notes` not null → include as "User context" at top of prompt
