# Research Agent Architecture

> Status: Active. Documents the research agent harness design,
> evolution roadmap, and key architectural decisions.
>
> Related: `13-system-layers.md` (Agent Layer), `10-architecture-decisions.md`

## Problem Statement

quarry CLI provides atomic tools (mesh, expand, bridge, shrink, info)
for paper discovery. Composing these into a coherent research workflow
requires LLM judgment: problem decomposition, seed selection, serendipity
detection, and feasibility synthesis.

A research agent automates this composition while preserving the
properties that make quarry valuable: citation-graph-based discovery
and MeSH-mediated serendipity.

## Validated Findings (from dogfood sessions S13-S16)

| Session | Finding | Impact on Design |
|---------|---------|-----------------|
| S13 | PE efficiency: 8 factor categories mapped in 24 commands | Agent must manage 20-50 tool calls per session |
| S14 | Cas9→PE lineage: bridge sp bug found, classic paper noise | Led to F1 (sp fix) and F2 (--min-citations) |
| S16 | Reverse central dogma: Rqc2p found via MeSH serendipity | Serendipity detection is core value, not optional |
| S16-agent | 50 tool calls, 4 serendipity, 15-paper reading list | Single agent works but context pressure at 50 calls |
| S16-agent | Fidelity as natural integration point (bridge sp=3 pattern) | Bridge analysis produces structural insights beyond individual expand |
| Pipeline-run1 | 94 calls, 6 serendipity, CSR error → SQL fallback, hooks not called | Hook invocation must be explicit bash command in SOUL.md |
| Pipeline-run2 | 95 calls, 6 serendipity, 41 state files, L1 hooks 4/4, novelty 89% | Pipeline validated. L1 deterministic flag caught "aaRS as bridge" |

## Architecture: Pipeline of Specialized Agents

### Why Pipeline, Not Single Agent

A single 454-line agent executing 50 tool calls in one context window
suffers from context pressure: early findings are compressed away,
serendipity detection degrades, and the agent cannot be partially re-run.

The pipeline pattern (validated by Ralph loop principle: "state on disk,
fresh context per iteration") gives each agent a clean context window
of 10-15 tool calls while sharing state via disk.

Cross-pollination between sub-problems — the key concern against splitting
— is handled by the orchestrator reading state files between agent spawns
and enriching the next agent's input. This is batch cross-pollination
(between agents) rather than real-time (within one context), but S16-agent
showed the "real-time" version was effectively batch anyway (sequential
sub-problem processing).

### Agent Inventory

| Agent | Role | Tools | Lines |
|-------|------|-------|-------|
| `research-scout` | Orchestrator: decompose, spawn, serendipity L1, checkpoints | Agent, Read, Write, Bash (quarry only) | ~120 |
| `research-explorer` | Worker: quarry mesh→info→expand for one sub-problem | Bash (quarry), Read, Write | ~80 |
| `research-bridger` | Worker: quarry bridge for one seed pair | Bash (quarry), Read, Write | ~60 |
| `research-synthesizer` | Worker: state→report, optional PMC deep read | Read, Write, PubMed MCP | ~80 |

### Shared Skill

```
.claude/skills/quarry-research/
  SKILL.md                     # trigger + mode routing
  references/
    quarry-reference.md        # CLI command reference
    serendipity.md             # 2-layer detection protocol
    context-discipline.md      # data-driven cutoff rules
    deep-read.md               # PMC full-text workflow
    eval-metrics.md            # quality measurement
    state-schema.md            # state directory spec
    report-template.md         # output format
```

All agents read from `quarry-research/references/` as needed.
This separates orchestration (agents) from procedural knowledge (skill).

### Pipeline Flow

```
USER → idea
         │
    ┌────▼─────────────────────────────────────────────────┐
    │  research-scout (orchestrator)                        │
    │                                                      │
    │  DECOMPOSE (self, no sub-agent)                      │
    │  → state/sub_problems.yaml                           │
    │  → CHECKPOINT: user approval                         │
    │                                                      │
    │  EXPLORE (sequential, per sub-problem)               │
    │  ┌────────────────────────────────────────┐          │
    │  │ for sp in sub_problems:                │          │
    │  │   Agent(research-explorer, sp + state) │          │
    │  │   read state/sp_{id}/*.yaml            │          │
    │  │   serendipity L1 grep (deterministic)  │          │
    │  │   enrich next sp's search strategy     │          │
    │  │   if needs_refinement: re-decompose sp │ (A)      │
    │  └────────────────────────────────────────┘          │
    │                                                      │
    │  BRIDGE (parallel possible, per pair)                │
    │  ┌────────────────────────────────────────┐          │
    │  │ for pair in sp_pairs:                  │          │
    │  │   Agent(research-bridger, pair + state)│          │
    │  │   if new sp found: add + re-explore    │ (B)      │
    │  └────────────────────────────────────────┘          │
    │  → CHECKPOINT: bridge summary → user                 │
    │                                                      │
    │  SYNTHESIZE                                          │
    │  ┌────────────────────────────────────────┐          │
    │  │ Agent(research-synthesizer, all state)  │          │
    │  │ if gap found: re-explore specific sp    │ (C)     │
    │  └────────────────────────────────────────┘          │
    │                                                      │
    │  → report.md                                         │
    └──────────────────────────────────────────────────────┘

    Loops (each max 1 iteration):
      (A) Refine: sp too broad → re-decompose + re-explore sub-sps
      (B) Redecompose: bridge reveals missing sp → add + explore
      (C) Gap: synthesis finds insufficient evidence → re-explore sp
```

### Serendipity Detection: 3-Layer Architecture

| Layer | Where | Method | Catches |
|-------|-------|--------|---------|
| L1: Deterministic | Orchestrator (between agent spawns) | grep state files: all papers vs all sp keywords/MeSH | Keyword/MeSH cross-hits |
| L2: LLM Evaluation | Explorer agent (per flagged paper) | Read abstract, assess relevance to other sp | Meaningful connections from L1 flags |
| L3: Semantic Escape | Explorer agent (end of each sp) | "Could top-5 papers be repurposed for other sp?" | Purely semantic connections (e.g., "mRNA-independent" ↔ "template-free") |

L1 is the harness (deterministic, never skipped). L2-L3 are the model
(non-deterministic, may miss things). This separation means serendipity
detection improves as tools improve, independent of model capability.

### Context Discipline: Data-Driven Cutoffs

Hard-coded top-N values were rejected. Instead:

| Result type | Retention criterion | Hard cap (circuit breaker) |
|------------|--------------------|----|
| Expand | score > 30% of rank-1 score, plus all laterals | 30 papers |
| Bridge | AA > median AA of type | 15 per type |
| Info | Abstract only (MeSH/metadata in state file) | — |

The agent decides the cutoff by reading the data distribution.
Hard caps prevent context overflow regardless of agent judgment.

## State Management Evolution

### Phase A: YAML (Current)

```
$HOME/.claude/outputs/research-scout-<slug>/
  sub_problems.yaml
  sp_1/{seeds,findings,serendipity_flags,expand_raw}.yaml
  sp_2/...
  bridge_1_2/{connections,bridge_raw}.yaml
  report.md
```

Sufficient for: 4-6 sub-problems, ~80 papers, single session.

### Phase B: SQLite (3-5 sessions)

Migration trigger: cross-sub-problem queries become frequent
(serendipity L1 doing N×M keyword matching across all papers
and all sub-problems).

Single `research.db` file replaces YAML directory. Schema:

```sql
papers(work_id, title, year, cited_by, mesh_tags, sub_problem_id, source_cmd)
serendipity(paper_id, from_sp, to_sp, match_type, mechanism)
findings(sub_problem_id, key, value)
```

### Phase C: memvid (Cross-Session Memory)

Migration trigger: user runs research agent repeatedly and wants
"remember what we found last week about Rqc2p."

memvid provides:

- **Branching**: decompose sp2 two different ways, compare results
- **Semantic search**: serendipity L1 upgraded from grep to embedding
- **Time-travel**: rewind to pre-bridge state, try different seed pairs
- **Single file**: `.mv2` capsule per research topic, portable

Evaluation criteria before adoption:

- [ ] memvid Python API callable from Claude Code Bash/scripts
- [ ] Nix packaging available (or pip install sufficient)
- [ ] Branching API stable for research workflow pattern
- [ ] Benchmark: semantic search recall vs grep L1 on dogfood data

## Orchestration Evolution

### Phase A: Markdown Agents (Current)

```
.claude/agents/research-scout.md      (orchestrator, ~120 lines)
.claude/agents/research-explorer.md   (worker)
.claude/agents/research-bridger.md    (worker)
.claude/agents/research-synthesizer.md (worker)
.claude/skills/quarry-research/       (shared references)
```

Orchestrator spawns workers via `Agent()` tool. State on disk.
Loops capped at max 1 iteration. Sufficient for sequential pipeline
with simple conditional branches.

### Phase B: Markdown + Critic (3-5 sessions)

Add critic loop to synthesis:

```
SYNTHESIZE → CRITIC → if issues: re-EXPLORE → re-SYNTHESIZE
```

md-based if:

- Loop is 1-level (no loop-in-loop)
- ≤2 parallel agents
- Critic applies to 1 phase only
- Incorrect loop decisions < 3 observed

### Phase C: Claude Agent SDK (Transition Trigger)

Transition when ANY of:

- Loop-in-loop needed (critic inside refine loop)
- 3+ parallel agents with synchronization
- Observed: agent ignores loop cap 3+ times
- Observed: state inconsistency from partial failure

SDK provides:

- Deterministic state machine (Python code, not LLM judgment)
- Programmatic retry with backoff
- Explicit tool allowlists enforced in code
- Checkpoint/resume across sessions
- Cost tracking per agent

md agents convert to SDK tool definitions. Skill references remain.

### Phase D: Structured Deep Read (AIM-SciQA Pattern)

Reference: AIM-SciQA (arXiv:2603.14257) — automatic inter-document
multi-hop scientific QA generation from PMC papers.

Key insight: reading a paper should produce structured (Q, A, Evidence)
triplets per section, not unstructured summaries. This enables:

1. **Serendipity via QA embedding**: cross-sp QA cosine similarity
   (τ=0.3 threshold, per AIM-SciQA) replaces keyword grep for L1.
1. **Multi-hop QA as quality metric**: combine QA pairs from different
   sub-problems → generate cross-document question → test if agent
   can answer → quantitative feasibility calibration.
1. **Evidence tracing**: each insight carries its source sentence,
   eliminating hallucination in feasibility claims.

Implementation in synthesizer agent:

- PMC full text via `get_full_text_article` MCP tool
- Section-level (Q,A,E) extraction via LLM MRC prompt
- Triplets stored in `state/sp_{id}/qa_triplets.yaml`
- Cross-sp matching stored in `state/qa_relations.yaml`

Prerequisites: PMC MCP plugin (already installed), lancedb for
QA embeddings (Phase D only — keyword grep fallback until then).

### Phase E: DSPy/BAML Optimization (eval data 200+)

Prerequisites:

- BAML functions extracted: `AssessSeedQuality`, `DetectSerendipity`,
  `JudgeRelevance`, `RateFeasibility`
- Eval dataset: 200+ (paper, judgment_label) pairs from agent runs
  (QA triplets from Phase D serve as natural eval data source)
- Metric defined: serendipity recall, feasibility calibration,
  multi-hop QA answerability (AIM-SciQA pattern)

DSPy optimizes BAML function prompts automatically.
Human-rated bridge results + QA triplet quality feed the loop.

## Decision Log

| # | Decision | Rationale | Alternatives Rejected |
|---|----------|-----------|----------------------|
| D1 | Pipeline of agents, not single fat agent | Context pressure at 50 calls; Ralph principle; partial re-run | Single agent (454 lines, context degradation) |
| D2 | Sub-agent mode, not agent team | No inter-worker communication needed; orchestrator manages cross-pollination via state | Team mode (overhead, workers don't need to talk) |
| D3 | Sequential explore, parallel bridge | Explore needs cross-pollination (sp_prev informs sp_next); bridge pairs are independent | All parallel (loses cross-pollination); all sequential (bridge is slow) |
| D4 | Data-driven cutoffs, not fixed top-N | Score distribution varies by seed (classic paper vs focused paper) | Top-10/top-5 (ignores data distribution) |
| D5 | 3-layer serendipity (deterministic + LLM + semantic) | L1 grep is reliable and cheap; L2 filters false positives; L3 catches what grep misses | LLM-only (unreliable, expensive); grep-only (misses semantic connections) |
| D6 | YAML now, SQLite later, memvid evaluated | YAML sufficient for 80 items; SQLite for cross-queries; memvid for cross-session + branching | SQLite now (over-engineering); memvid now (unvalidated dependency) |
| D7 | md agents now, SDK at complexity threshold | md works for sequential pipeline + 1-level branches; SDK needed when loops nest or critic introduced | SDK now (premature; adds dependency without proven need) |
| D8 | Shared skill with references, not inline | Progressive disclosure; references loaded only when needed; reusable across agent types | All-in-one agent md (hits 500-line limit); separate skills per agent (duplication) |
| D9 | No orchestrator skill (agent IS orchestrator) | Single-agent-type orchestration; SKILL.md is trigger wrapper only | Full orchestrator skill (adds layer without value for single-type pipeline) |
| D10 | Deep read = structured QA triplets (AIM-SciQA pattern) | (Q,A,E) per section enables embedding-based serendipity, multi-hop QA metrics, evidence tracing | Unstructured summary (loses structure); full text in context (overflows) |

## Future Agent Types (Not Yet Implemented)

| Agent | Input | Output | Reuses |
|-------|-------|--------|--------|
| `research-surveyor` | Topic keyword | Field map + reading list | explorer, synthesizer |
| `research-lineage` | Old paper + new paper | Technology timeline | explorer, bridger |
| `research-deep-reader` | Paper ID | Methods/results analysis | synthesizer (PMC) |

These share `quarry-research` skill and worker agents. Only the
orchestrator agent differs (phase sequence and decomposition logic).

## References

- S16 dogfood session: `docs/dogfood/2026-04-08-reverse-central-dogma.md`
- S16 agent test: agent output in `$HOME/.claude/outputs/`
- System layers: `docs/design/13-system-layers.md`
- Harness engineering analysis: conversation 2026-04-09 (memvid, SDK thresholds)
- AIM-SciQA: arXiv:2603.14257 (multi-hop scientific QA, QA triplet extraction)
- IM-SciQA dataset: https://huggingface.co/datasets/MRAGDATASET/Pubmed-MHQA
- memvid: https://github.com/memvid/memvid
- Claude Agent SDK: https://platform.claude.com/docs/en/agent-sdk/overview
