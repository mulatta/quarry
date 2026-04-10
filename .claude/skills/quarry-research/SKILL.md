---
name: quarry-research
description: >
  Procedural knowledge for research literature exploration using quarry CLI. Contains serendipity detection protocol, context discipline rules, state schema, report template, and evaluation metrics. Loaded by research-scout orchestrator and its sub-agents (explorer, bridger, synthesizer).
---

# quarry-research

Shared references for all research agents using quarry CLI.

## References

Load only the reference needed for the current task:

| Reference | When to load |
| --- | --- |
| `references/quarry-reference.md` | During EXPLORE and BRIDGE — CLI command synopsis, options, JSON schemas |
| `references/serendipity.md` | During EXPLORE, BRIDGE, and Phase 2.5 — serendipity detection + validation |
| `references/context-discipline.md` | During EXPLORE and BRIDGE — deciding what to keep in context |
| `references/state-schema.md` | Session start — state directory structure |
| `references/report-template.md` | During SYNTHESIZE — output format |
| `references/deep-read.md` | During SYNTHESIZE — PMC full-text extraction protocol |
| `references/eval-metrics.md` | After session — quality measurement |
