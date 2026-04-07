# Evaluation Framework

Use this framework after every dogfood session, regardless of persona.

## Five-Dimension Rating (1-5 each)

| Dimension | 1 (bad) | 5 (good) |
|-----------|---------|----------|
| **Discoverability** | Couldn't find relevant papers | Found what I needed quickly |
| **Output clarity** | Couldn't understand what the output means | Self-explanatory, could act on it immediately |
| **Workflow continuity** | Had to guess what to do next | Next step was obvious from the output |
| **Information sufficiency** | Needed external lookup to make decisions | All needed info present in output |
| **Command ergonomics** | Too many steps, copy-paste, flags to remember | Smooth, minimal friction |
| **Performance** | Commands too slow, broke my flow | All commands felt instant |

## Friction Log

For each friction point, record:

```
FRICTION: [what you were trying to do]
COMMAND: [what you ran]
OUTPUT: [what you got (brief)]
PROBLEM: [what was wrong or confusing]
WANTED: [what would have helped]
```

## Command Sequence Log

Record every command you ran, in order. This is the most valuable
data — patterns in the sequence reveal UX issues that friction
points alone miss.

```
CMD 1: quarry search "CRISPR delivery"           → 20 results      0.3s
CMD 2: quarry expand W3119813698                  → 200 results     1.2s
CMD 3: quarry bridge W3119813698 W4404577932      → 6 types         2.1s
...
TOTAL: 3 commands, 3.6s wall time
```

Measure wall time for every command (use `time` or note start/end).
This identifies performance bottlenecks — a command taking >2s breaks
flow, >5s is a context switch, >10s is unacceptable for interactive use.

After logging, annotate the sequence:

- **Unnecessary commands**: Steps you took only because a previous
  command's output lacked information (e.g., sql lookup for work_id)
- **Repeated patterns**: Same command pattern used 3+ times (should
  it be automated or batched?)
- **Backtracking**: Commands you re-ran with different params because
  the first attempt was wrong (why was it wrong?)
- **Transitions**: Where did you switch from one command to another?
  Was the transition obvious or did you have to think?
- **Copy-paste points**: Mark each place you manually transferred
  a value (work_id, DOI, etc.) between commands
- **Slow commands**: Any command >2s? Did it break your flow?
  What were you doing while waiting?
- **Performance bottleneck**: Which command in the sequence is the
  bottleneck? Would the overall workflow feel faster if that one
  command was 10x faster?

## Additional Observations

- **Surprises**: Anything unexpectedly good or bad
- **Dead ends**: Commands that returned nothing useful — why?
- **Missing commands**: Actions you wanted to take but couldn't
- **Workflow suggestions**: Based on your command sequence, what
  shortcuts, pipes, or combined commands would have helped?

## Persistence Format

Save every dogfood session to `docs/dogfood/YYYY-MM-DD-{slug}.md`.
The file must be self-contained — anyone reading it should understand
what was tested, what broke, and what was learned, without needing
the conversation that produced it.

### YAML Frontmatter

```yaml
---
session_id: dogfood-YYYY-MM-DD-{slug}
date: YYYY-MM-DD
persona: newbie | reader | bridger | surveyor
topic: "free-text description of what was explored"
seeds: [W1234567890, W9876543210]  # work_ids used as seeds
commands_used: [sql, info, mesh, expand, bridge, shrink]
commands_count: 10        # total commands executed
friction_count: 2         # number of FRICTION entries
copy_paste_count: 6       # work_id transfers between commands
scores:
  discoverability: 4
  output_clarity: 5
  workflow_continuity: 3
  information_sufficiency: 3
  command_ergonomics: 3
  total: 18
previous_session: dogfood-YYYY-MM-DD-{prev-slug}  # or null
score_delta: +2           # vs previous session, or null
improvements_tested:      # features first used in this session
  - "quarry mesh command"
  - "expand --mesh-summary"
design_refs:              # AD docs this session informed
  - "AD-9: docs/design/10-architecture-decisions.md#ad-9"
---
```

### Body Structure

The body follows the same structure as the evaluation output:

```markdown
# Dogfood: {topic}

## Session Context
Brief description of research goal and constraints (e.g.,
"FTS/vector search unavailable").

## Command Sequence Log
Full table with all commands, results, and timings.

## Five-Dimension Rating
Table with scores and justification.

## Friction Log
All FRICTION entries.

## Additional Observations
Surprises, dead ends, missing commands, workflow suggestions.

## Research Findings (if applicable)
Key papers found, reading lists, niche areas identified.
This preserves the scientific value of the session.
```

### README.md Index

Maintain `docs/dogfood/README.md` with a running table:

```markdown
# Dogfood Sessions

| Date | Persona | Topic | Score | Δ | Key Findings |
|------|---------|-------|-------|---|--------------|
| 2026-04-07 | surveyor | [cytidine deaminase](2026-04-07-cytidine-deaminase.md) | 15/25 | — | AD-8: info needed |
| 2026-04-07 | newbie | [retron](2026-04-07-retron.md) | 16/25 | +1 | AD-9: mesh needed |
```

When adding a new row, compute Δ as the difference from the
previous session's total score.
