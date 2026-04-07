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
