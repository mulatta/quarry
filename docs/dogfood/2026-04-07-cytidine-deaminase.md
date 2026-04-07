______________________________________________________________________

session_id: dogfood-2026-04-07-cytidine-deaminase
date: 2026-04-07
persona: surveyor
topic: "cytidine deaminase discovery for base editors"
seeds: [W4382247824, W4413279334]
commands_used: [sql, expand, bridge, info]
commands_count: 13
friction_count: 5
copy_paste_count: 6
scores:
discoverability: 4
output_clarity: 4
workflow_continuity: 2
information_sufficiency: 2
command_ergonomics: 3
total: 15
previous_session: null
score_delta: null
improvements_tested: []
design_refs:

- "AD-8: docs/design/10-architecture-decisions.md#ad-8-info-command-and-sql-lifecycle"

______________________________________________________________________

# Dogfood: Cytidine Deaminase Discovery for Base Editors

## Session Context

Surveyor persona exploring the landscape of cytidine deaminase discovery
as it relates to CRISPR base editors. Goal: map the approaches used to
find novel cytidine deaminases (beyond APOBEC1) and understand how the
field has branched.

Constraints: `search` and `vsearch` commands were unavailable because the
FTS index was still building. All paper discovery had to go through raw
`sql` queries against the works table.

## Command Sequence Log

| # | Command | Result | Time | Notes |
|---|---------|--------|------|-------|
| 1 | `quarry sql "SELECT id, title FROM works WHERE title ILIKE '%cytidine deaminase%' AND publication_year >= 2020 LIMIT 20"` | Error: UndefinedColumn `publication_year` | 0.1s | **FRICTION F2**: schema unknown |
| 2 | `quarry sql "SELECT column_name FROM information_schema.columns WHERE table_name = 'works'"` | Column list returned | 0.4s | Unnecessary command; needed schema discovery |
| 3 | `quarry sql "SELECT id, title FROM works WHERE title ILIKE '%cytidine deaminase%' AND year >= 2020 LIMIT 20"` | 12 results | 3.2s | Found seed candidates |
| 4 | `quarry sql "SELECT id, title, cited_by_count FROM works WHERE abstract ILIKE '%cytidine deaminase%base edit%' LIMIT 20"` | Timeout after 30s | 30s | **FRICTION F3**: abstract ILIKE is too slow |
| 5 | `quarry sql "SELECT id, title, cited_by_count FROM works WHERE title ILIKE '%cytidine deaminase%' ORDER BY cited_by_count DESC LIMIT 10"` | 10 results; top hit W4382247824 (Cell 2023, ~280 cites) | 2.8s | Identified seed 1: structure-based discovery paper |
| 6 | `quarry expand W4382247824` | 200 related works (references + cited_by) | 1.4s | Good coverage; **FRICTION F4**: no DOI or venue in output |
| 7 | `quarry sql "SELECT id, title FROM works WHERE title ILIKE '%AlphaCD%' OR title ILIKE '%machine learning%deaminase%' LIMIT 10"` | 4 results; found W4413279334 | 2.1s | Identified seed 2: ML characterization paper |
| 8 | `quarry expand W4413279334` | 85 related works | 1.1s | Smaller network; newer paper |
| 9 | `quarry bridge W4382247824 W4413279334` | 3 bridge types found, 12 bridge papers | 2.8s | **FRICTION F4**: bridge output also lacks DOI/venue |
| 10 | `quarry sql "SELECT id, title, doi FROM works WHERE id IN ('W4382247824', 'W4413279334')"` | DOI + title confirmed | 0.3s | **CP**: manual DOI lookup because expand/bridge didn't show it |
| 11 | `quarry sql "SELECT id, title FROM works WHERE title ILIKE '%directed evolution%deaminase%' LIMIT 10"` | 6 results | 2.5s | Found directed evolution axis |
| 12 | `quarry sql "SELECT id, title FROM works WHERE title ILIKE '%homolog%deaminase%' LIMIT 10"` | 3 results | 1.9s | Found homolog search axis |
| 13 | `quarry info W4382247824` | Error: unknown command `info` | 0.1s | **Missing command**: this is what prompted AD-8 |

**Total**: 13 commands, ~49s wall time

### Sequence Annotations

- **Unnecessary commands**: CMD 2 (schema discovery) would be eliminated by `sql --schema` or built-in column docs. CMD 10 (DOI lookup) would be eliminated if expand/bridge included DOI.
- **Repeated patterns**: `sql ... ILIKE '%X%deaminase%'` pattern used 5 times (CMDs 1, 3, 5, 11, 12). A `search` command would collapse these.
- **Copy-paste points (6)**: W4382247824 copied 4 times (CMDs 6, 9, 10, 13), W4413279334 copied 2 times (CMDs 8, 9). Every downstream command required pasting a work_id from a previous sql result.
- **Slow commands**: CMD 4 (abstract ILIKE, 30s timeout) broke flow completely. CMD 5 (2.8s) and CMD 11 (2.5s) were borderline.
- **Performance bottleneck**: Abstract-level ILIKE (CMD 4). Title-level ILIKE is tolerable (~2-3s) but abstract search needs an index.

## Five-Dimension Rating

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Discoverability | 4 | Found the key papers through title ILIKE, despite no FTS. Seeds were high-quality. |
| Output clarity | 4 | sql output was clean tabular. expand/bridge results were readable but lacked metadata. |
| Workflow continuity | 2 | No obvious next step after expand. Had to invent the bridge query manually. No `info` command existed to inspect a single work. |
| Information sufficiency | 2 | DOI and venue missing from expand/bridge forced extra sql lookups. Abstract not shown anywhere. |
| Command ergonomics | 3 | sql is powerful but required schema knowledge. Copy-paste of work_ids 6 times was tedious. |
| **Total** | **15** | |

## Friction Log

### F1: Hybrid search crashes without FTS index

```
FRICTION: Wanted to search papers by topic
COMMAND: quarry search "cytidine deaminase base editor"
OUTPUT:  Error: FTS index not found. Run `quarry index` first.
PROBLEM: No fallback when FTS is unavailable; error message doesn't
         suggest alternatives (e.g., sql ILIKE)
WANTED:  Graceful degradation: auto-fallback to ILIKE search, or at
         minimum suggest `sql` as alternative in error message
```

### F2: Table schema unknown (publication_year -> UndefinedColumn)

```
FRICTION: Wanted to filter by publication year
COMMAND: quarry sql "... WHERE publication_year >= 2020"
OUTPUT:  ERROR: column "publication_year" does not exist
PROBLEM: No way to discover schema without running information_schema
         query. Column is actually named `year`, not `publication_year`.
WANTED:  `quarry sql --schema works` or built-in column documentation.
         Tab-completion for column names would be ideal.
```

### F3: Abstract ILIKE timeout

```
FRICTION: Wanted to search within abstracts
COMMAND: quarry sql "... WHERE abstract ILIKE '%cytidine deaminase%base edit%'"
OUTPUT:  Timeout after 30s
PROBLEM: Abstract column has no index; ILIKE on full text is O(n) scan.
         This is the strongest argument for FTS.
WANTED:  FTS index covering abstracts, or at minimum a pg_trgm index.
```

### F4: expand/bridge output lacks DOI and venue

```
FRICTION: Needed DOI to look up papers externally
COMMAND: quarry expand W4382247824
OUTPUT:  Table with work_id, title, year, cited_by_count — but no DOI, no venue
PROBLEM: Had to run separate sql query (CMD 10) just to get DOI.
         Venue (journal name) is critical for assessing paper quality.
WANTED:  Optional columns: --columns doi,venue or always include DOI.
         At minimum, an `info` command for single-work detail view.
```

### F5: work_id manual copy-paste (6 times)

```
FRICTION: Every command requires pasting a 12-character work_id
COMMAND: quarry bridge W4382247824 W4413279334
OUTPUT:  (works fine, but getting the IDs there was the problem)
PROBLEM: Copied W4382247824 four times, W4413279334 twice across
         the session. No aliasing, no result referencing, no clipboard
         integration.
WANTED:  Result numbering (expand shows [1] [2] [3]..., then
         `bridge @1 @3`) or named bookmarks (`quarry bookmark seed1 W4382247824`).
```

## Additional Observations

### Surprises (positive)

- `expand` was fast (~1.2s for 200 results). The citation graph
  traversal is well-optimized.
- `bridge` found non-obvious connections between the structural
  biology and ML papers. The bridge concept works well for surveyor
  workflow.

### Dead ends

- Abstract ILIKE (CMD 4) was a complete dead end. 30s wait for nothing.
- `info` command (CMD 13) doesn't exist yet. This was the gap that
  directly motivated AD-8.

### Missing commands

- `info <work_id>`: single-work detail view (DOI, abstract, venue,
  MeSH terms, authors). This became AD-8.
- `sql --schema [table]`: schema introspection without raw
  information_schema queries.

### Workflow suggestions

- **search -> expand -> bridge** should be the natural pipeline, but
  search was unavailable and bridge required manual seed discovery.
- Result sets should carry work_ids forward so the next command can
  reference them by index rather than by pasting.

## Research Findings

### Four Approaches to Cytidine Deaminase Discovery

The session identified four distinct approaches in the literature:

| Approach | Key Paper (seed/found) | Method |
|----------|----------------------|--------|
| Structure-based discovery | W4382247824 (Cell 2023) | Protein structure databases (AlphaFold DB) to find novel deaminase folds |
| ML characterization | W4413279334 | AlphaCD and similar ML models to predict deaminase activity from sequence |
| Homolog search | Found via CMD 12 | BLAST/PSI-BLAST from known APOBEC family to find distant homologs |
| Directed evolution | Found via CMD 11 | Laboratory evolution of existing deaminases for altered specificity |

### Bridge Analysis: Structure vs ML

The bridge between W4382247824 and W4413279334 revealed that
structure-based and ML approaches converge on **protein language
models** — both use embedding representations of deaminase sequences,
but diverge on whether the downstream task is fold classification
(structural) or activity prediction (ML).

### Design Decision Triggered

**AD-8** (info command and sql lifecycle): This session demonstrated
that sql is a powerful escape hatch but should not be required for
basic metadata lookup. The `info` command was designed to provide
single-work detail view including DOI, abstract, venue, authors, and
MeSH terms.
