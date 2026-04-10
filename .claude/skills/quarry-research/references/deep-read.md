# Deep Read: PMC Full-Text Workflow

Structured protocol for extracting detailed evidence from full-text papers via PMC.
Used by research-synthesizer agent during Phase 3.

## When to Deep-Read

Deep-read is optional and expensive (uses PubMed MCP tools). Apply selectively:

1. **Top 2-3 papers only** — the most critical seeds or bridge papers
2. **Selection criteria** (pick papers that meet ≥2):
   - Seed with highest `cited_by` for a sub-problem
   - Bridge paper connecting the most sub-problem pairs
   - Paper where abstract alone is insufficient to assess feasibility gap
   - Serendipity-validated paper (mechanism needs detailed understanding)
3. **Skip if**: paper is a review (generic), or abstract already contains sufficient quantitative detail

## Step-by-Step Protocol

### 1. Get PMCID from Paper Metadata

Use `quarry info` to check if `pmc_id` is available:

```
quarry info <work_id> -f json
```

The `pmc_id` field (e.g., `PMC6776048`) is populated from the `id_crosswalk` table. If present, proceed to step 2.

**If pmc_id is null**: Try converting via PubMed MCP:

```
convert_article_ids(ids: "<pmid>", from_type: "pmid", to_type: "pmcid")
```

If neither PMID nor PMCID is available → **fallback to abstract-only** (step 5).

### 2. Retrieve Full Text

Use PubMed MCP tool:

```
get_full_text_article(pmc_id: "PMC6776048")
```

This returns the full XML/text of the article from PMC Open Access.

**If PMC access fails** (article not in OA subset) → **fallback to abstract-only** (step 5).

### 3. Structured Extraction (Q, A, Evidence)

For each section of the paper, extract structured triplets:

#### Methods Section
- **Q**: What experimental system/technique was used?
- **A**: [extracted method description]
- **Evidence**: [key parameters, conditions, organisms]

#### Results Section
- **Q**: What quantitative outcome supports the mechanism?
- **A**: [main finding statement]
- **Evidence**: [specific numbers: rates, efficiencies, error rates, fold-changes]

#### Discussion/Limitations
- **Q**: What are the acknowledged limitations?
- **A**: [limitation summary]
- **Evidence**: [specific constraints mentioned by authors]

### 4. Apply to Feasibility Assessment

Map extracted evidence to sub-problem gaps:

| Extracted Evidence | Relevant SP | Impact on Feasibility |
|---|---|---|
| "polymerase rate: 0.5 nt/min" | SP1 (polymerase) | ★★★☆ → quantified but slow |
| "error rate: 1/1000" | SP3 (fidelity) | Below error catastrophe threshold |

Update the feasibility rating if quantitative evidence changes the assessment.

### 5. Fallback: Abstract-Only Analysis

When full text is unavailable:

1. Use `quarry info <work_id> --full -f json` to get the abstract
2. Extract what's available from the abstract (usually: main claim, key metric, system used)
3. Note in the report: "Assessment based on abstract only — full text unavailable"
4. Consider citing the paper's DOI for manual follow-up

## Output Integration

Deep-read findings are NOT written to separate state files. Instead:

1. Directly incorporate into `report.md` in the relevant sub-problem analysis section
2. Add quantitative evidence to the gap assessment
3. If deep-read reveals a new gap → add to `gaps.yaml`

## Cost Control

- Max 3 deep-reads per session (each uses 1-2 MCP calls)
- Only the synthesizer agent performs deep-reads
- If the orchestrator doesn't explicitly request deep-read in the spawn prompt, skip this protocol entirely
