# Feasibility Report Template

```markdown
# Feasibility Report: [Idea Title]

**Overall feasibility**: ★★☆☆ (1-sentence summary)

## 1. Problem Decomposition

| # | Sub-problem | Function | Natural Precedent | Feasibility |
|---|-------------|----------|-------------------|-------------|

## 2. Sub-problem Analysis

### 2.1 [Sub-problem name]
- **Closest precedent**: [paper] — [one-line summary]
- **Key insight**: [from abstract or full text]
- **Gap**: [what's missing vs. the idea's requirement]
- **Feasibility**: ★★★☆

(repeat per sub-problem)

## 3. Cross-domain Connections

| Pair | sp | Key Bridge Paper | Implication |
|------|----|-----------------|-------------|

### Serendipity Log
- S1. [description] — source: [cmd], paper: [id], link: [sp→sp]
- S2. ...

(if none: "No unexpected connections detected. Possible reasons: ...")

## 4. Proposed Architecture
[description or ASCII diagram of how sub-systems combine]

## 5. Technical Challenges

| Priority | Challenge | Difficulty | Approach |
|----------|-----------|------------|----------|

## 6. Reading List

| # | Paper | Year | Field | Role in Architecture |
|---|-------|------|-------|---------------------|
```

## Feasibility Rating Scale

| Rating | Meaning |
|--------|---------|
| ★★★★ | Exists in nature; engineering challenge only |
| ★★★☆ | Partial precedent; significant engineering needed |
| ★★☆☆ | Conceptual precedent; fundamental research needed |
| ★☆☆☆ | No known precedent; speculative |
