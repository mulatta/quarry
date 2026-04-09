# Agent Landscape

> Capabilities inventory, agent definitions, shared infrastructure, and
> implementation priorities for quarry-based research agents.

## Capability Inventory

### A. Adoptable from Prior Art

Patterns from FutureHouse (PaperQA2, Aviary, Robin, BixBench), STORM,
and AI-Scientist that quarry can adopt.

| # | Capability | Source | quarry Adoption |
|---|---|---|---|
| A1 | 3-phase literature QA (search → evidence → answer) | PaperQA2 | Add evidence gathering (abstract/full-text read) after explore |
| A2 | Contextual Ranking and Summarization (RCS) | PaperQA2 | LLM reranking on expand top-K (currently graph score only) |
| A3 | Full-text reading + chunk + summarize | PaperQA2 (Docling) | PubMed MCP (PMC full-text) + PDF parser (Docling/Nougat) |
| A4 | Environment-based agent evaluation | Aviary | Wrap seed-recall as Aviary-style env (reset → step → reward) |
| A5 | Benchmark capsule structure | BixBench | Eval data as (hypothesis, data, answer) capsules |
| A6 | Multi-agent end-to-end discovery pipeline | Robin | Literature → experiment design → execution; quarry covers the literature leg |
| A7 | Metadata-aware retrieval | PaperQA2 | Incorporate RCR, citation count, retraction status into seed selection and LLM reranking |

### B. Unique to quarry

Capabilities no existing system provides, enabled by quarry's data and
graph algorithms.

| # | Capability | quarry Asset | Why Others Can't |
|---|---|---|---|
| B1 | Multi-hop citation graph traversal (APPR) | 2.4B-edge CSR, O(1/ε) | PaperQA2: LLM-guided 1-hop (expensive, slow, shallow) |
| B2 | 7-type structural bridge discovery | coupling, co-citation, path, ppr, steiner | No system computes structural relationships; embedding sees similarity only |
| B3 | MeSH ontology navigation + hierarchical topic decomposition | 379M annotations + 65K tree | S2/OpenAlex lack MeSH |
| B4 | Graph-based serendipity detection + scoring | bridge sp + cross-MeSH + 3-dim scoring (D14) | PaperQA2: embedding similarity only |
| B5 | Minimum covering set (shrink) | Graph algorithm | No equivalent: "minimum N papers to understand this field" |
| B6 | Cross-domain author network | 183M authors + 883M work_authors | S2 author API: single-author lookup only |
| B7 | Deterministic reproducibility | CSR mmap + sorted tiebreakers | LLM-dependent traversal varies per run |
| B8 | Fine-grained MeSH heading analysis | 392M headings (with qualifiers) | No system has qualifier-level MeSH |
| B9 | Chemical-literature cross-reference | 65M chemicals table | Requires separate ChEMBL or PubChem |
| B10 | Grant-research mapping | 14M grants table | Requires separate NIH Reporter or NSF |
| B11 | Clinical citation tracking | 32M cited_by_clin rows | No equivalent outside iCite |

### C. Novel (No System Does This Yet)

Capabilities that neither quarry nor any existing system currently
provides, but quarry's data makes feasible.

| # | Capability | Description | Enabler |
|---|---|---|---|
| C1 | Citation-aware serendipity scoring | Quantitative serendipity via bridge sp + MeSH cross-classification | quarry bridge + MeSH (designed: Phase 2.5, D14) |
| C2 | Graph-based systematic review automation | seed → expand → bridge → shrink recovers included studies | eval/seed-recall is this exact workflow |
| C3 | Cross-domain technology lineage tracking | path_bridges traces "technique A → field B → technique C" | Directed BFS + MeSH topic change detection |
| C4 | MeSH-guided problem decomposition | Hypothesis decomposition with MeSH hierarchy-defined SP scope | CLARIFY + DECOMPOSE + mesh tree |
| C5 | Author-bridged collaboration discovery | "Find papers co-authored by SP1 expert and SP2 expert" | work_authors JOIN + bridge |
| C6 | Concept graph real-time visualization | Live rendering of agent reasoning (SP, paper, concept nodes) with user intervention | quarry graph + agent state → concept graph |
| C7 | Chemical-literature bridge | Citation network of papers mentioning a compound | chemicals (65M) + CSR graph |
| C8 | Grant-linked research landscape | "Who funds this research area and what do they produce?" | grants (14M) + expand/mesh |
| C9 | Clinical translation pathway | Basic research → clinical trial connection via citation chain | cited_by_clin (32M) + MeSH + ClinicalTrials MCP |
| C10 | Literature-derived experiment design | Analyze methods sections → generate reproducible/variant protocol | Full-text reading (A3) + LLM protocol generation |

______________________________________________________________________

## Agent Definitions

### Agent Catalog

| Agent | Input | Output | Core Pipeline | Tier |
|---|---|---|---|---|
| **research-scout** | Research hypothesis | Feasibility report | CLARIFY → DECOMPOSE(DAG) → EXPLORE → BRIDGE → VALIDATE → SYNTHESIZE | 1 |
| **research-surveyor** | Topic keyword | Field map + reading list | mesh → expand → shrink + LLM summary | 1 |
| **systematic-reviewer** | Seed papers + review question | Included study candidates | expand → bridge → rank + recall measurement | 1 |
| **research-lineage** | Old paper + new paper | Technology timeline | bridge(path) → directed chain → MeSH topic evolution | 1 |
| **research-reader** | Paper ID(s) | Structured QA triplets + methods summary | Full-text retrieve → section parse → (Q,A,Evidence) | 2 |
| **collaborator-finder** | Research topic | Potential collaborators + connecting papers | mesh → expand → work_authors JOIN → bridge | 2 |
| **concept-bridger** | Two concepts (natural language) | Connection path + key papers + explanation | vsearch → bridge → mesh → LLM interpretation | 2 |
| **trend-detector** | MeSH term + year range | Emerging topic report | mesh → expand(per-year) → citation velocity time series | 2 |
| **experiment-designer** | Hypothesis + literature findings | Experiment protocol draft | research-reader output → LLM protocol generation | 3 |
| **clinical-translator** | Basic research paper | Bench-to-bedside pathway analysis | cited_by_clin → ClinicalTrials MCP → pathway mapping | 3 |
| **grant-mapper** | Research area | Funding landscape report | mesh → expand → grants JOIN → aggregation | 3 |
| **chem-literature-bridge** | Chemical name/ID | Related paper network | chemicals → work JOIN → expand → bridge | 3 |

______________________________________________________________________

## Requirement Matrices

### quarry Core Operations per Agent

| Agent | expand | bridge | mesh | info | search | vsearch | shrink | sql |
|---|---|---|---|---|---|---|---|---|
| research-scout | ✅ | ✅ | ✅ | ✅ | | | | △ |
| research-surveyor | ✅ | | ✅ | ✅ | ✅ | | ✅ | |
| systematic-reviewer | ✅ | ✅ | | ✅ | ✅ | | | |
| research-lineage | | ✅ | ✅ | ✅ | | | | |
| research-reader | | | | ✅ | | | | |
| collaborator-finder | ✅ | ✅ | ✅ | ✅ | | | | ✅ |
| concept-bridger | | ✅ | ✅ | ✅ | | ✅ | | |
| trend-detector | ✅ | | ✅ | | | | | ✅ |
| experiment-designer | | | | ✅ | | | | |
| clinical-translator | ✅ | | ✅ | ✅ | | | | ✅ |
| grant-mapper | ✅ | | ✅ | | | | | ✅ |
| chem-literature-bridge | ✅ | ✅ | | ✅ | | | | ✅ |

### LLM Functions per Agent

| Agent | EvaluateSeed | ScoreSerendipity | GenerateReport | ExtractQA | GenerateProtocol | Rerank |
|---|---|---|---|---|---|---|
| research-scout | ✅ | ✅ | ✅ | | | |
| research-surveyor | | | ✅ | | | ✅ |
| systematic-reviewer | ✅ | | | | | ✅ |
| research-lineage | | | ✅ | | | |
| research-reader | | | | ✅ | | |
| collaborator-finder | | | ✅ | | | |
| concept-bridger | | ✅ | ✅ | | | |
| trend-detector | | | ✅ | | | |
| experiment-designer | | | | ✅ | ✅ | |
| clinical-translator | | | ✅ | | | |
| grant-mapper | | | ✅ | | | |
| chem-literature-bridge | | | ✅ | | | |

### Data Sources per Agent

| Agent | PG works | CSR graph | LanceDB | MeSH | authors | chemicals | grants | cited_by_clin | PubMed MCP | ClinicalTrials MCP |
|---|---|---|---|---|---|---|---|---|---|---|
| research-scout | ✅ | ✅ | | ✅ | | | | | ✅ | |
| research-surveyor | ✅ | ✅ | | ✅ | | | | | | |
| systematic-reviewer | ✅ | ✅ | ✅ | | | | | | | |
| research-lineage | ✅ | ✅ | | ✅ | | | | | | |
| research-reader | ✅ | | | | | | | | ✅ | |
| collaborator-finder | ✅ | ✅ | | ✅ | ✅ | | | | | |
| concept-bridger | ✅ | ✅ | ✅ | ✅ | | | | | | |
| trend-detector | ✅ | ✅ | | ✅ | | | | | | |
| experiment-designer | ✅ | | | | | | | | ✅ | |
| clinical-translator | ✅ | ✅ | | ✅ | | | | ✅ | | ✅ |
| grant-mapper | ✅ | ✅ | | ✅ | | | ✅ | | | |
| chem-literature-bridge | ✅ | ✅ | | | | ✅ | | | | |

______________________________________________________________________

## Shared Infrastructure

### Common Modules (all agents)

| Module | Description | Used by |
|---|---|---|
| **QuarryClient** | Unified interface for expand/bridge/mesh/info/search (local import or HTTP) | 12/12 agents |
| **LLM client** | Provider-agnostic, OpenAI-compatible, model-per-function config | 12/12 agents |
| **State manager** | YAML state read/write, session directory management | 12/12 agents |
| **Report generator** | Pydantic schema → markdown report | 10/12 agents |

### Common LLM Functions (Pydantic schemas)

| Function | Description | Used by |
|---|---|---|
| `EvaluateSeed` | Assess paper suitability as seed (relevance, novelty) | scout, systematic-reviewer |
| `Rerank` | Combine graph score + LLM semantic evaluation | surveyor, systematic-reviewer |
| `GenerateReport` | Produce structured markdown report from findings | 9 agents (all except reader, experiment-designer) |
| `ExtractQA` | Extract (Question, Answer, Evidence) triplets from paper | reader, experiment-designer |
| `ScoreSerendipity` | Rate novelty + specificity + actionability (0/1 each) | scout, concept-bridger |
| `GenerateProtocol` | Draft experiment protocol from literature methods | experiment-designer |

### Common quarry Operation Patterns

| Pattern | Description | Used by |
|---|---|---|
| `seed_discover` | mesh/search → info → seed selection | scout, surveyor, systematic-reviewer |
| `landscape_map` | expand → (optional rerank) → collect top-K | scout, surveyor, systematic-reviewer, collaborator, trend |
| `connection_find` | bridge(pairs) → assess connections | scout, systematic-reviewer, lineage, concept-bridger, chem |
| `author_resolve` | work_authors JOIN → author info | collaborator-finder |
| `mesh_navigate` | mesh tree traversal → define topic scope | scout, surveyor, lineage, trend, clinical, grant |
| `fulltext_read` | PubMed MCP → section parse → QA extraction | reader, experiment-designer, scout (deep-read) |

______________________________________________________________________

## Implementation Priorities

| Priority | Target | Rationale | Depends on |
|---|---|---|---|
| **P0** | QuarryClient + LLM client + State manager | Common foundation for all agents | None |
| **P1** | research-scout (Python orchestrator) | Already designed, resolves Run 3 failure | P0 |
| **P1** | eval/seed-recall run.py + report.py | Establishes graph quality baseline | P0 |
| **P2** | research-surveyor | Simplest pipeline (expand + mesh + shrink) | P0 |
| **P2** | systematic-reviewer | Same workflow as eval data; eval reuse | P0, P1 (eval) |
| **P3** | research-reader | Full-text foundation; reusable as deep-read module | P0, PubMed MCP |
| **P3** | research-lineage | Uses path_bridges; relatively simple | P0 |
| **P4** | collaborator-finder, concept-bridger, trend-detector | Require additional data JOINs | P0, sql patterns |
| **P5** | experiment-designer, clinical-translator, grant-mapper, chem-bridge | External data or special tables | P3, external MCPs |

## References

- AD-13: Server-client architecture (10-architecture-decisions.md)
- AD-14: Research agent pipeline (14-research-agent.md)
- FutureHouse: PaperQA2, Aviary, Robin, BixBench, DISCO
- STORM: DSPy-based multi-perspective article generation
- AI-Scientist: Pure Python pipeline, Aider-based experiment automation
- eval/seed-recall: Systematic review recall benchmark (sysrev-seed-collection)
