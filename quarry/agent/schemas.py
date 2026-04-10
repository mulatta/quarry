"""Pydantic models for agent LLM function I/O.

These schemas define the structured data exchanged between agents via YAML
state files. They also serve as structured-output schemas for LLM calls
(decompose, evaluate-seed, score-serendipity, etc.).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class Paper(BaseModel):
    """Minimal paper representation used across all agent phases."""

    work_id: str
    title: str
    year: int | None = None
    cited_by: int | None = None
    mesh_major: list[str] = []


class NormalizedQuery(BaseModel):
    """Phase -1 output: PICO-like structured query."""

    original: str
    system: str
    mechanism: str
    context: str
    outcome: str
    scope_assumptions: list[str] = []
    clarification_asked: bool = False


class SubProblem(BaseModel):
    """Single node in the sub-problem DAG."""

    id: str
    name: str
    function: str
    analogue: str
    search: list[str]
    depends_on: list[str] = []
    parent_id: str | None = None
    depth: int = 0
    status: Literal["pending", "explored", "refined", "unable_to_assess"] = "pending"
    feasibility: str | None = None


class SubProblemsDAG(BaseModel):
    """Phase 0 output: directed acyclic graph of sub-problems."""

    sub_problems: list[SubProblem]

    @model_validator(mode="after")
    def validate_dag(self) -> SubProblemsDAG:
        ids = {sp.id for sp in self.sub_problems}
        for sp in self.sub_problems:
            for dep in sp.depends_on:
                if dep not in ids:
                    msg = f"SP '{sp.id}' depends on unknown SP '{dep}'"
                    raise ValueError(msg)
        # cycle detection via topological sort (Kahn's algorithm)
        in_degree: dict[str, int] = {sp.id: 0 for sp in self.sub_problems}
        adj: dict[str, list[str]] = {sp.id: [] for sp in self.sub_problems}
        for sp in self.sub_problems:
            for dep in sp.depends_on:
                adj[dep].append(sp.id)
                in_degree[sp.id] += 1
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited = 0
        while queue:
            node = queue.pop(0)
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if visited != len(self.sub_problems):
            msg = "Sub-problem DAG contains a cycle"
            raise ValueError(msg)
        return self

    def topological_levels(self) -> list[list[str]]:
        """Return sub-problem IDs grouped by topological level.

        Level 0 = roots (no dependencies). Level N = all SPs whose
        dependencies are all in levels < N. SPs within the same level
        can be explored in parallel.
        """
        resolved: set[str] = set()
        remaining = {sp.id: set(sp.depends_on) for sp in self.sub_problems}
        levels: list[list[str]] = []
        while remaining:
            level = [nid for nid, deps in remaining.items() if deps <= resolved]
            if not level:
                msg = "Cannot compute levels — cycle or missing dependency"
                raise ValueError(msg)
            levels.append(sorted(level))
            resolved.update(level)
            for nid in level:
                del remaining[nid]
        return levels


# ---------------------------------------------------------------------------
# Explorer output models
# ---------------------------------------------------------------------------


class Seed(BaseModel):
    """A seed paper selected by the explorer."""

    work_id: str
    title: str
    year: int | None = None
    cited_by: int | None = None
    why: str
    mesh_major: list[str] = []


class Finding(BaseModel):
    """A key insight extracted from paper abstracts."""

    key: str
    value: str
    source_paper: str | None = None


# ---------------------------------------------------------------------------
# Serendipity models
# ---------------------------------------------------------------------------


class SerendipityFlag(BaseModel):
    """L1/L3 raw serendipity candidate (before validation)."""

    current_sp: str
    matched_sp: str
    match_type: Literal["title_keyword", "mesh_cross", "semantic_escape"]
    matched_term: str
    source_file: str | None = None
    match_preview: str | None = None


class SerendipityValidated(BaseModel):
    """Phase 2.5 output: orchestrator-validated serendipity entry."""

    paper: Paper
    from_sp: str
    to_sp: str
    scores: dict[str, int]  # {novelty: 0|1, specificity: 0|1, actionability: 0|1}
    total: int
    bridge_sp: int | None = None
    mechanism: str
    verdict: Literal["validated", "related_finding", "rejected"]

    @field_validator("scores")
    @classmethod
    def validate_score_keys(cls, v: dict[str, int]) -> dict[str, int]:
        required = {"novelty", "specificity", "actionability"}
        if not required <= v.keys():
            msg = f"scores must contain keys {required}, got {set(v.keys())}"
            raise ValueError(msg)
        return v


# ---------------------------------------------------------------------------
# Bridge models
# ---------------------------------------------------------------------------


class BridgeConnection(BaseModel):
    """A single bridge connection between two sub-problems."""

    pair: tuple[str, str]
    sp: int | None = None
    overlap_refs: int = 0
    overlap_citers: int = 0
    key_paper: Paper | None = None
    implication: str


# ---------------------------------------------------------------------------
# Synthesis models
# ---------------------------------------------------------------------------


class Gap(BaseModel):
    """An unresolved gap identified during synthesis."""

    sp_id: str
    gap_description: str
    severity: Literal["critical", "major", "minor"]


class GenerateReportInput(BaseModel):
    """Aggregated input for report generation."""

    idea: str
    sub_problems: SubProblemsDAG
    findings: dict[str, list[Finding]]  # sp_id -> findings
    connections: list[BridgeConnection]
    serendipity: list[SerendipityValidated]
    gaps: list[Gap]


# ---------------------------------------------------------------------------
# LLM structured-output schemas
# ---------------------------------------------------------------------------


class EvaluateSeedResult(BaseModel):
    """LLM output: seed paper relevance evaluation."""

    paper: Paper
    relevance: float  # 0.0 - 1.0
    reasoning: str

    @field_validator("relevance")
    @classmethod
    def clamp_relevance(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            msg = f"relevance must be 0.0-1.0, got {v}"
            raise ValueError(msg)
        return v


class ScoreSerendipityResult(BaseModel):
    """LLM output: serendipity scoring for a single candidate."""

    novelty: Literal[0, 1]
    specificity: Literal[0, 1]
    actionability: Literal[0, 1]
    reasoning: str


# ---------------------------------------------------------------------------
# Multi-hop chain models
# ---------------------------------------------------------------------------

HopGrade = Literal["A", "B", "C", "D"]
"""Evidence grade for a single hop claim.

A = direct experimental evidence (in vivo / clinical)
B = strong mechanistic evidence (in vitro, validated pathway)
C = indirect / correlational evidence
D = hypothetical / analogical reasoning only
"""


class HopLink(BaseModel):
    """One logical step in a multi-hop reasoning chain."""

    claim: str
    grade: HopGrade
    refs: list[str] = []  # work_id strings supporting this claim
    flag: str | None = None  # optional warning (e.g. "single study", "model organism")
    validation_experiment: str | None = None  # suggested experiment to validate


class HopChain(BaseModel):
    """Full multi-hop chain from hypothesis to conclusion."""

    hypothesis: str
    hops: list[HopLink]

    @property
    def n_unvalidated(self) -> int:
        """Number of hops graded C or D (weak evidence)."""
        return sum(1 for h in self.hops if h.grade in ("C", "D"))


# ---------------------------------------------------------------------------
# Critic / Devil's Advocate models
# ---------------------------------------------------------------------------

IssueSeverity = Literal["fatal", "high", "medium", "low"]


class CriticIssue(BaseModel):
    """Single issue raised by a Devil's Advocate persona."""

    persona: str  # "Mechanistic Skeptic" | "Prior Art Investigator" | "Safety/Feasibility Auditor"
    issue: str
    severity: IssueSeverity
    mitigation: str | None = None  # suggested fix or experiment


class CriticReport(BaseModel):
    """Aggregated output from all Devil's Advocate personas."""

    hypothesis: str
    issues: list[CriticIssue] = []

    @property
    def fatal_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "fatal")


# ---------------------------------------------------------------------------
# Co-Scientist tournament models
# ---------------------------------------------------------------------------


class TournamentMatch(BaseModel):
    """Single pairwise match between two hypotheses."""

    winner: str  # hypothesis text or id
    loser: str
    reasoning: str
    grade_delta: float = 1.0  # Elo-like score change


class TournamentResult(BaseModel):
    """Full tournament outcome for a set of hypotheses."""

    ranked_hypotheses: list[str]  # ordered best → worst
    matches: list[TournamentMatch] = []
