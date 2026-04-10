"""ReportGenerator: Pydantic schemas → feasibility report markdown.

Follows the report-template.md format from quarry-research skill.
Uses f-strings (no Jinja2 dependency).
"""

from __future__ import annotations

from quarry.agent.schemas import (
    BridgeConnection,
    Finding,
    Gap,
    GenerateReportInput,
    SerendipityValidated,
    SubProblem,
)

# ★ rating scale (from report-template.md)
FEASIBILITY_SCALE = {
    "★★★★": "Exists in nature; engineering challenge only",
    "★★★☆": "Partial precedent; significant engineering needed",
    "★★☆☆": "Conceptual precedent; fundamental research needed",
    "★☆☆☆": "No known precedent; speculative",
}


class ReportGenerator:
    """Generate a feasibility report from aggregated agent findings."""

    def generate(self, inp: GenerateReportInput) -> str:
        """Produce full markdown report from GenerateReportInput."""
        sections = [
            self._header(inp),
            self._decomposition_table(inp.sub_problems.sub_problems),
            self._sp_analysis(inp.sub_problems.sub_problems, inp.findings),
            self._connections(inp.connections, inp.serendipity),
            self._architecture_placeholder(),
            self._challenges(inp.gaps),
            self._reading_list(inp),
        ]
        return "\n\n".join(sections) + "\n"

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self, inp: GenerateReportInput) -> str:
        # Compute overall feasibility from sub-problem ratings
        ratings = [
            sp.feasibility for sp in inp.sub_problems.sub_problems if sp.feasibility
        ]
        overall = self._aggregate_feasibility(ratings) if ratings else "★★☆☆"
        return f"# Feasibility Report: {inp.idea}\n\n**Overall feasibility**: {overall}"

    def _decomposition_table(self, sps: list[SubProblem]) -> str:
        lines = [
            "## 1. Problem Decomposition\n",
            "| # | Sub-problem | Function | Natural Precedent | Feasibility |",
            "|---|-------------|----------|-------------------|-------------|",
        ]
        for sp in sps:
            feas = sp.feasibility or "—"
            lines.append(
                f"| {sp.id} | {sp.name} | {sp.function} | {sp.analogue} | {feas} |"
            )
        return "\n".join(lines)

    def _sp_analysis(
        self, sps: list[SubProblem], findings: dict[str, list[Finding]]
    ) -> str:
        lines = ["## 2. Sub-problem Analysis"]
        for sp in sps:
            feas = sp.feasibility or "—"
            lines.append(f"\n### 2.{sp.id} {sp.name}")
            lines.append(f"- **Closest precedent**: {sp.analogue}")

            sp_findings = findings.get(sp.id, [])
            if sp_findings:
                insight = sp_findings[0]
                lines.append(f"- **Key insight**: {insight.key}: {insight.value}")
            else:
                lines.append("- **Key insight**: (no findings)")

            # Gap: infer from feasibility or findings
            gap_text = self._infer_gap(sp, sp_findings)
            lines.append(f"- **Gap**: {gap_text}")
            lines.append(f"- **Feasibility**: {feas}")
        return "\n".join(lines)

    def _connections(
        self,
        conns: list[BridgeConnection],
        serendipity: list[SerendipityValidated],
    ) -> str:
        lines = [
            "## 3. Cross-domain Connections\n",
            "| Pair | sp | Key Bridge Paper | Implication |",
            "|------|----|-----------------|-------------|",
        ]
        for c in conns:
            pair_str = f"{c.pair[0]}↔{c.pair[1]}"
            sp_str = str(c.sp) if c.sp is not None else "—"
            paper_str = c.key_paper.title[:50] if c.key_paper else "—"
            lines.append(f"| {pair_str} | {sp_str} | {paper_str} | {c.implication} |")

        lines.append("\n### Serendipity Log")
        validated = [s for s in serendipity if s.verdict == "validated"]
        if validated:
            for i, s in enumerate(validated, 1):
                lines.append(
                    f"- S{i}. {s.mechanism} — "
                    f"paper: {s.paper.work_id}, link: {s.from_sp}→{s.to_sp}"
                )
        else:
            lines.append(
                "No unexpected connections detected. "
                "Possible reasons: sub-problems are within the same domain, "
                "or graph data gaps limited cross-domain discovery."
            )
        return "\n".join(lines)

    def _architecture_placeholder(self) -> str:
        return (
            "## 4. Proposed Architecture\n\n"
            "[To be filled by synthesizer based on sub-problem integration analysis]"
        )

    def _challenges(self, gaps: list[Gap]) -> str:
        lines = [
            "## 5. Technical Challenges\n",
            "| Priority | Challenge | Difficulty | Approach |",
            "|----------|-----------|------------|----------|",
        ]
        for i, g in enumerate(gaps, 1):
            lines.append(f"| {i} | {g.gap_description} | {g.severity} | — |")
        if not gaps:
            lines.append("| — | No critical gaps identified | — | — |")
        return "\n".join(lines)

    def _reading_list(self, inp: GenerateReportInput) -> str:
        lines = [
            "## 6. Reading List\n",
            "| # | Paper | Year | Field | Role in Architecture |",
            "|---|-------|------|-------|---------------------|",
        ]
        seen: set[str] = set()
        rank = 0
        # Collect papers from findings (seeds), connections, serendipity
        for sp in inp.sub_problems.sub_problems:
            sp_findings = inp.findings.get(sp.id, [])
            for f in sp_findings:
                if f.source_paper and f.source_paper not in seen:
                    seen.add(f.source_paper)
                    rank += 1
                    lines.append(
                        f"| {rank} | {f.source_paper} | — | SP {sp.id}: {sp.name} | {f.key} |"
                    )
        for c in inp.connections:
            if c.key_paper and c.key_paper.work_id not in seen:
                seen.add(c.key_paper.work_id)
                rank += 1
                year = c.key_paper.year or "—"
                lines.append(
                    f"| {rank} | {c.key_paper.title[:60]} | {year} | Bridge {c.pair[0]}↔{c.pair[1]} | {c.implication[:40]} |"
                )
        for s in inp.serendipity:
            if s.verdict == "validated" and s.paper.work_id not in seen:
                seen.add(s.paper.work_id)
                rank += 1
                year = s.paper.year or "—"
                lines.append(
                    f"| {rank} | {s.paper.title[:60]} | {year} | Serendipity | {s.mechanism[:40]} |"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_feasibility(ratings: list[str]) -> str:
        """Aggregate feasibility: overall = min of individual ratings."""
        star_count = []
        for r in ratings:
            filled = r.count("★")
            star_count.append(filled)
        if not star_count:
            return "★★☆☆"
        min_stars = min(star_count)
        return "★" * min_stars + "☆" * (4 - min_stars)

    @staticmethod
    def _infer_gap(sp: SubProblem, findings: list[Finding]) -> str:
        """Infer gap description from feasibility and findings."""
        if sp.feasibility and sp.feasibility.count("★") >= 4:
            return "Engineering implementation only"
        if not findings:
            return "No literature found — fundamental gap"
        # Look for gap-related findings
        for f in findings:
            if "gap" in f.key.lower() or "missing" in f.value.lower():
                return f.value
        return "Gap between precedent and idea requirement needs assessment"
