"""YAML-based state management for research agents.

Implements the Ralph pattern: state on disk, fresh context per agent.
Each sub-agent reads inputs from state files and writes outputs there.
The orchestrator reads between spawns for cross-pollination decisions.

Directory layout follows state-schema.md in quarry-research skill.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from quarry.agent.schemas import (
    BridgeConnection,
    CriticReport,
    Finding,
    Gap,
    HopChain,
    NormalizedQuery,
    Seed,
    SerendipityFlag,
    SerendipityValidated,
    SubProblemsDAG,
    TournamentResult,
)


def _slug(text: str) -> str:
    """Convert free-form text to a filesystem-safe slug."""
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    return s[:60].rstrip("-")


class StateManager:
    """YAML state read/write for the research-scout pipeline.

    All state lives under a single base directory, structured as:
      base_dir/
        normalized_query.yaml
        sub_problems.yaml
        sp_{id}/seeds.yaml, findings.yaml, ...
        bridge_{i}_{j}/connections.yaml, ...
        serendipity_validated.yaml
        gaps.yaml
        report.md
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def init_session(cls, slug_text: str) -> StateManager:
        """Create a new session state directory and return a StateManager."""
        outputs = Path.home() / ".claude" / "outputs"
        session_dir = outputs / f"research-scout-{_slug(slug_text)}" / "state"
        session_dir.mkdir(parents=True, exist_ok=True)
        return cls(session_dir)

    # ------------------------------------------------------------------
    # Normalized query
    # ------------------------------------------------------------------

    def write_normalized_query(self, query: NormalizedQuery) -> None:
        self._write_yaml(
            self.base_dir / "normalized_query.yaml", query.model_dump(mode="json")
        )

    def read_normalized_query(self) -> NormalizedQuery:
        data = self._read_yaml(self.base_dir / "normalized_query.yaml")
        return NormalizedQuery.model_validate(data)

    # ------------------------------------------------------------------
    # Sub-problems DAG
    # ------------------------------------------------------------------

    def write_sub_problems(self, dag: SubProblemsDAG) -> None:
        self._write_yaml(
            self.base_dir / "sub_problems.yaml",
            {"sub_problems": [sp.model_dump(mode="json") for sp in dag.sub_problems]},
        )

    def read_sub_problems(self) -> SubProblemsDAG:
        data = self._read_yaml(self.base_dir / "sub_problems.yaml")
        return SubProblemsDAG.model_validate(data)

    def get_topological_order(self) -> list[list[str]]:
        """Return topological levels from the persisted DAG."""
        return self.read_sub_problems().topological_levels()

    # ------------------------------------------------------------------
    # Per-SP state
    # ------------------------------------------------------------------

    def _sp_dir(self, sp_id: str) -> Path:
        d = self.base_dir / f"sp_{sp_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_seeds(self, sp_id: str, seeds: list[Seed]) -> None:
        self._write_yaml(
            self._sp_dir(sp_id) / "seeds.yaml",
            {"seeds": [s.model_dump(mode="json") for s in seeds]},
        )

    def read_seeds(self, sp_id: str) -> list[Seed]:
        data = self._read_yaml(self._sp_dir(sp_id) / "seeds.yaml")
        return [Seed.model_validate(s) for s in data.get("seeds", [])]

    def write_findings(self, sp_id: str, findings: list[Finding]) -> None:
        self._write_yaml(
            self._sp_dir(sp_id) / "findings.yaml",
            {"findings": [f.model_dump(mode="json") for f in findings]},
        )

    def read_findings(self, sp_id: str) -> list[Finding]:
        data = self._read_yaml(self._sp_dir(sp_id) / "findings.yaml")
        return [Finding.model_validate(f) for f in data.get("findings", [])]

    def write_expand_raw(self, sp_id: str, content: str) -> None:
        (self._sp_dir(sp_id) / "expand_raw.txt").write_text(content)

    def read_expand_raw(self, sp_id: str) -> str:
        return (self._sp_dir(sp_id) / "expand_raw.txt").read_text()

    def write_summary(self, sp_id: str, text: str) -> None:
        (self._sp_dir(sp_id) / "summary.txt").write_text(text)

    def read_summary(self, sp_id: str) -> str:
        return (self._sp_dir(sp_id) / "summary.txt").read_text()

    # ------------------------------------------------------------------
    # Serendipity (per-SP L1/L3 flags)
    # ------------------------------------------------------------------

    def read_serendipity_flags(self, sp_id: str) -> list[SerendipityFlag]:
        path = self._sp_dir(sp_id) / "serendipity_flags.yaml"
        if not path.exists():
            return []
        data = self._read_yaml(path)
        return [SerendipityFlag.model_validate(f) for f in data.get("flags", [])]

    # ------------------------------------------------------------------
    # Serendipity validated (session-level)
    # ------------------------------------------------------------------

    def write_serendipity_validated(
        self,
        entries: list[SerendipityValidated],
        sp_distribution: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {}
        if sp_distribution:
            payload["sp_distribution"] = sp_distribution
        payload["validated"] = [
            e.model_dump(mode="json") for e in entries if e.verdict == "validated"
        ]
        payload["rejected"] = [
            e.model_dump(mode="json") for e in entries if e.verdict != "validated"
        ]
        self._write_yaml(self.base_dir / "serendipity_validated.yaml", payload)

    def read_serendipity_validated(self) -> list[SerendipityValidated]:
        path = self.base_dir / "serendipity_validated.yaml"
        if not path.exists():
            return []
        data = self._read_yaml(path)
        entries = []
        for key in ("validated", "rejected"):
            for item in data.get(key, []):
                entries.append(SerendipityValidated.model_validate(item))
        return entries

    # ------------------------------------------------------------------
    # Bridge state
    # ------------------------------------------------------------------

    def _bridge_dir(self, sp_i: str, sp_j: str) -> Path:
        d = self.base_dir / f"bridge_{sp_i}_{sp_j}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_connections(
        self, sp_i: str, sp_j: str, conns: list[BridgeConnection]
    ) -> None:
        self._write_yaml(
            self._bridge_dir(sp_i, sp_j) / "connections.yaml",
            {"connections": [c.model_dump(mode="json") for c in conns]},
        )

    def read_connections(self, sp_i: str, sp_j: str) -> list[BridgeConnection]:
        data = self._read_yaml(self._bridge_dir(sp_i, sp_j) / "connections.yaml")
        return [BridgeConnection.model_validate(c) for c in data.get("connections", [])]

    def write_bridge_raw(self, sp_i: str, sp_j: str, content: str) -> None:
        (self._bridge_dir(sp_i, sp_j) / "bridge_raw.txt").write_text(content)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def write_gaps(self, gaps: list[Gap]) -> None:
        self._write_yaml(
            self.base_dir / "gaps.yaml",
            {
                "has_gaps": len(gaps) > 0,
                "gaps": [g.model_dump(mode="json") for g in gaps],
            },
        )

    def read_gaps(self) -> list[Gap]:
        path = self.base_dir / "gaps.yaml"
        if not path.exists():
            return []
        data = self._read_yaml(path)
        return [Gap.model_validate(g) for g in data.get("gaps", [])]

    def write_report(self, content: str) -> None:
        (self.base_dir / "report.md").write_text(content)

    def read_report(self) -> str:
        return (self.base_dir / "report.md").read_text()

    # ------------------------------------------------------------------
    # Multi-hop chain (per-SP)
    # ------------------------------------------------------------------

    def write_hop_chain(self, sp_id: str, chain: HopChain) -> None:
        self._write_yaml(
            self._sp_dir(sp_id) / "hop_chain.yaml",
            chain.model_dump(mode="json"),
        )

    def read_hop_chain(self, sp_id: str) -> HopChain | None:
        path = self._sp_dir(sp_id) / "hop_chain.yaml"
        if not path.exists():
            return None
        return HopChain.model_validate(self._read_yaml(path))

    # ------------------------------------------------------------------
    # Critic / Devil's Advocate (session-level)
    # ------------------------------------------------------------------

    def write_critic_report(self, report: CriticReport) -> None:
        self._write_yaml(
            self.base_dir / "critic_report.yaml",
            report.model_dump(mode="json"),
        )

    def read_critic_report(self) -> CriticReport | None:
        path = self.base_dir / "critic_report.yaml"
        if not path.exists():
            return None
        return CriticReport.model_validate(self._read_yaml(path))

    # ------------------------------------------------------------------
    # Co-Scientist tournament (session-level)
    # ------------------------------------------------------------------

    def write_tournament_result(self, result: TournamentResult) -> None:
        self._write_yaml(
            self.base_dir / "tournament_result.yaml",
            result.model_dump(mode="json"),
        )

    def read_tournament_result(self) -> TournamentResult | None:
        path = self.base_dir / "tournament_result.yaml"
        if not path.exists():
            return None
        return TournamentResult.model_validate(self._read_yaml(path))

    # ------------------------------------------------------------------
    # Internal YAML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_yaml(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True))

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        return yaml.safe_load(path.read_text()) or {}
