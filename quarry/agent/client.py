"""QuarryClient: subprocess wrapper for quarry CLI.

Agents call quarry via Bash, but this client provides a Python-native
alternative with JSON output parsing and structured error handling.
Optional — agents can use Bash directly per their md definitions.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any


class QuarryError(Exception):
    """Raised when a quarry CLI command fails."""

    def __init__(self, args: list[str], returncode: int, stderr: str) -> None:
        self.args_used = args
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"quarry {' '.join(args)} failed (rc={returncode}): {stderr}")


class QuarryClient:
    """Quarry CLI subprocess wrapper with JSON output parsing.

    All methods return parsed JSON dicts. Raises QuarryError on non-zero exit.
    """

    def __init__(self, *, timeout: int = 120) -> None:
        self.timeout = timeout

    def mesh(
        self,
        query: str,
        *,
        tree: bool = False,
        limit: int = 15,
    ) -> str:
        """Search MeSH vocabulary. Returns raw text (mesh has no JSON mode)."""
        args = ["mesh", query, "-n", str(limit)]
        if tree:
            args.append("--tree")
        return self._run(args)

    def info(
        self,
        work_ids: list[str],
        *,
        full: bool = False,
        mesh: bool = False,
    ) -> list[dict[str, Any]]:
        """Lookup paper metadata. Returns list of paper dicts."""
        args = ["info", *work_ids, "-f", "json"]
        if full:
            args.append("--full")
        if mesh:
            args.append("--mesh")
        return json.loads(self._run(args))

    def expand(
        self,
        seed: str,
        *,
        limit: int = 200,
        mode: str = "fused",
        mesh_summary: bool = False,
        min_citations: int = 0,
        alpha: float = 0.15,
        epsilon: float = 1e-6,
    ) -> dict[str, Any]:
        """Expand seed into ranked subgraph. Returns full JSON result."""
        args = [
            "expand",
            seed,
            "-n",
            str(limit),
            "-m",
            mode,
            "-f",
            "json",
            "--alpha",
            str(alpha),
            "--epsilon",
            str(epsilon),
        ]
        if mesh_summary:
            args.append("--mesh-summary")
        if min_citations > 0:
            args.extend(["--min-citations", str(min_citations)])
        return json.loads(self._run(args))

    def bridge(
        self,
        seeds: list[str],
        *,
        types: list[str] | None = None,
        limit: int = 100,
        max_degree: int = 10_000,
        max_path_depth: int = 5,
        alpha: float = 0.15,
        epsilon: float = 1e-6,
    ) -> dict[str, Any]:
        """Discover bridge papers between seeds. Returns full JSON result."""
        args = [
            "bridge",
            *seeds,
            "-n",
            str(limit),
            "--max-degree",
            str(max_degree),
            "--max-path-depth",
            str(max_path_depth),
            "--alpha",
            str(alpha),
            "--epsilon",
            str(epsilon),
            "-f",
            "json",
        ]
        if types:
            for t in types:
                args.extend(["-t", t])
        return json.loads(self._run(args))

    def shrink(
        self,
        seed: str,
        *,
        top_n: int = 5,
        venue: str = "NCS+",
        limit: int = 200,
        no_foundation: bool = False,
        exclude: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find minimum covering set from top venues. Returns full JSON result."""
        args = [
            "shrink",
            seed,
            "-n",
            str(top_n),
            "-v",
            venue,
            "--limit",
            str(limit),
            "-f",
            "json",
        ]
        if no_foundation:
            args.append("--no-foundation")
        if exclude:
            args.extend(["--exclude", ",".join(exclude)])
        return json.loads(self._run(args))

    def sql(self, query: str) -> list[dict[str, Any]]:
        """Execute read-only SQL. Returns list of row dicts."""
        raw = self._run(["sql", query])
        return json.loads(raw)

    def _run(self, args: list[str]) -> str:
        """Execute quarry CLI and return stdout."""
        result = subprocess.run(
            ["quarry", *args],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            raise QuarryError(args, result.returncode, result.stderr.strip())
        return result.stdout
