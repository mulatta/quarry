"""STORMAdapter: quarry-backed retriever for the STORM knowledge curation pipeline.

STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective
Question Asking) generates comprehensive background articles by simulating
multi-perspective question-answering. This adapter plugs quarry's hybrid
search into STORM's retriever interface.

Installation requirement (optional dependency):
    pip install knowledge-storm

Usage:
    from quarry.agent.storm_adapter import STORMAdapter

    adapter = STORMAdapter(topic="RNA self-replication", sub_problem_id="sp_1")
    article = adapter.run()   # returns markdown background article
    print(article)

The adapter is designed for well-studied sub-problems where quarry's
semantic + citation graph context can enrich STORM's question generation.
For sparse sub-problems (< 5 seeds), prefer DeepReader instead.

Fallback behavior:
    If `knowledge-storm` is not installed, STORMAdapter.run() raises
    ImportError with installation instructions. The rest of quarry is
    not affected.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# QuarryRetriever — STORM retriever interface backed by quarry search
# ---------------------------------------------------------------------------


class QuarryRetriever:
    """STORM-compatible retriever that delegates to quarry hybrid search.

    STORM's retriever interface requires:
        retrieve(query: str, k: int) -> list[dict]

    Where each dict must have keys:
        - "url": unique identifier string
        - "title": document title
        - "description": abstract or snippet text

    This implementation calls `quarry search` (BM25 + embedding hybrid) and
    enriches results with abstracts via `quarry info`.
    """

    def __init__(self, *, limit: int = 20, timeout: int = 60) -> None:
        self.limit = limit
        self.timeout = timeout

    def retrieve(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve papers for a query, return STORM-compatible dicts."""
        n = k or self.limit
        results = self._quarry_search(query, n)
        enriched = self._enrich_with_abstracts(results)
        return enriched

    def _quarry_search(self, query: str, n: int) -> list[dict[str, Any]]:
        """Call quarry search and return raw results."""
        try:
            proc = subprocess.run(
                ["quarry", "search", query, "-n", str(n), "-f", "json"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if proc.returncode != 0:
                return []
            return json.loads(proc.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            return []

    def _enrich_with_abstracts(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Fetch abstracts for top results via quarry info."""
        if not results:
            return []
        work_ids = [r["work_id"] for r in results if "work_id" in r]
        if not work_ids:
            return self._format_without_abstracts(results)

        # Batch info lookup (max 20 at once to avoid CLI arg limits)
        abstracts: dict[str, str] = {}
        for batch_start in range(0, len(work_ids), 20):
            batch = work_ids[batch_start : batch_start + 20]
            try:
                proc = subprocess.run(
                    ["quarry", "info", *batch, "-f", "json", "--full"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if proc.returncode == 0:
                    for paper in json.loads(proc.stdout):
                        wid = paper.get("work_id", "")
                        abstracts[wid] = paper.get("abstract", "") or ""
            except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
                pass

        output = []
        for r in results:
            wid = r.get("work_id", "")
            output.append(
                {
                    "url": f"https://openalex.org/{wid}" if wid else wid,
                    "title": r.get("title", ""),
                    "description": abstracts.get(wid, r.get("abstract", "")),
                }
            )
        return output

    @staticmethod
    def _format_without_abstracts(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "url": f"https://openalex.org/{r.get('work_id', '')}",
                "title": r.get("title", ""),
                "description": r.get("abstract", ""),
            }
            for r in results
        ]


# ---------------------------------------------------------------------------
# STORMAdapter — high-level interface
# ---------------------------------------------------------------------------


class STORMAdapter:
    """Run STORM knowledge curation with quarry as the retriever.

    Args:
        topic: The topic string for STORM to research. Should match the
               sub-problem's `function` field from sub_problems.yaml.
        sub_problem_id: Optional SP id for output naming.
        output_dir: Where STORM writes intermediate and final files.
                    Defaults to a system temp directory.
        max_perspectives: Number of viewpoints STORM considers (default 3).
        retriever_limit: Papers fetched per STORM query (default 20).
        max_conv_turn: Max conversation turns per perspective (default 3).
        max_thread_num: Max parallel threads for STORM pipeline (default 3).
        lm: Language model instance for STORM. Must be provided — STORM
            requires an LM to generate outlines and articles.
    """

    def __init__(
        self,
        topic: str,
        *,
        sub_problem_id: str = "sp",
        output_dir: Path | None = None,
        max_perspectives: int = 3,
        retriever_limit: int = 20,
        max_conv_turn: int = 3,
        max_thread_num: int = 3,
        lm: Any = None,
    ) -> None:
        self.topic = topic
        self.sub_problem_id = sub_problem_id
        self.output_dir = output_dir or Path(tempfile.mkdtemp(prefix="storm_"))
        self.max_perspectives = max_perspectives
        self.retriever_limit = retriever_limit
        self.max_conv_turn = max_conv_turn
        self.max_thread_num = max_thread_num
        self.lm = lm

    def run(self) -> str:
        """Run STORM pipeline and return the generated article as markdown.

        Raises:
            ImportError: If `knowledge-storm` is not installed.
            RuntimeError: If STORM pipeline fails.
        """
        try:
            from knowledge_storm import STORMWikiRunner, STORMWikiRunnerArguments  # type: ignore[import]
            from knowledge_storm.retriever import Retriever  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "knowledge-storm is not installed. "
                "Install it with: pip install knowledge-storm\n"
                "STORMAdapter requires this optional dependency."
            ) from exc

        if self.lm is None:
            raise ValueError(
                "STORMAdapter requires a language model (lm=...). "
                "Pass a STORM-compatible LM instance when constructing "
                "STORMAdapter, e.g.: STORMAdapter(topic=..., lm=my_lm)"
            )

        retriever = self._build_retriever(Retriever)
        runner_args = STORMWikiRunnerArguments(
            output_dir=str(self.output_dir),
            max_conv_turn=self.max_conv_turn,
            max_perspective=self.max_perspectives,
            search_top_k=self.retriever_limit,
            max_thread_num=self.max_thread_num,
        )

        try:
            runner = STORMWikiRunner(runner_args, lm=self.lm, rm=retriever)
            runner.run(
                topic=self.topic,
                do_research=True,
                do_generate_outline=True,
                do_generate_article=True,
                do_polish_article=True,
            )
        except Exception as exc:
            raise RuntimeError(f"STORM pipeline failed: {exc}") from exc

        return self._read_output()

    def _build_retriever(self, base_class: type) -> Any:
        """Build a STORM-compatible retriever by subclassing the base."""
        quarry_retriever = QuarryRetriever(limit=self.retriever_limit)

        class _QuarrySTORMRetriever(base_class):  # type: ignore[misc]
            def retrieve(
                self, query: str, exclude_urls: list[str] | None = None
            ) -> list[dict]:
                results = quarry_retriever.retrieve(query)
                if exclude_urls:
                    excluded = set(exclude_urls)
                    results = [r for r in results if r.get("url") not in excluded]
                return results

        return _QuarrySTORMRetriever()

    def _read_output(self) -> str:
        """Read the polished article from STORM output directory."""
        slug = self.topic.lower().replace(" ", "_")[:60]
        candidates = [
            self.output_dir / slug / "storm_gen_article_polished.txt",
            self.output_dir / slug / "storm_gen_article.txt",
            self.output_dir / "storm_gen_article_polished.txt",
            self.output_dir / "storm_gen_article.txt",
        ]
        for path in candidates:
            if path.exists():
                return path.read_text()
        # Try any .txt in output_dir
        txts = list(self.output_dir.rglob("*.txt"))
        if txts:
            return max(txts, key=lambda p: p.stat().st_size).read_text()
        raise RuntimeError(
            f"STORM finished but no output article found in {self.output_dir}"
        )
