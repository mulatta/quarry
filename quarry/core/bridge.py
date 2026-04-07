"""Core bridge logic — shared by CLI and MCP server.

Given two or more seed papers, find papers that connect them.
Each bridge type answers a distinct structural question.
See docs/design/11-bridge.md for full design rationale.

Pipeline: resolve seeds → Rust bridge() → PG metadata enrichment → JSON assembly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import quarry_graph

from quarry.core.expand import _enrich_metadata, resolve_seed

_BRIDGE_TYPES = [
    "common_refs",
    "common_citers",
    "coupling",
    "cocitation",
    "path",
    "ppr",
]


def run_bridge(
    *,
    seeds: list[str],
    csr_dir: str,
    pg_conninfo: str,
    types: list[str] | None = None,
    limit: int = 100,
    max_neighbor_degree: int = 10_000,
    max_path_depth: int = 5,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    include_abstract: bool = False,
) -> dict[str, Any]:
    """Run bridge pipeline: resolve → compute → enrich → assemble.

    Returns a JSON-serializable dict matching the output schema
    defined in 11-bridge.md.
    """
    graph = quarry_graph.Graph(csr_dir)

    # Resolve seeds
    seed_ids = []
    for s in seeds:
        sid = resolve_seed(s, pg_conninfo)
        if not graph.has_node(sid):
            raise ValueError(f"Seed {sid} not found in graph")
        seed_ids.append(sid)

    # Rust bridge
    t0 = time.perf_counter()
    results, stats = graph.bridge(
        seed_ids,
        types=types,
        limit=limit,
        max_neighbor_degree=max_neighbor_degree,
        max_path_depth=max_path_depth,
        alpha=alpha,
        epsilon=epsilon,
    )
    elapsed = time.perf_counter() - t0

    # Collect all work_ids for PG enrichment
    all_wids: set[int] = set(seed_ids)
    for key in (
        "common_refs",
        "common_citers",
        "coupling_bridges",
        "cocitation_bridges",
        "path_bridges",
        "ppr_bridges",
    ):
        for entry in results.get(key, []):
            all_wids.add(entry["work_id"])

    metadata = _enrich_metadata(
        list(all_wids), pg_conninfo, include_abstract=include_abstract
    )

    # Seed metadata
    seed_infos = []
    for sid in seed_ids:
        meta = metadata.get(sid, {})
        seed_infos.append(
            {
                "work_id": sid,
                "title": meta.get("title"),
                "doi": meta.get("doi"),
                "year": meta.get("pub_year"),
            }
        )

    # Graph meta
    graph_meta = {"num_nodes": graph.num_nodes, "num_edges": graph.num_edges}
    meta_json_path = Path(csr_dir) / "meta.json"
    if meta_json_path.exists():
        with open(meta_json_path) as f:
            graph_meta["build_date"] = json.load(f).get("build_date")

    # Enrich each bridge type
    def _enrich_common(entries: list[dict]) -> list[dict]:
        out = []
        for e in entries:
            wid = e["work_id"]
            meta = metadata.get(wid, {})
            entry: dict[str, Any] = {
                "work_id": wid,
                "title": meta.get("title"),
                "doi": meta.get("doi"),
                "year": meta.get("pub_year"),
                "aa_weight": round(e["aa_weight"], 4),
                "seed_count": e["seed_count"],
                "quality": {
                    "cited_by": meta.get("cited_by_count"),
                    "cnp": meta.get("cnp"),
                    "fwci": meta.get("fwci"),
                    "rcr": meta.get("rcr"),
                },
            }
            if include_abstract:
                entry["abstract"] = meta.get("abstract")
            out.append(entry)
        return out

    def _enrich_scored(entries: list[dict]) -> list[dict]:
        out = []
        for e in entries:
            wid = e["work_id"]
            meta = metadata.get(wid, {})
            entry: dict[str, Any] = {
                "work_id": wid,
                "title": meta.get("title"),
                "doi": meta.get("doi"),
                "year": meta.get("pub_year"),
                "per_seed_scores": [round(s, 4) for s in e["per_seed_scores"]],
                "score": round(e["score"], 6),
                "quality": {
                    "cited_by": meta.get("cited_by_count"),
                    "cnp": meta.get("cnp"),
                    "fwci": meta.get("fwci"),
                    "rcr": meta.get("rcr"),
                },
            }
            if include_abstract:
                entry["abstract"] = meta.get("abstract")
            out.append(entry)
        return out

    def _enrich_path(entries: list[dict]) -> list[dict]:
        out = []
        for e in entries:
            wid = e["work_id"]
            meta = metadata.get(wid, {})
            entry: dict[str, Any] = {
                "work_id": wid,
                "title": meta.get("title"),
                "doi": meta.get("doi"),
                "year": meta.get("pub_year"),
                "hop_from": e["hop_from"],
                "path_count": e["path_count"],
                "quality": {
                    "cited_by": meta.get("cited_by_count"),
                    "cnp": meta.get("cnp"),
                    "fwci": meta.get("fwci"),
                    "rcr": meta.get("rcr"),
                },
            }
            if include_abstract:
                entry["abstract"] = meta.get("abstract")
            out.append(entry)
        return out

    return {
        "meta": {
            "quarry_version": "0.2.0",
            "graph": graph_meta,
        },
        "seeds": seed_infos,
        "params": {
            "types": types or _BRIDGE_TYPES,
            "limit": limit,
            "max_neighbor_degree": max_neighbor_degree,
            "max_path_depth": max_path_depth,
        },
        "common_refs": _enrich_common(results.get("common_refs", [])),
        "common_citers": _enrich_common(results.get("common_citers", [])),
        "coupling_bridges": _enrich_scored(results.get("coupling_bridges", [])),
        "cocitation_bridges": _enrich_scored(results.get("cocitation_bridges", [])),
        "path_bridges": _enrich_path(results.get("path_bridges", [])),
        "ppr_bridges": _enrich_scored(results.get("ppr_bridges", [])),
        "stats": {
            "seeds": seed_ids,
            "per_seed_ref_count": stats["per_seed_ref_count"],
            "per_seed_citer_count": stats["per_seed_citer_count"],
            "overlap_refs": stats["overlap_refs"],
            "overlap_citers": stats["overlap_citers"],
            "shortest_path_length": stats.get("shortest_path_length"),
            "path_quality_warning": (
                "sp >= 6: long citation chain, path_bridges may include off-topic papers"
                if (stats.get("shortest_path_length") or 0) >= 6
                else None
            ),
            "elapsed_ms": stats["elapsed_ms"],
            "elapsed_s": round(elapsed, 3),
        },
    }
