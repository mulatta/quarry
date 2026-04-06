"""Core expand logic — shared by CLI and MCP server.

This module is the exact scope of future Rust migration (AD-1).
All graph computation happens in Rust; this module handles:
  - Seed resolution (DOI/W-prefix → work_id_int)
  - Rust expand() call
  - PG metadata enrichment (title, doi, year, quality signals)
  - Relation tagging
  - Bridge title enrichment
  - JSON-ready dict assembly
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import psycopg
import quarry_graph


def run_expand(
    *,
    seed: str,
    csr_dir: str,
    pg_conninfo: str,
    mode: str = "fused",
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    limit: int = 200,
    include_abstract: bool = False,
) -> dict[str, Any]:
    """Run expand pipeline: resolve → compute → enrich → assemble.

    Returns a JSON-serializable dict matching the output schema
    defined in 08-expand-command.md.
    """
    graph = quarry_graph.Graph(csr_dir)
    seed_id = resolve_seed(seed, pg_conninfo)

    if not graph.has_node(seed_id):
        raise ValueError(f"Seed {seed_id} not found in graph")

    # Rust expand
    t0 = time.perf_counter()
    papers, stats = graph.expand(
        seed_id, alpha=alpha, epsilon=epsilon, mode=mode, limit=limit
    )
    elapsed = time.perf_counter() - t0

    # Relation tagging
    seed_fwd = set(graph.neighbors(seed_id, "forward"))
    seed_rev = set(graph.neighbors(seed_id, "reverse"))

    # Classify papers (Rust now returns list of dicts with bridges)
    classified = []
    for paper in papers:
        wid = paper["work_id"]
        cites_seed = wid in seed_rev
        cited_by_seed = wid in seed_fwd
        if cited_by_seed and cites_seed:
            relation = "mutual"
        elif cited_by_seed:
            relation = "foundation"
        elif cites_seed:
            relation = "follow-up"
        else:
            relation = "lateral"
        classified.append(
            {
                "work_id": wid,
                "fused_score": paper["fused_score"],
                "appr_score": paper["appr_score"],
                "bridges": paper["bridges"],
                "relation": relation,
            }
        )

    # PG enrichment
    all_wids = [p["work_id"] for p in classified]
    metadata = _enrich_metadata(
        all_wids, pg_conninfo, include_abstract=include_abstract
    )

    # Seed metadata
    seed_meta = _enrich_metadata([seed_id], pg_conninfo, include_abstract=False)
    seed_info = seed_meta.get(seed_id, {})

    # Graph meta
    graph_meta = {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
    }
    meta_json_path = Path(csr_dir) / "meta.json"
    if meta_json_path.exists():
        with open(meta_json_path) as f:
            csr_meta = json.load(f)
        graph_meta["build_date"] = csr_meta.get("build_date")

    # Bridge title enrichment
    bridge_wids: set[int] = set()
    for p in classified:
        for b in p["bridges"]:
            bridge_wids.add(b["work_id"])
    bridge_meta = (
        _enrich_metadata(list(bridge_wids), pg_conninfo) if bridge_wids else {}
    )

    # Assemble output
    result_papers = []
    for i, p in enumerate(classified):
        wid = p["work_id"]
        meta = metadata.get(wid, {})

        # Enrich bridges with titles
        bridges = None
        if p["bridges"]:
            bridges = []
            for b in p["bridges"]:
                bm = bridge_meta.get(b["work_id"], {})
                bridges.append(
                    {
                        "work_id": b["work_id"],
                        "type": b["type"],
                        "title": bm.get("title"),
                        "weight": round(b["weight"], 4),
                    }
                )

        entry: dict[str, Any] = {
            "rank": i + 1,
            "work_id": wid,
            "doi": meta.get("doi"),
            "title": meta.get("title"),
            "year": meta.get("pub_year"),
            "relation": p["relation"],
            "scores": {
                "fused": round(p["fused_score"], 6),
                "appr": round(p["appr_score"], 6)
                if p["appr_score"] is not None
                else None,
            },
            "quality": {
                "cited_by": meta.get("cited_by_count"),
                "cnp": meta.get("cnp"),
                "fwci": meta.get("fwci"),
                "rcr": meta.get("rcr"),
            },
            "bridges": bridges,
        }
        if include_abstract:
            entry["abstract"] = meta.get("abstract")
        result_papers.append(entry)

    return {
        "meta": {
            "quarry_version": "0.2.0",
            "graph": graph_meta,
        },
        "seed": {
            "work_id": seed_id,
            "title": seed_info.get("title"),
            "doi": seed_info.get("doi"),
            "year": seed_info.get("pub_year"),
        },
        "params": {
            "mode": mode,
            "alpha": alpha,
            "epsilon": epsilon,
            "limit": limit,
        },
        "papers": result_papers,
        "stats": {
            **{k: int(v) for k, v in stats.items()},
            "returned": len(result_papers),
            "elapsed_s": round(elapsed, 3),
        },
    }


def resolve_seed(seed: str, pg_conninfo: str) -> int:
    """Resolve seed string to work_id_int.

    Accepts: work_id_int, W<id>, DOI, https://doi.org/DOI.
    """
    # W-prefix
    if seed.upper().startswith("W") and seed[1:].isdigit():
        return int(seed[1:])

    # Direct integer
    try:
        return int(seed)
    except ValueError:
        pass

    # DOI — normalize to OA format
    if seed.startswith("10.") or seed.startswith("https://doi.org/"):
        doi = "https://doi.org/" + seed.removeprefix("https://doi.org/").lower()
        with psycopg.connect(pg_conninfo) as conn, conn.cursor() as cur:
            cur.execute("SELECT work_id_int FROM works WHERE doi = %s LIMIT 1", (doi,))
            row = cur.fetchone()
            if row:
                return row[0]
        raise ValueError(f"DOI not found: {doi}")

    raise ValueError(f"Cannot resolve seed: {seed}")


def _enrich_metadata(
    work_ids: list[int], pg_conninfo: str, *, include_abstract: bool = False
) -> dict[int, dict]:
    """Fetch metadata from PG for a batch of work_ids."""
    if not work_ids:
        return {}

    try:
        cols = (
            "work_id_int, title, pub_year, cited_by_count, doi, "
            "fwci, citation_normalized_percentile, rcr"
        )
        if include_abstract:
            cols += ", abstract"

        with psycopg.connect(pg_conninfo) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT {cols} FROM works WHERE work_id_int = ANY(%s)",  # noqa: S608
                (list(work_ids),),
            )
            result = {}
            for row in cur.fetchall():
                entry = {
                    "title": row[1],
                    "pub_year": row[2],
                    "cited_by_count": row[3],
                    "doi": row[4],
                    "fwci": float(row[5]) if row[5] is not None else None,
                    "cnp": float(row[6]) if row[6] is not None else None,
                    "rcr": float(row[7]) if row[7] is not None else None,
                }
                if include_abstract:
                    entry["abstract"] = row[8] or ""
                result[row[0]] = entry
            return result
    except Exception:
        return {}
