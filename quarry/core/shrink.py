"""Core shrink logic — minimum covering paper set from top venues.

Given a seed, runs expand internally, filters to specified venues,
then selects papers by greedy weighted citation coverage (AD-10).

Pipeline: expand(seed) → venue filter → coverage precompute → greedy select.

Centralization handling (AD-10 post-implementation):
In highly centralized fields, a single foundational paper can cover 90%+
of candidates via 1-hop citations (e.g., CBE 2016 covers 96% of ABE
expand results). This makes the remaining selections trivially marginal.
Two mitigations:
  --no-foundation: exclude relation="foundation" papers from pool, surfacing
    recent trends instead of historical origin.
  --exclude: manually exclude specific dominant papers.
When top-1 coverage exceeds 80%, a warning is emitted.
"""

from __future__ import annotations

from typing import Any

import quarry_graph

from quarry.core.expand import run_expand

NCS_PLUS = {
    "Nature",
    "Science",
    "Cell",
    "Nature Methods",
    "Nature Biotechnology",
    "Nature Genetics",
    "Nature Medicine",
    "Nature Chemical Biology",
    "Nature Communications",
    "Science Advances",
    "Molecular Cell",
    "Cell Reports",
    "Cell Systems",
    "Cell Stem Cell",
    "Nature Reviews Genetics",
    "Nature Reviews Molecular Cell Biology",
    "Nature Biomedical Engineering",
    "Nature Neuroscience",
}

# Threshold above which a single paper's coverage triggers a warning.
# Empirically determined: 80%+ means the field is dominated by one
# foundational paper and the remaining selections are near-trivial.
_CENTRALIZATION_THRESHOLD = 0.80


def run_shrink(
    *,
    seed: str,
    csr_dir: str,
    pg_conninfo: str,
    top_n: int = 5,
    venues: set[str] | None = None,
    expand_limit: int = 200,
    no_foundation: bool = False,
    exclude: set[int] | None = None,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Run shrink pipeline: expand → venue filter → greedy set cover.

    Args:
        no_foundation: Exclude relation="foundation" from pool. Use when
            the field is centralized around a single founder paper and you
            want to surface recent follow-up/lateral trends instead.
        exclude: Set of work_id_ints to exclude from pool. Use to manually
            remove known dominant papers.

    Returns a JSON-serializable dict with selected papers and coverage stats.
    """
    if venues is None:
        venues = NCS_PLUS
    if exclude is None:
        exclude = set()

    # Phase 1: expand
    expand_result = run_expand(
        seed=seed,
        csr_dir=csr_dir,
        pg_conninfo=pg_conninfo,
        mode="fused",
        alpha=alpha,
        epsilon=epsilon,
        limit=expand_limit,
    )

    candidates = expand_result["papers"]
    candidate_ids = {p["work_id"] for p in candidates}
    score_map = {p["work_id"]: p["scores"]["fused"] for p in candidates}
    expand_rank_map = {p["work_id"]: p["rank"] for p in candidates}

    # Phase 2: build venue-filtered pool with optional exclusions
    graph = quarry_graph.Graph(csr_dir)
    pool: list[dict[str, Any]] = []
    excluded_count = 0
    for p in candidates:
        if p.get("host_venue") not in venues:
            continue
        wid = p["work_id"]
        if wid in exclude:
            excluded_count += 1
            continue
        # --no-foundation: skip papers the seed cites (its foundations).
        # These are the centralization source — the seed's own references
        # that all follow-up papers also cite, creating trivial coverage.
        if no_foundation and p.get("relation") == "foundation":
            excluded_count += 1
            continue
        fwd = set(graph.neighbors(wid, "forward"))
        rev = set(graph.neighbors(wid, "reverse"))
        cov = (fwd | rev | {wid}) & candidate_ids
        pool.append({"paper": p, "coverage_set": cov})

    # Phase 3: greedy weighted selection
    covered: set[int] = set()
    selected: list[dict[str, Any]] = []
    centralization_warning: str | None = None

    for i in range(min(top_n, len(pool))):
        best = max(
            pool,
            key=lambda x: sum(score_map.get(c, 0) for c in x["coverage_set"] - covered),
        )
        marginal = best["coverage_set"] - covered
        covered |= best["coverage_set"]
        pool.remove(best)

        cum_coverage = len(covered) / len(candidate_ids) if candidate_ids else 0

        # Detect centralization: first selected paper covers >80%
        if i == 0 and cum_coverage > _CENTRALIZATION_THRESHOLD:
            centralization_warning = (
                f"Top-1 paper covers {cum_coverage:.0%} of candidates — "
                f"highly centralized field. Consider --no-foundation or "
                f"a more recent seed for diverse results."
            )

        p = best["paper"]
        selected.append(
            {
                "rank_in_shrink": i + 1,
                "rank_in_expand": expand_rank_map.get(p["work_id"]),
                "work_id": p["work_id"],
                "title": p.get("title"),
                "year": p.get("year"),
                "host_venue": p.get("host_venue"),
                "doi": p.get("doi"),
                "relation": p.get("relation"),
                "scores": p.get("scores"),
                "quality": p.get("quality"),
                "coverage_count": len(best["coverage_set"]),
                "marginal_count": len(marginal),
                "cumulative_coverage": round(cum_coverage, 4),
            }
        )

    total_covered = len(covered)
    total_candidates = len(candidate_ids)

    return {
        "seed": expand_result["seed"],
        "params": {
            "top_n": top_n,
            "venues": sorted(venues),
            "expand_limit": expand_limit,
            "no_foundation": no_foundation,
            "excluded": sorted(exclude),
        },
        "selected": selected,
        "stats": {
            "total_candidates": total_candidates,
            "venue_pool_size": len(pool) + len(selected),
            "excluded_count": excluded_count,
            "covered": total_covered,
            "uncovered": total_candidates - total_covered,
            "coverage": round(total_covered / total_candidates, 4)
            if total_candidates
            else 0,
            "centralization_warning": centralization_warning,
            "expand_elapsed_s": expand_result["stats"]["elapsed_s"],
        },
    }
