"""Run seed-recall evaluation: four retrieval methods on each review.

Reads reviews.jsonl (from just eval-prep) and runs four retrieval methods
per review, writing results to results/{review_id}/{method}.json.

Usage:
    python eval/seed-recall/run.py [--force] [--conninfo DSN]
    just eval-run
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).parent
REVIEWS = EVAL_DIR / "reviews.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

# Papers published within this many years are under-cited by definition.
RECENCY_YEARS = 3

# Methods run in this order; b_expand result is reused by c and d.
METHODS = ("a_bm25", "b_expand", "c_bridge", "d_mesh")


# ── helpers ─────────────────────────────────────────────────────────────────


def work_id_to_int(wid: str | int) -> int:
    """Convert "W12345" or "12345" or 12345 → 12345."""
    s = str(wid)
    if s.upper().startswith("W") and s[1:].isdigit():
        return int(s[1:])
    return int(s)


def work_int_to_str(wid: int) -> str:
    """Convert 12345 → "W12345"."""
    return f"W{wid}"


# ── method implementations ───────────────────────────────────────────────────


def _method_a_bm25(title: str, limit: int = 200) -> tuple[list[int], int]:
    """BM25 text search against LanceDB full-text index."""
    from quarry.search.hybrid import HybridSearcher

    searcher = HybridSearcher()
    results = searcher.search(title, limit=limit, mode="text", enrich=False)
    found = [work_id_to_int(r["work_id"]) for r in results if "work_id" in r]
    return found, 0


def _method_b_expand(seeds: list[str], limit: int = 200) -> tuple[list[int], int]:
    """APPR expand per seed, union ordered by best-rank across seeds."""
    from quarry.config import settings
    from quarry.core.expand import run_expand

    best_rank: dict[int, int] = {}  # work_id_int → best rank seen
    n_resolved = 0

    for seed in seeds:
        try:
            result = run_expand(
                seed=seed,
                csr_dir=str(settings.csr_dir),
                pg_conninfo=settings.pg_conninfo,
                limit=limit,
            )
            n_resolved += 1
            for paper in result["papers"]:
                wid = paper["work_id"]  # already int from Rust
                rank = paper["rank"]
                if wid not in best_rank or rank < best_rank[wid]:
                    best_rank[wid] = rank
        except Exception as exc:  # noqa: BLE001
            print(f"    expand({seed[:20]}): {exc}", file=sys.stderr)

    ordered = sorted(best_rank.keys(), key=lambda w: best_rank[w])
    return ordered, n_resolved


def _method_c_bridge(
    seeds: list[str],
    b_found: list[int],
    b_n_resolved: int,
    limit: int = 100,
) -> tuple[list[int], int]:
    """expand + bridge on top-2 most-cited seeds, union with B."""
    from quarry.config import settings
    from quarry.core.bridge import run_bridge
    from quarry.store.pg import PGStore

    if b_n_resolved < 2:
        return b_found, b_n_resolved

    # Rank seeds by recency-adjusted citation count to pick top-2 bridge anchors.
    # Recent papers (≤ RECENCY_YEARS old) are under-cited by definition, so we
    # apply a 1.5× multiplier to avoid systematically deprioritising new work.
    import datetime

    db = PGStore(settings.pg_conninfo)
    current_year = datetime.date.today().year
    seed_scores: list[tuple[str, float]] = []
    for seed in seeds:
        try:
            w = db.get_work(seed)
            if w:
                citations = w.get("cited_by_count") or 0
                pub_year = w.get("publication_year") or current_year
                recency_mult = (
                    1.5 if (current_year - pub_year) <= RECENCY_YEARS else 1.0
                )
                score = citations * recency_mult
            else:
                score = 0.0
        except Exception:  # noqa: BLE001
            score = 0.0
        seed_scores.append((seed, score))
    seed_scores.sort(key=lambda x: -x[1])
    top2 = [s for s, _ in seed_scores[:2]]

    try:
        bridge_result = run_bridge(
            seeds=top2,
            csr_dir=str(settings.csr_dir),
            pg_conninfo=settings.pg_conninfo,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    bridge({top2[0][:12]},{top2[1][:12]}): {exc}", file=sys.stderr)
        return b_found, b_n_resolved

    bridge_wids: set[int] = set()
    for key in (
        "common_refs",
        "common_citers",
        "coupling_bridges",
        "cocitation_bridges",
        "path_bridges",
        "ppr_bridges",
    ):
        for entry in bridge_result.get(key, []):
            bridge_wids.add(entry["work_id"])
    for wid in bridge_result.get("steiner_bridges", []):
        bridge_wids.add(wid)

    b_set = set(b_found)
    bridge_new = [w for w in bridge_wids if w not in b_set]
    return b_found + bridge_new, b_n_resolved


def _method_d_mesh(
    b_found: list[int],
    b_n_resolved: int,
    top_mesh_limit: int = 3,
    mesh_papers_limit: int = 50,
) -> tuple[list[int], int]:
    """expand + MeSH: top descriptors from B → mesh-tagged papers."""
    from quarry.store.pg import PGStore
    from quarry.config import settings

    db = PGStore(settings.pg_conninfo)

    b_work_ids = [work_int_to_str(w) for w in b_found[:200]]
    try:
        top_mesh = db.top_mesh_by_work_ids(
            b_work_ids, limit=top_mesh_limit, major_only=True
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    top_mesh: {exc}", file=sys.stderr)
        return b_found, b_n_resolved

    mesh_wids: set[int] = set()
    for descriptor in top_mesh:
        try:
            works, _ = db.get_top_works_by_mesh(
                descriptor["descriptor_ui"], limit=mesh_papers_limit
            )
            for w in works:
                wid = w.get("work_id")
                if wid:
                    mesh_wids.add(work_id_to_int(wid))
        except Exception as exc:  # noqa: BLE001
            print(f"    mesh({descriptor['descriptor_ui']}): {exc}", file=sys.stderr)

    b_set = set(b_found)
    mesh_new = [w for w in mesh_wids if w not in b_set]
    return b_found + mesh_new, b_n_resolved


# ── per-review runner ────────────────────────────────────────────────────────


def run_review(review: dict, force: bool = False) -> None:
    review_id = review["review_id"]
    seeds = review["seeds"]  # work_id strings: ["W12345", ...]
    gold_int = {work_id_to_int(g) for g in review["gold"]}
    title = review["title"]
    seed_int = {work_id_to_int(s) for s in seeds}

    out_dir = RESULTS_DIR / review_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # b_expand is reused by c and d; load from cache if available
    b_found: list[int] | None = None
    b_n_resolved: int = 0
    b_cache = out_dir / "b_expand.json"
    if b_cache.exists() and not force:
        with b_cache.open() as f:
            cached = json.load(f)
        b_found = cached["found"]
        b_n_resolved = cached["n_seeds_resolved"]

    for method in METHODS:
        out_file = out_dir / f"{method}.json"
        if out_file.exists() and not force:
            continue

        t0 = time.perf_counter()

        try:
            if method == "a_bm25":
                found, n_resolved = _method_a_bm25(title)
            elif method == "b_expand":
                found, n_resolved = _method_b_expand(seeds)
                b_found = found
                b_n_resolved = n_resolved
            elif method == "c_bridge":
                if b_found is None:
                    b_found, b_n_resolved = _method_b_expand(seeds)
                found, n_resolved = _method_c_bridge(seeds, b_found, b_n_resolved)
            elif method == "d_mesh":
                if b_found is None:
                    b_found, b_n_resolved = _method_b_expand(seeds)
                found, n_resolved = _method_d_mesh(b_found, b_n_resolved)
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"    [{method}] FAILED: {exc}", file=sys.stderr)
            continue

        # Exclude seeds from found set
        found = [w for w in found if w not in seed_int]
        elapsed = time.perf_counter() - t0

        result = {
            "review_id": review_id,
            "method": method,
            "found": found,
            "n_seeds_resolved": n_resolved,
            "elapsed_s": round(elapsed, 3),
        }
        with out_file.open("w") as f:
            json.dump(result, f)

        recall_200 = (
            len(set(found[:200]) & gold_int) / len(gold_int) if gold_int else 0.0
        )
        print(
            f"    [{method:10s}] n={len(found):4d}  recall@200={recall_200:.3f}"
            f"  ({elapsed:.1f}s)"
        )


# ── entry point ──────────────────────────────────────────────────────────────


def main(force: bool = False) -> None:
    if not REVIEWS.exists():
        print(
            f"Error: {REVIEWS} not found. Run `just eval-prep` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    RESULTS_DIR.mkdir(exist_ok=True)

    reviews: list[dict] = []
    with REVIEWS.open() as f:
        for line in f:
            reviews.append(json.loads(line))

    print(f"Evaluating {len(reviews)} reviews across {len(METHODS)} methods…")

    for i, review in enumerate(reviews, 1):
        print(
            f"\n[{i}/{len(reviews)}] #{review['review_id']}"
            f"  coverage={review['coverage']:.2f}"
            f"  seeds={len(review['seeds'])}"
            f"  gold={len(review['gold'])}"
        )
        print(f"  {review['title'][:70]}")
        run_review(review, force=force)

    print("\nDone. Run `just eval-report` for summary.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    main(force=force)
