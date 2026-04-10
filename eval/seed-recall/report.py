"""Compute seed-recall metrics and write summary reports.

Reads results/{review_id}/{method}.json and reviews.jsonl, then writes:
  results/summary.md         — method × Recall@K table (console + file)
  results/per_review.csv     — per-review breakdown
  results/unique_discovery.md — papers found only by one method
  results/method_overlap.md  — Jaccard similarity matrix

Usage:
    python eval/seed-recall/report.py
    just eval-report
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent
REVIEWS = EVAL_DIR / "reviews.jsonl"
RESULTS_DIR = EVAL_DIR / "results"

METHODS = ("a_bm25", "b_expand", "c_bridge", "d_mesh")
METHOD_LABELS = {
    "a_bm25": "A BM25",
    "b_expand": "B Expand",
    "c_bridge": "C Bridge",
    "d_mesh": "D MeSH",
}
K_VALUES = (50, 100, 200)


# ── helpers ──────────────────────────────────────────────────────────────────


def work_id_to_int(wid: str | int) -> int:
    s = str(wid)
    if s.upper().startswith("W") and s[1:].isdigit():
        return int(s[1:])
    return int(s)


def recall_at_k(found: list[int], gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(found[:k]) & gold) / len(gold)


def precision_at_k(found: list[int], gold: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(found[:k]) & gold) / k


def jaccard(hits_a: set[int], hits_b: set[int]) -> float:
    union = hits_a | hits_b
    if not union:
        return 0.0
    return len(hits_a & hits_b) / len(union)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


# ── data loading ─────────────────────────────────────────────────────────────


def load_reviews() -> dict[str, dict]:
    """Returns {review_id: review_dict}."""
    reviews = {}
    with REVIEWS.open() as f:
        for line in f:
            r = json.loads(line)
            reviews[r["review_id"]] = r
    return reviews


def load_result(review_id: str, method: str) -> list[int] | None:
    p = RESULTS_DIR / review_id / f"{method}.json"
    if not p.exists():
        return None
    with p.open() as f:
        data = json.load(f)
    return data.get("found", [])


# ── report sections ──────────────────────────────────────────────────────────


def compute_metrics(
    reviews: dict[str, dict],
) -> tuple[
    dict[str, dict[str, list[float]]],  # {method: {metric: [values]}}
    list[dict],  # per-review rows
]:
    """Compute all metrics across reviews."""
    agg: dict[str, dict[str, list[float]]] = {m: defaultdict(list) for m in METHODS}
    rows: list[dict] = []

    for review_id, review in sorted(reviews.items()):
        gold = {work_id_to_int(g) for g in review["gold"]}
        coverage = review.get("coverage", 0.0)

        for method in METHODS:
            found = load_result(review_id, method)
            if found is None:
                continue

            row: dict = {
                "review_id": review_id,
                "method": method,
                "coverage": round(coverage, 4),
                "n_gold": len(gold),
                "n_found": len(found),
            }
            for k in K_VALUES:
                r = recall_at_k(found, gold, k)
                p = precision_at_k(found, gold, k)
                row[f"recall@{k}"] = round(r, 4)
                row[f"precision@{k}"] = round(p, 4)
                agg[method][f"recall@{k}"].append(r)
                agg[method][f"precision@{k}"].append(p)
            agg[method]["coverage"].append(coverage)
            rows.append(row)

    return agg, rows


def write_summary_md(agg: dict[str, dict[str, list[float]]]) -> str:
    """Return markdown table string and write to results/summary.md."""
    header = "| Method | n | R@50 | R@100 | R@200 | P@200 |"
    sep = "|--------|---|------|-------|-------|-------|"
    lines = [header, sep]

    for method in METHODS:
        d = agg[method]
        n = len(d.get("recall@200", []))
        if n == 0:
            continue
        r50 = mean(d["recall@50"])
        r100 = mean(d["recall@100"])
        r200 = mean(d["recall@200"])
        p200 = mean(d["precision@200"])
        s200 = std(d["recall@200"])
        label = METHOD_LABELS[method]
        lines.append(
            f"| {label:12s} | {n} "
            f"| {r50:.3f} "
            f"| {r100:.3f} "
            f"| {r200:.3f}±{s200:.3f} "
            f"| {p200:.4f} |"
        )

    content = "\n".join(lines) + "\n"
    out = RESULTS_DIR / "summary.md"
    out.write_text(f"# Seed-Recall Evaluation Summary\n\n{content}")
    return content


def write_per_review_csv(rows: list[dict]) -> None:
    out = RESULTS_DIR / "per_review.csv"
    if not rows:
        return
    fieldnames = [
        "review_id",
        "method",
        "coverage",
        "n_gold",
        "n_found",
        "recall@50",
        "recall@100",
        "recall@200",
        "precision@50",
        "precision@100",
        "precision@200",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_unique_discovery(reviews: dict[str, dict]) -> None:
    """Papers found by exactly one method (in gold set)."""
    unique_counts: dict[str, int] = {m: 0 for m in METHODS}
    total_hits: dict[str, int] = {m: 0 for m in METHODS}

    for review_id, review in reviews.items():
        gold = {work_id_to_int(g) for g in review["gold"]}
        method_hits: dict[str, set[int]] = {}

        for method in METHODS:
            found = load_result(review_id, method)
            if found is None:
                continue
            method_hits[method] = set(found[:200]) & gold
            total_hits[method] += len(method_hits[method])

        for method, hits in method_hits.items():
            others = set.union(
                *(h for m, h in method_hits.items() if m != method),
                set(),
            )
            unique_counts[method] += len(hits - others)

    lines = [
        "# Unique Discovery per Method\n",
        "Papers in gold set found by exactly one method (at K=200).\n",
        "| Method | Unique hits | Total hits |",
        "|--------|------------|------------|",
    ]
    for method in METHODS:
        lines.append(
            f"| {METHOD_LABELS[method]:12s} "
            f"| {unique_counts[method]:11d} "
            f"| {total_hits[method]:10d} |"
        )
    (RESULTS_DIR / "unique_discovery.md").write_text("\n".join(lines) + "\n")


def write_method_overlap(reviews: dict[str, dict]) -> None:
    """Mean Jaccard similarity between method pairs."""
    pairs: dict[tuple[str, str], list[float]] = defaultdict(list)

    for review_id, review in reviews.items():
        gold = {work_id_to_int(g) for g in review["gold"]}
        method_hits: dict[str, set[int]] = {}

        for method in METHODS:
            found = load_result(review_id, method)
            if found is not None:
                method_hits[method] = set(found[:200]) & gold

        for i, m1 in enumerate(METHODS):
            for m2 in METHODS[i + 1 :]:
                if m1 in method_hits and m2 in method_hits:
                    pairs[(m1, m2)].append(jaccard(method_hits[m1], method_hits[m2]))

    # Header row
    labels = [METHOD_LABELS[m] for m in METHODS]
    header = "| | " + " | ".join(labels) + " |"
    sep = "|---|" + "---|" * len(METHODS)
    matrix_rows: list[str] = [header, sep]

    for m1 in METHODS:
        cells = [METHOD_LABELS[m1]]
        for m2 in METHODS:
            if m1 == m2:
                cells.append("1.000")
            else:
                key = (m1, m2) if (m1, m2) in pairs else (m2, m1)
                vals = pairs.get(key, [])
                cells.append(f"{mean(vals):.3f}" if vals else "—")
        matrix_rows.append("| " + " | ".join(cells) + " |")

    content = "\n".join(matrix_rows) + "\n"
    (RESULTS_DIR / "method_overlap.md").write_text(
        "# Method Overlap (mean Jaccard on gold hits, K=200)\n\n" + content
    )


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    if not REVIEWS.exists():
        print(
            f"Error: {REVIEWS} not found. Run `just eval-prep` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not RESULTS_DIR.exists():
        print(
            "Error: results/ not found. Run `just eval-run` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    reviews = load_reviews()
    agg, rows = compute_metrics(reviews)

    if not rows:
        print("No results found. Run `just eval-run` first.", file=sys.stderr)
        sys.exit(1)

    # Write all outputs
    summary = write_summary_md(agg)
    write_per_review_csv(rows)
    write_unique_discovery(reviews)
    write_method_overlap(reviews)

    # Console output
    n_reviews = len({r["review_id"] for r in rows})
    print(f"\nSeed-Recall Evaluation ({n_reviews} reviews)\n")
    print(summary)
    print(f"Full results: {RESULTS_DIR}/")
    print("  summary.md, per_review.csv, unique_discovery.md, method_overlap.md")


if __name__ == "__main__":
    main()
