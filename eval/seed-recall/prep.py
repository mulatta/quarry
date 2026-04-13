"""Prepare seed-recall evaluation data: PMID → OpenAlex work_id mapping.

Reads overall_collection.jsonl, resolves PMIDs against the PG works table,
and writes reviews.jsonl with quarry-ready work_ids.

Requires: PG running with works table populated.

Usage: python eval/seed-recall/prep.py [--conninfo DSN]
"""

import json
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

EVAL_DIR = Path(__file__).parent
INPUT = EVAL_DIR / "overall_collection.jsonl"
OUTPUT = EVAL_DIR / "reviews.jsonl"
SUMMARY = EVAL_DIR / "summary.json"


def resolve_pmids(conn: psycopg.Connection, pmids: list[str]) -> dict[str, str]:
    """Map PMID strings → OpenAlex work_id. Returns {pmid: work_id}."""
    if not pmids:
        return {}
    int_pmids = [int(p) for p in pmids]
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT pmid, work_id FROM works WHERE pmid = ANY(%s)",
            (int_pmids,),
        )
        return {str(row["pmid"]): row["work_id"] for row in cur.fetchall()}


def main(conninfo: str | None = None) -> None:
    if conninfo is None:
        # Try quarry config, fall back to default
        try:
            from quarry.config import settings

            conninfo = settings.pg_conninfo
        except ImportError:
            conninfo = "dbname=quarry"

    reviews_raw: list[dict] = []
    with INPUT.open() as f:
        for line in f:
            reviews_raw.append(json.loads(line))

    # Collect all unique PMIDs
    all_pmids: set[str] = set()
    for r in reviews_raw:
        all_pmids.update(r.get("seed_studies", []))
        all_pmids.update(r.get("included_studies", []))

    print(f"Total unique PMIDs to resolve: {len(all_pmids)}")

    with psycopg.connect(conninfo) as conn:
        pmid_map = resolve_pmids(conn, list(all_pmids))

    print(
        f"Resolved: {len(pmid_map)}/{len(all_pmids)} ({len(pmid_map) / len(all_pmids):.1%})"
    )

    # Build output reviews
    reviews_out: list[dict] = []
    total_seeds_resolved = 0
    total_seeds = 0
    total_gold_resolved = 0
    total_gold = 0
    skipped = 0

    for r in reviews_raw:
        seed_pmids = r.get("seed_studies", [])
        gold_pmids = r.get("included_studies", [])

        seeds_resolved = {p: pmid_map[p] for p in seed_pmids if p in pmid_map}
        gold_resolved = {p: pmid_map[p] for p in gold_pmids if p in pmid_map}

        total_seeds += len(seed_pmids)
        total_seeds_resolved += len(seeds_resolved)
        total_gold += len(gold_pmids)
        total_gold_resolved += len(gold_resolved)

        # Skip reviews with <2 seeds or <5 gold resolved
        if len(seeds_resolved) < 2 or len(gold_resolved) < 5:
            skipped += 1
            continue

        coverage = len(gold_resolved) / len(gold_pmids) if gold_pmids else 0.0

        reviews_out.append(
            {
                "review_id": r["id"],
                "title": r["title"],
                "link": r.get("link_to_review", ""),
                "seeds": list(seeds_resolved.values()),
                "seeds_pmid": list(seeds_resolved.keys()),
                "gold": list(gold_resolved.values()),
                "gold_pmid": list(gold_resolved.keys()),
                "gold_missing_pmid": [p for p in gold_pmids if p not in pmid_map],
                "coverage": round(coverage, 4),
            }
        )

    # Sort by coverage descending
    reviews_out.sort(key=lambda r: -r["coverage"])

    with OUTPUT.open("w") as f:
        for r in reviews_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "total_raw_reviews": len(reviews_raw),
        "usable_reviews": len(reviews_out),
        "skipped_reviews": skipped,
        "skip_reason": "seeds<2 or gold<5 after PMID resolution",
        "pmid_resolution": {
            "total_unique": len(all_pmids),
            "resolved": len(pmid_map),
            "rate": round(len(pmid_map) / len(all_pmids), 4) if all_pmids else 0,
        },
        "seed_resolution": {
            "total": total_seeds,
            "resolved": total_seeds_resolved,
            "rate": round(total_seeds_resolved / total_seeds, 4) if total_seeds else 0,
        },
        "gold_resolution": {
            "total": total_gold,
            "resolved": total_gold_resolved,
            "rate": round(total_gold_resolved / total_gold, 4) if total_gold else 0,
        },
    }

    with SUMMARY.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutput: {OUTPUT} ({len(reviews_out)} reviews)")
    print(f"Summary: {SUMMARY}")
    print(f"Skipped: {skipped} reviews (insufficient seed/gold coverage)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    conninfo = None
    if len(sys.argv) > 2 and sys.argv[1] == "--conninfo":
        conninfo = sys.argv[2]
    main(conninfo)
