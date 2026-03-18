"""bioRxiv API daily fetch → DuckDB preprints table.

Fetches recent preprints from bioRxiv/medRxiv API and upserts into DuckDB.
"""

import time
from datetime import date, timedelta

import httpx
import pyarrow as pa

from quarry.store.duckdb import DuckDBStore

BIORXIV_API = "https://api.biorxiv.org/details"


def fetch_preprints(
    server: str = "biorxiv",
    from_date: date | None = None,
    to_date: date | None = None,
    cursor: int = 0,
    page_size: int = 100,
) -> list[dict]:
    """Fetch preprints from bioRxiv/medRxiv API.

    API: GET /details/{server}/{from}/{to}/{cursor}
    Returns up to 100 per page.
    """
    if to_date is None:
        to_date = date.today()
    if from_date is None:
        from_date = to_date - timedelta(days=7)

    all_results = []
    current_cursor = cursor

    with httpx.Client(timeout=httpx.Timeout(60, connect=10)) as client:
        while True:
            url = f"{BIORXIV_API}/{server}/{from_date}/{to_date}/{current_cursor}"
            for attempt in range(3):
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    break
                except (httpx.TimeoutException, httpx.HTTPStatusError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
                    continue
            data = resp.json()

            messages = data.get("messages", [{}])
            total = int(messages[0].get("total", 0)) if messages else 0
            collection = data.get("collection", [])

            if not collection:
                break

            for item in collection:
                all_results.append(
                    {
                        "doi": item.get("doi", ""),
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "date": item.get("date"),
                        "server": server,
                        "category": item.get("category", ""),
                        "version": int(item.get("version", 1)),
                        "published_doi": item.get("published") or None,
                    }
                )

            current_cursor += len(collection)
            if current_cursor >= total:
                break

    return all_results


def load_preprints(
    server: str = "biorxiv",
    from_date: date | None = None,
    to_date: date | None = None,
    db: DuckDBStore | None = None,
):
    """Fetch and upsert preprints into DuckDB."""
    close_db = False
    if db is None:
        db = DuckDBStore()
        db.init_schema()
        close_db = True

    print(f"Fetching {server} preprints ...")
    t0 = time.time()
    preprints = fetch_preprints(server=server, from_date=from_date, to_date=to_date)
    print(f"  Fetched {len(preprints)} preprints in {time.time() - t0:.1f}s")

    if preprints:
        # Deduplicate: keep only latest version per DOI
        by_doi: dict[str, dict] = {}
        for p in preprints:
            existing = by_doi.get(p["doi"])
            if existing is None or p["version"] > existing["version"]:
                by_doi[p["doi"]] = p
        preprints = list(by_doi.values())

        # Upsert via DELETE + pyarrow bulk INSERT (DuckDB has no INSERT OR REPLACE)
        columns = [
            "doi",
            "title",
            "abstract",
            "date",
            "server",
            "category",
            "version",
            "published_doi",
        ]
        arrow_table = pa.table(
            {
                "doi": pa.array([p["doi"] for p in preprints], type=pa.string()),
                "title": pa.array([p["title"] for p in preprints], type=pa.string()),
                "abstract": pa.array(
                    [p["abstract"] for p in preprints], type=pa.string()
                ),
                "date": pa.array(
                    [
                        date.fromisoformat(p["date"]) if p["date"] else None
                        for p in preprints
                    ],
                    type=pa.date32(),
                ),
                "server": pa.array([p["server"] for p in preprints], type=pa.string()),
                "category": pa.array(
                    [p["category"] for p in preprints], type=pa.string()
                ),
                "version": pa.array([p["version"] for p in preprints], type=pa.int16()),
                "published_doi": pa.array(
                    [p["published_doi"] for p in preprints], type=pa.string()
                ),
            }
        )
        # Delete existing DOIs
        db.conn.register("_tmp_preprints", arrow_table)
        db.conn.execute(
            "DELETE FROM preprints WHERE doi IN (SELECT doi FROM _tmp_preprints)"
        )
        cols = ", ".join(columns)
        db.conn.execute(
            f"INSERT INTO preprints ({cols}) SELECT {cols} FROM _tmp_preprints"
        )
        db.conn.unregister("_tmp_preprints")
        print(f"  Upserted {len(preprints)} preprints into DuckDB")

    if close_db:
        db.close()

    return len(preprints)
