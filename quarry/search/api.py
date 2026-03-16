"""Supplementary API clients: bioRxiv + OpenAlex on-demand enrichment."""

import httpx


def enrich_with_openalex(doi: str) -> dict | None:
    """Fetch per-paper enrichment from OpenAlex API (topics, FWCI, institutions).

    On-demand only — not for bulk use.
    """
    if not doi:
        return None
    url = f"https://api.openalex.org/works/doi:{doi}"
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, headers={"Accept": "application/json"})
            if resp.status_code != 200:
                return None
            work = resp.json()
            return {
                "openalex_id": work.get("id"),
                "fwci": work.get("fwci"),
                "cited_by_count": work.get("cited_by_count"),
                "topics": [
                    {"name": t.get("display_name"), "score": t.get("score")}
                    for t in (work.get("topics") or [])[:5]
                ],
                "primary_topic": (work.get("primary_topic") or {}).get("display_name"),
                "is_oa": (work.get("open_access") or {}).get("is_oa"),
            }
    except httpx.HTTPError:
        return None
