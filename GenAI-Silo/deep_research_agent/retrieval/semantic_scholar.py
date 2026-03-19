"""Semantic Scholar API for paper search (optional; no key required for basic use)."""
from typing import Optional

import requests

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"


def search_semantic_scholar(
    query: str,
    max_results: int = 20,
    fields: str = "paperId,title,abstract,url,year,authors,externalIds",
) -> list[dict]:
    """
    Search Semantic Scholar. Returns list of paper dicts.
    No API key required for basic search (rate limit applies).
    """
    url = f"{SEMANTIC_SCHOLAR_API}/paper/search"
    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": fields,
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    papers = []
    for hit in data.get("data", []):
        if not hit.get("paperId"):
            continue
        authors = [a.get("name") for a in hit.get("authors", []) if a.get("name")]

        external_ids = hit.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv")  # e.g. "2211.07525"

        papers.append({
            "id": hit["paperId"],
            "arxiv_id": arxiv_id,
            "title": hit.get("title", ""),
            "summary": hit.get("abstract", ""),
            "url": hit.get("url", f"https://www.semanticscholar.org/paper/{hit['paperId']}"),
            "date": str(hit.get("year", "")),
            "authors": authors,
            "source": "semantic_scholar",
        })
    return papers