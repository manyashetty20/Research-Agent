"""arXiv search client for scientific literature."""
from datetime import datetime
from typing import Optional
import os
import time
import arxiv
import sys


def _disable_ssl_verification():
    verify_env = os.getenv("ARXIV_VERIFY_SSL", "1").lower()
    if verify_env in ("0", "false"):
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        import requests
        requests.Session.verify = False


def search_arxiv(
    query: str,
    max_results: int = 15,
    end_date: Optional[str] = None,
) -> list[dict]:
    """
    Search arXiv. end_date filters to papers published BEFORE that date.
    """
    _disable_ssl_verification()
    time.sleep(8)  # always wait before hitting arXiv

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=5,
        num_retries=2
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    cutoff = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    try:
        for result in client.results(search):
            pub_date = result.published.replace(tzinfo=None) if result.published else None
            if cutoff and pub_date and pub_date > cutoff:
                continue
            arxiv_id = result.entry_id.split("/")[-1]
            papers.append({
                "id": arxiv_id,
                "arxiv_id": arxiv_id,
                "title": result.title,
                "summary": result.summary,
                "url": result.entry_id,
                "date": result.published.isoformat() if result.published else None,
                "authors": [a.name for a in result.authors],
                "source": "arxiv",
            })
    except Exception as e:
        # Catch ALL errors (429, 503, etc.) and return empty
        print(f"ArXiv error: {e}. Skipping and continuing...", file=sys.stderr)
        time.sleep(20)
        return []

    return papers