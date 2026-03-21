from typing import List, Dict, Optional, Any
import concurrent.futures
import time
import requests
from tavily import TavilyClient

from deep_research_agent.config import get_config
from deep_research_agent.retrieval import (
    search_arxiv,
    search_semantic_scholar,
    rerank
)

_SEARCH_CACHE = {}


def relevance_score(paper: Dict, query: str) -> float:
    text = (
        (paper.get("title") or "") + " " + (paper.get("summary") or "")
    ).lower()
    query_terms = [t.lower() for t in query.split() if len(t) > 3]
    score = sum(1 for term in query_terms if term in text)
    return score


def fetch_references_from_s2(arxiv_id: str) -> List[Dict]:
    """Fetch papers directly cited by the query paper via Semantic Scholar."""
    try:
        clean_id = arxiv_id.split("v")[0]

        for attempt in range(3):
            r = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_id}/references",
                params={"fields": "title,year,authors,externalIds,abstract", "limit": 50},
                timeout=15
            )
            if r.status_code == 429:
                wait = (attempt + 1) * 10
                print(f"S2 rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            papers = []
            for ref in data.get("data", []):
                p = ref.get("citedPaper", {})
                if not p.get("title"):
                    continue
                ext = p.get("externalIds") or {}
                ref_arxiv_id = ext.get("ArXiv")
                authors = [a.get("name", "") for a in p.get("authors", [])]
                papers.append({
                    "id": ref_arxiv_id or p.get("paperId", ""),
                    "arxiv_id": ref_arxiv_id,
                    "title": p.get("title", ""),
                    "summary": p.get("abstract", ""),
                    "url": f"https://arxiv.org/abs/{ref_arxiv_id}" if ref_arxiv_id else "",
                    "date": str(p.get("year", "")),
                    "authors": authors,
                    "source": "s2_references",
                })
            print(f"Fetched {len(papers)} direct references from Semantic Scholar")
            return papers

        print("S2 references fetch failed after 3 retries")
        return []

    except Exception as e:
        print(f"S2 references fetch failed: {e}")
        return []


def expand_queries(main_query: str, llm: Any) -> List[str]:
    prompt = f"""
You are an expert academic search assistant.

Generate 5 high-quality literature search queries for finding papers related to:

"{main_query}"

Rules:
- Each query should target a DIFFERENT aspect of the topic
- Use terminology specific to this research area
- Include queries for: core methods, benchmarks, recent advances, related tasks, key architectures
- Do NOT include generic terms like "survey" or "overview"
- Keep each query SHORT and specific (5-8 words max)

Return only the queries, one per line, no numbering or bullets.
"""
    resp = llm.invoke(prompt)
    queries = [
        q.strip("- ").strip()
        for q in resp.split("\n")
        if len(q.strip()) > 5
    ]
    return queries[:5]


def tavily_search(query, api_key):
    try:
        client = TavilyClient(api_key=api_key)
        results = client.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )
        papers = []
        for r in results["results"]:
            papers.append({
                "title": r["title"],
                "summary": r["content"],
                "url": r["url"]
            })
        return papers
    except Exception:
        return []


def search_agent(
    search_queries: List[str],
    main_query: Optional[str] = None,
    end_date: Optional[str] = None,
    top_k: Optional[int] = None,
    llm=None,
    arxiv_id: Optional[str] = None,
):
    cfg = get_config()
    rerank_query = main_query or search_queries[0]

    # Step 1: Fetch direct references from S2 (gold standard papers)
    reference_papers = []
    if arxiv_id:
        reference_papers = fetch_references_from_s2(arxiv_id)

    # Step 2: Expand queries using LLM
    expanded = list(search_queries)
    if llm:
        expanded += expand_queries(rerank_query, llm)

    # Deduplicate queries
    seen_q = set()
    unique_queries = []
    for q in expanded:
        if q not in seen_q:
            seen_q.add(q)
            unique_queries.append(q)

    all_papers = []
    seen_ids = set()

    # Step 3: Add reference papers first (highest priority)
    for p in reference_papers:
        pid = p.get("id") or p.get("title")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            all_papers.append(p)

    # Step 4: Run search queries
    def run_query(q):
        if q in _SEARCH_CACHE:
            return _SEARCH_CACHE[q]
        # Run arXiv separately (rate limit sensitive)
        arxiv_papers = search_arxiv(q, 15, end_date)
        # Run Semantic Scholar and Tavily in parallel
        with concurrent.futures.ThreadPoolExecutor() as ex:
            f2 = ex.submit(search_semantic_scholar, q, 20)
            f3 = ex.submit(tavily_search, q, cfg.retrieval.tavily_api_key)
            other_papers = f2.result() + f3.result()
        papers = arxiv_papers + other_papers
        _SEARCH_CACHE[q] = papers
        return papers

    for q in unique_queries[:4]:
        batch = run_query(q)
        for p in batch:
            pid = p.get("id") or p.get("title")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(p)

    if not all_papers:
        return []

    # Step 5: Score by relevance
    scored = [(p, relevance_score(p, rerank_query)) for p in all_papers]
    scored.sort(key=lambda x: x[1], reverse=True)
    candidates = [p for p, s in scored[:60]]

    # Step 6: Rerank
    ranked = rerank(
        rerank_query,
        candidates,
        text_key="summary",
        top_k=top_k or cfg.retrieval.top_k_after_rerank,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return ranked