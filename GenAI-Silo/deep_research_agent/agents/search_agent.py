from typing import List, Dict, Optional, Any
import concurrent.futures
from tavily import TavilyClient

from deep_research_agent.config import get_config
from deep_research_agent.retrieval import (
    search_arxiv,
    search_semantic_scholar,
    rerank
)

_SEARCH_CACHE = {}


def relevance_score(paper: Dict, query: str) -> float:
    """Score relevance based on query terms instead of hardcoded keywords."""
    text = (
        (paper.get("title") or "") + " " + (paper.get("summary") or "")
    ).lower()
    query_terms = [t.lower() for t in query.split() if len(t) > 3]
    score = sum(1 for term in query_terms if term in text)
    return score


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
    llm=None
):
    cfg = get_config()
    rerank_query = main_query or search_queries[0]

    expanded = list(search_queries)

    # Always expand queries using LLM for better coverage
    if llm:
        expanded += expand_queries(rerank_query, llm)

    # Deduplicate while preserving order
    seen_q = set()
    unique_queries = []
    for q in expanded:
        if q not in seen_q:
            seen_q.add(q)
            unique_queries.append(q)

    all_papers = []
    seen_ids = set()

    def run_query(q):
        if q in _SEARCH_CACHE:
            return _SEARCH_CACHE[q]

        # Run arXiv separately and sequentially (rate limit sensitive)
        arxiv_papers = search_arxiv(q, 15, end_date)

        # Run Semantic Scholar and Tavily in parallel
        with concurrent.futures.ThreadPoolExecutor() as ex:
            f2 = ex.submit(search_semantic_scholar, q, 20)
            f3 = ex.submit(tavily_search, q, cfg.retrieval.tavily_api_key)
            other_papers = f2.result() + f3.result()

        papers = arxiv_papers + other_papers
        _SEARCH_CACHE[q] = papers
        return papers

    for q in unique_queries[:4]:  # reduced from 6 to 4 to ease arXiv load
        batch = run_query(q)
        for p in batch:
            pid = p.get("id") or p.get("title")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(p)

    if not all_papers:
        return []

    # Score by query relevance (dynamic, not hardcoded keywords)
    scored = [(p, relevance_score(p, rerank_query)) for p in all_papers]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep top candidates for reranking
    candidates = [p for p, s in scored[:50]]

    # Rerank using cross-encoder
    ranked = rerank(
        rerank_query,
        candidates,
        text_key="summary",
        top_k=top_k or cfg.retrieval.top_k_after_rerank,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return ranked