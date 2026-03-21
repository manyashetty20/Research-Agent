"""
Plan → Execute → Verify loop for agentic deep research.
"""
from typing import Any, List, Dict, Optional
import os

from deep_research_agent.agents import (
    plan_research,
    search_agent,
    extract_nuggets,
    synthesize_report,
    verify_citations,
)
from deep_research_agent.config import get_config


def generate_reference_block(papers: List[Dict]) -> str:
    ref_lines = []
    unique_papers = {p.get("id") or p.get("arxiv_id") or p.get("title"): p for p in papers}.values()
    for i, p in enumerate(unique_papers, 1):
        authors_list = p.get("authors", [])
        authors = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors += " et al."
        title = p.get("title", "Unknown Title")
        url = p.get("url") or f"https://arxiv.org/abs/{p.get('arxiv_id', '')}"
        ref_lines.append(f"[{i}] {authors}. {title}. {url}")
    return "\n\n## References\n" + "\n".join(ref_lines)


def run_research(
    query: str,
    llm: Any,
    end_date: str | None = None,
    arxiv_id: str | None = None,
) -> tuple[str, List[dict], dict[str, Any]]:

    cfg = get_config().agent

    # 1. Plan
    plan = plan_research(query, llm=llm)

    # 2. Search
    papers = search_agent(
        search_queries=plan.get("search_queries", [query]),
        main_query=query,
        end_date=end_date,
        llm=llm,
        arxiv_id=arxiv_id,
    )

    if not papers:
        return "No relevant papers found.", [], {"verified": False, "num_papers": 0}

    # 3. Extract nuggets
    nuggets = extract_nuggets(papers, llm=llm)

    # 4. Synthesize
    report = synthesize_report(
        query=query,
        plan=plan,
        papers=papers,
        nuggets=nuggets,
        llm=llm
    )

    # 5. Verify loop
    verified = False
    verify_result: dict[str, Any] | None = None
    iteration = 0

    while iteration < cfg.max_verify_iterations:
        iteration += 1
        verify_result = verify_citations(report, papers, llm=llm)
        verified = bool(verify_result.get("valid", False))
        report = verify_result.get("corrected_report", report)
        if "## References" not in report:
            report += generate_reference_block(papers)
        if verified:
            break

    stats = {
        "verified": verified,
        "num_papers": len(papers),
        "total_nuggets": len(nuggets),
        "iterations": iteration,
        "search_depth": len(plan.get("search_queries", []))
    }

    return report, papers, stats