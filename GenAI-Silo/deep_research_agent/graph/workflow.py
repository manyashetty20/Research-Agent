"""
Plan → Execute → Verify loop for agentic deep research.
Optimized for 2026 DeepScholar/FACT Benchmarks.
"""
from typing import Any, List, Dict
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
    """Helper to ensure references are consistently formatted and appended."""
    ref_lines = []
    # Deduplicate papers by ID to prevent reference bloat
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
    llm: Any,  # Injecting the LLM directly for Query Expansion
    end_date: str | None = None,
) -> tuple[str, List[dict], dict[str, Any]]:
    """
    Run a Plan → Execute → Verify pipeline.
    Optimized for 'Nugget Coverage' and 'Citation Traceability'.
    """
    cfg = get_config().agent
    
    # 1. Plan: Decompose query into sub-questions
    # We pass the LLM to the planner to ensure high-quality decomposition
    plan = plan_research(query, llm=llm)

    # 2. Execute: Recursive search with Query Expansion (Benchmark Booster)
    # We pass the LLM here so search_agent.py can generate 5x more targeted queries
    papers = search_agent(
        search_queries=plan.get("search_queries", [query]),
        main_query=query,
        end_date=end_date,
        llm=llm  # <--- Crucial for expanded retrieval
    )
    
    if not papers:
        return "No relevant papers found.", [], {"verified": False, "num_papers": 0}

    # 3. Extraction: Convert papers into atomic 'nuggets'
    nuggets = extract_nuggets(papers, llm=llm)
    
    # 4. Initial Synthesis
    #report = synthesize_report(query, plan, papers, nuggets, llm=llm)
    report = synthesize_report(
    query=query,
    plan=plan,
    papers=papers,
    nuggets=nuggets,
    llm=llm
)
    # 5. Verify loop: Audit claim-citation pairs (The Verifiability Metric)
    verified = False
    verify_result: dict[str, Any] | None = None
    iteration = 0

    while iteration < cfg.max_verify_iterations:
        iteration += 1
        
        # Verify current report against nuggets to detect hallucinations
        verify_result = verify_citations(report, papers, llm=llm)
        verified = bool(verify_result.get("valid", False))
        
        # Update report with corrections from the 'Verifier' LLM
        report = verify_result.get("corrected_report", report)
        
        # Check for missing Reference block (common failure mode in benchmarks)
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