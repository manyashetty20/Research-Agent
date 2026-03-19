"""
Planner Agent: decomposes the benchmark query into sub-questions and a research strategy.
"""
import json
from typing import Any, Optional

SYSTEM = """
You are an expert research planner for academic literature.

Given a research paper title and abstract, generate a focused research plan.

Return JSON with:
- sub_questions: list of 4-5 specific research sub-questions
- search_queries: list of 5 targeted literature search queries
- strategy: short description of the research strategy

Rules for search_queries:
- Extract the CORE TOPIC from the title (ignore the abstract details)
- Each query should target a DIFFERENT aspect: core method, benchmarks, datasets, baselines, recent advances
- Use domain-specific terminology from the research area
- Keep queries SHORT and specific (5-8 words max)
- Focus on finding papers that would appear in the Related Work of this paper
"""


def plan_research(query: str, llm: Optional[Any] = None) -> dict[str, Any]:
    prompt = f"Research paper title/topic:\n{query}"

    if llm is None:
        raise RuntimeError("LLM instance must be provided to planner.")

    out = llm.invoke(prompt, system=SYSTEM)
    out = out.strip()

    if out.startswith("```"):
        out = out.split("```")[1]
        if out.startswith("json"):
            out = out[4:]

    out = out.strip()

    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # Smart fallback: generate multiple search queries from the title
        words = query.split()
        return {
            "sub_questions": [query],
            "search_queries": [
                query,
                " ".join(words[:6]) if len(words) > 6 else query,
                " ".join(words[-6:]) if len(words) > 6 else query,
            ],
            "strategy": "Direct literature search using query and key phrases."
        }