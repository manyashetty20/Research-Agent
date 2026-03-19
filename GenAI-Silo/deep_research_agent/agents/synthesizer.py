"""Synthesizer / Writer Agent: produces a structured, cited scientific answer."""

from typing import Any, Optional

SYSTEM = """You write a highly technical "Related Work" section in Markdown for an academic paper.

Structure rules:
- You MUST start with the header "# Related Work"
- Organize content into 3-4 thematic subsections using ## headers (e.g. "## Domain Adaptation Methods", "## Benchmark Datasets", "## Transformer-based Approaches")
- Each subsection must have 2-3 paragraphs
- Use double newlines between every paragraph
- End with a short ## Summary paragraph that synthesizes the themes

Citation rules:
- Every factual claim MUST be followed by an inline citation in this EXACT format:
  [Author et al.](https://arxiv.org/abs/ARXIV_ID)
- Use ONLY the arXiv URLs provided in the reference list
- NEVER use numbered citations like [1] or [2]
- DO NOT generate a References section

Writing rules:
- Write like a top NeurIPS/CVPR paper — precise, technical, comparative
- Compare methods directly: "Method A achieves X% on dataset Y, outperforming Method B by Z%"
- Group related works thematically, not chronologically
- Each paragraph should discuss 2-3 papers and compare their approaches
- Minimum 600 words
"""


def synthesize_report(
    query: str,
    plan: dict[str, Any],
    papers: list[dict],
    nuggets: list[dict],
    llm: Optional[Any] = None,
    max_tokens: int = 4096,
) -> str:

    if llm is None:
        raise RuntimeError("LLM instance must be provided to synthesize_report().")

    # Map paper id -> nuggets
    nugget_map = {n.get("id"): n.get("nuggets", []) for n in nuggets if n.get("id")}

    # 1. Build reference list for the prompt
    ref_lines = []
    for p in papers:
        authors_list = p.get("authors", [])
        authors = ", ".join(authors_list[:3])
        if len(authors_list) > 3:
            authors += " et al."
        title = p.get("title", "Unknown Title")
        arxiv_id = p.get("arxiv_id") or p.get("externalIds", {}).get("ArXiv", "")
        year = p.get("date", p.get("year", ""))[:4] if p.get("date") or p.get("year") else ""
        if arxiv_id:
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
            ref_lines.append(
                f"- [{authors}] {title} ({year}) → {arxiv_url}"
            )

    reference_context = "\n".join(ref_lines)

    # 2. Convert nuggets into prompt context
    nugget_lines = []
    for pid, nug_list in nugget_map.items():
        paper = next((p for p in papers if p.get("id") == pid), None)
        if not paper:
            continue
        arxiv_id = paper.get("arxiv_id") or ""
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
        authors_list = paper.get("authors", [])
        first_author = authors_list[0].split()[-1] if authors_list else "Unknown"
        citation = f"[{first_author} et al.]({arxiv_url})" if arxiv_url else ""
        for n in nug_list[:4]:
            nugget_lines.append(f"- {n} {citation}")

    nugget_text = "\n".join(nugget_lines)

    prompt = f"""
You are writing the Related Work section for an academic paper on:
"{query}"

Research strategy: {plan.get('strategy', '')}

Sub-questions to address:
{chr(10).join(f"- {q}" for q in plan.get('sub_questions', []))}

Available papers to cite (use ONLY these URLs):
{reference_context}

Key research nuggets extracted from papers:
{nugget_text}

Instructions:
1. Organize into 3-4 thematic ## subsections relevant to "{query}"
2. Each subsection should group and compare related methods
3. Cite every claim using [Author et al.](URL) format with URLs from the list above
4. Be specific: mention dataset names, metric values, and architectural details from the nuggets
5. Write at least 600 words
"""

    report_body = llm.invoke(prompt, system=SYSTEM)
    return report_body.strip()