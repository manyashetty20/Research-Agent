"""Reader / Nugget Agent: extracts specific facts, metrics, and mechanisms from papers."""
import json
from typing import Any, Optional


SYSTEM = """You are a scientific reader. For each paper, extract 2-5 short "nuggets": specific facts, numbers, methods, or claims that would support a related-work synthesis. Each nugget must be grounded in the paper. Output a JSON array of objects, one per paper, with keys:
- "id": same as the paper id provided
- "nuggets": list of strings (each one sentence, no citation yet)

Output only valid JSON array, no markdown."""


def extract_nuggets(
    papers: list[dict],
    llm: Optional[Any] = None
) -> list[dict[str, Any]]:
    """For each paper, extract nuggets (facts/claims) for synthesis."""

    if llm is None:
        raise RuntimeError("LLM instance must be provided to extract_nuggets().")

    if not papers:
        return []

    # Batch into chunks to avoid token limits
    batch_size = 5
    results = []

    for i in range(0, len(papers), batch_size):

        batch = papers[i : i + batch_size]

        parts = []

        for p in batch:

            parts.append(
                f"Paper id: {p.get('id', '')}\n"
                f"Title: {p.get('title', '')}\n"
                f"Abstract/summary: {(p.get('summary') or '')[:1500]}\n"
            )

        prompt = "Extract nuggets from these papers:\n\n" + "\n---\n".join(parts)

        out = llm.invoke(prompt, system=SYSTEM)

        out = out.strip()

        if out.startswith("```"):
            out = out.split("```")[1]
            if out.startswith("json"):
                out = out[4:]

        out = out.strip()

        try:

            arr = json.loads(out)

            if isinstance(arr, list):
                results.extend(arr)
            else:
                results.append(arr)

        except json.JSONDecodeError:

            # fallback if LLM returns invalid JSON
            for p in batch:
                results.append(
                    {
                        "id": p.get("id"),
                        "nuggets": [p.get("title", "")]
                    }
                )

    return results