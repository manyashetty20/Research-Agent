"""
Dataset preparation for writer LoRA fine-tuning.

High-level idea:
- Input: research query + nuggets + references.
- Output: related-work style section with inline [1], [2], ... citations.

This script adapts DeepScholar-bench data into (input_text, output_text) pairs for SFT.
It extracts query (from title/abstract), nuggets (from abstract/related works), 
and references (from the related works text).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import csv
import re
from collections import defaultdict


@dataclass
class WriterExample:
    input_text: str
    output_text: str


def build_prompt(query: str, nuggets_block: str, references_block: str) -> str:
    """Construct the writer prompt used for training and inference."""
    return (
        "You are writing a Related Work section for a scientific paper.\n"
        "Use the following nuggets (facts) and references. Write well-structured "
        "paragraphs and include inline citations like [1], [2], ... matching the "
        "reference indices.\n\n"
        f"[QUERY]\n{query}\n\n"
        f"[NUGGETS]\n{nuggets_block}\n\n"
        f"[REFERENCES]\n{references_block}\n"
    )


def extract_citations_from_text(text: str) -> list[tuple[str, int]]:
    """Extract citation numbers and surrounding context from text."""
    citations = []
    # Find all [N] patterns
    for match in re.finditer(r'\[(\d+)\]', text):
        ref_num = int(match.group(1))
        # Get context around citation
        start = max(0, match.start() - 100)
        end = min(len(text), match.end() + 100)
        context = text[start:end].replace('\n', ' ').strip()
        citations.append((context, ref_num))
    return citations


def extract_references_from_text(text: str) -> dict[int, str]:
    """Extract [N] Title/Info patterns from text."""
    references = {}
    # Pattern: [N] Some citation text (until next [M] or end)
    pattern = r'\[(\d+)\]\s+([^[\n]+(?:\n(?!\[)[^[\n]*)*)'
    for match in re.finditer(pattern, text):
        ref_num = int(match.group(1))
        ref_text = match.group(2).strip()
        references[ref_num] = ref_text
    return references


def iter_from_deepscholar_csv(path: Path, limit: int = None) -> Iterable[WriterExample]:
    """
    Read from DeepScholar-bench CSV and generate training examples.
    
    Expected columns:
    - title (query)
    - abstract (source for nuggets)
    - pdf_related_works (target output)
    """
    count = 0
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if limit and count >= limit:
                break
            
            title = row.get("title", "").strip()
            abstract = row.get("abstract", "").strip()
            related_works = row.get("pdf_related_works", "").strip()
            
            # Skip if missing critical fields
            if not title or not related_works:
                continue
            
            # Use title as query
            query = title
            
            # Extract nuggets from abstract (split into sentences)
            nuggets = []
            if abstract:
                # Split into sentences
                sentences = [s.strip() for s in re.split(r'[.!?]+', abstract) if s.strip()]
                for i, sent in enumerate(sentences[:5]):  # Take first 5 sentences as nuggets
                    if len(sent) > 10:  # Skip very short sentences
                        nuggets.append(f"- {sent} [ref_{i+1}]")
            
            nuggets_block = "\n".join(nuggets) if nuggets else "- No specific nuggets provided"
            
            # Extract references from related works text
            references = extract_references_from_text(related_works)
            if not references:
                # Fallback: create dummy references
                references = {i: f"Reference {i}" for i in range(1, min(6, len(nuggets) + 1))}
            
            references_block = "\n".join(
                f"[{i}] {text[:200]}" for i, text in sorted(references.items())
            ) if references else "- No references"
            
            input_text = build_prompt(query, nuggets_block, references_block)
            output_text = related_works[:4000]  # Limit output length
            
            yield WriterExample(input_text=input_text, output_text=output_text)
            count += 1


def export_to_jsonl(examples: Iterable[WriterExample], out_path: Path) -> None:
    """Write examples to JSONL file with keys: input, output."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"input": ex.input_text, "output": ex.output_text}) + "\n")
            count += 1
    print(f"Wrote {count} examples to {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Prepare writer dataset JSONL from DeepScholar-bench CSV."
    )
    p.add_argument("--csv", type=str, required=True, help="Input CSV path (related_works_combined.csv)")
    p.add_argument("--out", type=str, required=True, help="Output JSONL path")
    p.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    args = p.parse_args()

    examples = iter_from_deepscholar_csv(Path(args.csv), limit=args.limit)
    export_to_jsonl(examples, Path(args.out))


