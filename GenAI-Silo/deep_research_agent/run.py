#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from pathlib import Path

from deep_research_agent.graph import run_research
from deep_research_agent.llm import get_llm


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", required=True)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--query-id", default="0")
    parser.add_argument("--arxiv-id", default=None)

    args = parser.parse_args()

    llm = get_llm()

    report, papers, stats = run_research(
        query=args.query,
        llm=llm,
        end_date=args.end_date,
        arxiv_id=args.arxiv_id,
    )

    if args.output:
        Path(args.output).write_text(report)

    if args.output_dir:
        query_dir = Path(args.output_dir) / str(args.query_id)
        query_dir.mkdir(parents=True, exist_ok=True)

        (query_dir / "intro.md").write_text(report)

        rows = []
        for p in papers:
            arxiv_id = p.get("arxiv_id") or ""
            if not arxiv_id:
                continue
            rows.append({
                "id": arxiv_id,
                "title": p.get("title", ""),
                "snippet": p.get("summary") or p.get("abstract", "")
            })

        pd.DataFrame(rows).to_csv(query_dir / "paper.csv", index=False)

    if not args.output and not args.output_dir:
        print(report)


if __name__ == "__main__":
    main()