import csv
import subprocess
import sys

DATASET = "deepscholar-bench/dataset/related_works_combined.csv"
OUTPUT_DIR = "results/deepscholar_base"
NUM_QUERIES = 10

queries = []
with open(DATASET) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= NUM_QUERIES:
            break
        pub_date = row.get("publication_date", "")[:10]
        queries.append((
            i,
            row["arxiv_id"],
            row["title"].replace("\n", " ").strip(),
            row["abstract"].replace("\n", " ").strip(),
            pub_date
        ))

for query_id, arxiv_id, title, abstract, pub_date in queries:
    print(f"\n{'='*60}")
    print(f"Running query {query_id}: {title}")
    print(f"End date: {pub_date}")
    print(f"{'='*60}")

    # Combine title + abstract as the query for richer retrieval
    full_query = f"{title}. {abstract[:300]}"

    cmd = [
        sys.executable, "-m", "deep_research_agent.run",
        "--query", full_query,
        "--output-dir", OUTPUT_DIR,
        "--query-id", str(query_id),
    ]

    if pub_date and len(pub_date) == 10:
        cmd += ["--end-date", pub_date]

    result = subprocess.run(cmd, cwd="GenAI-Silo")

    if result.returncode != 0:
        print(f"WARNING: query {query_id} failed, skipping...")

print("\nDone! Now run eval.")