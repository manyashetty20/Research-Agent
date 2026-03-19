# GenAI-Silo 🔬
### Agentic Deep Research System — DeepScholar Benchmark

An agentic pipeline that automatically generates **Related Work sections** for academic papers using a **Plan → Search → Read → Write → Verify** loop. Built to be evaluated on the [DeepScholar-Bench](https://github.com/guestrin-lab/deepscholar-bench) framework.

---

## How It Works

```
Query + Abstract
      │
      ▼
 [Planner Agent]  → generates sub-questions + search queries
      │
      ▼
 [Search Agent]   → queries arXiv + Semantic Scholar + Tavily
      │            → expands queries using LLM
      │            → reranks results using cross-encoder
      ▼
 [Reader Agent]   → extracts key nuggets (facts, metrics, claims) from each paper
      │
      ▼
 [Synthesizer]    → writes thematic Related Work section in Markdown
      │            → uses [Author et al.](https://arxiv.org/abs/ID) citations
      ▼
 [Verifier Agent] → audits claim–citation pairs, corrects hallucinations
      │
      ▼
 intro.md + paper.csv  →  DeepScholar-Bench Eval
```

---

## Benchmark Results

Evaluated on 10 queries from the DeepScholar-Bench dataset using `gpt-4o-mini` as judge:

| Metric | Score | What it measures |
|---|---|---|
| **organization** | 0.65 | Structure and logical flow of the Related Work section |
| **nugget_coverage** | 0.17 | Key facts/findings captured from ground-truth papers |
| **reference_coverage** | 0.06 | Overlap with exact ground-truth references |
| **cite_p** | 0.24 | Citation precision — claims supported by cited papers |

---

## Project Structure

```
GenAI-Silo/
└── deep_research_agent/
    ├── config.py               # LLM, retrieval, agent settings
    ├── llm.py                  # LLM client (OpenAI)
    ├── run.py                  # CLI entry point
    ├── .env                    # API keys (never commit this)
    ├── agents/
    │   ├── planner.py          # Query → sub-questions + search queries
    │   ├── search_agent.py     # arXiv + Semantic Scholar + Tavily + rerank
    │   ├── reader.py           # Nugget extraction from papers
    │   ├── synthesizer.py      # Cited Markdown report writer
    │   └── verifier.py         # Claim–citation auditor
    ├── retrieval/
    │   ├── arxiv_client.py     # arXiv API client (rate-limit aware)
    │   ├── semantic_scholar.py # Semantic Scholar API (returns arXiv IDs)
    │   └── embeddings.py       # Sentence-transformer embeddings + reranker
    ├── graph/
    │   └── workflow.py         # Orchestrates the full pipeline
    └── training/
        ├── prepare_writer_dataset.py
        └── train_lora_writer.py
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/manyashetty20/GenAI-Silo.git
cd GenAI-Silo
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r deep_research_agent/requirements.txt
```

### 3. Set up API keys

Create `deep_research_agent/.env`:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

> **Important:** Never commit your `.env` file. It's already in `.gitignore`.

### 4. Set up DeepScholar-Bench (for evaluation)

```bash
# From the parent directory (one level above GenAI-Silo)
git clone https://github.com/guestrin-lab/deepscholar-bench.git
```

### 5. Download NLTK data (required for eval)

```bash
python3 -c "import nltk; nltk.download('punkt_tab')"
```

---

## Configuration

Edit `deep_research_agent/config.py` to tune the system:

```python
@dataclass
class LLMConfig:
    backend: str = "openai"
    model: str = "gpt-4o-mini"       # use gpt-4o for better quality
    base_url: str = "https://api.openai.com/v1"
    api_key: str = os.getenv("OPENAI_API_KEY")

@dataclass
class RetrievalConfig:
    max_arxiv_results: int = 15
    max_semantic_scholar_results: int = 15
    top_k_after_rerank: int = 15      # increase for more papers
    max_tavily_results: int = 8
```

---

## Usage

### Run a single query

```bash
cd GenAI-Silo
python -m deep_research_agent.run \
  --query "Universal Domain Adaptation for Semantic Segmentation" \
  --end-date 2025-05-28 \
  --output report.md
```

### Run batch for DeepScholar evaluation

From the **parent directory** (one level above `GenAI-Silo`):

```bash
python run_batch.py
```

This reads the first 10 queries from the DeepScholar dataset and saves outputs to:
```
GenAI-Silo/results/deepscholar_base/
    0/
        intro.md      ← generated Related Work section
        paper.csv     ← retrieved papers (id, title, snippet)
    1/
    2/ ...
```

### Run evaluation

```bash
cd deepscholar-bench
export OPENAI_API_KEY="sk-..."

python -m eval.main \
  --modes deepscholar_base \
  --evals organization nugget_coverage reference_coverage cite_p \
  --input-folder ../GenAI-Silo/results/deepscholar_base \
  --output-folder ../eval_outputs \
  --dataset-path dataset/related_works_combined.csv \
  --model-name gpt-4o-mini
```

---

## Output Format

Each generated `intro.md` uses markdown hyperlink citations:

```markdown
## Domain Adaptation Methods

Recent work on UDA for semantic segmentation has focused on pseudo-label refinement.
[Pan et al.](https://arxiv.org/abs/2211.07525) propose MoDA, which uses object motion
cues to align source and target domains, achieving significant improvements on
GTA5→Cityscapes benchmarks.

[Zhao et al.](https://arxiv.org/abs/2301.12345) extend this with a self-training
framework that combines contrastive learning with Fourier-based style adaptation...
```

And `paper.csv`:
```
id,title,snippet
2211.07525,MoDA: Leveraging Motion Priors...,"Abstract text..."
```

---

## run_batch.py

Place this file in the **parent directory** (alongside `GenAI-Silo/` and `deepscholar-bench/`):

```python
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
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `cite_p = 0` | Check `intro.md` uses `[Author](https://arxiv.org/abs/ID)` format, not `[1]` |
| `No results to save` | Make sure `--input-folder` points to `results/deepscholar_base` not its parent |
| `ArXiv rate limit (429)` | The client auto-waits and retries. If it keeps failing, wait a few minutes and re-run |
| `OPENAI_API_KEY not set` | Run `export OPENAI_API_KEY="sk-..."` or add to `~/.zshrc` |
| `punkt_tab not found` | Run `python3 -c "import nltk; nltk.download('punkt_tab')"` |
| `citations_for_cite_quality is None` | Apply the fix in `deepscholar-bench/eval/parsers/deepscholar_base.py` — see Technical Report |

---

## Known Limitations

- **reference_coverage is low (0.06)** — the system retrieves topically relevant papers but rarely the exact papers cited by the original authors. Improving this requires either direct reference lookup via Semantic Scholar's references API or training a specialized retriever.
- **arXiv rate limits** — running many queries in quick succession triggers HTTP 429. The client handles this automatically but slows down the pipeline.
- **10-query evaluation** — scores have high variance at this sample size. Run 50+ queries for reliable numbers.

---

## Dependencies

- `openai` — LLM backend
- `arxiv` — arXiv search client
- `sentence-transformers` — embeddings + reranker
- `tavily-python` — web search
- `langchain-openai` — LLM invocation
- `pandas` — data handling

Install all: `pip install -r deep_research_agent/requirements.txt`
