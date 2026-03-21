# Research Agent 🔬

A multi-agent pipeline that automatically generates **Related Work sections** for academic papers, evaluated against the **DeepScholar-Bench** benchmark.

Built on top of [GenAI-Silo](https://github.com/manyashetty20/GenAI-Silo) and [DeepScholar-Bench](https://github.com/guestrin-lab/deepscholar-bench).

---

## What This Does

Given a paper title and abstract, the system:
1. Plans a research strategy and generates search queries
2. Fetches papers directly cited by the query paper via Semantic Scholar References API
3. Retrieves additional relevant papers from arXiv, Semantic Scholar, and the web
4. Extracts key findings (nuggets) from each paper
5. Writes a structured, cited Related Work section in Markdown
6. Verifies citations and corrects unsupported claims

The output is scored by DeepScholar-Bench across four metrics.

---

## Workspace Layout

```
Research-Agent/
├── GenAI-Silo/                        ← Our agentic system
│   ├── deep_research_agent/
│   │   ├── config.py                  ← All settings (LLM, retrieval, agents)
│   │   ├── llm.py                     ← OpenAI client
│   │   ├── run.py                     ← CLI entry point
│   │   ├── .env                       ← API keys — never commit this
│   │   ├── agents/
│   │   │   ├── planner.py             ← Breaks query into search queries
│   │   │   ├── search_agent.py        ← Retrieves + reranks papers
│   │   │   ├── reader.py              ← Extracts nuggets from papers
│   │   │   ├── synthesizer.py         ← Writes the Related Work section
│   │   │   └── verifier.py            ← Audits and corrects citations
│   │   ├── retrieval/
│   │   │   ├── arxiv_client.py        ← arXiv search (rate-limit aware)
│   │   │   ├── semantic_scholar.py    ← Semantic Scholar search
│   │   │   └── embeddings.py          ← Reranker
│   │   └── graph/
│   │       └── workflow.py            ← Orchestrates the full pipeline
│   └── results/
│       └── deepscholar_base/          ← Pipeline outputs go here
│           ├── 0/
│           │   ├── intro.md           ← Generated Related Work section
│           │   └── paper.csv         ← Retrieved papers
│           ├── 1/ ...
│
├── deepscholar-bench/                 ← Evaluation framework
│   ├── dataset/
│   │   └── related_works_combined.csv ← 6,323 ground-truth papers
│   └── eval/                          ← Scoring scripts
│
├── eval_outputs/                      ← Eval scores saved here
├── run_batch.py                       ← Runs pipeline on multiple queries
└── venv/                              ← Python virtual environment
```

---

## Current Scores

Evaluated on 10 queries using `gpt-4o-mini` as judge:

| Metric | Score | What it measures |
|---|---|---|
| **organization** | 0.55 | How well-structured the output is |
| **nugget_coverage** | 0.11 | Key facts captured from ground-truth papers |
| **reference_coverage** | 0.25 | Overlap with exact ground-truth references |
| **cite_p** | 0.28 | Are cited papers actually supporting the claims? |

### Score History

| Run | Key Change | organization | nugget_coverage | reference_coverage | cite_p |
|---|---|---|---|---|---|
| 1 | Everything broken | 0 | 0 | 0 | 0 |
| 2 | First working run | 0.55 | 0.16 | 0.00 | 0.13 |
| 3 | Fixed OpenAI backend | 0.55 | 0.16 | 0.00 | 0.16 |
| 4 | Dynamic relevance scoring | 0.40 | 0.12 | 0.03 | 0.24 |
| 5 | Thematic subsections | 0.50 | 0.17 | 0.03 | 0.20 |
| 6 | End-date filtering | 0.55 | 0.13 | 0.03 | 0.19 |
| 7 | Abstract in query | 0.65 | 0.14 | 0.06 | 0.18 |
| 8 | arXiv rate limit fix | 0.60 | 0.10 | 0.05 | 0.23 |
| 9 | S2 References API | 0.55 | 0.11 | **0.25** | **0.28** |

> reference_coverage jumped 4x (0.06 → 0.25) by directly fetching papers cited by the query paper via Semantic Scholar's References API.

---

## Setup (Do This Once)

### 1. Clone the repo

```bash
git clone https://github.com/manyashetty20/Research-Agent.git
cd Research-Agent
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r GenAI-Silo/deep_research_agent/requirements.txt
pip install -r deepscholar-bench/requirements.txt
```

### 4. Set up API keys

Create `GenAI-Silo/deep_research_agent/.env`:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

Add to your shell so you don't have to export every time:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

### 5. Download NLTK data

```bash
python3 -c "import nltk; nltk.download('punkt_tab')"
```

### 6. Apply the eval parser fix

Open `deepscholar-bench/eval/parsers/deepscholar_base.py`, find `_load_file()` and add one line after `self.clean_text, self.docs = self._to_autoais(...)`:

```python
self.citations_for_cite_quality = [
    (doc.get("title", ""), doc.get("sent", ""))
    for doc in self.docs
]
```

> This fixes a bug in the original eval code that causes `cite_p` to crash.

---

## Running the System

Every time you work on this, start from the repo root with the venv active:

```bash
source venv/bin/activate
```

### Full pipeline + eval (the usual flow)

```bash
# Step 1 — clear old results
rm -rf GenAI-Silo/results/deepscholar_base/

# Step 2 — run pipeline for 10 queries (takes ~15-20 min)
python run_batch.py

# Step 3 — run evaluation
cd deepscholar-bench
python -m eval.main \
  --modes deepscholar_base \
  --evals organization nugget_coverage reference_coverage cite_p \
  --input-folder ../GenAI-Silo/results/deepscholar_base \
  --output-folder ../eval_outputs \
  --dataset-path dataset/related_works_combined.csv \
  --model-name gpt-4o-mini

# Step 4 — go back to root when done
cd ..
```

### Test a single query

```bash
cd GenAI-Silo
python -m deep_research_agent.run \
  --query "Universal Domain Adaptation for Semantic Segmentation" \
  --end-date 2025-05-28 \
  --arxiv-id 2505.22458v1 \
  --output report.md
cd ..
```

---

## Key Files to Know

If you're making changes, these are the files that matter most:

| File | What to change here |
|---|---|
| `GenAI-Silo/deep_research_agent/config.py` | LLM model, number of papers to retrieve |
| `GenAI-Silo/deep_research_agent/agents/planner.py` | How search queries are generated |
| `GenAI-Silo/deep_research_agent/agents/search_agent.py` | Retrieval logic, S2 references, query expansion |
| `GenAI-Silo/deep_research_agent/retrieval/arxiv_client.py` | arXiv search, rate limiting |
| `GenAI-Silo/deep_research_agent/retrieval/semantic_scholar.py` | Semantic Scholar search |
| `GenAI-Silo/deep_research_agent/agents/synthesizer.py` | How the Related Work is written |
| `run_batch.py` | Number of queries, dataset used |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| All scores are 0 | Make sure `--input-folder` ends with `/deepscholar_base` not its parent |
| `cite_p = 0` | Check `intro.md` uses `[Author](https://arxiv.org/abs/ID)` not `[1]` style |
| `OPENAI_API_KEY not set` | Run `export OPENAI_API_KEY="sk-..."` or add to `~/.zshrc` |
| arXiv rate limits (429/503) | Client retries automatically — just wait, it will continue |
| S2 rate limits (429) | Client retries with backoff — just wait, it will continue |
| `punkt_tab not found` | Run `python3 -c "import nltk; nltk.download('punkt_tab')"` |
| `cd: no such file or directory` | Check `pwd` — you are probably already inside that folder |
| Query failed, skipping | Check your OpenAI key is valid and has credits |

---

## Things to Improve

Open areas if you want to contribute:

- **nugget_coverage** — reference papers fetched from S2 sometimes lack abstracts, giving the synthesizer less content. Supplement missing abstracts by fetching them separately.
- **organization** — recover the 0.65 score from Run 7 while keeping the reference_coverage gains.
- **More queries** — currently evaluating on 10. Run 50+ for reliable scores.
- **Better reranking** — try a domain-specific reranker trained on scientific text.
- **LoRA fine-tuning** — the `training/` folder has scripts to fine-tune a local writer model on the DeepScholar dataset.
