# Research Agent 🔬

A multi-agent pipeline that automatically generates **Related Work sections** for academic papers, evaluated against the **DeepScholar-Bench** benchmark.


---

## Overview

Given a paper title and abstract, the pipeline:

1. **Plans** a research strategy and generates targeted search queries
2. **Fetches** papers directly cited by the query paper via the Semantic Scholar References API
3. **Retrieves** additional relevant papers from arXiv, Semantic Scholar, and the web
4. **Extracts** key findings (nuggets) from each paper
5. **Writes** a structured, cited Related Work section in Markdown
6. **Verifies** citations and corrects any unsupported claims

The output is automatically scored by DeepScholar-Bench across four metrics.

---

## Workspace Layout

```
Research-Agent/
├── GenAI-Silo/                        ← Agentic pipeline
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
| **organization** | 0.65 | How well-structured the output is |
| **nugget_coverage** | 0.14 | Key facts captured from ground-truth papers |
| **reference_coverage** | 0.25 | Overlap with exact ground-truth references |
| **cite_p** | 0.25 | Whether cited papers actually support the claims |
| **G-Mean** | 0.28 | Overall composite score |



> `reference_coverage` jumped 4x (0.06 → 0.25) in Run 9 by directly fetching papers cited by the query paper via Semantic Scholar's References API. Run 10 recovers the `organization` score of 0.65 while maintaining those gains.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/manyashetty20/Research-Agent.git
cd Research-Agent
```

### 2. Create a virtual environment

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

To avoid re-exporting keys every session, add them to your shell profile:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

### 5. Download NLTK data

```bash
python3 -c "import nltk; nltk.download('punkt_tab')"
```

### 6. Apply the eval parser fix

Open `deepscholar-bench/eval/parsers/deepscholar_base.py`, find `_load_file()`, and add one line after `self.clean_text, self.docs = self._to_autoais(...)`:

```python
self.citations_for_cite_quality = [
    (doc.get("title", ""), doc.get("sent", ""))
    for doc in self.docs
]
```

This fixes a bug in the original eval code that causes `cite_p` to crash.

---

## Running the Pipeline

Always start from the repo root with your virtual environment active:

```bash
source venv/bin/activate
```

### Full pipeline + eval (recommended flow)

```bash
# Step 1 — clear old results
rm -rf GenAI-Silo/results/deepscholar_base/

# Step 2 — run the pipeline on 10 queries (~15–20 min)
python run_batch.py

# Step 3 — evaluate
cd deepscholar-bench
python -m eval.main \
  --modes deepscholar_base \
  --evals organization nugget_coverage reference_coverage cite_p \
  --input-folder ../GenAI-Silo/results/deepscholar_base \
  --output-folder ../eval_outputs \
  --dataset-path dataset/related_works_combined.csv \
  --model-name gpt-4o-mini

# Step 4 — return to root
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

## Key Files

| File | What to change here |
|---|---|
| `config.py` | LLM model, number of papers to retrieve |
| `agents/planner.py` | How search queries are generated |
| `agents/search_agent.py` | Retrieval logic, S2 references, query expansion |
| `retrieval/arxiv_client.py` | arXiv search and rate limiting |
| `retrieval/semantic_scholar.py` | Semantic Scholar search |
| `agents/synthesizer.py` | How the Related Work section is written |
| `run_batch.py` | Number of queries, dataset used |

All files above live under `GenAI-Silo/deep_research_agent/` unless noted.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| All scores are 0 | Ensure `--input-folder` ends with `/deepscholar_base`, not its parent |
| `cite_p = 0` | Check that `intro.md` uses `[Author](https://arxiv.org/abs/ID)` format, not `[1]` style |
| `OPENAI_API_KEY not set` | Run `export OPENAI_API_KEY="sk-..."` or add it to `~/.zshrc` |
| arXiv 429/503 errors | The client retries automatically — just wait |
| Semantic Scholar 429 errors | The client retries with backoff — just wait |
| `punkt_tab not found` | Run `python3 -c "import nltk; nltk.download('punkt_tab')"` |
| `cd: no such file or directory` | Run `pwd` — you may already be inside that folder |
| Query failed / skipping | Verify your OpenAI key is valid and has available credits |

---

