"""
Configuration for the agentic deep research system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ------------------------------
# LLM CONFIG
# ------------------------------

@dataclass
class LLMConfig:

    # backend options:
    # openai | groq | hf_router
    backend: str = "openai"

    model: str = "gpt-4o-mini"

    base_url: str = "https://api.openai.com/v1"

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    temperature: float = 0.1
    max_tokens: int = 4096


# ------------------------------
# EMBEDDINGS
# ------------------------------

@dataclass
class EmbeddingConfig:
    """
    Embedding model used for retrieval ranking
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Better scientific embedding (optional)
    # model_name = "allenai/scibert_scivocab_uncased"

    device: str = "cpu"


# ------------------------------
# RERANKER CONFIG
# ------------------------------

@dataclass
class RerankConfig:
    """
    Cross encoder reranker for better paper selection
    """

    model_name: str = "BAAI/bge-reranker-base"
    device: str = "cpu"


# ------------------------------
# RETRIEVAL
# ------------------------------

@dataclass
class RetrievalConfig:

    max_arxiv_results: int = 20
    max_semantic_scholar_results: int = 15

    # Tavily search results
    max_tavily_results: int = 8

    # final reranked papers
    top_k_after_rerank: int = 15

    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )

    end_date: Optional[str] = None


# ------------------------------
# AGENT LOOP
# ------------------------------

@dataclass
class AgentConfig:

    max_plan_iterations: int = 3
    max_verify_iterations: int = 2

    # ensures claim traceability
    min_citations_per_claim: int = 1


# ------------------------------
# ROOT CONFIG
# ------------------------------

@dataclass
class Config:

    llm: LLMConfig = field(default_factory=LLMConfig)

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    rerank: RerankConfig = field(default_factory=RerankConfig)

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    agent: AgentConfig = field(default_factory=AgentConfig)


def get_config() -> Config:
    return Config()