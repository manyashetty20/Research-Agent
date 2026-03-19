"""Embeddings and reranking for retrieval (CPU-friendly models)."""
from typing import Optional
from sentence_transformers import CrossEncoder

_embedding_model = None


def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
    """Lazy-load sentence-transformers model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(model_name, device=device)
    return _embedding_model


def embed_texts(texts: list[str], model_name: Optional[str] = None, device: str = "cpu") -> list[list[float]]:
    """Embed a list of texts; returns list of vectors."""
    model = get_embeddings(model_name or "sentence-transformers/all-MiniLM-L6-v2", device)
    return model.encode(texts, convert_to_numpy=True).tolist()




def rerank(
    query: str,
    documents: list[dict],
    text_key: str = "summary",
    top_k: int = 15,
    model_name: Optional[str] = None,
    device: str = "cpu",
) -> list[dict]:
    """
    Cross-encoder reranking (HIGH QUALITY)
    """

    if not documents:
        return []

    model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"

    model = CrossEncoder(model_name, device=device)

    pairs = [
        (query, d.get(text_key) or d.get("title", ""))
        for d in documents
    ]

    scores = model.predict(pairs)

    indexed = list(zip(scores, documents))
    indexed.sort(key=lambda x: x[0], reverse=True)

    top = indexed[:top_k]

    out = []
    for score, doc in top:
        d = dict(doc)
        d["score"] = float(score)
        out.append(d)

    return out