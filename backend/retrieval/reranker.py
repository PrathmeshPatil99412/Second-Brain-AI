"""
Cross-encoder reranking — second-stage relevance scoring.

WHY THIS EXISTS:
Vector search (ChromaDB) is fast but approximate — it compares
pre-computed embeddings independently, without directly comparing the
query and each candidate together. A cross-encoder is slower but more
accurate: it looks at the query and each chunk TOGETHER, producing a
more precise relevance score.

PATTERN: retrieve MORE candidates from vector search than needed (e.g.
top 20), then rerank them down to the final top-K (e.g. top 5) using
the cross-encoder. This gets speed (Chroma narrows 1000s down to 20
candidates fast) AND precision (cross-encoder refines those 20 into
the best 5).
"""
from sentence_transformers import CrossEncoder

from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # standard, fast, well-benchmarked reranker
_model = CrossEncoder(_MODEL_NAME)

logger.info(f"Loaded reranker model: {_MODEL_NAME}")


def rerank(query: str, results: list[dict], top_k: int) -> list[dict]:
    """
    Re-score and re-sort results using a cross-encoder, return the new top_k.
    `results` is the output of retriever.retrieve_top_k() (already has 'content').
    """
    if not results:
        return results

    pairs = [(query, r["content"]) for r in results]
    scores = _model.predict(pairs)  # raw relevance scores, higher = more relevant

    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)

    reranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
    return reranked[:top_k]