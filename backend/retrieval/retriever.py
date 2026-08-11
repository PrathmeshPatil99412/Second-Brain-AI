"""
Core retrieval logic: embed a query, search ChromaDB, enrich with SQLite data.

WHAT THIS FILE DOES:
Takes a natural-language query string, returns the top-K most relevant
chunks — each enriched with the full chunk text, filename, and score.
Used by BOTH chat.py (as context for Gemini) and search.py (as direct results).
"""
from sqlalchemy.orm import Session

from database import crud
from ingestion.embeddings import embed_text
from retrieval.chroma_client import query_collection
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def retrieve_top_k(db: Session, query: str, k: int | None = None,use_reranking: bool = False) -> list[dict]:
    """
    Embed the query, search Chroma, and return enriched results.

    Returns a list of dicts, each shaped like:
        {
            "document_id": str,
            "chunk_id": str,
            "content": str,
            "score": float,       # 0.0-1.0, higher = more relevant
            "metadata": dict,     # e.g. {"filename": ..., "chunk_index": ...}
        }
    """
    top_k = k or settings.top_k_default
    fetch_k = top_k * 4 if use_reranking else top_k  # fetch more candidates when reranking

    # Step 1: embed the query using the SAME model used for chunks
    query_vector = embed_text(query)

    # Step 2: search Chroma
    raw_results = query_collection(query_vector, top_k=fetch_k)

    # Chroma returns nested lists (supports batch queries) — we only sent
    # one query, so we always take index [0] to unwrap the single result set.
    ids = raw_results["ids"][0]
    documents = raw_results["documents"][0]
    distances = raw_results["distances"][0]
    metadatas = raw_results["metadatas"][0]

    if not ids:
        logger.info(f"No results found for query: {query!r}")
        return []

    # Step 3: convert Chroma's "distance" (lower = more similar) into a
    # 0-1 "score" (higher = more similar) — matches your schema's
    # score: float = Field(..., ge=0.0, le=1.0) expectation.
    # Using a simple inverse — good enough for a hackathon, not perfectly
    # calibrated, but monotonic (closer chunks always score higher).
    results = []
    for chunk_id, content, distance, metadata in zip(ids, documents, distances, metadatas):
        score = 1.0 / (1.0 + distance)  # maps distance [0, inf) -> score (0, 1]

        results.append({
            "document_id": metadata.get("document_id", ""),
            "chunk_id": chunk_id,
            "content": content,
            "score": round(score, 4),
            "metadata": metadata,
        })

    if use_reranking:
        from retrieval.reranker import rerank
        results = rerank(query, results, top_k)
        logger.info(f"Reranked {len(results)} results for query: {query!r}")
    else:
        logger.info(f"Retrieved {len(results)} chunks for query: {query!r}")

    return results

if __name__ == "__main__":
    # Standalone test — run from backend/:
    #   python -m retrieval.retriever
    # REQUIRES: at least one document already uploaded via the API
    # (so Chroma actually has real data to search, not the empty/fake test data)
    from database.session import SessionLocal

    db = SessionLocal()
    test_query = "What is a large language model?"  # adjust to match your uploaded PDF's topic
    results = retrieve_top_k(db, test_query, k=3,use_reranking=True)

    print(f"Query: {test_query}")
    print(f"Results: {len(results)}\n")
    for r in results:
        print(f"Score: {r['score']} | Doc: {r['metadata'].get('filename')}")
        print(f"Content: {r['content'][:150]}...\n")