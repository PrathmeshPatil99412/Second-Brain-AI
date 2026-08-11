"""
Before/after comparison: vector-only retrieval vs. cross-encoder reranked.

Run from backend/:
    python -m tests.benchmark_reranking
"""
import time

from database.session import SessionLocal
from retrieval.retriever import retrieve_top_k


def compare(query: str, k: int = 5):
    db = SessionLocal()

    print(f"\n{'=' * 70}")
    print(f"QUERY: {query}")
    print("=" * 70)

    # Without reranking
    start = time.perf_counter()
    baseline = retrieve_top_k(db, query, k=k, use_reranking=False)
    baseline_time = time.perf_counter() - start

    print(f"\n--- WITHOUT reranking ({round(baseline_time * 1000, 2)}ms) ---")
    for i, r in enumerate(baseline, 1):
        print(f"  #{i} | vector_score={r['score']} | {r['content'][:90]}...")

    # With reranking
    start = time.perf_counter()
    reranked = retrieve_top_k(db, query, k=k, use_reranking=True)
    reranked_time = time.perf_counter() - start

    print(f"\n--- WITH reranking ({round(reranked_time * 1000, 2)}ms) ---")
    for i, r in enumerate(reranked, 1):
        print(f"  #{i} | rerank_score={round(r['rerank_score'], 4)} (vector_score={r['score']}) | {r['content'][:90]}...")

    # Did the ranking order actually change?
    baseline_order = [r["chunk_id"] for r in baseline]
    reranked_order = [r["chunk_id"] for r in reranked]
    order_changed = baseline_order != reranked_order

    print(f"\nOrder changed by reranking: {order_changed}")
    print(f"Latency overhead: +{round((reranked_time - baseline_time) * 1000, 2)}ms")


if __name__ == "__main__":
    test_queries = [
        "What is a large language model?",
        "What is the difference between TCP/IP and OSI?",
        "How does self-attention work in transformers?",
    ]

    for q in test_queries:
        compare(q)