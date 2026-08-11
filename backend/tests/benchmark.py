"""
Benchmark script — produces resume-worthy metrics:
1. Latency: ingestion pipeline stages, retrieval, chat response time
2. Retrieval accuracy: top-K hit rate against a set of known queries

Run from backend/:
    python -m tests.benchmark
"""
import time

from database.session import SessionLocal
from ingestion.chunker import chunk_text
from ingestion.embeddings import embed_batch, embed_text
from ingestion.parser import extract_text
from retrieval.retriever import retrieve_top_k
from services.chat_service import run_chat_query


def benchmark_ingestion(pdf_path: str) -> dict:
    """Times each stage of the ingestion pipeline separately."""
    timings = {}

    start = time.perf_counter()
    text = extract_text(pdf_path)
    timings["parse_seconds"] = round(time.perf_counter() - start, 3)

    start = time.perf_counter()
    chunks = chunk_text(text)
    timings["chunk_seconds"] = round(time.perf_counter() - start, 3)

    start = time.perf_counter()
    vectors = embed_batch(chunks)
    timings["embed_seconds"] = round(time.perf_counter() - start, 3)

    timings["total_seconds"] = round(sum(timings.values()), 3)
    timings["num_chunks"] = len(chunks)
    timings["chars_processed"] = len(text)
    timings["chunks_per_second"] = round(len(chunks) / timings["embed_seconds"], 2) if timings["embed_seconds"] > 0 else 0

    return timings


def benchmark_retrieval(queries: list[str], k: int = 5, runs: int = 3) -> dict:
    """Times retrieve_top_k() over multiple queries, averaged over multiple runs."""
    db = SessionLocal()
    all_times = []

    for query in queries:
        for _ in range(runs):
            start = time.perf_counter()
            retrieve_top_k(db, query, k=k)
            all_times.append(time.perf_counter() - start)

    return {
        "avg_seconds": round(sum(all_times) / len(all_times), 4),
        "min_seconds": round(min(all_times), 4),
        "max_seconds": round(max(all_times), 4),
        "num_queries_tested": len(queries),
        "runs_per_query": runs,
    }


def benchmark_chat(queries: list[str]) -> dict:
    """Times full chat pipeline (retrieval + Gemini generation) — real latency users would feel."""
    db = SessionLocal()
    all_times = []

    for query in queries:
        start = time.perf_counter()
        run_chat_query(db, query)
        all_times.append(time.perf_counter() - start)

    return {
        "avg_seconds": round(sum(all_times) / len(all_times), 3),
        "min_seconds": round(min(all_times), 3),
        "max_seconds": round(max(all_times), 3),
        "num_queries_tested": len(queries),
    }


def benchmark_retrieval_accuracy(test_cases: list[dict], k: int = 5) -> dict:
    """
    Measures top-K retrieval accuracy.
    Each test_case: {"query": str, "expected_keyword": str}
    A "hit" = expected_keyword appears in the content of at least one of the top-K results.
    """
    db = SessionLocal()
    hits = 0
    results_log = []

    for case in test_cases:
        results = retrieve_top_k(db, case["query"], k=k)
        found = any(case["expected_keyword"].lower() in r["content"].lower() for r in results)
        hits += int(found)
        results_log.append({
            "query": case["query"],
            "expected_keyword": case["expected_keyword"],
            "hit": found,
            "top_score": results[0]["score"] if results else None,
        })

    accuracy = round(hits / len(test_cases) * 100, 1)
    return {"accuracy_percent": accuracy, "hits": hits, "total": len(test_cases), "details": results_log}


if __name__ == "__main__":
    print("=" * 60)
    print("INGESTION BENCHMARK")
    print("=" * 60)
    ingestion_results = benchmark_ingestion("../data/uploads/test.pdf")
    for k, v in ingestion_results.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("RETRIEVAL LATENCY BENCHMARK")
    print("=" * 60)
    sample_queries = [
        "What is a large language model?",
        "What is the difference between TCP/IP and OSI?",
        "What is CHAP authentication?",
    ]
    retrieval_results = benchmark_retrieval(sample_queries)
    for k, v in retrieval_results.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("CHAT LATENCY BENCHMARK (includes Gemini generation)")
    print("=" * 60)
    chat_results = benchmark_chat(sample_queries[:2])  # fewer, since Gemini calls are slower/costlier
    for k, v in chat_results.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("RETRIEVAL ACCURACY BENCHMARK")
    print("=" * 60)
    # Fill these in based on YOUR actual uploaded documents' content
    accuracy_test_cases = [
        {"query": "What is a large language model?", "expected_keyword": "Large Language Models"},
        {"query": "What is the Transformer architecture?", "expected_keyword": "self-attention"},
        {"query": "How many layers does TCP/IP have?", "expected_keyword": "4 layers"},
        {"query": "What is CHAP?", "expected_keyword": "CHAP"},
        {"query": "What does PPP stand for?", "expected_keyword": "PPP"},
    ]
    accuracy_results = benchmark_retrieval_accuracy(accuracy_test_cases)
    print(f"  Accuracy: {accuracy_results['accuracy_percent']}% ({accuracy_results['hits']}/{accuracy_results['total']})")
    for detail in accuracy_results["details"]:
        status = "✅" if detail["hit"] else "❌"
        print(f"    {status} '{detail['query']}' (score: {detail['top_score']})")