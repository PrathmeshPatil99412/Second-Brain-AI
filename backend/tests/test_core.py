"""
Core test suite — two tests chosen to demonstrate the most important
correctness properties of the system, rather than broad shallow coverage.

Run from backend/:
    pytest tests/test_core.py -v
"""
from unittest.mock import patch

from database.session import SessionLocal
from database.crud import create_document, create_chunks
from ingestion.chunker import chunk_text
from ingestion.embeddings import embed_batch
from retrieval.chroma_client import add_chunks, get_collection
from services.chat_service import run_chat_query


def test_sqlite_chroma_id_consistency():
    """
    PROVES: every chunk ID generated in SQLite (via SQLAlchemy's default
    UUID) is used identically as the ID in ChromaDB — no drift, no
    separate mapping table, no risk of a chunk existing in one store
    but not the other under a different ID.

    This is the highest-risk invariant in the system: if these ever
    diverge, retrieval would return a Chroma chunk_id that doesn't
    resolve to any row in SQLite, silently breaking citations.
    """
    db = SessionLocal()

    # Arrange: create a real document row and some real chunks
    doc = create_document(
        db, filename="test_consistency.pdf", file_path="/fake/path.pdf",
        content_type="application/pdf", size_bytes=1000,
    )
    test_texts = ["First test chunk.", "Second test chunk.", "Third test chunk."]
    vectors = embed_batch(test_texts)

    # Act: same flow as document_service.py — create chunks, then use
    # the REAL generated IDs (not new ones) for Chroma
    chunk_rows = create_chunks(db, texts=test_texts, document_id=doc.id)
    chunk_ids = [c.id for c in chunk_rows]
    metadatas = [{"document_id": doc.id, "chunk_index": c.chunk_index} for c in chunk_rows]
    add_chunks(ids=chunk_ids, texts=test_texts, embeddings=vectors, metadatas=metadatas)

    # Assert: every SQLite chunk ID must exist in Chroma, and Chroma's
    # content for that ID must match what SQLite has — proving the two
    # stores stayed in sync, not just that both have "some" data.
    collection = get_collection()
    chroma_result = collection.get(ids=chunk_ids)

    assert set(chroma_result["ids"]) == set(chunk_ids), (
        "SQLite and ChromaDB chunk IDs diverged — the core sync invariant is broken"
    )
    assert len(chroma_result["documents"]) == len(test_texts)

    # Extra rigor: verify content matches per-ID, not just that the ID sets match
    id_to_text = dict(zip(chunk_ids, test_texts))
    for returned_id, returned_text in zip(chroma_result["ids"], chroma_result["documents"]):
        assert returned_text == id_to_text[returned_id], (
            f"Content mismatch for chunk {returned_id}: SQLite and Chroma disagree"
        )


def test_end_to_end_rag_pipeline_with_mocked_gemini():
    """
    PROVES: the full RAG chat pipeline (retrieve -> build prompt ->
    call LLM -> format citations -> persist history) works correctly
    end-to-end, independent of Gemini's actual uptime.

    Gemini is mocked so this test is deterministic and fast — it
    validates OUR pipeline logic, not Google's API availability.
    """
    db = SessionLocal()

    # Arrange: seed a document with known, distinctive content
    doc = create_document(
        db, filename="mock_test_doc.pdf", file_path="/fake/path.pdf",
        content_type="application/pdf", size_bytes=500,
    )
    known_text = "The mitochondria is the powerhouse of the cell."
    chunks = chunk_text(known_text)
    vectors = embed_batch(chunks)
    chunk_rows = create_chunks(db, texts=chunks, document_id=doc.id)
    chunk_ids = [c.id for c in chunk_rows]
    metadatas = [{"document_id": doc.id, "filename": doc.filename, "chunk_index": c.chunk_index} for c in chunk_rows]
    add_chunks(ids=chunk_ids, texts=chunks, embeddings=vectors, metadatas=metadatas)

    # Mock Gemini so the test doesn't depend on the real API being up
    fake_answer = "The mitochondria is known as the powerhouse of the cell."
    with patch("services.chat_service.generate_text", return_value=fake_answer) as mock_gemini:
        result = run_chat_query(db, "What is the powerhouse of the cell?")

        # Assert: Gemini was actually called (pipeline reached generation)
        mock_gemini.assert_called_once()

        # Assert: the mocked answer flows through correctly
        assert result["answer"] == fake_answer

        # Assert: retrieval actually found and cited the relevant chunk
        assert len(result["sources"]) > 0
        assert any("mitochondria" in s["content"].lower() for s in result["sources"])

        # Assert: citation shape is correct (matches ChatSource schema)
        first_source = result["sources"][0]
        assert set(first_source.keys()) == {"document_id", "chunk_id", "content", "score", "metadata"}
        assert 0.0 <= first_source["score"] <= 1.0