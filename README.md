# 🧠 Second Brain AI

![Banner](assets/banner.png)

An AI-powered Personal Knowledge Management (PKM) system that transforms your documents and notes into a searchable, conversational knowledge base using Retrieval-Augmented Generation (RAG).

Upload PDFs or notes, search semantically, chat with your knowledge, generate summaries, and organize information—all in one place.

---

## ✨ Features

- 📄 Upload PDF documents
- 📝 Create and store notes
- 🔍 Semantic search across all knowledge
- 💬 Chat with your documents using RAG
- 📚 Source citations for every response
- 📖 AI-generated document summaries
- 🏷️ Automatic document tagging
- 📊 Dashboard with knowledge base statistics
- 🗂️ Document management (view, list, delete)

---

## 🏗️ System Architecture

To understand the core mechanics of the **Second Brain AI** platform, review the architecture and data flow diagrams detailed below:

### 🌐 High-Level System Architecture
The overall architecture maps the interaction between the Streamlit UI frontend, the FastAPI backend ecosystem, and the underlying storage engines (SQLite & ChromaDB).

![System Architecture](assets/SecondBrain-Asset-Architecture.png)

---

### 📥 Ingestion Pipeline
The ingestion framework handles document parsing, chunking strategy via LangChain, embedding generation, and simultaneous population of relational metadata and vector storage.

![Asset Ingestion Pipeline](assets/SecondBrain-Asset-InjestionPipeline-V2.png)

---

### 🔍 Retrieval Pipeline
The query processing and RAG execution flow—spanning semantic vector search, context synthesis, prompt formatting, and final response generation using Google Gemini.

![Asset Retrieval Pipeline](assets/SecondBrain-Asset-RetrivalPipeline.png)



## ⚙️ Tech Stack

### Frontend
- Streamlit

### Backend
- FastAPI
- Uvicorn

### AI & RAG
- Google Gemini
- sentence-transformers
- LangChain Text Splitter

### Databases
- ChromaDB
- SQLite
- SQLAlchemy

### PDF Processing
- PyMuPDF

---

## 📂 Project Structure

```
backend/
├── api/
├── services/
├── ingestion/
├── retrieval/
├── intelligence/
├── database/
├── vectorstore/
├── utils/
└── main.py

frontend/
└── app.py

data/
├── uploads/
└── chroma_db/
```

---


## 🚀 Running the Project

### Clone

```bash
git clone <repository-url>
cd second-brain-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key
```

### Start Backend

```bash
cd backend
uvicorn main:app --reload
```

### Start Frontend

In a separate terminal:

```bash
cd frontend
streamlit run app.py
```
Both the backend (`http://127.0.0.1:8000`) and frontend (`http://localhost:8501`) must be running simultaneously.

---

## 📡 API Overview

| Endpoint | Description |
|----------|-------------|
| POST `/api/v1/documents/upload` | Upload a PDF |
| POST `/api/v1/notes` | Create a note |
| POST `/api/v1/chat/query` | Chat with knowledge base |
| POST `/api/v1/search` | Semantic search |
| GET `/api/v1/documents` | List documents |
| GET `/api/v1/documents/{id}` | Document details |
| DELETE `/api/v1/documents/{id}` | Delete document |
| POST `/api/v1/summary/{id}` | Generate summary + tags |
| GET `/api/v1/dashboard` | Dashboard statistics |
| GET `/health` | Health check |

Full interactive API docs available at `/docs` once the backend is running.

---

## 🧪 Testing & Performance

- Targeted `pytest` suite (`backend/tests/test_core.py`) covering:
  - Cross-store (SQLite ↔ ChromaDB) chunk ID consistency
  - Semantic retrieval discrimination across unrelated documents
  - A mocked end-to-end RAG pipeline (retrieval independent of LLM uptime)
- Ingestion and retrieval benchmarking (`backend/tests/benchmark.py`):
  - ~50 chunks/sec ingestion throughput
  - Sub-15ms average retrieval latency
  - 100% top-K retrieval accuracy on a representative query set
- Cross-encoder reranking benchmark (`backend/tests/benchmark_reranking.py`):
  - Measures precision/latency trade-off of an optional reranking stage against baseline vector retrieval

Run tests:
```bash
cd backend
pytest tests/test_core.py -v
```

---

## 📊 Current Capabilities

- ✅ PDF ingestion
- ✅ Note ingestion
- ✅ Text extraction
- ✅ Chunking
- ✅ Local embeddings
- ✅ ChromaDB integration
- ✅ SQLite metadata
- ✅ Semantic search
- ✅ RAG chat with citations
- ✅ AI summaries
- ✅ Auto-tagging
- ✅ Dashboard
- ✅ Full Streamlit frontend integration
- ✅ Cross-encoder reranking (benchmarked, opt-in)

---

## 🎯 Key Design Decisions

- **Local embeddings, not Gemini embeddings** — avoids API rate limits/cost on a high-frequency operation (embedding runs on every chunk of every upload).
- **UUID-matched IDs across SQLite and ChromaDB** — every chunk's ID is generated once (via SQLAlchemy) and reused identically in Chroma, avoiding a separate ID-mapping table.
- **Thin routers, fat services** — API routers only handle HTTP concerns; orchestration logic lives in `services/`, making each piece independently testable.
- **Graceful LLM failure handling** — Gemini API overload (503) returns a clear fallback message instead of crashing the request.
- **Reranking kept opt-in, not default** — benchmarking showed cross-encoder reranking adds significant latency (~400-500ms on CPU) with mixed precision gains on a small corpus; kept as an available capability rather than a default trade-off.

---


## 🔮 Future Enhancements

- Wire reranking into live chat/search endpoints (currently standalone, benchmarked)
- Knowledge Graph visualization
- Hybrid search (Keyword + Vector)
- OCR support for scanned PDFs
- Multi-user authentication
- Provider-agnostic LLM layer (beyond Gemini)
- Docker deployment
- Cloud storage support

---

## 🛡️ Error Handling

- Graceful fallback when Gemini API is unavailable
- Metadata consistency between SQLite and ChromaDB
- End-to-end ingestion pipeline validation
- Modular architecture for maintainability

