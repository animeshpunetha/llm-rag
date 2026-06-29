# Enterprise-Grade Agentic RAG Architecture

A fully local, enterprise-grade Retrieval-Augmented Generation (RAG) microservice optimized for low-latency inference, high concurrency, context awareness, and strict data privacy.

Built with Ollama, ChromaDB, Redis Stack, FastAPI, and LangChain, the system features semantic caching, intent routing, conversational memory, real-time token streaming, and strict hallucination guardrails.

---

## Features

* **Asynchronous Microservice (FastAPI)** – Handles concurrent user requests without blocking the main thread.
* **Real-Time Token Streaming** – Streams LLM responses via Server-Sent Events (SSE) for zero-wait-time UX.
* **Multi-User Session Management** – Maintains independent conversational memory buffers for simultaneous users.
* **Semantic Cache (Redis Stack)** – Serves instant responses for duplicate or semantically similar queries.
* **Intent Routing** – Classifies queries as `CHAT` or `SEARCH` to bypass unnecessary database retrieval.
* **Query Reformulation** – Converts pronoun-heavy follow-up questions into standalone vector search queries.
* **Vector Search (ChromaDB)** – Retrieves highly relevant document chunks using local embeddings.
* **Hallucination Control** – Enforces a `0.0` temperature and strict prompt isolation.
* **100% Air-Gapped Execution** – No external APIs; complete proprietary data privacy.

---

## Architecture

```text
Client Request (HTTP POST)
    │
    ▼
FastAPI Gateway & Session Manager
    │
    ▼
Semantic Cache (Redis)
    │
 ┌──┴──┐
 │Hit? │──► Stream Cached Response
 └──┬──┘
    ▼
Intent Router (CHAT / SEARCH)
    │
 ┌──┴──┐
 │CHAT │──► Stream Direct LLM Response
 └──┬──┘
    ▼
Query Reformulation
    ▼
ChromaDB Retrieval
    ▼
Context Injection
    ▼
Qwen2.5 Generation
    ▼
Stream Tokens to Client
```

---

## Tech Stack

| Component      | Technology                 |
| -------------- | -------------------------- |
| Core API       | FastAPI, Uvicorn, Pydantic |
| LLM            | Ollama + Qwen2.5:7B        |
| Embeddings     | nomic-embed-text           |
| Vector DB      | ChromaDB                   |
| Semantic Cache | Redis Stack (RediSearch)   |
| Orchestration  | LangChain                  |

---

## Setup

### Option A: Docker (recommended)

1. Place PDFs in the `data/` directory (default ingest target: `sample.pdf`).

2. Start all services (API, Redis, Ollama, and model download):

```bash
docker compose up --build
```

3. Build the vector database (first run, or after adding new PDFs):

```bash
docker compose --profile tools run --rm build-db
```

4. Stream a query:

```bash
curl -N -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the full name of APJ Abdul Kalam?", "session_id": "user_123"}'
```

**Docker services**

| Service          | Purpose                          | Port  |
| ---------------- | -------------------------------- | ----- |
| `rag-api`        | FastAPI RAG microservice         | 8000  |
| `semantic-cache` | Redis Stack semantic cache       | 6379  |
| `ollama`         | Local LLM + embedding inference  | 11434 |
| `ollama-init`    | One-shot model pull on startup   | —     |
| `build-db`       | Ingest PDFs into ChromaDB        | —     |

Persistent data is stored in `./data`, `./db`, and Docker volumes `redis_data` / `ollama_data`.

Copy `.env.example` to `.env` for local (non-Docker) overrides.

---

### Option B: Local development

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure the following packages are included:

```text
fastapi
uvicorn
pydantic
redis
langchain
langchain-ollama
langchain-chroma
chromadb
```

#### 2. Pull Local Models

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

#### 3. Start Redis Stack

```bash
sudo docker run -d \
--name semantic-cache \
-p 6379:6379 \
redis/redis-stack-server:latest
```

---

## Usage

### 1. Build the Vector Database

Place source PDFs inside the `data/` directory and run:

```bash
python build_db.py
```

### 2. Start the API Server

Launch the ASGI server:

```bash
python api.py
```

The service will be available at:

```text
http://0.0.0.0:8000
```

### 3. Stream a Query

Open a new terminal and send a request:

```bash
curl -N -X POST http://localhost:8000/v1/chat \
-H "Content-Type: application/json" \
-d '{
  "query": "What is the full name of APJ Abdul Kalam?",
  "session_id": "user_123"
}'
```

The response will stream token-by-token in real time.

---

## Project Structure

```text
local-rag-project/
├── data/                  # Source PDF documents
├── db/                    # ChromaDB persistent storage
├── src/
│   ├── cache.py           # Redis Semantic Cache engine
│   ├── generation.py      # LLM invocation, routing, and streaming logic
│   ├── memory.py          # Session state management
│   ├── retrieval.py       # ChromaDB vector search
│   ├── intent_router.py   # Query classification
│   └── query_rewriter.py  # Follow-up query reformulation
│
├── config.py              # Global configurations
├── build_db.py            # Document ingestion pipeline
├── query.py               # Legacy CLI interface
├── api.py                 # FastAPI microservice entry point
└── README.md
```

---

## Validation & Testing

### Semantic Cache

Ask the same question twice:

```text
What is the full name of APJ Abdul Kalam?
```

Then ask a paraphrased version:

```text
What was Abdul Kalam's complete name?
```

Expected:

```text
[CACHE HIT]
```

---

### Intent Routing

```text
Hello
```

Expected:

```text
CHAT Route
```

```text
What is ISO 27001?
```

Expected:

```text
SEARCH Route
```

---

### Conversational Memory

```text
Tell me about Abdul Kalam.
Where was he born?
```

Expected:

```text
Pronoun correctly resolved
```

---

### Multi-User Session Isolation

Session A:

```json
{
  "query": "Tell me about Abdul Kalam",
  "session_id": "user_a"
}
```

Session B:

```json
{
  "query": "Tell me about ISO 27001",
  "session_id": "user_b"
}
```

Expected:

```text
Independent memory buffers
No context leakage
```

---

### Real-Time Streaming

Verify responses arrive incrementally rather than waiting for complete generation.

Expected:

```text
Token-by-token streaming
```

---

## Future Improvements (Tier 4)

* Cross-Encoder Re-ranking (`BAAI/bge-reranker-base`)
* Hybrid Search (Dense + BM25)
* Streamlit / Next.js Frontend
* Rate Limiting via Redis Token Bucket
* Observability & Tracing (Langfuse, OpenTelemetry)
* Kubernetes Support
* Authentication & RBAC
* Multi-Document Collections
* Agentic Tool Calling

---

## Why This Project?

Most RAG implementations stop at retrieval and generation.

This project demonstrates production-oriented RAG patterns used in modern AI systems:

* Semantic Caching
* Intent Routing
* Conversational Memory
* Query Reformulation
* Streaming Responses
* Multi-User Session Management
* Retrieval Grounding
* Hallucination Prevention
* Fully Local Deployment

Together, these techniques significantly improve latency, scalability, retrieval quality, and user experience while maintaining complete privacy and zero external dependencies.
