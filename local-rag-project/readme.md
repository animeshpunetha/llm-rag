# Agentic RAG

A fully local, enterprise-grade Retrieval-Augmented Generation (RAG) system optimized for low-latency inference, high concurrency, context awareness, and data privacy.

Built with Ollama, ChromaDB, Redis Stack, and LangChain, the system features semantic caching, intent routing, conversational memory, and strict hallucination guardrails.

## Features

* **Semantic Cache (Redis Stack)** – Instant responses for duplicate or semantically similar queries.
* **Intent Routing** – Classifies queries as `CHAT` or `SEARCH` to avoid unnecessary retrieval.
* **Conversational Memory** – Maintains session context across multiple turns.
* **Query Reformulation** – Converts follow-up questions into standalone retrieval queries.
* **Vector Search (ChromaDB)** – Retrieves relevant document chunks using embeddings.
* **Hallucination Control** – Temperature `0.0` and strict prompt isolation.
* **100% Local Execution** – No external APIs; complete data privacy.

## Architecture

```text
User Query
    │
    ▼
Semantic Cache (Redis)
    │
 ┌──┴──┐
 │Hit? │──► Return Cached Response
 └──┬──┘
    ▼
Intent Router (CHAT / SEARCH)
    │
 ┌──┴──┐
 │CHAT │──► Direct LLM Response
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
Response
```

## Tech Stack

| Component      | Technology          |
| -------------- | ------------------- |
| LLM            | Ollama + Qwen2.5:7B |
| Embeddings     | nomic-embed-text    |
| Vector DB      | ChromaDB            |
| Semantic Cache | Redis Stack         |
| Framework      | LangChain           |

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Pull Models

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### Start Redis Stack

```bash
docker run -d --name semantic-cache \
-p 6379:6379 redis/redis-stack-server:latest
```

## Usage

### 1. Build Vector Database

Place PDFs inside the `data/` directory and run:

```bash
python build_db.py
```

### 2. Start the RAG System

```bash
python query.py
```

## Project Structure

```text
local-rag-project/
├── data/
├── db/
├── src/
│   ├── cache.py
│   ├── generation.py
│   ├── memory.py
│   └── retrieval.py
├── config.py
├── build_db.py
└── query.py
```

## Validation

* Ask the same question twice → **Cache Hit**
* Ask "Hello" → **CHAT route**
* Ask follow-up questions with pronouns → **Memory + Reformulation**

## Future Improvements

* Cross-Encoder Re-ranking
* FastAPI Microservices
* Streamlit UI
* Token Streaming
* Multi-user Session Management

