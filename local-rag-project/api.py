import uvicorn
from fastapi import FastAPI 
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.cache import SemanticCache
from src.memory import SessionMemory
from src.retrieval import retrieve_context
from src.generation import (
    rewrite_query,
    get_query_intent,
    generate_answer_stream,
    generate_chat_response_stream
)


# initialise API and Singleton Cache
app = FastAPI(title = "Enterprise Agentic RAG API")
semantic_cache = SemanticCache()

# dictionary to hold independent memory buffers for concurrent users
active_sessions = {}

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_user"

async def rag_stream_generator(user_query: str, session_id: str):
    """The core engine that yields data back to the client."""
    # Fetch or create session memory
    if session_id not in active_sessions:
        active_sessions[session_id] = SessionMemory(max_turns=3)
    memory = active_sessions[session_id]
    
    full_response = ""

    # Check the Semantic Cache
    cached_answer = semantic_cache.check_cache(user_query)
    if cached_answer:
        yield f"[Cache Hit]\n\n{cached_answer}"
        memory.add_turn(user_query, cached_answer)
        return

    # Intent Routing
    intent = get_query_intent(user_query)

    if intent=="CHAT":
        async for token in generate_chat_response_stream(user_query, memory.get_history()):
            full_response += token
            yield token
        memory.add_turn(user_query, full_response)
        return

    
    # Standard RAG Pipeline
    search_query = rewrite_query(user_query, memory.get_history())
    matching_chunks = retrieve_context(search_query, k=6)

    # Stream the main answer
    async for token in generate_answer_stream(user_query, matching_chunks, memory.get_history()):
        full_response += token
        yield token

    # Append the sources to the end of the stream
    yield "\n\n---Sources---\n"
    for idx, doc in enumerate(matching_chunks):
        page = doc.metadata.get('page', 'Unknown')
        yield f"Chunk {idx+1} (Page {page})\n"

    # Save to cache and memory
    semantic_cache.set_cache(user_query, full_response)
    memory.add_turn(user_query, full_response)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/v1/chat")
async def chat_endpoint(request: ChatRequest):
    """API Endpoint that consumes a query and returns a streaming response."""
    return StreamingResponse(
        rag_stream_generator(request.query, request.session_id),
        media_type = "text/plain"
    )

if __name__ == "__main__":
    print("Starting the RAG MicroService...")
    uvicorn.run("api:app", host = "0.0.0.0", port = 8000, reload = True)


