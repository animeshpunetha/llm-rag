# query.py
import sys
import traceback
from src.retrieval import retrieve_context
from src.generation import generate_answer, rewrite_query, get_query_intent, generate_chat_response
from src.memory import SessionMemory
from src.cache import SemanticCache

def run_rag_query(user_query: str, memory:SessionMemory, cache: SemanticCache):
    """Coordinates retrieval and generation pipelines to answer a query."""
    if not user_query.strip():
        return
        
    print(f"\nQuestion: {user_query}")
    print("-" * 50)

    
    #===Sementic Cache Layer========

    cached_answer = cache.check_cache(user_query)

    if cached_answer:
        print("\n===Cached Answer===")
        print(cached_answer)
        print("=====================")

        # We still need to save this to our session memory so the LLM 
        # understands follow-up pronouns on the *next* turn!
        memory.add_turn(user_query, cached_answer)
        return  # Exit the function early

    #===============================
    #==============================
    # NEW: Route by Intent
    intent = get_query_intent(user_query)
    
    if intent == "CHAT":
        print("[Router] Intent classified as CHAT. Bypassing Vector DB...")
        answer = generate_chat_response(user_query, memory.get_history())

        print("\n=== ANSWER ===")
        print(answer)
        print("==============\n")
        
        memory.add_turn(user_query, answer)
        return

    print("[Router] Intent classified as SEARCH. Proceeding to Vector DB...")

    #=====Cache Miss================

    # 1. NEW: Reformulate query using conversation history to handle pronouns
    search_query = rewrite_query(user_query, memory.get_history())
    
    # 2. Retrieve context matching the REWRITTEN query
    matching_chunks = retrieve_context(search_query, k=6)
    
    # 3. Generate answer and showcase dynamic metadata references
    answer, docs = generate_answer(user_query, matching_chunks, memory.get_history())
    
    print("\n=== ANSWER ===")
    print(answer)
    print("==============\n")
    
    print("Sources Used:")
    for idx, doc in enumerate(docs):
        page = doc.metadata.get('page', 'Unknown')
        print(f" -> Chunk {idx+1} | Page Reference: {page}")
    print("-" * 50)

    # save into Redis as cache
    cache.set_cache(user_query, answer)

    # CRITICAL: Save this interaction to memory for the next loop
    memory.add_turn(user_query, answer)

def main():
    print("=== Local Naive RAG Interface ===")
    print("Type 'exit' or 'quit' to close the interface.\n")

    # Initialize the memory buffer ONCE before the loop starts
    session_memory = SessionMemory(max_turns=3)

    # Initialise the Redis connection once 
    print("Connecting to Redis Semantic Cache...")
    semantic_cache = SemanticCache()
    print("Cache Ready.")
    
    while True:
        try:
            query = input("\nAsk something about your document: ")
            if query.lower() in ['exit', 'quit']:
                print("Closing RAG system. Goodbye!")
                break
                
            run_rag_query(query, session_memory, semantic_cache)
            
        except KeyboardInterrupt:
            print("\nClosing RAG system. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n[System Error] {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()