from src.database import get_vector_db

def retrieve_context(query: str, k: int = 4):
    """
    Queries the local ChromaDB database and returns the top K matching chunks.
    """
    # 1. Connect to the existing vector store
    db = get_vector_db()
    
    # 2. Perform similarity search (handles query embedding under the hood via nomic-embed-text)
    print(f"[Retrieval] Searching for top {k} relevant chunks...")
    docs = db.similarity_search(query, k=k)
    
    return docs