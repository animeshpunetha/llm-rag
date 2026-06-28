import os
import shutil
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import config

def get_embedding_model():
    # Initializes the local embedding model.
    return OllamaEmbeddings(model = config.EMBEDDING_MODEL)

def clear_db():
    """Deletes the existing vector database directory to ensure a clean slate."""
    if os.path.exists(config.DB_DIR):
        print(f"[Database] Clearing existing database at {config.DB_DIR}...")
        shutil.rmtree(config.DB_DIR)
        print("[Database] Database cleared.")

def save_to_chroma(chunks, reset: bool = False):
    # Embeds text chunks and saves them to the local disk.
    if reset:
        clear_db()
    
    embeddings = get_embedding_model()

    vector_db = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        persist_directory = config.DB_DIR
    )

    return vector_db


def get_vector_db():
    # Loads the existing database from disk for querying.

    embeddings = get_embedding_model()
    return Chroma(
        persist_directory = config.DB_DIR,
        embedding_function = embeddings
    )