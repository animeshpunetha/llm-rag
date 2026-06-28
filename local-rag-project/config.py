# Centralizing variables here means so never have to hunt through different files to change a model name or adjust a chunk size.

import os

# Model Settings
LLM_MODEL = "qwen2.5:7b"
EMBEDDING_MODEL = "nomic-embed-text"

#---Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

#---Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
