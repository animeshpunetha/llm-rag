# This module only cares about one thing: turning a PDF into clean text chunks based on your config file.

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


def load_and_chunk_pdf(file_name: str):
    #Loads a PDF from the data directory and splits it into chunks.
    file_path = os.path.join(config.DATA_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap = config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", "", ]
    )

    return text_splitters.split_documents(documents)