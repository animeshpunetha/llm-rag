import os
import config
from src.ingest import load_and_chunk_pdf
from src.database import save_to_chroma

def main():
    target_pdf = "sample.pdf"

    # Check if the data directory exists, create if not
    os.makedirs(config.DATA_DIR, exist_ok = True)

    print(f"----Building local vector db----")

    try:
        # 1. Ingest
        print(f"[1/2] Loading and chunking '{target_pdf}")
        chunks = load_and_chunk_pdf(target_pdf)
        print(f"----Created {len(chunks)} chunks.")

        # 2. Embed and store
        print(f"[2/2] Embedding chunks using {config.EMBEDDING_MODEL}")
        save_to_chroma(chunks, reset=True)
        print(f"----Success! Database saved to {config.DB_DIR}---")

    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print(f"Please place '{target_pdf}' inside the '{config.DATA_DIR}/' folder.")

if __name__ == "__main__":
    main()