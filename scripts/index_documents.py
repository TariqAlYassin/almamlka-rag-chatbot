from pathlib import Path

from mamlka.chunking import chunk_sections
from mamlka.ingestion import load_pdfs
from mamlka.vector_store import (
    get_qdrant_client,
    index_chunks,
    recreate_collection,
)

ROOT = Path(__file__).resolve().parents[1]
PDF_DIRECTORY = ROOT / "data" / "pdfs"
QDRANT_DIRECTORY = ROOT / "data" / "qdrant"


def main():
    sections = load_pdfs(PDF_DIRECTORY)
    chunks = chunk_sections(sections)

    client = get_qdrant_client(QDRANT_DIRECTORY)

    try:
        recreate_collection(client)
        count = index_chunks(client, chunks)
        print(f"Sections extracted: {len(sections)}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Chunks indexed: {count}")
        print("Qdrant indexing complete.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
