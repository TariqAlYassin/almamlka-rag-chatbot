from pathlib import Path

from mamlka.chunking import chunk_sections
from mamlka.ingestion import load_pdfs

PDF_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "pdfs"


def test_chunking():
    sections = load_pdfs(PDF_DIRECTORY)
    chunks = chunk_sections(sections)

    assert chunks, "No chunks were created."

    for chunk in chunks:
        assert chunk["text"].strip()
        assert chunk["metadata"]["source"]
        assert chunk["metadata"]["heading"]
        assert chunk["metadata"]["pages"]

    print(f"Sections: {len(sections)}")
    print(f"Chunks: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n--- CHUNK ---")
        print(chunk["metadata"])
        print(chunk["text"][:300])


if __name__ == "__main__":
    test_chunking()
