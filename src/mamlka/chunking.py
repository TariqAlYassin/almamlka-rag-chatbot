from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_sections(
    sections: list[dict],
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[dict]:
    """Split document sections into retrieval-friendly chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = []

    for section in sections:
        text = section["text"]

        split_docs = splitter.create_documents(
            texts=[text],
            metadatas=[
                {
                    "heading": section["heading"],
                    "source": section["metadata"]["source"],
                    "pages": section["metadata"]["pages"],
                }
            ],
        )

        for doc in split_docs:
            chunks.append(
                {
                    "text": doc.page_content,
                    "metadata": {
                        **doc.metadata,
                        "start_index": doc.metadata["start_index"],
                    },
                }
            )

    return chunks
