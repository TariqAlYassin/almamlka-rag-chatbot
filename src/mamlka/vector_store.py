import os
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient, models

from mamlka.embeddings import embed_documents, embed_query

COLLECTION_NAME = "almamlaka_knowledge"
VECTOR_SIZE = 384
DEFAULT_DB_PATH = Path("data/qdrant")


DEFAULT_DB_PATH = Path("data/qdrant")


def get_qdrant_client(db_path: Path = DEFAULT_DB_PATH) -> QdrantClient:
    """Connect to Qdrant Server in Docker, or use local mode."""

    qdrant_url = os.getenv("QDRANT_URL")

    if qdrant_url:
        return QdrantClient(url=qdrant_url)

    db_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(db_path))


def create_collection(client: QdrantClient) -> None:
    """Create the collection if it does not already exist."""
    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )


def index_chunks(
    client: QdrantClient,
    chunks: list[dict],
    batch_size: int = 32,
) -> int:
    """Embed and index chunks in batches."""
    if not chunks:
        return 0

    create_collection(client)
    indexed_count = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk["text"] for chunk in batch]
        vectors = embed_documents(texts)

        points = []

        for chunk, vector in zip(batch, vectors):
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        **chunk["metadata"],
                    },
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        indexed_count += len(points)

    return indexed_count


def search_chunks(
    client: QdrantClient,
    query: str,
    limit: int = 5,
) -> list[dict]:
    """Find the most relevant chunks for a question."""
    if not query.strip():
        return []

    create_collection(client)
    query_vector = embed_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points

    return [
        {
            "text": result.payload.get("text", ""),
            "metadata": {
                key: value for key, value in result.payload.items() if key != "text"
            },
            "score": result.score,
        }
        for result in results
    ]


def recreate_collection(client: QdrantClient) -> None:
    """Delete and recreate the collection for a clean full reindex."""

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE,
            distance=models.Distance.COSINE,
        ),
    )
