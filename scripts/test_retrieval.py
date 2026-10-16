from pathlib import Path

from mamlka.vector_store import (
    get_qdrant_client,
    search_chunks,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    client = get_qdrant_client(ROOT / "data" / "qdrant")

    try:
        questions = [
            "What is the revised project budget?",
            "When is the public launch scheduled?",
            "Who leads the project?",
            "ما هي الميزانية المعدلة للمشروع؟",
        ]

        for question in questions:
            print(f"\nQUESTION: {question}")

            results = search_chunks(client, question, limit=3)

            for result in results:
                print(f"Score: {result['score']:.3f}")
                print(f"Source: {result['metadata'].get('source')}")
                print(f"Pages: {result['metadata'].get('pages')}")
                print(result["text"][:500])
                print("-" * 50)
    finally:
        client.close()


if __name__ == "__main__":
    main()
