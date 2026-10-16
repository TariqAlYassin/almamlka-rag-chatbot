from pathlib import Path

from mamlka.answering import answer_question
from mamlka.vector_store import get_qdrant_client

ROOT = Path(__file__).resolve().parents[1]


def main():
    client = get_qdrant_client(ROOT / "data" / "qdrant")

    try:
        questions = [
            "What is the revised project budget?",
            "When is the public launch scheduled?",
            "ما هي الميزانية المعدلة للمشروع؟",
            "What is the capital of France?",
        ]

        for question in questions:
            print(f"\nQUESTION: {question}")
            print("ANSWER:")
            print(answer_question(question, client))
            print("-" * 70)
    finally:
        client.close()


if __name__ == "__main__":
    main()
