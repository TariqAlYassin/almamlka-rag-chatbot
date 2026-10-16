from pathlib import Path

from dotenv import load_dotenv

from mamlka.answering import answer_question
from mamlka.vector_store import get_qdrant_client


ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from the project root .env file
load_dotenv(ROOT / ".env", override=False)


def main():
    client = get_qdrant_client()

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

            answer = answer_question(question, client)
            print(answer)
            print("-" * 70)

    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
