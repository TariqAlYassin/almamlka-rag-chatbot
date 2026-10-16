import re

from mamlka.vector_store import search_chunks

CONFLICT_TOPICS = {
    "budget": {
        "keywords": [
            "budget",
            "cost",
            "funding",
            "financial",
            "ميزانية",
            "الميزانيه",
            "التكلفة",
            "التكاليف",
            "تمويل",
        ],
        "queries": [
            "Approved project budget amount in the project overview",
            "Finance Committee revised total project budget amount",
        ],
    },
    "launch_date": {
        "keywords": [
            "launch",
            "release date",
            "go live",
            "public launch",
            "موعد الإطلاق",
            "موعد الاطلاق",
            "تاريخ الإطلاق",
            "تاريخ الاطلاق",
            "الإطلاق",
            "الاطلاق",
        ],
        "queries": [
            "Planned public project launch date in the project overview",
            "Revised launch date following the steering committee meeting",
        ],
    },
}


def detect_conflict_topics(question: str) -> list[str]:
    """Identify conflict topics relevant to the user's question."""

    normalized = question.casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)

    return [
        topic
        for topic, config in CONFLICT_TOPICS.items()
        if any(keyword.casefold() in normalized for keyword in config["keywords"])
    ]


def retrieve_with_conflicts(
    client,
    question: str,
    limit: int = 5,
) -> list[dict]:
    """
    Retrieve ordinary semantic results and additional targeted results
    for known conflict topics. Deduplicate overlapping chunks.
    """

    results_by_key = {}

    def add_results(results: list[dict]) -> None:
        for result in results:
            metadata = result.get("metadata", {})
            key = (
                metadata.get("source", ""),
                metadata.get("heading", ""),
                result.get("text", ""),
            )

            existing = results_by_key.get(key)
            if existing is None or result.get("score", 0) > existing.get("score", 0):
                results_by_key[key] = result

    # Keep the normal semantic search for general relevance.
    add_results(search_chunks(client, question, limit=limit))

    # Search specifically for both sides of any relevant conflict.
    for topic in detect_conflict_topics(question):
        for targeted_query in CONFLICT_TOPICS[topic]["queries"]:
            add_results(search_chunks(client, targeted_query, limit=limit))

    return sorted(
        results_by_key.values(),
        key=lambda result: result.get("score", 0),
        reverse=True,
    )
