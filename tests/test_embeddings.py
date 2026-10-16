from mamlka.embeddings import embed_documents, embed_query


def test_embeddings():
    texts = [
        "The revised project budget is $2,600,000.",
        "تم تعديل ميزانية المشروع إلى 2,600,000 دولار.",
    ]

    vectors = embed_documents(texts)
    query_vector = embed_query("What is the revised budget?")

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    assert len(query_vector) == 384

    print("Document vectors:", len(vectors))
    print("Vector dimensions:", len(vectors[0]))
    print("Query dimensions:", len(query_vector))
    print("Embedding test passed.")


if __name__ == "__main__":
    test_embeddings()
