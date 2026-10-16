import mamlka.answering as answering


def test_system_prompt_requires_grounded_answers():
    prompt = answering.SYSTEM_PROMPT.lower()

    assert "only" in prompt
    assert "conflict" in prompt
    assert "instructions" in prompt


def test_context_preserves_source_citations():
    results = [
        {
            "text": "The revised budget is $2,600,000.",
            "metadata": {
                "source": "Almamlaka_Budget_Timeline.pdf",
                "pages": [2],
                "heading": "2. Budget Notes",
            },
            "score": 0.95,
        }
    ]

    context = answering.format_context(results)

    assert "Almamlaka_Budget_Timeline.pdf" in context
    assert "Pages: 2" in context
    assert "2. Budget Notes" in context
    assert "$2,600,000" in context


def test_empty_context_is_explicit():
    assert "No relevant" in answering.format_context([])


def test_no_retrieved_results_returns_refusal(monkeypatch):
    monkeypatch.setattr(
        answering,
        "retrieve_with_conflicts",
        lambda **kwargs: [],
    )

    result = answering.answer_question(
        question="What is the project budget?",
        client=object(),
    )

    assert "couldn't find relevant information" in result.lower()


def test_arabic_empty_result_refusal(monkeypatch):
    monkeypatch.setattr(
        answering,
        "retrieve_with_conflicts",
        lambda **kwargs: [],
    )

    result = answering.answer_question(
        question="ما هي الميزانية؟",
        client=object(),
    )

    assert "لم أجد" in result
