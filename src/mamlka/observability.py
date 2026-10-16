import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register


def setup_observability() -> None:
    """Send LangChain traces to the Phoenix container."""

    endpoint = os.getenv(
        "PHOENIX_COLLECTOR_ENDPOINT",
        "http://localhost:6006",
    )

    tracer_provider = register(
        project_name="almamlaka-rag",
        endpoint=f"{endpoint.rstrip('/')}/v1/traces",
    )

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
