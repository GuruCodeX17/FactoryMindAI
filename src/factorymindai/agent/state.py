
from typing import Any, Literal, TypedDict


class FactoryMindState(TypedDict, total=False):

    # User input
    question: str

    # Agent routing decision
    intent: Literal["rag", "prediction", "combined", "unknown"]

    # Machine sensor input
    machine_data: dict[str, Any]

    # RAG configuration
    top_k: int

    # ML prediction output
    prediction: dict[str, Any]

    # RAG output
    rag_result: dict[str, Any]

    # Final response
    final_answer: str

    # Retrieved source information
    sources: list[dict[str, Any]]

    # Error handling
    error: str
