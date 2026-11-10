
from src.factorymindai.agent.state import FactoryMindState
from src.factorymindai.rag.generator import generate_rag_answer


def rag_node(state: FactoryMindState) -> dict:
    """
    Retrieve maintenance knowledge and generate an answer.
    """

    question = state.get("question", "").strip()
    top_k = state.get("top_k", 3)

    if not question:
        return {
            "error": "Question cannot be empty."
        }

    try:
        result = generate_rag_answer(
            question=question,
            top_k=top_k
        )

        return {
            "rag_result": result,
            "sources": result.get("sources", []),
            "final_answer": result.get("answer", "")
        }

    except Exception as error:
        return {
            "error": f"RAG processing failed: {error}"
        }

from src.factorymindai.services.prediction_service import (
    predict_machine_failure
)


def prediction_node(state: FactoryMindState) -> dict:
    """
    Predict machine failure using the trained ML model.
    """

    machine_data = state.get("machine_data")

    if not machine_data:
        return {
            "error": "Machine sensor data is required."
        }

    try:
        result = predict_machine_failure(machine_data)

        return {
            "prediction": result,
            "final_answer": (
                f"{result['status']}. "
                f"Failure probability: "
                f"{result['failure_probability']:.2%}."
            )
        }

    except Exception as error:
        return {
            "error": f"ML prediction failed: {error}"
        }

def supervisor_node(state: FactoryMindState) -> dict:
    """
    Classify the user request and decide which agent should handle it.
    """

    question = state.get("question", "").strip().lower()
    machine_data = state.get("machine_data")

    if not question:
        return {
            "intent": "unknown",
            "error": "Question cannot be empty."
        }

    # Keywords related to machine failure prediction
    prediction_keywords = [
        "predict",
        "prediction",
        "failure risk",
        "failure probability",
        "machine failure",
        "will fail",
        "sensor anomaly",
        "anomaly detection"
    ]

    # Keywords related to maintenance knowledge
    rag_keywords = [
        "maintenance",
        "preventive",
        "troubleshoot",
        "recommend",
        "what should i do",
        "repair",
        "root cause",
        "diagnose",
        "equipment manual",
        "maintenance procedure"
    ]

    needs_prediction = any(
        keyword in question
        for keyword in prediction_keywords
    )

    needs_rag = any(
        keyword in question
        for keyword in rag_keywords
    )

    # Both prediction and maintenance knowledge are required
    if needs_prediction and needs_rag and machine_data:
        intent = "combined"

    # Only ML prediction is required
    elif needs_prediction:
        intent = "prediction"

    # Only RAG retrieval is required
    elif needs_rag:
        intent = "rag"

    # Request cannot be classified
    else:
        intent = "unknown"

    return {
        "intent": intent
    }
