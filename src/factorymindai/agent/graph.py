
from langgraph.graph import StateGraph, START, END

from src.factorymindai.agent.state import FactoryMindState

from src.factorymindai.agent.nodes import (
    supervisor_node,
    rag_node,
    prediction_node
)


# Fallback node for unknown questions
def unknown_node(state: FactoryMindState) -> dict:
    return {
        "final_answer": (
            "I'm FactoryMind AI, an industrial predictive "
            "maintenance assistant. I can help with machine "
            "failure prediction and maintenance knowledge. "
            "Please provide a relevant question."
        )
    }


# Combine ML prediction and RAG results
def combined_response_node(state: FactoryMindState) -> dict:

    prediction = state.get("prediction", {})
    rag_result = state.get("rag_result", {})

    prediction_status = prediction.get(
        "status", "Prediction unavailable"
    )

    probability = prediction.get(
        "failure_probability"
    )

    rag_answer = rag_result.get(
        "answer", "No maintenance information retrieved."
    )

    if probability is not None:
        prediction_text = (
            f"{prediction_status}. "
            f"Failure probability: {probability:.2%}."
        )
    else:
        prediction_text = prediction_status

    final_answer = (
        f"Machine Analysis:\n{prediction_text}\n\n"
        f"Maintenance Guidance:\n{rag_answer}"
    )

    return {
        "final_answer": final_answer
    }


# Routing after Supervisor
def route_intent(state: FactoryMindState) -> str:

    intent = state.get("intent", "unknown")

    if intent in ["rag", "prediction", "combined"]:
        return intent

    return "unknown"


# Decide what happens after ML prediction
def route_after_prediction(state: FactoryMindState) -> str:

    if state.get("intent") == "combined":
        return "rag"

    return "end"


# Decide what happens after RAG
def route_after_rag(state: FactoryMindState) -> str:

    if state.get("intent") == "combined":
        return "combined_response"

    return "end"


# Initialize graph
workflow = StateGraph(FactoryMindState)


# Register nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("rag", rag_node)
workflow.add_node("prediction", prediction_node)
workflow.add_node("combined_response", combined_response_node)
workflow.add_node("unknown", unknown_node)


# Entry point
workflow.add_edge(START, "supervisor")


# Conditional routing from Supervisor
workflow.add_conditional_edges(
    "supervisor",
    route_intent,
    {
        "rag": "rag",
        "prediction": "prediction",
        "combined": "prediction",
        "unknown": "unknown"
    }
)


# Conditional routing after prediction
workflow.add_conditional_edges(
    "prediction",
    route_after_prediction,
    {
        "rag": "rag",
        "end": END
    }
)


# Conditional routing after RAG
workflow.add_conditional_edges(
    "rag",
    route_after_rag,
    {
        "combined_response": "combined_response",
        "end": END
    }
)


# Final connections
workflow.add_edge("combined_response", END)
workflow.add_edge("unknown", END)


# Compile graph
factorymind_graph = workflow.compile()
