
from src.factorymindai.agent.graph import factorymind_graph


def test_graph_rag_workflow(monkeypatch):
    from src.factorymindai.agent import nodes

    def mock_rag_answer(question, top_k=3):
        return {
            "answer": "Preventive maintenance helps reduce machine failures.",
            "sources": []
        }

    monkeypatch.setattr(
        nodes,
        "generate_rag_answer",
        mock_rag_answer
    )

    result = factorymind_graph.invoke({
        "question": "What is preventive maintenance?"
    })

    assert result["intent"] == "rag"
    assert result["final_answer"]


def test_graph_prediction_workflow(monkeypatch):
    from src.factorymindai.agent import nodes

    def mock_prediction(machine_data):
        return {
            "machine_failure_prediction": 0,
            "failure_probability": 0.05,
            "threshold": 0.15,
            "status": "No failure predicted"
        }

    monkeypatch.setattr(
        nodes,
        "predict_machine_failure",
        mock_prediction
    )

    result = factorymind_graph.invoke({
        "question": "Predict machine failure using sensor data.",
        "machine_data": {
            "machine_type": "L",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500.0,
            "torque": 40.0,
            "tool_wear": 100.0
        }
    })

    assert result["intent"] == "prediction"
    assert "prediction" in result
    assert result["final_answer"]


def test_graph_unknown_workflow():
    result = factorymind_graph.invoke({
        "question": "Hello, how are you?"
    })

    assert result["intent"] == "unknown"
    assert result["final_answer"]

def test_graph_combined_workflow(monkeypatch):
    from src.factorymindai.agent import nodes

    # Mock ML prediction
    def mock_prediction(machine_data):
        return {
            "machine_failure_prediction": 1,
            "failure_probability": 0.75,
            "threshold": 0.15,
            "status": "Warning"
        }

    # Capture the question passed to RAG
    captured_questions = []

    def mock_rag_answer(question, top_k=3):
        captured_questions.append(question)

        return {
            "answer": (
                "Inspect machine bearings and check "
                "the lubrication system."
            ),
            "sources": []
        }

    monkeypatch.setattr(
        nodes,
        "predict_machine_failure",
        mock_prediction
    )

    monkeypatch.setattr(
        nodes,
        "generate_rag_answer",
        mock_rag_answer
    )

    result = factorymind_graph.invoke({
        "question": (
            "Predict machine failure and recommend "
            "preventive maintenance."
        ),
        "machine_data": {
            "machine_type": "L",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500.0,
            "torque": 40.0,
            "tool_wear": 100.0
        }
    })

    # Verify combined intent
    assert result["intent"] == "combined"

    # Verify ML prediction
    assert result["prediction"]["status"] == "Warning"
    assert result["prediction"]["failure_probability"] == 0.75

    # Verify prediction context reaches RAG
    assert len(captured_questions) == 1
    assert "Machine prediction context" in captured_questions[0]
    assert "75.00%" in captured_questions[0]

    # Verify RAG result
    assert "lubrication" in result["rag_result"]["answer"]

    # Verify combined final response
    assert "Machine Analysis" in result["final_answer"]
    assert "Maintenance Guidance" in result["final_answer"]
