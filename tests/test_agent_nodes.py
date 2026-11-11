
import pytest

from src.factorymindai.agent.nodes import rag_node


def test_rag_node_success(monkeypatch):

    def mock_generate_rag_answer(question, top_k):
        return {
            "question": question,
            "answer": "Preventive maintenance reduces equipment failures.",
            "sources": [
                {
                    "id": "S1",
                    "source": "maintenance_guide.pdf",
                    "page": 10,
                    "score": 0.85
                }
            ]
        }

    monkeypatch.setattr(
        "src.factorymindai.agent.nodes.generate_rag_answer",
        mock_generate_rag_answer
    )

    state = {
        "question": "What is preventive maintenance?",
        "top_k": 3
    }

    result = rag_node(state)

    assert "Preventive maintenance" in result["final_answer"]
    assert len(result["sources"]) == 1
    assert "error" not in result


def test_rag_node_empty_question():

    result = rag_node({"question": ""})

    assert result["error"] == "Question cannot be empty."


def test_rag_node_invalid_probability(monkeypatch):

    captured = {}

    def mock_generate_rag_answer(question, top_k):
        captured["question"] = question

        return {
            "answer": "Check machine bearings.",
            "sources": []
        }

    monkeypatch.setattr(
        "src.factorymindai.agent.nodes.generate_rag_answer",
        mock_generate_rag_answer
    )

    state = {
        "question": "Recommend maintenance",
        "prediction": {
            "status": "Warning",
            "failure_probability": "invalid"
        }
    }

    result = rag_node(state)

    assert "error" not in result
    assert "Machine prediction context" in captured["question"]
    assert "Machine status: Warning" in captured["question"]
    assert "Estimated failure probability" not in captured["question"]


def test_rag_node_invalid_prediction_structure(monkeypatch):

    def mock_generate_rag_answer(question, top_k):
        return {
            "answer": "Maintenance guidance retrieved.",
            "sources": []
        }

    monkeypatch.setattr(
        "src.factorymindai.agent.nodes.generate_rag_answer",
        mock_generate_rag_answer
    )

    state = {
        "question": "Recommend maintenance",
        "prediction": "invalid prediction"
    }

    result = rag_node(state)

    assert "error" not in result
    assert result["final_answer"] == "Maintenance guidance retrieved."


def test_rag_node_probability_out_of_range(monkeypatch):

    captured = {}

    def mock_generate_rag_answer(question, top_k):
        captured["question"] = question

        return {
            "answer": "Check the machine.",
            "sources": []
        }

    monkeypatch.setattr(
        "src.factorymindai.agent.nodes.generate_rag_answer",
        mock_generate_rag_answer
    )

    state = {
        "question": "Recommend maintenance",
        "prediction": {
            "status": "Warning",
            "failure_probability": 1.5
        }
    }

    result = rag_node(state)

    assert "error" not in result
    assert "Machine prediction context" in captured["question"]
    assert "Estimated failure probability" not in captured["question"]


def test_rag_node_non_string_question():

    result = rag_node({"question": 123})

    assert result["error"] == "Question cannot be empty."
