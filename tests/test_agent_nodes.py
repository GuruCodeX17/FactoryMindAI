
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
