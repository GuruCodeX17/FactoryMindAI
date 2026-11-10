
from fastapi.testclient import TestClient
from src.factorymindai.main import app

client = TestClient(app)


def test_rag_ask_success(monkeypatch):
    def mock_generate_rag_answer(question, top_k):
        return {
            "question": question,
            "answer": "Preventive maintenance reduces equipment wear. [S1]",
            "sources": [
                {
                    "id": "S1",
                    "source": "maintenance_guide.pdf",
                    "page": 51,
                    "score": 0.70
                }
            ]
        }

    monkeypatch.setattr(
        "src.factorymindai.api.routes.generate_rag_answer",
        mock_generate_rag_answer
    )

    response = client.post(
        "/rag/ask",
        json={
            "question": "What is preventive maintenance?",
            "top_k": 3
        }
    )

    assert response.status_code == 200
    data = response.json()

    assert data["question"] == "What is preventive maintenance?"
    assert "Preventive maintenance" in data["answer"]
    assert len(data["sources"]) == 1


def test_rag_ask_invalid_question():
    response = client.post(
        "/rag/ask",
        json={
            "question": "Hi",
            "top_k": 3
        }
    )

    assert response.status_code == 422
