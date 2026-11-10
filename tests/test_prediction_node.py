
from src.factorymindai.agent.nodes import prediction_node


def test_prediction_node_success(monkeypatch):

    def mock_predict_machine_failure(machine_data):
        return {
            "machine_failure_prediction": 1,
            "failure_probability": 0.85,
            "threshold": 0.5,
            "status": "Potential failure detected"
        }

    monkeypatch.setattr(
        "src.factorymindai.agent.nodes.predict_machine_failure",
        mock_predict_machine_failure
    )

    state = {
        "machine_data": {
            "machine_type": "L",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500.0,
            "torque": 40.0,
            "tool_wear": 100.0
        }
    }

    result = prediction_node(state)

    assert result["prediction"]["machine_failure_prediction"] == 1
    assert result["prediction"]["failure_probability"] == 0.85
    assert "Potential failure detected" in result["final_answer"]
    assert "error" not in result


def test_prediction_node_missing_data():

    result = prediction_node({})

    assert result["error"] == "Machine sensor data is required."
