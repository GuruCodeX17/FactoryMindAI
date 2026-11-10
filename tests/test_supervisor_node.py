
from src.factorymindai.agent.nodes import supervisor_node


def test_supervisor_routes_to_rag():
    state = {
        "question": "What is preventive maintenance?"
    }

    result = supervisor_node(state)

    assert result["intent"] == "rag"


def test_supervisor_routes_to_prediction():
    state = {
        "question": "Predict machine failure using sensor data."
    }

    result = supervisor_node(state)

    assert result["intent"] == "prediction"


def test_supervisor_routes_to_combined():
    state = {
        "question": (
            "Predict machine failure and recommend "
            "what maintenance action should be taken."
        ),
        "machine_data": {
            "machine_type": "L",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500.0,
            "torque": 40.0,
            "tool_wear": 100.0
        }
    }

    result = supervisor_node(state)

    assert result["intent"] == "combined"


def test_supervisor_routes_to_unknown():
    state = {
        "question": "Hello, how are you?"
    }

    result = supervisor_node(state)

    assert result["intent"] == "unknown"


def test_supervisor_handles_empty_question():
    result = supervisor_node({})

    assert result["intent"] == "unknown"
    assert result["error"] == "Question cannot be empty."
