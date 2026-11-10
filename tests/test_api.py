
from fastapi.testclient import TestClient

from src.factorymindai.main import app

client = TestClient(app)


# Test 1: Home endpoint
def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "API is running"


# Test 2: Health endpoint
def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# Test 3: Machine failure prediction
def test_prediction():
    payload = {
        "machine_type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500,
        "torque": 40,
        "tool_wear": 120
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["machine_failure_prediction"] in [0, 1]
    assert 0 <= result["failure_probability"] <= 1
    assert 0 <= result["threshold"] <= 1
    assert "status" in result


# Test 4: Invalid machine type
def test_invalid_machine_type():
    payload = {
        "machine_type": "X",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1500,
        "torque": 40,
        "tool_wear": 120
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# Test 5: Invalid negative sensor value
def test_invalid_sensor_value():
    payload = {
        "machine_type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": -1500,
        "torque": 40,
        "tool_wear": 120
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
