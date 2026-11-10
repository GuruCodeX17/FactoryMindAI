
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "factorymind_model.pkl"


@lru_cache(maxsize=1)
def load_prediction_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ML model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    return (
        artifact["pipeline"],
        artifact["threshold"],
        artifact["features"]
    )


def predict_machine_failure(machine_data: dict) -> dict:
    model, threshold, features = load_prediction_artifact()

    temperature_difference = (
        machine_data["process_temperature"]
        - machine_data["air_temperature"]
    )

    input_data = {
        "Type": machine_data["machine_type"],
        "Air temperature [K]": machine_data["air_temperature"],
        "Process temperature [K]": machine_data["process_temperature"],
        "Rotational speed [rpm]": machine_data["rotational_speed"],
        "Torque [Nm]": machine_data["torque"],
        "Tool wear [min]": machine_data["tool_wear"],
        "Temperature Difference [K]": temperature_difference
    }

    input_df = pd.DataFrame([input_data], columns=features)

    probability = float(model.predict_proba(input_df)[0][1])
    prediction = int(probability >= threshold)

    return {
        "machine_failure_prediction": prediction,
        "failure_probability": round(probability, 4),
        "threshold": round(float(threshold), 4),
        "status": (
            "Potential failure detected"
            if prediction == 1
            else "No failure predicted"
        )
    }
