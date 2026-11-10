
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.factorymindai.api.routes import router as rag_router
from src.factorymindai.api.agent_routes import router as agent_router
from src.factorymindai.services.prediction_service import (
    predict_machine_failure
)


# Initialize FastAPI
app = FastAPI(
    title="FactoryMind AI",
    description="Industrial Predictive Maintenance and RAG Assistant",
    version="1.0.0"
)


# Register RAG routes
app.include_router(rag_router)


# Input validation
class MachineData(BaseModel):
    machine_type: Literal["L", "M", "H"]
    air_temperature: float = Field(gt=0)
    process_temperature: float = Field(gt=0)
    rotational_speed: float = Field(gt=0)
    torque: float = Field(gt=0)
    tool_wear: float = Field(ge=0)
# Register RAG routes
app.include_router(rag_router)

# Register Agent routes
app.include_router(agent_router)

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to FactoryMind AI",
        "status": "API is running"
    }


# Health endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Machine failure prediction endpoint
@app.post("/predict")
def predict_failure(data: MachineData):
    try:
        # Convert validated input into a dictionary
        machine_data = data.model_dump()

        # Call the reusable prediction service
        result = predict_machine_failure(machine_data)

        return result

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        ) from error
