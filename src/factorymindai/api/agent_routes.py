
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.factorymindai.agent.graph import factorymind_graph


router = APIRouter(prefix="/agent", tags=["Agent"])


# Machine sensor input validation
class MachineDataInput(BaseModel):
    machine_type: Literal["L", "M", "H"]
    air_temperature: float = Field(gt=0)
    process_temperature: float = Field(gt=0)
    rotational_speed: float = Field(gt=0)
    torque: float = Field(gt=0)
    tool_wear: float = Field(ge=0)


# Agent API request
class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    machine_data: MachineDataInput | None = None
    top_k: int = Field(default=3, ge=1, le=10)


# Agent API endpoint
@router.post("/ask")
def ask_agent(request: AgentRequest):
    try:
        # Convert machine data into a dictionary
        machine_data = (
            request.machine_data.model_dump()
            if request.machine_data
            else None
        )

        # Invoke the LangGraph workflow
        result = factorymind_graph.invoke({
            "question": request.question,
            "machine_data": machine_data,
            "top_k": request.top_k
        })

        # Prepare the API response
        return {
            "question": request.question,
            "intent": result.get("intent", "unknown"),
            "answer": (
                result.get("final_answer")
                or result.get("error")
                or "Unable to generate an answer."
            ),
            "prediction": result.get("prediction"),
            "sources": result.get("sources", []),
            "error": result.get("error")
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {str(error)}"
        ) from error
