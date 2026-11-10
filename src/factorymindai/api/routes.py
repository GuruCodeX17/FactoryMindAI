
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.factorymindai.rag.generator import generate_rag_answer

router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)


class RAGSource(BaseModel):
    id: str
    source: str
    page: int
    score: float


class RAGResponse(BaseModel):
    question: str
    answer: str
    sources: list[RAGSource]


@router.post("/ask", response_model=RAGResponse)
def ask_rag(request: RAGRequest):
    try:
        result = generate_rag_answer(
            question=request.question,
            top_k=request.top_k
        )
        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
