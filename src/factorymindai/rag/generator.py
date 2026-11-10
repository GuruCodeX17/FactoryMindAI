
from langchain_groq import ChatGroq

from src.factorymindai.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    RAG_TOP_K,
    RAG_SCORE_THRESHOLD
)

from src.factorymindai.rag.retriever import retrieve_documents


def get_llm():
    """Initialize the Groq LLM."""

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing.")

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=1024,
        api_key=GROQ_API_KEY
    )


def generate_rag_answer(
    question: str,
    top_k: int = RAG_TOP_K
) -> dict:
    """Generate an answer using retrieved maintenance knowledge."""

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    # 1. Retrieve documents
    documents = retrieve_documents(
        query=question,
        top_k=top_k,
        score_threshold=RAG_SCORE_THRESHOLD
    )

    if not documents:
        return {
            "answer": (
                "I couldn't find sufficiently relevant information "
                "in the maintenance documents."
            ),
            "sources": []
        }

    # 2. Prepare context
    context_parts = []
    sources = []

    for index, doc in enumerate(documents, start=1):

        source_id = f"S{index}"

        context_parts.append(
            f"""
            [{source_id}]
            Document: {doc['source']}
            Page: {doc['page']}
            Content: {doc['text']}
            """
        )

        sources.append({
            "id": source_id,
            "source": doc["source"],
            "page": doc["page"],
            "score": doc["score"]
        })

    context = "\n\n".join(context_parts)

    # 3. Create grounded prompt
    prompt = f"""
    You are FactoryMind AI, an industrial maintenance assistant.

    Answer using ONLY the supplied context.

    Rules:
    - Never invent information.
    - Cite factual claims using [S1], [S2], etc.
    - Only use the supplied source labels.
    - If information is unavailable, say so.
    - Use simple and clear language.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # 4. Generate answer
    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        "question": question,
        "answer": response.content,
        "sources": sources
    }
