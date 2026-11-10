
from src.factorymindai.config import (
    COLLECTION_NAME,
    RAG_TOP_K,
    RAG_SCORE_THRESHOLD
)

from src.factorymindai.rag.embeddings import generate_embedding
from src.factorymindai.rag.vector_store import (
    get_qdrant_client,
    validate_collection
)


def retrieve_documents(
    query: str,
    top_k: int = RAG_TOP_K,
    score_threshold: float = RAG_SCORE_THRESHOLD
) -> list[dict]:
    """
    Retrieve relevant maintenance document chunks from Qdrant.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if not 0 <= score_threshold <= 1:
        raise ValueError("Score threshold must be between 0 and 1.")

    # 1. Generate query embedding
    query_vector = generate_embedding(query)

    # 2. Connect to Qdrant
    client = get_qdrant_client()

    # 3. Validate existing collection
    validate_collection()

    # 4. Search for relevant documents
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    # 5. Filter and prepare results
    retrieved_documents = []

    for result in search_results:

        if result.score < score_threshold:
            continue

        payload = result.payload or {}

        retrieved_documents.append({
            "chunk_id": payload.get("chunk_id"),
            "source": payload.get("source"),
            "page": payload.get("page"),
            "text": payload.get("text", ""),
            "score": float(result.score)
        })

    return retrieved_documents
