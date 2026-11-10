
from functools import lru_cache

from qdrant_client import QdrantClient

from src.factorymindai.config import (
    QDRANT_PATH,
    COLLECTION_NAME,
    VECTOR_SIZE
)


@lru_cache(maxsize=1)
def get_qdrant_client():
    """
    Connect to the existing persistent Qdrant database.
    """
    return QdrantClient(
        path=str(QDRANT_PATH)
    )


def get_collection_info():
    """
    Verify that the existing collection is available.
    """
    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        raise ValueError(
            f"Qdrant collection '{COLLECTION_NAME}' not found."
        )

    collection = client.get_collection(COLLECTION_NAME)

    return {
        "collection_name": COLLECTION_NAME,
        "points_count": collection.points_count,
        "status": str(collection.status),
        "expected_vector_size": VECTOR_SIZE
    }


def validate_collection():
    """
    Check the existing collection and vector dimensions.
    """
    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME):
        raise ValueError("Qdrant collection does not exist.")

    collection = client.get_collection(COLLECTION_NAME)

    vector_config = collection.config.params.vectors

    if hasattr(vector_config, "size"):
        actual_size = vector_config.size
    else:
        raise ValueError(
            "Expected an unnamed vector configuration. "
            "Inspect the Qdrant collection configuration."
        )

    if actual_size != VECTOR_SIZE:
        raise ValueError(
            f"Vector dimension mismatch: "
            f"expected {VECTOR_SIZE}, got {actual_size}"
        )

    return True
