
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.factorymindai.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Load the embedding model once and reuse it.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


def generate_embedding(text: str) -> list[float]:
    """
    Convert input text into a normalized embedding vector.
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    model = get_embedding_model()

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert multiple text inputs into embedding vectors.
    """
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Texts must be non-empty.")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings.tolist()
