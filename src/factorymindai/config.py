import os
from pathlib import Path

from dotenv import load_dotenv


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")


# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"


# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Qdrant Configuration
QDRANT_PATH = PROJECT_ROOT / "qdrant_storage"
COLLECTION_NAME = "factorymind_knowledge"
VECTOR_SIZE = 384


# RAG Configuration
RAG_TOP_K = 3
RAG_SCORE_THRESHOLD = 0.25