@'
# FactoryMind AI

An AI-powered industrial predictive maintenance assistant that combines machine learning, Retrieval-Augmented Generation (RAG), and agent-based workflows.

## Features

- **Machine Failure Prediction:** Predicts potential machine failures using a trained ML pipeline.
- **Knowledge Retrieval:** Retrieves relevant maintenance guidance from technical documents using semantic search.
- **RAG Assistant:** Generates answers grounded in retrieved maintenance documentation.
- **LangGraph Agent:** Routes user questions to the appropriate prediction or retrieval workflow.
- **FastAPI Backend:** Exposes prediction, RAG, and agent endpoints.
- **Maintenance MCP Server:** Provides tools to check machine status, retrieve maintenance history, and create demo work orders with an approval check.

## Technology Stack

- Python
- Scikit-learn, Pandas, NumPy
- FastAPI
- LangChain and LangGraph
- Qdrant
- Sentence Transformers
- Groq LLM
- Model Context Protocol (MCP)
- uv

## Project Structure

```text
FactoryMindAI/
├── data/
├── knowledge_base/
├── models/
├── notebooks/
├── src/
│   └── factorymindai/
│       ├── agent/
│       ├── api/
│       ├── mcp_servers/
│       ├── rag/
│       └── services/
├── tests/
├── pyproject.toml
└── README.md