# Graph RAG System

A modular implementation of Graph-based Retrieval Augmented Generation (Graph RAG) using LangChain and Neo4j.

## Overview

Traditional RAG systems retrieve documents based on semantic similarity, but Graph RAG enhances this by building a knowledge graph from documents where:
- Nodes represent entities, concepts, or document chunks
- Edges represent relationships between these elements
- Retrieval involves traversing the graph to find related information

This approach provides richer context and captures relationships that traditional vector search might miss.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│  Graph Builder   │───▶│ Knowledge Graph │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                    │
                                                    ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Graph RAG      │───▶│    Response     │
└─────────────────┘    │   Retriever      │    └─────────────────┘
                       └──────────────────┘
```

### Components

1. **Document Processor**: Handles document loading, preprocessing, and chunking
2. **Neo4j Graph Storage**: Manages the knowledge graph using Neo4j database with vector embeddings
3. **Graph Builder**: Builds the knowledge graph from documents
4. **Graph RAG Retriever**: Retrieves relevant information using graph traversal
5. **Graph RAG Engine**: Orchestrates the entire process
6. **UI (Chainlit)**: Interactive chat interface for querying the system

## Installation

```bash
pip install uv
uv pip install -r requirements.txt
```

Or using pip:
```bash
pip install -r requirements.txt
```

## Configuration

The system supports flexible configuration through environment variables. Create a `.env` file in the root directory with the following options:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Model Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LLM_MODEL_NAME=gpt-4o-mini
OLLAMA_MODEL_NAME=llama3.1

# API Keys (optional - only needed if using hosted services)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
COHERE_API_KEY=

# Service endpoints
OLLAMA_BASE_URL=http://localhost:11434

# Paths
DATA_PATH=./data
GRAPH_STORAGE_PATH=./graphs

# Graph settings
MAX_NEIGHBORS=5
SIMILARITY_THRESHOLD=0.7

# Processing settings
CHUNK_SIZE=512
CHUNK_OVERLAP=64
```

All configuration values can be customized to suit your needs. The system will fall back to default values if environment variables are not set.

## Setup with Neo4j

Run the setup script to initialize the system:

```bash
python setup_neo4j_rag.py
```

## Usage

### Run the example:
```bash
python example.py
```

### Start the Chainlit UI:
```bash
chainlit run ui/app.py -h
```

Then open your browser to `http://localhost:8000` to interact with the chat interface.

## Key Features

- **Modular Design**: Each component is isolated and reusable
- **Graph-based Retrieval**: Captures relationships between concepts
- **Expandable Queries**: Uses graph connections to enhance query understanding
- **Vector Embeddings in Neo4j**: Combines graph structure with semantic search
- **Interactive UI**: Chainlit-based chat interface for easy interaction
- **Configurable**: Easily adjustable parameters for different use cases

## Customization

To customize the system for your own documents:

1. Create documents in LangChain format
2. Use the `GraphRAGEngine.ingest_documents()` method to build your knowledge graph
3. Query the system using `GraphRAGEngine.query()`

```python
from src.graph_rag.engine import GraphRAGEngine
from langchain.schema import Document

engine = GraphRAGEngine()

# Add your documents
documents = [
    Document(page_content="Your document content here...", metadata={"source": "my_doc.txt"})
]

engine.ingest_documents(documents)

# Query the system
answer = engine.query("Your question here?")
```

## Project Structure

```
/workspace/
├── requirements.txt          # Dependencies for uv/pip
├── setup_neo4j_rag.py       # Setup script for Neo4j-based system
├── example.py               # Example usage
├── README.md               # This file
├── __init__.py             # Package initialization
├── .env                    # Environment variables
├── ui/                     # Chainlit UI components
│   ├── app.py              # Chainlit application
│   ├── requirements.txt    # UI-specific dependencies
│   └── README.md           # UI documentation
├── src/
│   ├── utils/
│   │   └── config.py       # Configuration management
│   ├── components/
│   │   ├── neo4j_graph_storage.py # Neo4j-based graph storage
│   │   ├── graph_storage.py        # Legacy NetworkX-based storage
│   │   └── document_processor.py   # Document processing
│   └── graph_rag/
│       ├── engine.py           # Main Graph RAG engine
│       ├── graph_builder.py    # Graph construction logic
│       └── retriever.py        # Graph-based retrieval
```