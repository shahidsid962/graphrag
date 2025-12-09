# Graph RAG Chat Interface

This is a Chainlit-based chat interface for interacting with the Graph RAG system using Neo4j as the graph database.

## Setup

1. Make sure you have Neo4j running locally or accessible at the configured URI
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables in a `.env` file (or use the existing one in the root directory)

## Running the Application

To start the Chainlit application, run:

```bash
cd /workspace
chainlit run ui/app.py -h
```

Then open your browser to `http://localhost:8000` to access the chat interface.

## Features

- Interactive chat interface to query your knowledge graph
- Real-time visualization of graph statistics
- Shows relationships between entities in the knowledge graph
- Semantic search capabilities powered by Neo4j and vector embeddings

## Requirements

- Neo4j database (version 5.x recommended)
- Python 3.8+
- Dependencies listed in requirements.txt