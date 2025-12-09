import chainlit as cl
from typing import List
import os
import sys
import asyncio

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import config
from src.components.neo4j_graph_storage import Neo4jGraphStorage
from src.graph_rag.graph_builder import GraphBuilder
from src.graph_rag.retriever import GraphRAGRetriever
from src.components.document_processor import DocumentProcessor
from langchain.schema import Document as LCDocument


@cl.on_chat_start
async def start_chat():
    """Initialize the chat session with the RAG system."""
    await cl.Message(content="Initializing Graph RAG system with Neo4j...").send()
    
    try:
        # Initialize Neo4j Graph Storage
        graph_storage = Neo4jGraphStorage(config)
        
        # Initialize Document Processor
        document_processor = DocumentProcessor(config)
        
        # Initialize Graph Builder
        graph_builder = GraphBuilder(config, graph_storage, document_processor)
        
        # Initialize Retriever
        retriever = GraphRAGRetriever(graph_storage, config)
        
        # Store components in the session
        cl.user_session.set("graph_storage", graph_storage)
        cl.user_session.set("graph_builder", graph_builder)
        cl.user_session.set("retriever", retriever)
        cl.user_session.set("document_processor", document_processor)
        
        await cl.Message(content="Graph RAG system initialized successfully! You can now ask questions.").send()
        
    except Exception as e:
        error_msg = f"Error initializing Graph RAG system: {str(e)}"
        await cl.Message(content=error_msg).send()
        raise e


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and respond using the RAG system."""
    user_query = message.content
    
    # Retrieve the RAG components from session
    retriever = cl.user_session.get("retriever")
    graph_storage = cl.user_session.get("graph_storage")
    
    try:
        # Retrieve relevant documents from the graph
        retrieved_docs = retriever.retrieve(user_query, top_k=5)
        
        if not retrieved_docs:
            response = "I couldn't find any relevant information in the knowledge graph to answer your question."
            await cl.Message(content=response).send()
            return
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content  # Truncate long content
            context_parts.append(f"Document {i+1}: {content}")
        
        context = "\n\n".join(context_parts)
        
        # Create a response message with context information
        response = f"Based on the knowledge graph, here's what I found:\n\n{context}\n\nQuery: {user_query}"
        
        # Also show graph information
        graph_info = graph_storage.get_graph_info()
        graph_stats = f"\n\nGraph Statistics:\n- Nodes: {graph_info['num_nodes']}\n- Relationships: {graph_info['num_edges']}\n- Relationship Types: {', '.join(graph_info['relationship_types'])}"
        
        response += graph_stats
        
        # Send the response
        msg = cl.Message(content=response)
        await msg.send()
        
        # If there are relationships, show them as well
        if len(retrieved_docs) > 0:
            # Get relationships for the first few nodes
            node_ids = [doc.metadata.get('node_id') for doc in retrieved_docs if doc.metadata.get('node_id')]
            if node_ids:
                relationships_info = []
                for node_id in node_ids[:3]:  # Show relationships for first 3 nodes
                    try:
                        relationships = graph_storage.get_node_relationships(node_id)
                        if relationships:
                            relationships_info.append(f"Node '{node_id}' relationships: {len(relationships)}")
                    except Exception:
                        continue  # Skip if there's an error getting relationships
                
                if relationships_info:
                    rel_msg = cl.Message(content="Relationships found: " + "; ".join(relationships_info))
                    await rel_msg.send()
    
    except Exception as e:
        error_msg = f"Error processing your query: {str(e)}"
        await cl.Message(content=error_msg).send()
        raise e


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # In a real application, you would validate against a user database
    # For this example, we'll allow any user
    return cl.User(
        identifier=username, 
        metadata={"role": "user", "provider": "basic"}
    )


if __name__ == "__main__":
    # For testing purposes, you can run this directly
    # But normally chainlit will handle execution
    pass