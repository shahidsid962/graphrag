"""
Example usage of the Graph RAG system.
This demonstrates how to build a knowledge graph from documents and query it.
"""
from langchain.schema import Document
from src.graph_rag.engine import GraphRAGEngine
from src.utils.config import config


def create_sample_documents():
    """Create sample documents to demonstrate the Graph RAG system."""
    documents = [
        Document(
            page_content="Artificial Intelligence (AI) is a branch of computer science that aims to create software or machines that exhibit human-like intelligence. This can include learning from experience, understanding natural language, solving problems, and recognizing patterns.",
            metadata={"source": "ai_intro.txt", "page": 1}
        ),
        Document(
            page_content="Machine Learning is a subset of AI that focuses on algorithms that can learn from data. Instead of being explicitly programmed, these algorithms build a model based on training data to make predictions or decisions.",
            metadata={"source": "ml_basics.txt", "page": 1}
        ),
        Document(
            page_content="Deep Learning is a specialized subset of Machine Learning that uses neural networks with multiple layers. These deep neural networks can automatically discover representations needed for feature detection or classification from raw data.",
            metadata={"source": "deep_learning.txt", "page": 1}
        ),
        Document(
            page_content="Natural Language Processing (NLP) is a field of AI focused on the interaction between computers and humans through natural language. The ultimate objective is to enable computers to understand, interpret, and generate human language in a valuable way.",
            metadata={"source": "nlp_overview.txt", "page": 1}
        ),
        Document(
            page_content="Computer Vision is an interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.",
            metadata={"source": "computer_vision.txt", "page": 1}
        ),
        Document(
            page_content="Neural Networks are computing systems inspired by the human brain. They consist of interconnected nodes (neurons) that process information using dynamic state responses to external inputs. They are fundamental to deep learning.",
            metadata={"source": "neural_networks.txt", "page": 1}
        ),
        Document(
            page_content="Robotics combines AI with mechanical engineering to create robots that can perform tasks autonomously or semi-autonomously. Modern robots use AI for perception, decision making, and learning from experience.",
            metadata={"source": "robotics.txt", "page": 1}
        )
    ]
    return documents


def main():
    print(f"Using embedding model: {config.embedding_model_name}")
    print(f"Using LLM model: {config.llm_model_name}")
    print(f"Using Ollama model: {config.ollama_model_name}")
    print(f"Ollama base URL: {config.ollama_base_url}")
    
    print("\nInitializing Graph RAG Engine...")
    engine = GraphRAGEngine(config)
    
    print("\nCreating sample documents...")
    sample_docs = create_sample_documents()
    
    print("\nIngesting documents into knowledge graph...")
    engine.ingest_documents(sample_docs)
    
    print("\nGraph information:")
    graph_info = engine.get_graph_info()
    print(f"  Nodes: {graph_info['num_nodes']}")
    print(f"  Edges: {graph_info['num_edges']}")
    print(f"  Density: {graph_info['density']:.4f}")
    print(f"  Connected Components: {graph_info['components']}")
    
    print("\n" + "="*50)
    print("Graph RAG System Demo")
    print("="*50)
    
    questions = [
        "What is Artificial Intelligence?",
        "How is Machine Learning related to Deep Learning?",
        "What is Natural Language Processing used for?",
        "Explain Neural Networks and their connection to Deep Learning"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        answer = engine.query(question)
        print(f"Answer: {answer}")
        print("-" * 80)


if __name__ == "__main__":
    main()