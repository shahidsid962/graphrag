import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Config(BaseModel):
    """Configuration class for the Graph RAG system"""
    
    # Model configurations
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
    ollama_model_name: str = os.getenv("OLLAMA_MODEL_NAME", "llama3.1")
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")
    hf_token: str = os.getenv("HF_TOKEN", "")
    
    # Service endpoints
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Paths
    data_path: str = os.getenv("DATA_PATH", "./data")
    graph_storage_path: str = os.getenv("GRAPH_STORAGE_PATH", "./graphs")
    
    # Graph settings
    max_neighbors: int = int(os.getenv("MAX_NEIGHBORS", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Processing settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    
    # Neo4j settings
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

config = Config()