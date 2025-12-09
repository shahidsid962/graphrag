import networkx as nx
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class GraphStorage:
    """
    Graph storage component for Graph RAG system.
    Manages the knowledge graph with nodes representing entities/concepts
    and edges representing relationships between them.
    """
    
    def __init__(self, config, model_name: str = "all-MiniLM-L6-v2"):
        self.config = config
        self.graph = nx.Graph()
        self.embedding_model = SentenceTransformer(model_name)
        self.storage_path = Path(config.graph_storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
    def add_node(self, node_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a node to the graph with content and metadata."""
        embeddings = self.embedding_model.encode([content])
        self.graph.add_node(
            node_id,
            content=content,
            embeddings=embeddings[0],
            metadata=metadata or {}
        )
        
    def add_edge(self, node1_id: str, node2_id: str, relationship: str = "related", weight: float = 1.0):
        """Add an edge between two nodes with a relationship type and weight."""
        self.graph.add_edge(node1_id, node2_id, relationship=relationship, weight=weight)
        
    def find_similar_nodes(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find nodes similar to the query based on semantic similarity."""
        query_embedding = self.embedding_model.encode([query])
        similarities = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if 'embeddings' in node_data:
                node_embedding = node_data['embeddings'].reshape(1, -1)
                sim = cosine_similarity(query_embedding, node_embedding)[0][0]
                if sim >= threshold:
                    similarities.append((node_id, float(sim)))
        
        # Sort by similarity score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_neighbors(self, node_id: str, max_neighbors: int = 5) -> List[str]:
        """Get neighboring nodes of a given node."""
        if node_id not in self.graph:
            return []
        
        neighbors = list(self.graph.neighbors(node_id))
        return neighbors[:max_neighbors]
    
    def get_subgraph_by_nodes(self, node_ids: List[str]) -> nx.Graph:
        """Extract subgraph containing specified nodes."""
        return self.graph.subgraph(node_ids).copy()
    
    def get_node_content(self, node_id: str) -> Optional[str]:
        """Get content of a specific node."""
        if node_id in self.graph.nodes:
            return self.graph.nodes[node_id].get('content', '')
        return None
    
    def save_graph(self, filename: str = "knowledge_graph.pkl"):
        """Save the graph to disk."""
        filepath = self.storage_path / filename
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
            
    def load_graph(self, filename: str = "knowledge_graph.pkl"):
        """Load the graph from disk."""
        filepath = self.storage_path / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                self.graph = pickle.load(f)
                
    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'components': nx.number_connected_components(self.graph)
        }