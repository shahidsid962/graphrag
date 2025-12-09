from typing import List, Dict, Any, Optional
from langchain.schema import Document as LCDocument
from src.components.graph_storage import GraphStorage
from src.utils.config import config
from jinja2 import Template
import os
import json


class GraphRAGRetriever:
    """
    Graph RAG retriever component.
    Retrieves relevant information from the knowledge graph based on a query.
    """
    
    def __init__(self, graph_storage: GraphStorage, config=config):
        self.graph_storage = graph_storage
        self.config = config
        
    def retrieve(self, query: str, top_k: int = 5) -> List[LCDocument]:
        """
        Retrieve relevant documents from the graph based on the query.
        Uses graph traversal to find related nodes.
        """
        # Find nodes similar to the query
        similar_nodes = self.graph_storage.find_similar_nodes(
            query, 
            top_k=top_k, 
            threshold=self.config.similarity_threshold
        )
        
        retrieved_docs = []
        
        for node_id, similarity_score in similar_nodes:
            # Get the content of the node
            content = self.graph_storage.get_node_content(node_id)
            if content:
                # Get neighbors of the node to provide more context
                neighbors = self.graph_storage.get_neighbors(
                    node_id, 
                    max_neighbors=self.config.max_neighbors
                )
                
                # Collect content from neighbors
                neighbor_contents = []
                for neighbor_id in neighbors:
                    neighbor_content = self.graph_storage.get_node_content(neighbor_id)
                    if neighbor_content:
                        neighbor_contents.append(neighbor_content)
                
                # Combine the main content with neighbor content
                full_content = content
                if neighbor_contents:
                    full_content += "\n\nRelated information:\n" + "\n".join(neighbor_contents)
                
                # Create a LangChain document
                doc = LCDocument(
                    page_content=full_content,
                    metadata={
                        'node_id': node_id,
                        'similarity_score': similarity_score,
                        'neighbors': neighbors,
                        'retrieval_method': 'graph_rag'
                    }
                )
                
                retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def retrieve_with_path(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve with path information showing how nodes are connected.
        """
        similar_nodes = self.graph_storage.find_similar_nodes(
            query, 
            top_k=top_k, 
            threshold=self.config.similarity_threshold
        )
        
        results = []
        
        for node_id, similarity_score in similar_nodes:
            # Get neighbors and their relationships
            neighbors = self.graph_storage.get_neighbors(
                node_id, 
                max_neighbors=self.config.max_neighbors
            )
            
            # Get relationship details
            relationships = []
            for neighbor_id in neighbors:
                edge_data = self.graph_storage.graph.get_edge_data(node_id, neighbor_id)
                relationships.append({
                    'neighbor_id': neighbor_id,
                    'relationship': edge_data.get('relationship', 'unknown'),
                    'weight': edge_data.get('weight', 1.0),
                    'neighbor_content': self.graph_storage.get_node_content(neighbor_id)
                })
            
            result = {
                'node_id': node_id,
                'content': self.graph_storage.get_node_content(node_id),
                'similarity_score': similarity_score,
                'relationships': relationships
            }
            
            results.append(result)
        
        return results
    
    def expand_query_with_graph(self, query: str) -> str:
        """
        Expand the query using related concepts from the graph.
        """
        # Find similar nodes to the query
        similar_nodes = self.graph_storage.find_similar_nodes(
            query, 
            top_k=3,  # Only get top 3 for expansion
            threshold=self.config.similarity_threshold
        )
        
        expanded_terms = [query]
        
        for node_id, _ in similar_nodes:
            # Get neighbors to expand the query
            neighbors = self.graph_storage.get_neighbors(
                node_id, 
                max_neighbors=2  # Only get a few neighbors to avoid too much expansion
            )
            
            for neighbor_id in neighbors:
                neighbor_content = self.graph_storage.get_node_content(neighbor_id)
                if neighbor_content:
                    # Extract key terms from neighbor content
                    terms = neighbor_content.split()[:5]  # Take first 5 terms
                    expanded_terms.extend(terms)
        
        # Return expanded query
        return " ".join(expanded_terms)
    
    def generate_response_with_template(self, query: str, relevant_nodes: List[Dict]) -> str:
        """
        Generate a response using a Jinja2 template with graph context.
        """
        # Load the graph retrieval prompt template
        prompt_template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'graph_retrieval.j2')
        with open(prompt_template_path, 'r') as f:
            template_str = f.read()
        
        template = Template(template_str)
        prompt = template.render(query=query, relevant_nodes=relevant_nodes)
        
        # Return the formatted prompt (in a real implementation, you would call an LLM here)
        return prompt