import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class Neo4jGraphStorage:
    """
    Neo4j-based graph storage component for Graph RAG system.
    Manages the knowledge graph with nodes representing entities/concepts
    and edges representing relationships between them in Neo4j database.
    """
    
    def __init__(self, config, model_name: str = "all-MiniLM-L6-v2"):
        self.config = config
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize Neo4j connection
        self.uri = config.neo4j_uri or "bolt://localhost:7687"
        self.username = config.neo4j_username or "neo4j"
        self.password = config.neo4j_password or "password"
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def add_node(self, node_id: str, content: str, metadata: Dict[str, Any] = None):
        """Add a node to the Neo4j graph with content and metadata."""
        embeddings = self.embedding_model.encode([content])
        embedding_list = embeddings[0].tolist()  # Convert numpy array to list for Neo4j
        
        with self.driver.session() as session:
            query = """
            MERGE (n:Node {id: $node_id})
            SET n.content = $content,
                n.embedding = $embedding,
                n.metadata = $metadata
            """
            session.run(
                query,
                node_id=node_id,
                content=content,
                embedding=embedding_list,
                metadata=json.dumps(metadata or {})
            )

    def add_edge(self, node1_id: str, node2_id: str, relationship: str = "RELATED", weight: float = 1.0):
        """Add an edge between two nodes with a relationship type and weight."""
        with self.driver.session() as session:
            query = """
            MATCH (n1:Node {id: $node1_id})
            MATCH (n2:Node {id: $node2_id})
            MERGE (n1)-[r:RELATIONSHIP {type: $relationship}]->(n2)
            SET r.weight = $weight,
                r.created_at = datetime()
            """
            session.run(
                query,
                node1_id=node1_id,
                node2_id=node2_id,
                relationship=relationship,
                weight=weight
            )

    def find_similar_nodes(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find nodes similar to the query based on semantic similarity using vector search."""
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        with self.driver.session() as session:
            # First, get all nodes with embeddings
            query_all = """
            MATCH (n:Node)
            WHERE n.embedding IS NOT NULL
            RETURN n.id as id, n.embedding as embedding
            """
            result = session.run(query_all)
            
            similarities = []
            for record in result:
                node_embedding = record["embedding"]
                if node_embedding:
                    # Calculate cosine similarity
                    node_embedding_array = np.array(node_embedding).reshape(1, -1)
                    query_embedding_array = np.array(query_embedding).reshape(1, -1)
                    sim = cosine_similarity(query_embedding_array, node_embedding_array)[0][0]
                    
                    if sim >= threshold:
                        similarities.append((record["id"], float(sim)))
            
            # Sort by similarity score descending
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

    def get_neighbors(self, node_id: str, max_neighbors: int = 5) -> List[str]:
        """Get neighboring nodes of a given node."""
        with self.driver.session() as session:
            query = """
            MATCH (n:Node {id: $node_id})--(neighbor:Node)
            RETURN neighbor.id as id
            LIMIT $max_neighbors
            """
            result = session.run(query, node_id=node_id, max_neighbors=max_neighbors)
            
            neighbors = [record["id"] for record in result]
            return neighbors

    def get_node_content(self, node_id: str) -> Optional[str]:
        """Get content of a specific node."""
        with self.driver.session() as session:
            query = """
            MATCH (n:Node {id: $node_id})
            RETURN n.content as content
            """
            result = session.run(query, node_id=node_id)
            
            record = result.single()
            return record["content"] if record else None

    def get_node_metadata(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata of a specific node."""
        with self.driver.session() as session:
            query = """
            MATCH (n:Node {id: $node_id})
            RETURN n.metadata as metadata
            """
            result = session.run(query, node_id=node_id)
            
            record = result.single()
            if record and record["metadata"]:
                return json.loads(record["metadata"])
            return None

    def get_node_relationships(self, node_id: str) -> List[Dict[str, Any]]:
        """Get relationships of a specific node."""
        with self.driver.session() as session:
            query = """
            MATCH (n:Node {id: $node_id})-[r]->(connected:Node)
            RETURN r.type as type, r.weight as weight, connected.id as connected_id
            UNION
            MATCH (n:Node {id: $node_id})<-[r]-(connected:Node)
            RETURN r.type as type, r.weight as weight, connected.id as connected_id
            """
            result = session.run(query, node_id=node_id)
            
            relationships = []
            for record in result:
                relationships.append({
                    "type": record["type"],
                    "weight": record["weight"],
                    "connected_node_id": record["connected_id"]
                })
            return relationships

    def get_subgraph_by_nodes(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Extract subgraph containing specified nodes and their relationships."""
        with self.driver.session() as session:
            query = """
            MATCH (n:Node)
            WHERE n.id IN $node_ids
            OPTIONAL MATCH (n)-[r]-(connected:Node)
            WHERE connected.id IN $node_ids
            RETURN n.id as node_id, n.content as content, n.metadata as metadata,
                   r.type as rel_type, r.weight as rel_weight, connected.id as connected_id
            """
            result = session.run(query, node_ids=node_ids)
            
            nodes = {}
            relationships = []
            
            for record in result:
                # Add node if not already present
                node_id = record["node_id"]
                if node_id not in nodes:
                    nodes[node_id] = {
                        "id": node_id,
                        "content": record["content"],
                        "metadata": json.loads(record["metadata"]) if record["metadata"] else {}
                    }
                
                # Add relationship if exists
                if record["rel_type"]:
                    relationships.append({
                        "from": node_id,
                        "to": record["connected_id"],
                        "type": record["rel_type"],
                        "weight": record["rel_weight"]
                    })
            
            return {"nodes": list(nodes.values()), "relationships": relationships}

    def get_graph_info(self) -> Dict[str, Any]:
        """Get information about the graph."""
        with self.driver.session() as session:
            # Count nodes
            node_count_query = "MATCH (n:Node) RETURN count(n) as count"
            node_count = session.run(node_count_query).single()["count"]
            
            # Count relationships
            rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
            rel_count = session.run(rel_count_query).single()["count"]
            
            # Get some sample relationship types
            rel_types_query = "MATCH ()-[r]->() RETURN DISTINCT type(r) as rel_type"
            rel_types = [record["rel_type"] for record in session.run(rel_types_query)]
            
            return {
                "num_nodes": node_count,
                "num_edges": rel_count,
                "relationship_types": rel_types
            }

    def clear_graph(self):
        """Clear all nodes and relationships from the graph."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self):
        """Close the Neo4j driver connection."""
        if hasattr(self, 'driver'):
            self.driver.close()

    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close()