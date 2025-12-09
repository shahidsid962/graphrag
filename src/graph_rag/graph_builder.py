from typing import List, Dict, Any
from langchain.schema import Document as LCDocument
from src.components.graph_storage import GraphStorage
from src.components.document_processor import DocumentProcessor
from jinja2 import Template
import os


class GraphBuilder:
    """
    Graph builder component for the Graph RAG system.
    Builds and maintains the knowledge graph from documents.
    """
    
    def __init__(self, config, graph_storage: GraphStorage, document_processor: DocumentProcessor):
        self.config = config
        self.graph_storage = graph_storage
        self.document_processor = document_processor
        
    def build_graph_from_documents(self, documents: List[LCDocument]):
        """
        Build knowledge graph from a list of documents.
        Creates nodes for chunks and entities, and edges for relationships.
        """
        print("Processing documents...")
        processed_docs = self.document_processor.process_documents(documents)
        
        print("Adding document chunks to graph...")
        for i, doc in enumerate(processed_docs):
            node_id = f"chunk_{i}"
            content = doc.page_content
            metadata = doc.metadata
            
            self.graph_storage.add_node(
                node_id=node_id,
                content=content,
                metadata={
                    'type': 'document_chunk',
                    'source': metadata.get('source', 'unknown'),
                    'page': metadata.get('page', -1)
                }
            )
        
        print("Extracting and adding entities to graph...")
        entity_id_counter = 0
        for doc in processed_docs:
            entities = self.document_processor.extract_entities(doc.page_content)
            
            for entity_data in entities:
                entity = entity_data['entity']
                context = entity_data['context']
                
                # Create a unique ID for the entity
                entity_node_id = f"entity_{entity_id_counter}_{entity.lower().replace(' ', '_')}"
                entity_id_counter += 1
                
                # Add entity node to graph
                self.graph_storage.add_node(
                    node_id=entity_node_id,
                    content=f"Entity: {entity}. Context: {context}",
                    metadata={
                        'type': 'entity',
                        'name': entity,
                        'original_context': context
                    }
                )
                
                # Connect entity to the document chunk it came from
                chunk_node_id = f"chunk_{processed_docs.index(doc)}"
                self.graph_storage.add_edge(
                    chunk_node_id,
                    entity_node_id,
                    relationship="mentions",
                    weight=1.0
                )
        
        print("Connecting related entities...")
        # Connect related entities based on co-occurrence in similar contexts
        self._connect_related_entities()
        
        print(f"Graph built successfully with {self.graph_storage.graph.number_of_nodes()} nodes "
              f"and {self.graph_storage.graph.number_of_edges()} edges")
        
    def _connect_related_entities(self):
        """
        Connect entities that appear in similar contexts or are semantically related.
        """
        entity_nodes = [n for n, data in self.graph_storage.graph.nodes(data=True) 
                       if data.get('metadata', {}).get('type') == 'entity']
        
        # Simple approach: connect entities that appear in documents with similar content
        for i, node1 in enumerate(entity_nodes):
            for j, node2 in enumerate(entity_nodes[i+1:], i+1):
                if node1 != node2:
                    # Get content of both entities
                    content1 = self.graph_storage.get_node_content(node1)
                    content2 = self.graph_storage.get_node_content(node2)
                    
                    # If entities share common words in their contexts, connect them
                    if content1 and content2:
                        # Extract context parts to check for similarity
                        ctx1 = content1.lower()
                        ctx2 = content2.lower()
                        
                        # Simple heuristic: if they share at least 2 common words, connect
                        words1 = set(ctx1.split())
                        words2 = set(ctx2.split())
                        common_words = words1.intersection(words2)
                        
                        if len(common_words) >= 2:
                            self.graph_storage.add_edge(
                                node1, 
                                node2, 
                                relationship="related", 
                                weight=min(1.0, len(common_words) / 5.0)  # Normalize weight
                            )
    
    def update_graph(self, new_documents: List[LCDocument]):
        """
        Update the existing graph with new documents.
        """
        self.build_graph_from_documents(new_documents)