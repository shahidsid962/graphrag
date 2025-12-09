from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document as LCDocument
from src.components.graph_storage import GraphStorage
from src.components.document_processor import DocumentProcessor
from src.graph_rag.graph_builder import GraphBuilder
from src.graph_rag.retriever import GraphRAGRetriever
from src.utils.config import config
from jinja2 import Template
import os


class GraphRAGEngine:
    """
    Main Graph RAG engine that orchestrates the entire process:
    - Building knowledge graphs from documents
    - Retrieving relevant information using graph traversal
    - Generating answers using an LLM
    """
    
    def __init__(self, config=config):
        self.config = config
        
        # Initialize components
        self.graph_storage = GraphStorage(config, model_name=config.embedding_model_name)
        self.document_processor = DocumentProcessor(config)
        self.graph_builder = GraphBuilder(config, self.graph_storage, self.document_processor)
        self.retriever = GraphRAGRetriever(self.graph_storage, config)
        
        # Initialize LLM
        if config.openai_api_key:
            self.llm = ChatOpenAI(
                model_name=config.llm_model_name,
                temperature=0.1,
                api_key=config.openai_api_key
            )
        else:
            # For demo purposes, we'll use a mock response if no API key
            self.llm = None
            
        # Load existing graph if available
        self.graph_storage.load_graph()
        
        # Store the path to the answer generation template
        self.answer_template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'answer_generation.j2')
    
    def ingest_documents(self, documents: List[LCDocument]):
        """
        Ingest documents into the knowledge graph.
        """
        print("Building knowledge graph from documents...")
        self.graph_builder.build_graph_from_documents(documents)
        
        # Save the updated graph
        self.graph_storage.save_graph()
        print("Knowledge graph saved.")
    
    def query(self, question: str, top_k: int = 5) -> str:
        """
        Query the Graph RAG system and return an answer.
        """
        # First, expand the query using the graph
        expanded_query = self.retriever.expand_query_with_graph(question)
        
        # Retrieve relevant documents from the graph
        retrieved_docs = self.retriever.retrieve(expanded_query, top_k=top_k)
        
        if not retrieved_docs:
            return "I couldn't find any relevant information to answer your question."
        
        # Combine retrieved documents into context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Generate answer using LLM
        if self.llm:
            # Load and render the answer generation prompt template
            with open(self.answer_template_path, 'r') as f:
                template_str = f.read()
            
            template = Template(template_str)
            prompt = template.render(context=context, query=question)
            
            # Create a chat prompt from the rendered template
            chat_prompt = ChatPromptTemplate.from_messages([
                ("human", prompt)
            ])
            
            # Format the prompt
            formatted_prompt = chat_prompt.format_messages()
            
            # Get response from LLM
            response = self.llm(formatted_prompt)
            return response.content
        else:
            # Mock response if no LLM is configured
            return f"Mock response: Based on the context provided, here's an answer to your question '{question}'. " \
                   f"The system retrieved {len(retrieved_docs)} relevant documents from the knowledge graph."
    
    def get_graph_info(self):
        """
        Get information about the current knowledge graph.
        """
        return self.graph_storage.get_graph_info()
    
    def save_graph(self, filename: str = "knowledge_graph.pkl"):
        """
        Save the current graph to disk.
        """
        self.graph_storage.save_graph(filename)
    
    def load_graph(self, filename: str = "knowledge_graph.pkl"):
        """
        Load a graph from disk.
        """
        self.graph_storage.load_graph(filename)