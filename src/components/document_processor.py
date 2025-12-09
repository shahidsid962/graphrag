from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LCDocument
import re
import json
from jinja2 import Template
import os


class DocumentProcessor:
    """
    Document processor component for the Graph RAG system.
    Handles document loading, preprocessing, and chunking.
    """
    
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
    def process_documents(self, documents: List[LCDocument]) -> List[LCDocument]:
        """
        Process a list of documents by chunking them into smaller pieces.
        """
        processed_docs = []
        
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            processed_docs.extend(chunks)
            
        return processed_docs
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using LLM with a Jinja2 template.
        This provides a more sophisticated approach than regex patterns.
        """
        # Load the entity extraction prompt template
        prompt_template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'entity_extraction.j2')
        with open(prompt_template_path, 'r') as f:
            template_str = f.read()
        
        template = Template(template_str)
        prompt = template.render(text=text)
        
        # Return a mock response structure since we're not calling an actual LLM
        # In a real implementation, you would call an LLM here with the prompt
        # and parse the JSON response
        
        # For demonstration purposes, we'll simulate the response
        # In a real application, you'd use an LLM to generate this
        return self._parse_mock_entity_response(text)
    
    def _parse_mock_entity_response(self, text: str) -> List[Dict[str, Any]]:
        """
        Mock method to simulate parsing of LLM response.
        In a real implementation, this would parse actual LLM response.
        """
        entities = []
        
        # Extract potential entities using regex as fallback
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Extract potential entities (nouns, proper nouns, etc.)
            words = re.findall(r'\b[A-Z][a-z]+\b', sentence)
            for word in words:
                if len(word) > 2:  # Filter out short words
                    entities.append({
                        'entity': word,
                        'context': sentence,
                        'type': 'entity'
                    })
                    
        return entities
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        return text.strip()