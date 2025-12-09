"""
Graph RAG System - A modular implementation of Graph-based Retrieval Augmented Generation
"""

from .src.graph_rag.engine import GraphRAGEngine
from .src.utils.config import config

__all__ = ["GraphRAGEngine", "config"]