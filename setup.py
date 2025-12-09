from setuptools import setup, find_packages

setup(
    name="graph-rag-system",
    version="0.1.0",
    description="A modular Graph-based Retrieval Augmented Generation system",
    author="AI Developer",
    author_email="developer@example.com",
    packages=find_packages(),
    install_requires=[
        "langchain==0.1.16",
        "langchain-community==0.0.35",
        "langchain-core==0.1.42",
        "langgraph==0.0.61",
        "tiktoken==0.7.0",
        "sentence-transformers==2.6.1",
        "faiss-cpu==1.8.0",
        "pydantic==2.6.4",
        "python-dotenv==1.0.1",
        "unstructured==0.13.2",
        "beautifulsoup4==4.12.3",
        "lxml==5.2.1",
        "networkx==3.3",
        "matplotlib==3.8.4",
        "numpy==1.24.3",
        "pandas==2.2.2",
    ],
    python_requires=">=3.8",
)