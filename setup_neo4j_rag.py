#!/usr/bin/env python3
"""
Setup script for Neo4j-based Graph RAG system.
This script helps initialize and configure the system with Neo4j.
"""

import os
import sys
from pathlib import Path

def check_neo4j_connection():
    """Check if Neo4j is accessible."""
    try:
        from neo4j import GraphDatabase
        from src.utils.config import config
        
        print("Testing Neo4j connection...")
        driver = GraphDatabase.driver(config.neo4j_uri, auth=(config.neo4j_username, config.neo4j_password))
        
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            record = result.single()
            if record and record["test"] == 1:
                print("✓ Successfully connected to Neo4j!")
                driver.close()
                return True
            else:
                print("✗ Failed to execute test query")
                driver.close()
                return False
    except Exception as e:
        print(f"✗ Neo4j connection failed: {e}")
        print("\nMake sure:")
        print("- Neo4j is running at the configured URI")
        print("- Username and password are correct in .env file")
        print("- Neo4j driver is installed (pip install neo4j)")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    os.system("pip install -r requirements.txt")

def create_directories():
    """Create necessary directories."""
    data_path = Path("data")
    graphs_path = Path("graphs")
    
    data_path.mkdir(exist_ok=True)
    graphs_path.mkdir(exist_ok=True)
    
    print(f"✓ Created directories: {data_path}, {graphs_path}")

def main():
    print("Graph RAG System with Neo4j - Setup")
    print("=" * 40)
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Check Neo4j connection
    if check_neo4j_connection():
        print("\n✓ Setup completed successfully!")
        print("\nTo start the Chainlit UI:")
        print("chainlit run ui/app.py -h")
    else:
        print("\n⚠ Setup incomplete. Please fix Neo4j connection issues.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())