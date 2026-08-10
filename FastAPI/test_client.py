"""
Test client to interact with the RAG system API
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_query(query: str, top_k: int = 3, threshold: float = 0.1, permissions: list = None):
    """Send a query to the RAG system"""
    url = f"{BASE_URL}/query"
    payload = {
        "query": query,
        "top_k": top_k,
        "threshold": threshold
    }
    
    if permissions is not None:
        payload["permissions"] = permissions
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n{'='*80}")
        print(f"QUERY: {result['query']}")
        print(f"{'='*80}")
        print(f"\nANSWER:")
        print(result['answer'])
        print(f"\n{'='*80}")
        print(f"Found {result['total_documents_found']} relevant document(s)")
        print(f"{'='*80}\n")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def list_documents():
    """List all available documents"""
    url = f"{BASE_URL}/documents"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n{'='*80}")
        print(f"AVAILABLE DOCUMENTS ({result['total']} total)")
        print(f"{'='*80}")
        
        for doc in result['documents']:
            print(f"• [{doc['id']}] {doc['title']} - {doc['department']}")
        
        print(f"{'='*80}\n")
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def check_health():
    """Check API health"""
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n{'='*80}")
        print("SYSTEM HEALTH CHECK")
        print(f"{'='*80}")
        print(f"Status: {result['status']}")
        print(f"Documents loaded: {result['documents_loaded']}")
        print(f"Vectorizer ready: {result['vectorizer_ready']}")
        print(f"{'='*80}\n")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    print("\n🚀 RAG System Test Client")
    print("=" * 80)
    
    # Check health
    print("\n1. Checking system health...")
    check_health()
    
    # List documents
    print("\n2. Listing available documents...")
    list_documents()
    
    # Example queries
    print("\n3. Running example queries...")
    
    queries = [
        "How do I handle a production deployment rollback?",
        "What are the API authentication requirements?",
        "Tell me about AI response validation",
        "What is the data retention policy?",
        "How does predictive maintenance work?"
    ]
    
    for query in queries:
        test_query(query, top_k=2, threshold=0.1)
        input("Press Enter to continue to next query...")
