"""
Test client to demonstrate permission-based filtering in RAG system
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_query_with_permissions(query: str, permissions: list, top_k: int = 3, threshold: float = 0.1):
    """Send a query to the RAG system with permission filtering"""
    url = f"{BASE_URL}/query"
    payload = {
        "query": query,
        "top_k": top_k,
        "threshold": threshold,
        "permissions": permissions
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"\n{'='*80}")
        print(f"QUERY: {result['query']}")
        print(f"USER PERMISSIONS: {', '.join(permissions)}")
        print(f"{'='*80}")
        print(f"\nANSWER:")
        print(result['answer'])
        print(f"\n{'='*80}")
        print(f"Found {result['total_documents_found']} relevant document(s)")
        
        if result['sources']:
            print("\nSOURCE DOCUMENTS:")
            for i, source in enumerate(result['sources'], 1):
                print(f"\n  {i}. {source['title']}")
                print(f"     Department: {source['department']}")
                print(f"     Permissions: {', '.join(source['permissions'])}")
                print(f"     Relevance Score: {source['similarity_score']:.2%}")
        
        print(f"{'='*80}\n")
        
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None


def list_documents_by_permission(permissions: list = None):
    """List documents filtered by permissions"""
    url = f"{BASE_URL}/documents"
    
    if permissions:
        params = {"permissions": ",".join(permissions)}
        response = requests.get(url, params=params)
    else:
        response = requests.get(url)
    
    try:
        response.raise_for_status()
        result = response.json()
        
        print(f"\n{'='*80}")
        if permissions:
            print(f"DOCUMENTS ACCESSIBLE WITH PERMISSIONS: {', '.join(permissions)}")
        else:
            print("ALL DOCUMENTS (NO PERMISSION FILTER)")
        print(f"{'='*80}")
        print(f"Total: {result['total']} documents\n")
        
        for doc in result['documents']:
            print(f"• [{doc['id']}] {doc['title']}")
            print(f"  Department: {doc['department']}")
            print(f"  Required Permissions: {', '.join(doc['permissions'])}\n")
        
        print(f"{'='*80}\n")
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


def demonstrate_permission_filtering():
    """Demonstrate different permission scenarios"""
    
    print("\n" + "="*80)
    print("🔐 RAG SYSTEM - PERMISSION-BASED FILTERING DEMONSTRATION")
    print("="*80)
    
    # Scenario 1: Operations team member
    print("\n" + "="*80)
    print("SCENARIO 1: Operations Team Member (permissions: ['ops'])")
    print("="*80)
    list_documents_by_permission(['ops'])
    test_query_with_permissions(
        query="How do I handle a production deployment?",
        permissions=['ops'],
        top_k=3
    )
    
    # Scenario 2: AI Engineering team member
    print("\n" + "="*80)
    print("SCENARIO 2: AI Engineering Team Member (permissions: ['ai'])")
    print("="*80)
    list_documents_by_permission(['ai'])
    test_query_with_permissions(
        query="Tell me about AI response validation",
        permissions=['ai'],
        top_k=3
    )
    
    # Scenario 3: Admin with full access
    print("\n" + "="*80)
    print("SCENARIO 3: Administrator (permissions: ['admin'])")
    print("="*80)
    list_documents_by_permission(['admin'])
    test_query_with_permissions(
        query="What security policies should I know about?",
        permissions=['admin'],
        top_k=3
    )
    
    # Scenario 4: Multiple permissions (platform engineer + ops)
    print("\n" + "="*80)
    print("SCENARIO 4: Platform Engineer (permissions: ['platform', 'ops'])")
    print("="*80)
    list_documents_by_permission(['platform', 'ops'])
    test_query_with_permissions(
        query="What are the deployment procedures?",
        permissions=['platform', 'ops'],
        top_k=3
    )
    
    # Scenario 5: Compliance officer
    print("\n" + "="*80)
    print("SCENARIO 5: Compliance Officer (permissions: ['compliance'])")
    print("="*80)
    list_documents_by_permission(['compliance'])
    test_query_with_permissions(
        query="What are the data retention requirements?",
        permissions=['compliance'],
        top_k=3
    )
    
    # Scenario 6: No permissions (should return no results with sensitive info)
    print("\n" + "="*80)
    print("SCENARIO 6: Guest User (no permissions)")
    print("="*80)
    test_query_with_permissions(
        query="Tell me about the company policies",
        permissions=[],
        top_k=3
    )
    
    # Scenario 7: Query without permission filter (returns all matching docs)
    print("\n" + "="*80)
    print("SCENARIO 7: Query WITHOUT Permission Filter (shows all matches)")
    print("="*80)
    url = f"{BASE_URL}/query"
    payload = {
        "query": "What are the security requirements?",
        "top_k": 3,
        "threshold": 0.1
    }
    response = requests.post(url, json=payload)
    result = response.json()
    
    print(f"\nQUERY: {result['query']}")
    print(f"USER PERMISSIONS: None (no filter applied)")
    print(f"\nFound {result['total_documents_found']} documents")
    if result['sources']:
        print("\nSOURCE DOCUMENTS:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['title']} - Permissions: {', '.join(source['permissions'])}")
    print("="*80 + "\n")


if __name__ == "__main__":
    demonstrate_permission_filtering()
