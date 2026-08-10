"""
Simple RAG System using FastAPI and Cosine Similarity
This system retrieves relevant information from data.json based on user queries
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn

app = FastAPI(
    title="Simple RAG System",
    description="Retrieval Augmented Generation system using cosine similarity",
    version="1.0.0"
)


# Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3  # Number of top results to return
    threshold: Optional[float] = 0.1  # Minimum similarity threshold
    permissions: Optional[List[str]] = None  # User permissions for filtering


class DocumentResponse(BaseModel):
    id: str
    title: str
    department: str
    permissions: List[str]
    content: str
    similarity_score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[DocumentResponse]
    total_documents_found: int


# RAG System Class
class SimpleRAG:
    def __init__(self, data_file: str = "data.json"):
        self.documents = []
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.document_vectors = None
        self.load_data(data_file)
        self.vectorize_documents()
    
    def load_data(self, data_file: str):
        """Load documents from JSON file"""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"Loaded {len(self.documents)} documents")
        except FileNotFoundError:
            raise Exception(f"Data file {data_file} not found")
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON in {data_file}")
    
    def vectorize_documents(self):
        """Create TF-IDF vectors for all documents"""
        # Combine title and content for better matching
        texts = [
            f"{doc['title']} {doc['content']}" 
            for doc in self.documents
        ]
        self.document_vectors = self.vectorizer.fit_transform(texts)
        print(f"Vectorized {len(texts)} documents")
    
    def search(self, query: str, top_k: int = 3, threshold: float = 0.1, user_permissions: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for relevant documents using cosine similarity
        
        Args:
            query: User query string
            top_k: Number of top results to return
            threshold: Minimum similarity score (0-1)
            user_permissions: List of user permissions for filtering documents
        
        Returns:
            List of relevant documents with similarity scores
        """
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold, permissions, and prepare results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            doc = self.documents[idx]
            
            # Check if score meets threshold
            if score < threshold:
                continue
            
            # Check permissions if provided
            if user_permissions is not None:
                doc_permissions = doc.get('permissions', [])
                # User must have at least one matching permission
                if not any(perm in doc_permissions for perm in user_permissions):
                    continue
            
            # Add document to results
            result_doc = doc.copy()
            result_doc['similarity_score'] = score
            results.append(result_doc)
        
        # If we don't have enough results, try to get more
        if len(results) < top_k:
            # Get more indices beyond top_k
            remaining_indices = np.argsort(similarities)[::-1][top_k:top_k*3]
            for idx in remaining_indices:
                if len(results) >= top_k:
                    break
                    
                score = float(similarities[idx])
                doc = self.documents[idx]
                
                if score < threshold:
                    continue
                
                if user_permissions is not None:
                    doc_permissions = doc.get('permissions', [])
                    if not any(perm in doc_permissions for perm in user_permissions):
                        continue
                
                result_doc = doc.copy()
                result_doc['similarity_score'] = score
                results.append(result_doc)
        
        return results
    
    def generate_answer(self, query: str, documents: List[Dict]) -> str:
        """
        Generate an answer based on retrieved documents
        
        Args:
            query: User query
            documents: List of relevant documents
        
        Returns:
            Generated answer string
        """
        if not documents:
            return "I couldn't find any relevant information in the knowledge base to answer your query."
        
        # Create answer from top documents
        answer_parts = []
        answer_parts.append(f"Based on the available information:\n")
        
        for i, doc in enumerate(documents, 1):
            answer_parts.append(
                f"{i}. **{doc['title']}** (Department: {doc['department']}, "
                f"Relevance: {doc['similarity_score']:.2%}):\n"
                f"   {doc['content']}\n"
            )
        
        return "\n".join(answer_parts)


# Initialize RAG system
rag_system = SimpleRAG("data.json")


# API Endpoints
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to the Simple RAG System API",
        "endpoints": {
            "POST /query": "Submit a query to get relevant information",
            "GET /documents": "List all available documents",
            "GET /health": "Check system health"
        },
        "total_documents": len(rag_system.documents)
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Query the RAG system with a natural language question
    
    Args:
        request: QueryRequest containing the query and optional parameters
    
    Returns:
        QueryResponse with answer and source documents
    """
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Search for relevant documents with permission filtering
        results = rag_system.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            user_permissions=request.permissions
        )
        
        # Generate answer
        answer = rag_system.generate_answer(request.query, results)
        
        # Format sources
        sources = [
            DocumentResponse(
                id=doc['id'],
                title=doc['title'],
                department=doc['department'],
                permissions=doc.get('permissions', []),
                content=doc['content'],
                similarity_score=doc['similarity_score']
            )
            for doc in results
        ]
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            total_documents_found=len(results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/documents")
def list_documents(permissions: Optional[str] = None):
    """
    List all available documents in the knowledge base
    
    Args:
        permissions: Comma-separated list of permissions to filter by (e.g., "ops,admin")
    """
    filtered_docs = rag_system.documents
    
    # Filter by permissions if provided
    if permissions:
        user_perms = [p.strip() for p in permissions.split(',')]
        filtered_docs = [
            doc for doc in rag_system.documents
            if any(perm in doc.get('permissions', []) for perm in user_perms)
        ]
    
    return {
        "total": len(filtered_docs),
        "documents": [
            {
                "id": doc['id'],
                "title": doc['title'],
                "department": doc['department'],
                "permissions": doc.get('permissions', [])
            }
            for doc in filtered_docs
        ]
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents_loaded": len(rag_system.documents),
        "vectorizer_ready": rag_system.document_vectors is not None
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
