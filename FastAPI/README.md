# Simple RAG System with FastAPI

A Retrieval-Augmented Generation (RAG) system that uses **cosine similarity** to find and retrieve relevant information from a JSON knowledge base (`data.json`) and respond to user queries.

## 🚀 Features

- **FastAPI** backend with automatic API documentation
- **TF-IDF vectorization** for text representation
- **Cosine similarity** for semantic matching
- Configurable top-k results and similarity threshold
- RESTful API endpoints
- Interactive API documentation (Swagger UI)
- Test client for easy testing

## 📋 Prerequisites

- Python 3.8+
- pip

## 🛠️ Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## 🏃 Running the Application

### Start the FastAPI Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

### Access Interactive API Documentation

Once the server is running, open your browser:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

### 2. Query Endpoint
```
POST /query
```

**Request Body:**
```json
{
  "query": "How do I handle a production deployment rollback?",
  "top_k": 3,
  "threshold": 0.1
}
```

**Response:**
```json
{
  "query": "How do I handle a production deployment rollback?",
  "answer": "Based on the available information...",
  "sources": [
    {
      "id": "doc-001",
      "title": "Deployment Rollback Procedure",
      "department": "Operations",
      "content": "...",
      "similarity_score": 0.85
    }
  ],
  "total_documents_found": 1
}
```

### 3. List Documents
```
GET /documents
```
Returns all available documents in the knowledge base.

### 4. Health Check
```
GET /health
```
Returns system health status.

## 🧪 Testing with the Test Client

Run the interactive test client:

```bash
python test_client.py
```

This will:
1. Check system health
2. List all available documents
3. Run example queries

## 💻 Using the API Programmatically

### Python Example

```python
import requests

# Query the RAG system
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What are the API authentication requirements?",
        "top_k": 3,
        "threshold": 0.1
    }
)

result = response.json()
print(result['answer'])
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does predictive maintenance work?",
    "top_k": 2,
    "threshold": 0.15
  }'
```

### JavaScript/Fetch Example

```javascript
fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'Tell me about AI response validation',
    top_k: 3,
    threshold: 0.1
  })
})
.then(response => response.json())
.then(data => console.log(data.answer));
```

## 🔧 Configuration

You can adjust the RAG system parameters:

- **top_k**: Number of most relevant documents to return (default: 3)
- **threshold**: Minimum similarity score (0-1) to include a document (default: 0.1)

### In the code:
```python
# main.py - Modify TfidfVectorizer parameters
self.vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),  # Use 1-2 word phrases
    max_features=1000    # Maximum vocabulary size
)
```

## 📊 How It Works

1. **Loading**: System loads documents from `data.json`
2. **Vectorization**: Documents are converted to TF-IDF vectors
3. **Query Processing**: User query is vectorized using the same vectorizer
4. **Similarity Calculation**: Cosine similarity computed between query and all documents
5. **Ranking**: Top-k most similar documents are retrieved
6. **Response Generation**: Answer is formatted with relevant document content

## 🎯 Example Queries

- "How do I handle a production deployment rollback?"
- "What are the API authentication requirements?"
- "Tell me about AI response validation"
- "What is the data retention policy?"
- "How does predictive maintenance work?"

## 📁 Project Structure

```
.
├── main.py              # FastAPI application and RAG system
├── test_client.py       # Test client for API interaction
├── data.json            # Knowledge base (your documents)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔍 Understanding Cosine Similarity

The system uses **cosine similarity** to measure how similar the user's query is to each document:

- **Score of 1.0**: Perfect match
- **Score of 0.5**: Moderate similarity
- **Score of 0.0**: No similarity

The `threshold` parameter filters out documents below a certain similarity score.

## 🚧 Troubleshooting

### Port already in use
```bash
# Change port in main.py or use:
uvicorn main:app --reload --port 8001
```

### Module not found
```bash
pip install -r requirements.txt
```

### No results found
- Try lowering the `threshold` parameter
- Check if your query keywords match document content
- Ensure `data.json` is properly formatted

## 🔮 Future Enhancements

- Add semantic embeddings (e.g., Sentence Transformers)
- Implement permission-based filtering
- Add caching for faster responses
- Support for multiple languages
- Add query history and analytics

## 📝 License

This is a simple educational project. Feel free to use and modify as needed!
