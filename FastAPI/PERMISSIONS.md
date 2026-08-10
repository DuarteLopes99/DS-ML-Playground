# Permission-Based Filtering in RAG System

## 🔐 Overview

The RAG system now supports **permission-based filtering**! This ensures users only see documents they have access to based on their role permissions.

## 📋 Available Permissions in data.json

Based on your data, here are the available permission types:

- **`admin`** - Full access to all documents (12 documents)
- **`ops`** - Operations team access (4 documents)
- **`ai`** - AI Engineering team access (3 documents)
- **`platform`** - Platform Engineering team access (3 documents)
- **`compliance`** - Compliance team access (2 documents)
- **`security`** - Security team access (1 document)
- **`support`** - Support team access (1 document)
- **`industrial`** - Industrial AI team access (2 documents)
- **`hr`** - HR team access (1 document)

## 🚀 How to Use Permission Filtering

### Option 1: API Request with Permissions

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I handle a production deployment?",
    "permissions": ["ops", "platform"],
    "top_k": 3,
    "threshold": 0.1
  }'
```

### Option 2: Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What are the API authentication requirements?",
        "permissions": ["security", "admin"],
        "top_k": 3
    }
)

result = response.json()
print(result['answer'])
```

### Option 3: Swagger UI (Interactive)

1. Go to http://localhost:8000/docs
2. Click on **POST /query**
3. Click "Try it out"
4. Enter your query and permissions:
   ```json
   {
     "query": "Tell me about AI validation",
     "permissions": ["ai", "admin"],
     "top_k": 3,
     "threshold": 0.1
   }
   ```
5. Click "Execute"

### Option 4: Test Script

```bash
python test_permissions.py
```

## 📊 Permission Scenarios

### Scenario 1: Operations Team Member
```python
# User has 'ops' permission
permissions = ["ops"]
# Will see: Deployment Rollback, Incident Escalation, Production Deployment Checklist
```

### Scenario 2: AI Engineer
```python
# User has 'ai' permission
permissions = ["ai"]
# Will see: AI Response Validation, Enterprise AI Search, AI Service Observability
```

### Scenario 3: Administrator
```python
# User has 'admin' permission
permissions = ["admin"]
# Will see: ALL 12 documents (admin has access to everything)
```

### Scenario 4: Platform Engineer
```python
# User has both 'platform' and 'ops' permissions
permissions = ["platform", "ops"]
# Will see: All platform + ops documents (5 unique documents)
```

### Scenario 5: No Permissions
```python
# User has no permissions
permissions = []
# Will see: NO documents (properly secured)
```

### Scenario 6: No Permission Filter
```python
# Don't include permissions in the request
# Will see: ALL matching documents regardless of permissions
```

## 🔍 How Permission Filtering Works

1. **User sends query** with their permission list
2. **System searches** all documents using cosine similarity
3. **Permission check**: For each document, system verifies:
   - Does the user have ANY permission that matches the document's required permissions?
   - If YES → Document is included in results
   - If NO → Document is filtered out
4. **Return results** with only authorized documents

### Permission Logic

```python
# Document permissions: ["ops", "admin"]
# User permissions: ["ops"]
# Result: ✅ ALLOWED (user has "ops")

# Document permissions: ["compliance", "admin"]
# User permissions: ["ops"]
# Result: ❌ DENIED (user has neither "compliance" nor "admin")

# Document permissions: ["ai", "platform", "admin"]
# User permissions: ["platform", "security"]
# Result: ✅ ALLOWED (user has "platform")
```

## 📡 API Endpoints with Permissions

### 1. Query with Permission Filter
```
POST /query
Body: {
  "query": "string",
  "permissions": ["permission1", "permission2"],
  "top_k": 3,
  "threshold": 0.1
}
```

### 2. List Documents by Permission
```
GET /documents?permissions=ops,admin
```

Example:
```bash
# List all documents accessible to ops team
curl "http://localhost:8000/documents?permissions=ops"

# List all documents accessible to admin
curl "http://localhost:8000/documents?permissions=admin"

# List all documents (no filter)
curl "http://localhost:8000/documents"
```

## 🎯 Example Usage

### Example 1: Operations Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I rollback a deployment?",
    "permissions": ["ops"]
  }'
```

**Result**: Only returns documents with `ops` in their permissions list.

### Example 2: AI Team Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the AI validation requirements?",
    "permissions": ["ai"]
  }'
```

**Result**: Only returns AI-related documents.

### Example 3: Multi-Permission Query
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about security policies",
    "permissions": ["security", "compliance", "admin"]
  }'
```

**Result**: Returns documents matching ANY of the listed permissions.

## 🧪 Testing Permission Features

Run the comprehensive permission test:

```bash
python test_permissions.py
```

This will demonstrate:
- ✅ Operations team access
- ✅ AI Engineering team access
- ✅ Administrator full access
- ✅ Multi-permission scenarios
- ✅ Compliance officer access
- ✅ No permission (denied) scenarios
- ✅ Unfiltered queries

## 🔒 Security Best Practices

1. **Always validate permissions** on the server side (already implemented)
2. **Use authentication** to get user permissions (integrate with your auth system)
3. **Log permission checks** for audit trails
4. **Test edge cases** (empty permissions, invalid permissions, etc.)

## 📝 Response Format with Permissions

```json
{
  "query": "How do I handle deployments?",
  "answer": "Based on the available information:\n...",
  "sources": [
    {
      "id": "doc-001",
      "title": "Deployment Rollback Procedure",
      "department": "Operations",
      "permissions": ["ops", "admin"],
      "content": "...",
      "similarity_score": 0.85
    }
  ],
  "total_documents_found": 1
}
```

Note: Each source now includes the **`permissions`** field showing which permissions are required to access that document.

## 🎨 Integration with Authentication

To integrate with your authentication system:

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

def get_user_permissions(token: str = Depends(security)) -> List[str]:
    # Verify token and extract permissions
    # This is where you'd validate JWT, OAuth, etc.
    user = verify_token(token.credentials)
    return user.permissions

@app.post("/query")
def query_rag(
    request: QueryRequest,
    user_permissions: List[str] = Depends(get_user_permissions)
):
    # Use user_permissions from authentication
    request.permissions = user_permissions
    # ... rest of the code
```

## 🔮 Future Enhancements

- [ ] Role-based access control (RBAC)
- [ ] Hierarchical permissions (admin inherits all)
- [ ] Permission expiration/time-based access
- [ ] Audit logging for permission checks
- [ ] Integration with external auth providers (OAuth, LDAP, etc.)

## 📊 Permission Matrix

| Permission | Documents | Description |
|------------|-----------|-------------|
| `admin` | 12 | Full access to all documents |
| `ops` | 4 | Operations and deployment docs |
| `ai` | 3 | AI engineering documents |
| `platform` | 3 | Platform engineering docs |
| `compliance` | 2 | Compliance and regulations |
| `security` | 1 | Security policies |
| `support` | 1 | Support procedures |
| `industrial` | 2 | Industrial AI documents |
| `hr` | 1 | HR policies |

---

**Ready to test?** Run `python test_permissions.py` to see permission filtering in action! 🚀
