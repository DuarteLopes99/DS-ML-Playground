# Nintendo Product Recommendation System - Challenge Documentation

## Project Overview

A Retrieval-Augmented Generation (RAG) system designed to provide intelligent product recommendations for Nintendo products. The system combines semantic search, vector databases, and large language models to deliver personalized recommendations based on natural language queries.

## Project Structure

### Core Files

**`dev.ipynb`** - Development notebook containing all experimentation, prototyping, and iterative improvements. Includes data exploration, algorithm testing, and feature development.

**`rag_classes.py`** - Production-ready RAG implementations. Contains four distinct RAG architectures:
- `DualEmbeddingNintendoRAG` - Baseline cosine similarity approach
- `DualEmbeddingNintendoRAG_exclusions` - Enhanced with automatic filtering
- `ChromaDBNintendoRAG` - ChromaDB with HNSW indexing
- `ChromaDBNintendoRAG_exclusions` - ChromaDB with intelligent filtering

**`auxiliar_functions.py`** - Utility functions supporting RAG operations, including embedding generation, semantic search, ChromaDB management, and filtering operations.

**`main.ipynb`** - Clean evaluation notebook. Initializes all RAG models, runs comprehensive tests across multiple queries, and generates comparative analysis with detailed metrics.

**`data_exploration.ipynb`** - Initial data analysis with visualizations of product distributions, relationship patterns, and catalog statistics.

## RAG Architectures

### 1. Dual Embedding (Cosine Similarity)
Traditional semantic search using cosine similarity on product and relationship embeddings. No filtering capabilities.

### 2. Dual Embedding + Exclusions
Cosine similarity enhanced with automatic franchise exclusion detection and age-based filtering using LLM-powered query analysis.

### 3. ChromaDB (HNSW)
Vector database approach using ChromaDB's HNSW algorithm for efficient approximate nearest neighbor search. No filtering capabilities.

### 4. ChromaDB + Exclusions
ChromaDB efficiency combined with automatic exclusion detection and age-based filtering for optimal performance.

## Technical Components

**Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)  
**Vector Store**: ChromaDB with HNSW indexing  
**Language Model**: GPT-4o-mini for query analysis and response generation  
**Core Libraries**: sentence-transformers, chromadb, pandas, scikit-learn, openai

## Data Files

**`products_rag.json`** - Product catalog with embeddings and metadata (name, category, franchise, age rating, popularity, store availability)

**`relationships_rag.json`** - Product relationships with co-occurrence scores and strength metrics

## Test Scenarios

The evaluation includes 12 diverse queries testing:
- Out-of-scope detection (non-product queries)
- Age-based filtering (child vs teenager recommendations)
- Store availability constraints
- Interest-based matching (animals, sports, films)
- Exclusion handling (franchise filtering)
- Complex scenarios (multi-product bundles, collector items)

## Evaluation Metrics

**Performance**: Query execution time, token usage, success rate  
**Retrieval**: Number of products/relationships retrieved, relevance scores  
**Filtering**: Exclusions detected, age appropriateness  
**Quality**: Response length, recommendation accuracy

## Usage

1. Install dependencies:
```bash
pip install sentence-transformers chromadb openai pandas numpy scikit-learn
```

2. Set API key in main.ipynb or main.py

3. Run evaluation:
```bash
jupyter notebook main.ipynb
```
or
```bash
python main.py
```

## Results

The system generates:
- Comprehensive DataFrame with all test results
- CSV export with detailed metrics
- Performance comparison by model
- Query-specific analysis

**Output file**: `rag_evaluation_results_[timestamp].csv`

## Key Features

- Automatic exclusion detection from natural language
- Age-based product filtering
- Dual embedding system (products + relationships)
- Out-of-scope query handling
- Store availability filtering
- Comprehensive evaluation framework

## Development Timeline

1. Data exploration and analysis
2. RAG architecture design and prototyping
3. Exclusion detection implementation
4. Age filtering enhancement
5. Performance optimization
6. Comprehensive testing and evaluation

This implementation demonstrates practical applications of RAG systems in e-commerce, combining semantic search accuracy with intelligent business logic for improved user experience.
