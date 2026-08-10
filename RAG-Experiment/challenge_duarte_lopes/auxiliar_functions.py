import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import re
from typing import List, Dict, Optional



def semantic_search_relationships(
    query: str,
    df_relationships: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 5,
    filters: dict = None
) -> pd.DataFrame:
    """
    Perform semantic search on relationships
    
    Args:
        query: User's search query
        df_relationships: DataFrame with relationship embeddings
        model: SentenceTransformer model
        top_k: Number of results to return
        filters: Optional filters (strength, category, etc.)
    
    Returns:
        DataFrame with top K most similar relationships
    """
    # Apply filters
    filtered_df = df_relationships.copy()
    
    if filters:        
        if 'strength' in filters:
            # Filter by relationship strength
            filtered_df = filtered_df[filtered_df['strength'] == filters['strength']]
        
        if 'min_cooccurrence' in filters:
            # Filter by minimum co-occurrence score
            filtered_df = filtered_df[
                filtered_df['cooccurrence_score'] >= filters['min_cooccurrence']
            ]
        
        if 'category' in filters:
            # Filter by product category
            category = filters['category']
            filtered_df = filtered_df[
                (filtered_df['product_a_category'] == category) |
                (filtered_df['product_b_category'] == category)
            ]
        
    
    if len(filtered_df) == 0:
        print(" No relationships match the filters")
        return pd.DataFrame()
    
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    
    # Get embeddings matrix
    embeddings_matrix = np.vstack(filtered_df['embedding'].values)
    
    # Compute cosine similarities
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings_matrix
    )[0]
    
    # Add similarity scores
    filtered_df = filtered_df.copy()
    filtered_df['similarity'] = similarities
    
    # Sort and get top K
    results = filtered_df.nlargest(top_k, 'similarity')
    
    # for i, row in results.iterrows():
        # print(f"   {row['product_a_name']} ↔ {row['product_b_name']}: {row['similarity']:.3f}")
    
    return results.reset_index(drop=True)

def semantic_search(
    query: str,
    df: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 5,
    filters: dict = None
) -> pd.DataFrame:
    """
    Perform semantic search using cosine similarity
    
    Args:
        query: User's search query
        df: DataFrame with embeddings
        model: SentenceTransformer model
        top_k: Number of results to return
        filters: Optional filters (dict with column: value pairs)
    
    Returns:
        DataFrame with top K most similar products
    """

    # Apply filters first (before computing similarities)
    filtered_df = df.copy()
    
    if filters:
        for col, value in filters.items():
            if col == 'min_age':
                # Age filter: products suitable for age <= value
                filtered_df = filtered_df[filtered_df['min_age'] <= value]
            elif col == 'available_stores':
                # Store filter: check if store is in the list
                filtered_df = filtered_df[
                    filtered_df['available_stores'].apply(lambda x: value in x)
                ]
            elif col in filtered_df.columns:
                # Exact match filter
                filtered_df = filtered_df[filtered_df[col] == value]
        
    
    if len(filtered_df) == 0:
        print(" No products match the filters")
        return pd.DataFrame()
    
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    
    # Get embeddings matrix from filtered DataFrame
    embeddings_matrix = np.vstack(filtered_df['embedding'].values)
    
    # Compute cosine similarities
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings_matrix
    )[0]
    
    # Add similarity scores to DataFrame
    filtered_df = filtered_df.copy()
    filtered_df['similarity'] = similarities
    
    # Sort by similarity and get top K
    results = filtered_df.nlargest(top_k, 'similarity')
    
    # for i, row in results.iterrows():
    #     print(f"   {row['name']}: {row['similarity']:.3f}")
    
    return results.reset_index(drop=True)

def create_embeddings_dataframe(
    products_data: dict,
    model: SentenceTransformer
) -> pd.DataFrame:
    """
    Create pandas DataFrame with embeddings for all products
    
    Args:
        products_data: Dictionary from products_rag.json
        model: SentenceTransformer model for generating embeddings
    
    Returns:
        DataFrame with products and their embeddings
    """
    

    # Prepare lists for DataFrame
    rows = []
    
    for product in products_data['products']:
        # Extract metadata
        meta = product['metadata']
        
        # Create row with flattened structure
        row = {
            # Core identifiers
            'id': product['id'],
            'product_id': meta['product_id'],
            'name': meta['name'],
            'text': product['text'],  # Full description for embedding
            
            # Categories
            'category': meta['category'],
            'main_category': meta['main_category'],
            
            # Game info
            'type': meta.get('type'),
            'franchise': meta.get('franchise'),
            'min_age': meta.get('min_age', 0),
            'release_date': meta.get('release_date'),
            
            # Business metrics
            'times_sold': meta['times_sold'],
            'popularity_rank': meta['popularity_rank'],
            'total_stock': meta['total_stock'],
            'is_available': meta['is_available'],
            
            # Store availability (keep as list for easier filtering)
            'available_stores': meta['available_stores'],
            
            # Store-specific sales
            'times_sold_Store_A': meta['times_sold_Store_A'],
            'times_sold_Store_B': meta['times_sold_Store_B'],
            'times_sold_Store_C': meta['times_sold_Store_C'],
        }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    
    # Store embeddings as numpy array (more efficient than list)
    df['embedding'] = list(embeddings)
    return df

def create_relationships_embeddings_dataframe(
    relationships_data: dict,
    model: SentenceTransformer
) -> pd.DataFrame:
    """
    Create pandas DataFrame with embeddings for all relationships
    
    Args:
        relationships_data: Dictionary from relationships_rag.json
        model: SentenceTransformer model
    
    Returns:
        DataFrame with relationships and their embeddings
    """
    
    rows = []
    
    for relationship in relationships_data['relationships']:
        meta = relationship['metadata']
        
        row = {
            # Core identifiers
            'id': relationship['id'],
            'text': relationship['text'],  # Full relationship description
            
            # Product A
            'product_a_id': meta['product_a_id'],
            'product_a_name': meta['product_a_name'],
            'product_a_category': meta['product_a_category'],
            
            # Product B
            'product_b_id': meta['product_b_id'],
            'product_b_name': meta['product_b_name'],
            'product_b_category': meta['product_b_category'],
            
            # Relationship metrics
            'cooccurrence_score': meta['cooccurrence_score'],
            'strength': meta['strength'],
        }
        
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    df['embedding'] = list(embeddings)
    
    return df

def create_chromadb_from_json(
    products_data: dict,
    collection_name: str = "nintendo_products",
    persist_directory: str = "./chroma_db",
    model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
) -> chromadb.Collection:
    """
    Create ChromaDB collection from products_rag.json
    
    Args:
        products_data: Dictionary from products_rag.json
        collection_name: Name for the ChromaDB collection
        persist_directory: Where to store the ChromaDB data
    
    Returns:
        ChromaDB collection with all products embedded
    """
    
    # Initialize ChromaDB client (persistent mode)
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Delete existing collection if it exists (fresh start)
    try:
        client.delete_collection(name=collection_name)
        print(f"✓ Deleted existing collection: {collection_name}")
    except:
        print(f"✓ No existing collection found")
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    # Prepare data for ChromaDB
    ids = []
    documents = []  # Text descriptions to embed
    metadatas = []  # Metadata (cannot contain nested dicts or lists)
    embeddings = []  # Pre-computed embeddings
    
    for product in products_data['products']:
        # 1. ID (must be string)
        product_id = product['id']  # e.g., "prod_0001"
        ids.append(product_id)
        
        # 2. Document (text to embed)
        # Use the clean description from JSON
        document = product['text']
        documents.append(document)
        
        # 3. Metadata (flatten nested structures)
        # ChromaDB only accepts: str, int, float, bool (no lists, no nested dicts)
        metadata = {
            # Core identifiers
            'name': product['metadata']['name'],
            'product_id': product['metadata']['product_id'],
            
            # Categories
            'category': product['metadata']['category'],
            'main_category': product['metadata']['main_category'],
            
            # Game info (handle nulls)
            'type': product['metadata']['type'] if product['metadata']['type'] else 'Unknown',
            'franchise': product['metadata']['franchise'] if product['metadata']['franchise'] else 'Unknown',
            'min_age': product['metadata']['min_age'] if product['metadata']['min_age'] else 0,
            'release_date': product['metadata']['release_date'] if product['metadata']['release_date'] else 'Unknown',
            
            # Business metrics
            'times_sold': product['metadata']['times_sold'],
            'popularity_rank': product['metadata']['popularity_rank'],
            'total_stock': product['metadata']['total_stock'],
            'is_available': product['metadata']['is_available'],
            
            # Store availability (flatten list to comma-separated string)
            'available_stores': ','.join(product['metadata']['available_stores']) if product['metadata']['available_stores'] else 'None',
            
            # Store-specific sales
            'times_sold_Store_A': product['metadata']['times_sold_Store_A'],
            'times_sold_Store_B': product['metadata']['times_sold_Store_B'],
            'times_sold_Store_C': product['metadata']['times_sold_Store_C'],
        }
        metadatas.append(metadata)
        
        # 4. Generate embedding
        embedding = model.encode(document).tolist()
        embeddings.append(embedding)
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    # Verify
    count = collection.count()
    print(f"✓ Collection now contains {count} items")
    
    print(" CHROMADB COLLECTION CREATED SUCCESSFULLY")
    print(f"  • Collection name: {collection_name}")
    print(f"  • Storage location: {persist_directory}")
    print(f"  • Total products: {count}")
    print(f"  • Embedding model: all-MiniLM-L6-v2")
    print(f"  • Embedding dimension: 384")
    
    return collection

def semantic_search(
    query: str,
    df: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 5,
    filters: dict = None
) -> pd.DataFrame:
    """
    Perform semantic search using cosine similarity
    
    Args:
        query: User's search query
        df: DataFrame with embeddings
        model: SentenceTransformer model
        top_k: Number of results to return
        filters: Optional filters (dict with column: value pairs)
    
    Returns:
        DataFrame with top K most similar products
    """
    
    # Apply filters first (before computing similarities)
    filtered_df = df.copy()
    
    if filters:
        print(f"📋 Applying filters: {filters}")
        for col, value in filters.items():
            if col == 'min_age':
                # Age filter: products suitable for age <= value
                filtered_df = filtered_df[filtered_df['min_age'] <= value]
            elif col == 'available_stores':
                # Store filter: check if store is in the list
                filtered_df = filtered_df[
                    filtered_df['available_stores'].apply(lambda x: value in x)
                ]
            elif col in filtered_df.columns:
                # Exact match filter
                filtered_df = filtered_df[filtered_df[col] == value]
    
    if len(filtered_df) == 0:
        print("❌ No products match the filters")
        return pd.DataFrame()
    
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    
    # Get embeddings matrix from filtered DataFrame
    embeddings_matrix = np.vstack(filtered_df['embedding'].values)
    
    # Compute cosine similarities
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings_matrix
    )[0]
    
    # Add similarity scores to DataFrame
    filtered_df = filtered_df.copy()
    filtered_df['similarity'] = similarities
    
    # Sort by similarity and get top K
    results = filtered_df.nlargest(top_k, 'similarity')
    
    # print(f"✅ Found {len(results)} results:")
    # for i, row in results.iterrows():
    #     print(f"   {row['name']}: {row['similarity']:.3f}")
    
    return results.reset_index(drop=True)

def semantic_search_relationships(
    query: str,
    df_relationships: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int = 5,
    filters: dict = None
) -> pd.DataFrame:
    """
    Perform semantic search on relationships
    
    Args:
        query: User's search query
        df_relationships: DataFrame with relationship embeddings
        model: SentenceTransformer model
        top_k: Number of results to return
        filters: Optional filters (strength, category, etc.)
    
    Returns:
        DataFrame with top K most similar relationships
    """
    
    # Apply filters
    filtered_df = df_relationships.copy()
    
    if filters:
        print(f"📋 Applying filters: {filters}")
        
        if 'strength' in filters:
            # Filter by relationship strength
            filtered_df = filtered_df[filtered_df['strength'] == filters['strength']]
        
        if 'min_cooccurrence' in filters:
            # Filter by minimum co-occurrence score
            filtered_df = filtered_df[
                filtered_df['cooccurrence_score'] >= filters['min_cooccurrence']
            ]
        
        if 'category' in filters:
            # Filter by product category
            category = filters['category']
            filtered_df = filtered_df[
                (filtered_df['product_a_category'] == category) |
                (filtered_df['product_b_category'] == category)
            ]
    
    if len(filtered_df) == 0:
        print(" No relationships match the filters")
        return pd.DataFrame()
    
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    
    # Get embeddings matrix
    embeddings_matrix = np.vstack(filtered_df['embedding'].values)
    
    # Compute cosine similarities
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        embeddings_matrix
    )[0]
    
    # Add similarity scores
    filtered_df = filtered_df.copy()
    filtered_df['similarity'] = similarities
    
    # Sort and get top K
    results = filtered_df.nlargest(top_k, 'similarity')
    
    # print(f"✅ Found {len(results)} relationships:")
    # for i, row in results.iterrows():
    #     print(f"   {row['product_a_name']} ↔ {row['product_b_name']}: {row['similarity']:.3f}")
    
    return results.reset_index(drop=True)


### EXCLSUIONS 
class SimpleExclusionDetector:
    """
    Lightweight exclusion detector using regex patterns
    Now includes age extraction
    """
    
    def __init__(self):
        # Franchise keywords mapping
        self.franchise_keywords = {
            'Super Mario': ['super mario', 'mario', 'mario games', 'mario family'],
            'Pokemon': ['pokemon', 'pokémon'],
            'Zelda': ['zelda', 'legend of zelda'],
            'Splatoon': ['splatoon'],
            'Animal Crossing': ['animal crossing'],
            'Kirby': ['kirby'],
        }
        
        # Exclusion patterns (regex)
        self.exclusion_patterns = [
            r'(?:cannot|can\'t|don\'t|do not|exclude)\s+(?:buy|want|get|recommend)?\s*(?:any|a|from)?\s*(.+?)(?:game|product|family)',
            r'(?:already|has|have)\s+(?:all|every)?\s*(.+?)(?:game|product)',
            r'(?:not|no)\s+(?:more)?\s*(.+?)(?:game|product)',
            r'except\s+(.+?)(?:game|product)',
            r'(?:but|excluding)\s+(.+?)(?:game|product)',
            r'(?:no|not)\s+(?:interested|looking)\s+(?:in|for)\s+(.+?)(?:game|product)',
        ]
        
        # Age extraction patterns
        self.age_patterns = [
            r'(?:is|age|aged)\s+(\d+)\s+(?:years?\s+old|year)',  # "is 5 years old"
            r'(\d+)\s*(?:-|\s)?year\s*(?:-|\s)?old',  # "5-year-old"
            r'for\s+(?:a\s+)?(\d+)\s+year',  # "for a 5 year"
            r'(?:child|kid|nephew|niece)\s+(?:who\s+is\s+)?(\d+)',  # "nephew who is 5"
            r'age\s+(\d+)',  # "age 5"
        ]
    
    def extract_age(self, query: str) -> Optional[int]:
        """
        Extract age from query
        
        Returns:
            Age as integer or None if not found
        """
        query_lower = query.lower()
        
        for pattern in self.age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                age = int(match.group(1))
                print(f"   🎂 Detected age: {age}")
                return age
        
        return None
    
    def detect_exclusions(self, query: str) -> Dict[str, any]:
        """
        Detect exclusions AND age from natural language query
        
        Returns:
            {
                'franchises': ['Super Mario', 'Pokemon'],
                'products': [],
                'categories': [],
                'age': 5
            }
        """
        query_lower = query.lower()
        
        excluded_franchises = []
        excluded_products = []
        
        # Extract exclusions
        for pattern in self.exclusion_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            
            for match in matches:
                excluded_text = match.group(1).strip()
                
                # Match against known franchises
                for franchise, keywords in self.franchise_keywords.items():
                    if any(kw in excluded_text for kw in keywords):
                        if franchise not in excluded_franchises:
                            excluded_franchises.append(franchise)
                            print(f"  Detected exclusion: {franchise}")
        
        # Extract age
        age = self.extract_age(query)
        
        return {
            'franchises': excluded_franchises,
            'products': excluded_products,
            'categories': [],
            'age': age
        }


# ============================================================================
# LLM-BASED EXCLUSION DETECTOR WITH AGE EXTRACTION
# ============================================================================

class LLMExclusionDetector:
    """
    Use GPT-4o-mini to extract exclusions AND age from complex queries
    More accurate but requires API call
    """
    
    def __init__(self, openai_client):
        self.client = openai_client
    
    def detect_exclusions(self, query: str) -> Dict[str, any]:
        """
        Use LLM to extract exclusions and age from query
        """
        
        prompt = f"""Analyze this query and extract:
1. What the user wants to EXCLUDE
2. The AGE of the person (if mentioned)

Query: "{query}"

Return ONLY valid JSON in this format:
{{
    "franchises": ["Super Mario", "Pokemon"],
    "products": ["specific product names"],
    "categories": ["Accessories"],
    "age": 5,
    "reasoning": "brief explanation"
}}

Common franchises: Super Mario, Pokemon, Zelda, Splatoon, Animal Crossing, Kirby

Exclusion indicators:
- "cannot buy", "can't buy"
- "already has all X"
- "don't want", "exclude"
- "not interested in"
- "except X"

Age indicators:
- "5 years old"
- "for a 5 year old"
- "nephew who is 5"
- "age 5"

If no exclusions found, return empty lists.
If no age mentioned, return null for age.
Return ONLY valid JSON."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        # Log detected exclusions and age
        if result['franchises']:
            print(f"  LLM detected franchise exclusions: {result['franchises']}")
        if result['products']:
            print(f"  LLM detected product exclusions: {result['products']}")
        if result.get('age'):
            print(f"  LLM detected age: {result['age']}")
        
        return result
