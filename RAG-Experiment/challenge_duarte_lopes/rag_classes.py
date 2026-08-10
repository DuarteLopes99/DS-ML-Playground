import json
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from auxiliar_functions import semantic_search, semantic_search_relationships, LLMExclusionDetector, SimpleExclusionDetector
import chromadb

class DualEmbeddingNintendoRAG_exclusions:
    """
    Enhanced RAG with BOTH product and relationship embeddings + Exclusion Detection
    
    Architecture:
    1. User Query → Detect Exclusions (LLM)
    2. Semantic search in BOTH with exclusion filtering:
       - Products DataFrame (find relevant products)
       - Relationships DataFrame (find relevant combinations)
    3. Merge results intelligently
    4. Build rich context with exclusion warnings
    5. Generate LLM response
    """
    
    def __init__(
        self,
        df_products: pd.DataFrame,
        df_relationships: pd.DataFrame,
        products_rag_path: str,
        relationships_rag_path: str,
        openai_api_key: str,
        model: SentenceTransformer,
        use_llm_for_exclusions: bool = True
    ):
        self.df_products = df_products
        self.df_relationships = df_relationships
        self.model = model
        
        # Load JSON data
        with open(products_rag_path, 'r', encoding='utf-8') as f:
            self.products_data = json.load(f)
        
        with open(relationships_rag_path, 'r', encoding='utf-8') as f:
            self.relationships_data = json.load(f)
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        
        # Create lookup dicts
        self.product_by_id = {
            p['metadata']['product_id']: p 
            for p in self.products_data['products']
        }
        
        # Initialize exclusion detector
        if use_llm_for_exclusions:
            self.exclusion_detector = LLMExclusionDetector(self.client)
            print("✅ Using LLM-based exclusion detection")
        else:
            self.exclusion_detector = SimpleExclusionDetector()
            print("✅ Using regex-based exclusion detection")
        
        print("✅ Dual Embedding RAG System Initialized (with Exclusions)")
    
    def retrieve(self, query: str, top_k: int = 5, filters: dict = None, exclude_franchises: list = None) -> dict:
        """
        Step 1: RETRIEVE from BOTH products AND relationships with exclusion filtering
        
        Args:
            query: User's natural language query
            top_k: Number of results to retrieve
            filters: Optional filters
            exclude_franchises: Franchises to exclude from results
        
        Returns:
            Dict with products and relationships
        """
        # Over-retrieve for exclusion filtering
        retrieval_k = top_k * 2 if exclude_franchises else top_k
        
        # Search products
        products_results = semantic_search(
            query,
            self.df_products,
            self.model,
            top_k=retrieval_k,
            filters=filters
        )
        
        # Convert products to full JSON format
        retrieved_products = []
        for _, row in products_results.iterrows():
            product_id = row['product_id']
            product = self.product_by_id.get(product_id)
            if product:
                product_copy = product.copy()
                product_copy['_similarity'] = row['similarity']
                retrieved_products.append(product_copy)
        
        # Apply exclusion filtering
        if exclude_franchises:
            filtered_products = []
            for p in retrieved_products:
                franchise = p['metadata'].get('franchise')
                if franchise in exclude_franchises:
                    print(f" Excluded: {p['metadata']['name']} (franchise: {franchise})")
                    continue
                filtered_products.append(p)
            
            retrieved_products = filtered_products
        
        # Trim to top_k
        retrieved_products = retrieved_products[:top_k]
        
        # Search relationships
        relationships_results = semantic_search_relationships(
            query,
            self.df_relationships,
            self.model,
            top_k=top_k,
            filters=None
        )
        
        # Convert relationships to list of dicts
        retrieved_relationships = relationships_results.to_dict('records')
        
        return {
            'products': retrieved_products,
            'relationships': retrieved_relationships,
            'products_df': products_results,
            'relationships_df': relationships_results
        }
    
    def _augment_with_exclusions(
        self,
        products: list,
        relationships: list,
        query: str,
        exclude_franchises: list = None
    ) -> str:
        """
        Step 2: AUGMENT - Build context with exclusion warnings
        """
        print("\n📝 AUGMENT: Building dual context with exclusions...")
        
        context_parts = []
        
        # Add query
        context_parts.append("# User Query")
        context_parts.append(f"User is asking: {query}")
        context_parts.append("")
        
        # Add exclusion warnings (CRITICAL!)
        if exclude_franchises:
            context_parts.append("# ⚠️ CRITICAL EXCLUSIONS")
            context_parts.append(f"**DO NOT RECOMMEND products from:** {', '.join(exclude_franchises)}")
            context_parts.append("**Reason:** User explicitly stated they cannot/don't want these")
            context_parts.append("")
        
        # Add products
        context_parts.append("# Available Products (After Exclusions)")
        context_parts.append("")

        for i, product in enumerate(products, 1):
            meta = product['metadata']
            similarity = product.get('_similarity', 0)
            
            context_parts.append(f"## {i}. {meta['name']}")
            context_parts.append(f"**Relevance:** {similarity:.2%}")
            context_parts.append(f"**Category:** {meta['main_category']}")
            
            if meta.get('franchise'):
                context_parts.append(f"**Franchise:** {meta['franchise']}")
            
            if meta.get('min_age'):
                context_parts.append(f"**Age Rating:** {meta['min_age']}+")
            
            context_parts.append(f"**Popularity:** {meta['times_sold']:,} sales")
            
            if meta['available_stores']:
                context_parts.append(f"**Available at:** {', '.join(meta['available_stores'])}")
            
            context_parts.append("")
        
        # Add relationships
        if relationships:
            context_parts.append("# Relevant Product Combinations")
            context_parts.append("")
            
            for i, rel in enumerate(relationships[:5], 1):
                context_parts.append(
                    f"{i}. **{rel['product_a_name']}** + **{rel['product_b_name']}**"
                )
                context_parts.append(f"   - Strength: {rel['strength']}")
                context_parts.append(f"   - Bought together: {rel['cooccurrence_score']:.0f} times")
                context_parts.append(f"   - Relevance: {rel.get('similarity', 0):.2%}")
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        print(f"✅ Context built ({len(context)} characters)")
        
        return context
    
    def generate(self, query: str, context: str) -> tuple:
        """
        Step 3: GENERATE with LLM
        """
        print("\n🤖 GENERATE: Calling GPT-4o-mini...")
        
        system_prompt = """You are a Nintendo product recommendation assistant.

Your role:
- Recommend products from the catalog
- **CRITICAL:** NEVER recommend excluded products/franchises marked with ⚠️
- **CRITICAL:** Dont recommend products which are not age appropriate for the user
- Suggest complementary product combinations
- Consider user needs (age, preferences, budget)
- Consider availability in stores

Important:
- Only recommend from available products in context
- Respect ALL exclusions marked with ⚠️
- Use the relevance scores to prioritize
- Mention product combinations when relevant
- Be concise and helpful
- If you cannot find suitable products, say so clearly"""

        user_prompt = f"""{context}

Based on the products and combinations above (respecting exclusions), provide personalized recommendations."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        usage_info = {
            'total_tokens': response.usage.total_tokens,
        }
        print(f"✅ Response generated ({usage_info['total_tokens']} tokens)")
        return response.choices[0].message.content, usage_info
    
    def query(self, user_query: str, top_k: int = 5, filters: dict = None) -> dict:
        """
        Complete RAG pipeline with dual embeddings and automatic exclusion detection
        """
        
        # Step 0: AUTOMATICALLY DETECT EXCLUSIONS
        exclusions = self.exclusion_detector.detect_exclusions(user_query)
        
        exclude_franchises = exclusions.get('franchises', [])
        exclude_products = exclusions.get('products', [])
        exclude_categories = exclusions.get('categories', [])
        
        if not any([exclude_franchises, exclude_products, exclude_categories]):
            print("   ✅ No exclusions detected")
        
        # Step 1: Retrieve with exclusions
        retrieved_data = self.retrieve(user_query, top_k, filters, exclude_franchises)
        
        # Step 2: Augment with exclusion warnings
        context = self._augment_with_exclusions(
            retrieved_data['products'],
            retrieved_data['relationships'],
            user_query,
            exclude_franchises
        )
        
        # Step 3: Generate
        response, usage_info = self.generate(user_query, context)

        return {
            'query': user_query,
            'auto_detected_exclusions': {
                'franchises': exclude_franchises,
                'products': exclude_products,
                'categories': exclude_categories
            },
            'retrieved_products': retrieved_data['products'],
            'retrieved_relationships': retrieved_data['relationships'],
            'context': context,
            'response': response,
            'usage': usage_info,
            'search_method': 'Cosine Sim. (Dual) with Auto Exclusions'
        }

class DualEmbeddingNintendoRAG:
    """
    Enhanced RAG with BOTH product and relationship embeddings
    
    Architecture:
    1. User Query → Semantic search in BOTH:
       - Products DataFrame (find relevant products)
       - Relationships DataFrame (find relevant combinations)
    2. Merge results intelligently
    3. Build rich context
    4. Generate LLM response
    """
    
    def __init__(
        self,
        df_products: pd.DataFrame,
        df_relationships: pd.DataFrame,
        products_rag_path: str,
        relationships_rag_path: str,
        openai_api_key: str,
        model: SentenceTransformer
    ):
        self.df_products = df_products
        self.df_relationships = df_relationships
        self.model = model
        
        # Load JSON data
        with open(products_rag_path, 'r', encoding='utf-8') as f:
            self.products_data = json.load(f)
        
        with open(relationships_rag_path, 'r', encoding='utf-8') as f:
            self.relationships_data = json.load(f)
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        
        # Create lookup dicts
        self.product_by_id = {
            p['metadata']['product_id']: p 
            for p in self.products_data['products']
        }
        
        print("✅ Dual Embedding RAG System Initialized")
    
    def retrieve(self, query: str, top_k: int = 5, filters: dict = None) -> dict:
        """
        Step 1: RETRIEVE from BOTH products AND relationships
        
        Args:
            query: User's natural language query
            top_k: Number of results to retrieve
            filters: Optional filters
        
        Returns:
            Dict with products and relationships
        """
        # Search products
        products_results = semantic_search(
            query,
            self.df_products,
            self.model,
            top_k=top_k,
            filters=filters
        )
        
        # Search relationships
        relationships_results = semantic_search_relationships(
            query,
            self.df_relationships,
            self.model,
            top_k=top_k,
            filters=None  # Can add relationship-specific filters
        )
        
        # Convert products to full JSON format
        retrieved_products = []
        for _, row in products_results.iterrows():
            product_id = row['product_id']
            product = self.product_by_id.get(product_id)
            if product:
                product_copy = product.copy()
                product_copy['_similarity'] = row['similarity']
                retrieved_products.append(product_copy)
        
        # Convert relationships to list of dicts
        retrieved_relationships = relationships_results.to_dict('records')
        
        print(f"\n✅ Retrieved {len(retrieved_products)} products")
        print(f"✅ Retrieved {len(retrieved_relationships)} relationships")
        
        return {
            'products': retrieved_products,
            'relationships': retrieved_relationships,
            'products_df': products_results,
            'relationships_df': relationships_results
        }
    
    def augment(self, retrieved_data: dict, query: str) -> str:
        """
        Step 2: AUGMENT - Build context from BOTH products and relationships
        """
        print("\n📝 AUGMENT: Building dual context...")
        
        context_parts = []
        
        # Add query
        context_parts.append("# User Query")
        context_parts.append(f"User is asking: {query}")
        context_parts.append("")
        
        # Add products
        context_parts.append("# Relevant Products")
        context_parts.append("")

        for i, product in enumerate(retrieved_data['products'], 1):
            meta = product['metadata']
            similarity = product.get('_similarity', 0)
            
            context_parts.append(f"## {i}. {meta['name']}")
            context_parts.append(f"**Relevance:** {similarity:.2%}")
            context_parts.append(f"**Category:** {meta['main_category']}")
            
            if meta.get('min_age'):
                context_parts.append(f"**Age Rating:** {meta['min_age']}+")
            
            context_parts.append(f"**Popularity:** {meta['times_sold']:,} sales")
            
            if meta['available_stores']:
                context_parts.append(f"**Available at:** {', '.join(meta['available_stores'])}")
            
            context_parts.append("")
        
        # Add relationships
        if retrieved_data['relationships']:
            context_parts.append("# Relevant Product Combinations")
            context_parts.append("")
            
            for i, rel in enumerate(retrieved_data['relationships'][:5], 1):
                context_parts.append(
                    f"{i}. **{rel['product_a_name']}** + **{rel['product_b_name']}**"
                )
                context_parts.append(f"   - Strength: {rel['strength']}")
                context_parts.append(f"   - Bought together: {rel['cooccurrence_score']:.0f} times")
                context_parts.append(f"   - Relevance: {rel['similarity']:.2%}")
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        return context
    
    def generate(self, query: str, context: str) -> tuple:
        """
        Step 3: GENERATE with LLM
        """
        print("\n🤖 GENERATE: Calling GPT-4o-mini...")
        
        system_prompt = """You are a Nintendo product recommendation assistant.

Your role:
- Recommend products from the catalog
- Suggest complementary product combinations
- Consider user needs (age, preferences, budget)
- Do not invent products or details
- Consider availability in stores
- Use retrieved context to inform recommendations
- Highlight popular bundles

Important:
- Only recommend from available products
- Use the relevance scores to prioritize
- Mention product combinations when relevant
- Be concise and helpful
- If you cannot find the answer in the context, say so. Do not make up information."""

        user_prompt = f"""{context}

Based on the products and combinations above, provide personalized recommendations."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        usage_info = {
            'total_tokens': response.usage.total_tokens,
        }
        print(f" Response generated ({usage_info['total_tokens']} tokens)")
        return response.choices[0].message.content, usage_info
    
    def query(self, user_query: str, top_k: int = 5, filters: dict = None) -> dict:
        """
        Complete RAG pipeline with dual embeddings
        """
        # Step 1: Retrieve from both sources
        retrieved_data = self.retrieve(user_query, top_k, filters)
        
        # Step 2: Augment
        context = self.augment(retrieved_data, user_query)
        print(f"\nContext Preview:\n{context[:40]}")
        # Step 3: Generate
        response, usage_info = self.generate(user_query, context)

        return {
            'query': user_query,
            'retrieved_products': retrieved_data['products'],
            'retrieved_relationships': retrieved_data['relationships'],
            'context': context,
            'response': response,
            'usage': usage_info,
            'search_method': 'Cosine Sim.'
        }

class ChromaDBNintendoRAG:
    """
    RAG System using ChromaDB for vector storage and retrieval
    
    Architecture:
    1. User Query → ChromaDB semantic search (HNSW algorithm)
    2. Top K products → Get product IDs
    3. Lookup relationships from JSON (by product IDs)
    4. Build rich context (products + relationships)
    5. GPT-4o-mini → Natural language response
    """
    
    def __init__(
        self,
        collection_chroma: chromadb.Collection,
        products_rag_path: str,
        relationships_rag_path: str,
        openai_api_key: str,
        model: SentenceTransformer
    ):
        # Store references
        self.collection = collection_chroma
        self.model = model
        
        # Load JSON data
        with open(products_rag_path, 'r', encoding='utf-8') as f:
            self.products_data = json.load(f)
        
        with open(relationships_rag_path, 'r', encoding='utf-8') as f:
            self.relationships_data = json.load(f)
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        
        # Create product lookup dict for fast access
        self.product_by_id = {
            p['metadata']['product_id']: p 
            for p in self.products_data['products']
        }
        
        # Create relationships lookup dict (by product_id)
        self.relationships_by_product = {}
        for rel in self.relationships_data['relationships']:
            # Extract product_id from "prod_0001" format
            prod_a_id = int(rel['metadata']['product_a_id'].split('_')[1])
            
            if prod_a_id not in self.relationships_by_product:
                self.relationships_by_product[prod_a_id] = []
            
            self.relationships_by_product[prod_a_id].append(rel)
        
        print("✅ ChromaDB RAG System Initialized")
        # print(f"   • Products in ChromaDB: {self.collection.count()}")
        # print(f"   • Relationships: {len(self.relationships_data['relationships'])}")
        # print(f"   • Embedding model: all-MiniLM-L6-v2")
        # print(f"   • Search method: HNSW (ChromaDB)")
        # print(f"   • LLM: GPT-4o-mini")
    
    def retrieve(self, query: str, top_k: int = 5, filters: dict = None) -> dict:
        """
        Step 1: RETRIEVE - Semantic search using ChromaDB
        
        Args:
            query: User's natural language query
            top_k: Number of products to retrieve
            filters: Optional ChromaDB filters (where clause)
        
        Returns:
            Dict with products and their relationships
        """
        
        # Prepare ChromaDB query
        query_params = {
            'query_texts': [query],
            'n_results': top_k
        }
        
        # Add filters if provided
        if filters:
            query_params['where'] = filters
        
        # Query ChromaDB
        results = self.collection.query(**query_params)

        
        # Convert ChromaDB results to structured format
        retrieved_products = []
        
        for i, (id, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            product_id = metadata['product_id']
            similarity = 1 - distance  # Convert distance to similarity
            
            # Get full product data from JSON
            product = self.product_by_id.get(product_id)
            if product:
                product_copy = product.copy()
                product_copy['_similarity'] = similarity
                product_copy['_distance'] = distance
                retrieved_products.append(product_copy)

        
        # Get relationships for retrieved products
        all_relationships = []
        for product in retrieved_products:
            product_id = product['metadata']['product_id']
            rels = self.relationships_by_product.get(product_id, [])
            
            # Sort by co-occurrence score and take top 3
            rels_sorted = sorted(
                rels,
                key=lambda x: x['metadata']['cooccurrence_score'],
                reverse=True
            )[:3]
            
            all_relationships.extend(rels_sorted)
        
        return {
            'products': retrieved_products,
            'relationships': all_relationships,
            'chromadb_results': results  # Keep raw ChromaDB results
        }
    
    def augment(self, retrieved_data: dict, query: str) -> str:
        """
        Step 2: AUGMENT - Build rich context for LLM
        
        Args:
            retrieved_data: Output from retrieve()
            query: Original user query
        
        Returns:
            Formatted context string
        """
        print("\n📝 AUGMENT: Building context...")
        
        context_parts = []
        
        # Add query context
        context_parts.append("# User Query")
        context_parts.append(f"User is asking: {query}")
        context_parts.append("")
        
        # Add retrieved products
        context_parts.append("# Available Products (Ranked by Relevance)")
        context_parts.append("")
        for i, product in enumerate(retrieved_data['products'], 1):
            meta = product['metadata']
            similarity = product.get('_similarity', 0)
            
            context_parts.append(f"## {i}. {meta['name']}")
            context_parts.append(f"**Relevance Score:** {similarity:.2%}")
            context_parts.append(f"**Category:** {meta['main_category']}")
            
            if meta.get('franchise'):
                context_parts.append(f"**Franchise:** {meta['franchise']}")
            
            if meta.get('min_age'):
                context_parts.append(f"**Age Rating:** {meta['min_age']}+")
            
            context_parts.append(f"**Popularity:** {meta['times_sold']:,} sales (Rank #{meta['popularity_rank']})")
            
            # Store availability
            if meta['available_stores']:
                context_parts.append(f"**Available at:** {', '.join(meta['available_stores'])}")
            else:
                context_parts.append("**Availability:** Out of stock")
            
            context_parts.append("")
        # Add relationships
        if retrieved_data['relationships']:
            context_parts.append("# Frequently Bought Together")
            context_parts.append("")
            
            # Group relationships by source product
            by_product = {}
            for rel in retrieved_data['relationships']:
                source = rel['metadata']['product_a_name']
                if source not in by_product:
                    by_product[source] = []
                by_product[source].append(rel)
            
            for source_product, rels in by_product.items():
                context_parts.append(f"**If customer buys {source_product}:**")
                for rel in rels[:3]:  # Top 3 per product
                    meta = rel['metadata']
                    context_parts.append(
                        f"- {meta['product_b_name']} "
                        f"(co-occurrence: {meta['cooccurrence_score']:.0f}, {meta['strength']})"
                    )
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        return context
    
    def generate(self, query: str, context: str) -> tuple:
        """
        Step 3: GENERATE - Get LLM response
        
        Args:
            query: User query
            context: Augmented context
        
        Returns:
            (response_text, usage_info)
        """
        
        system_prompt = """You are a Nintendo product recommendation assistant.

        Your role:
        - Recommend products from the catalog
        - Suggest complementary product combinations
        - Consider user needs (age, preferences, budget)
        - Consider availability in stores
        - Use retrieved context to inform recommendations
        - Highlight popular bundles

        Important:
        - Only recommend from available products
        - Use the relevance scores to prioritize
        - Mention product combinations when relevant
        - Be concise and helpful
        - If you cannot find the answer in the context, say so. Do not make up information."""

        user_prompt = f"""{context}

        Based on the products and combinations above, provide personalized recommendations."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        usage_info = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        }
        
        
        return response.choices[0].message.content, usage_info
    
    def query(self, user_query: str, top_k: int = 5, filters: dict = None) -> dict:
        """
        Complete RAG pipeline using ChromaDB
        
        Args:
            user_query: Natural language query
            top_k: Number of products to retrieve
            filters: Optional ChromaDB filters
        
        Returns:
            Complete response with all intermediate results
        """
        
        # Step 1: Retrieve
        retrieved_data = self.retrieve(user_query, top_k, filters)
        
        # Step 2: Augment
        context = self.augment(retrieved_data, user_query)
        # Step 3: Generate
        response, usage_info = self.generate(user_query, context)
        
        return {
            'query': user_query,
            'retrieved_products': retrieved_data['products'],
            'relationships': retrieved_data['relationships'],
            'context': context,
            'response': response,
            'usage': usage_info,
            'search_method': 'ChromaDB (HNSW)'
        }

class ChromaDBNintendoRAG_exclusions:
    """
    RAG System using ChromaDB with Exclusion Detection
    
    Architecture:
    1. User Query → Detect Exclusions (LLM)
    2. ChromaDB semantic search (HNSW algorithm)
    3. Filter excluded products
    4. Lookup relationships from JSON (by product IDs)
    5. Build rich context (products + relationships + exclusion warnings)
    6. GPT-4o-mini → Natural language response
    """
    
    def __init__(
        self,
        collection_chroma: chromadb.Collection,
        products_rag_path: str,
        relationships_rag_path: str,
        openai_api_key: str,
        model: SentenceTransformer,
        use_llm_for_exclusions: bool = True
    ):
        # Store references
        self.collection = collection_chroma
        self.model = model
        
        # Load JSON data
        with open(products_rag_path, 'r', encoding='utf-8') as f:
            self.products_data = json.load(f)
        
        with open(relationships_rag_path, 'r', encoding='utf-8') as f:
            self.relationships_data = json.load(f)
        
        # Initialize OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        
        # Create product lookup dict for fast access
        self.product_by_id = {
            p['metadata']['product_id']: p 
            for p in self.products_data['products']
        }
        
        # Create relationships lookup dict (by product_id)
        self.relationships_by_product = {}
        for rel in self.relationships_data['relationships']:
            # Extract product_id from "prod_0001" format
            prod_a_id = int(rel['metadata']['product_a_id'].split('_')[1])
            
            if prod_a_id not in self.relationships_by_product:
                self.relationships_by_product[prod_a_id] = []
            
            self.relationships_by_product[prod_a_id].append(rel)
        
        # Initialize exclusion detector
        if use_llm_for_exclusions:
            self.exclusion_detector = LLMExclusionDetector(self.client)
            print("✅ Using LLM-based exclusion detection")
        else:
            self.exclusion_detector = SimpleExclusionDetector()
            print("✅ Using regex-based exclusion detection")
        
        print("✅ ChromaDB RAG System Initialized (with Exclusions)")
        # print(f"   • Products in ChromaDB: {self.collection.count()}")
        # print(f"   • Relationships: {len(self.relationships_data['relationships'])}")
        # print(f"   • Embedding model: all-MiniLM-L6-v2")
        # print(f"   • Search method: HNSW (ChromaDB)")
        # print(f"   • Exclusion detection: ENABLED")
        # print(f"   • LLM: GPT-4o-mini")
    
    def retrieve(self, query: str, top_k: int = 5, filters: dict = None, exclude_franchises: list = None) -> dict:
        """
        Step 1: RETRIEVE - Semantic search using ChromaDB with exclusion filtering
        
        Args:
            query: User's natural language query
            top_k: Number of products to retrieve
            filters: Optional ChromaDB filters (where clause)
            exclude_franchises: Franchises to exclude from results
        
        Returns:
            Dict with products and their relationships
        """
        
        # Over-retrieve for exclusion filtering
        retrieval_k = top_k * 2 if exclude_franchises else top_k
        
        # Prepare ChromaDB query
        query_params = {
            'query_texts': [query],
            'n_results': retrieval_k
        }
        
        # Add filters if provided
        if filters:
            query_params['where'] = filters
        
        # Query ChromaDB
        results = self.collection.query(**query_params)
        # Convert ChromaDB results to structured format
        retrieved_products = []
        
        for i, (id, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            product_id = metadata['product_id']
            similarity = 1 - distance  # Convert distance to similarity
            
            # Get full product data from JSON
            product = self.product_by_id.get(product_id)
            if product:
                product_copy = product.copy()
                product_copy['_similarity'] = similarity
                product_copy['_distance'] = distance
                retrieved_products.append(product_copy)
        
        # Apply exclusion filtering
        if exclude_franchises:
            filtered_products = []
            for p in retrieved_products:
                franchise = p['metadata'].get('franchise')
                if franchise in exclude_franchises:
                    print(f"  Excluded: {p['metadata']['name']} (franchise: {franchise})")
                    continue
                filtered_products.append(p)
            
            retrieved_products = filtered_products
        
        # Trim to top_k
        retrieved_products = retrieved_products[:top_k]
        # Get relationships for retrieved products
        all_relationships = []
        for product in retrieved_products:
            product_id = product['metadata']['product_id']
            rels = self.relationships_by_product.get(product_id, [])
            
            # Sort by co-occurrence score and take top 3
            rels_sorted = sorted(
                rels,
                key=lambda x: x['metadata']['cooccurrence_score'],
                reverse=True
            )[:3]
            
            all_relationships.extend(rels_sorted)
        
        return {
            'products': retrieved_products,
            'relationships': all_relationships,
            'chromadb_results': results  # Keep raw ChromaDB results
        }
    
    def _augment_with_exclusions(
        self,
        products: list,
        relationships: list,
        query: str,
        exclude_franchises: list = None
    ) -> str:
        """
        Step 2: AUGMENT - Build rich context with exclusion warnings
        """
        print("\n📝 AUGMENT: Building context with exclusions...")
        
        context_parts = []
        
        # Add query context
        context_parts.append("# User Query")
        context_parts.append(f"User is asking: {query}")
        context_parts.append("")
        
        # Add exclusion warnings (CRITICAL!)
        if exclude_franchises:
            context_parts.append("# ⚠️ CRITICAL EXCLUSIONS")
            context_parts.append(f"**DO NOT RECOMMEND products from:** {', '.join(exclude_franchises)}")
            context_parts.append("**Reason:** User explicitly stated they cannot/don't want these")
            context_parts.append("")
        
        # Add retrieved products
        context_parts.append("# Available Products (Ranked by Relevance)")
        context_parts.append("")
        
        for i, product in enumerate(products, 1):
            meta = product['metadata']
            similarity = product.get('_similarity', 0)
            
            context_parts.append(f"## {i}. {meta['name']}")
            context_parts.append(f"**Relevance Score:** {similarity:.2%}")
            context_parts.append(f"**Category:** {meta['main_category']}")
            
            if meta.get('franchise'):
                context_parts.append(f"**Franchise:** {meta['franchise']}")
            
            if meta.get('min_age'):
                context_parts.append(f"**Age Rating:** {meta['min_age']}+")
            
            context_parts.append(f"**Popularity:** {meta['times_sold']:,} sales (Rank #{meta['popularity_rank']})")
            
            # Store availability
            if meta['available_stores']:
                context_parts.append(f"**Available at:** {', '.join(meta['available_stores'])}")
            else:
                context_parts.append("**Availability:** Out of stock")
            
            context_parts.append("")
        
        # Add relationships
        if relationships:
            context_parts.append("# Frequently Bought Together")
            context_parts.append("")
            
            # Group relationships by source product
            by_product = {}
            for rel in relationships:
                source = rel['metadata']['product_a_name']
                if source not in by_product:
                    by_product[source] = []
                by_product[source].append(rel)
            
            for source_product, rels in by_product.items():
                context_parts.append(f"**If customer buys {source_product}:**")
                for rel in rels[:3]:  # Top 3 per product
                    meta = rel['metadata']
                    context_parts.append(
                        f"- {meta['product_b_name']} "
                        f"(co-occurrence: {meta['cooccurrence_score']:.0f}, {meta['strength']})"
                    )
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        return context
    
    def generate(self, query: str, context: str) -> tuple:
        """
        Step 3: GENERATE - Get LLM response
        """
        print("\n🤖 GENERATE: Calling GPT-4o-mini...")
        
        system_prompt = """You are a Nintendo product recommendation assistant.

Your role:
- Recommend products from the catalog
- **CRITICAL:** NEVER recommend excluded products/franchises marked with ⚠️
- **CRITICAL:** Dont recommend products which are not age appropriate for the user
- Suggest complementary product combinations
- Consider user needs (age, preferences, budget)
- Consider availability in stores

Important:
- Only recommend from available products in context
- Respect ALL exclusions marked with ⚠️
- Use the relevance scores to prioritize
- Mention product combinations when relevant
- Be concise and helpful
- If you cannot find suitable products, say so clearly"""

        user_prompt = f"""{context}

Based on the products and combinations above (respecting exclusions), provide personalized recommendations."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        usage_info = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
        }
        
        print(f"✅ Response generated ({usage_info['total_tokens']} tokens)")
        
        return response.choices[0].message.content, usage_info
    
    def query(self, user_query: str, top_k: int = 5, filters: dict = None) -> dict:
        """
        Complete RAG pipeline using ChromaDB with automatic exclusion detection
        """
        # Step 0: AUTOMATICALLY DETECT EXCLUSIONS
        exclusions = self.exclusion_detector.detect_exclusions(user_query)
        
        exclude_franchises = exclusions.get('franchises', [])
        exclude_products = exclusions.get('products', [])
        exclude_categories = exclusions.get('categories', [])
        
        if not any([exclude_franchises, exclude_products, exclude_categories]):
            print("   ✅ No exclusions detected")
        
        # Step 1: Retrieve with exclusions
        retrieved_data = self.retrieve(user_query, top_k, filters, exclude_franchises)
        
        # Step 2: Augment with exclusion warnings
        context = self._augment_with_exclusions(
            retrieved_data['products'],
            retrieved_data['relationships'],
            user_query,
            exclude_franchises
        )
        
        # Step 3: Generate
        response, usage_info = self.generate(user_query, context)
        
        return {
            'query': user_query,
            'auto_detected_exclusions': {
                'franchises': exclude_franchises,
                'products': exclude_products,
                'categories': exclude_categories
            },
            'retrieved_products': retrieved_data['products'],
            'relationships': retrieved_data['relationships'],
            'context': context,
            'response': response,
            'usage': usage_info,
            'search_method': 'ChromaDB (HNSW) with Auto Exclusions'
        }