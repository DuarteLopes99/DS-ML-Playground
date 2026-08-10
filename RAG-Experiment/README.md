# RAG Experiment — RAG Retrieval Evaluation & Exclusion Logic

This project implements and evaluates a Retrieval-Augmented Generation (RAG) system for natural-language product recommendations over a small e-commerce-style catalog. It originated as a take-home technical assessment and was extended into a personal exploration of RAG architecture trade-offs, comparing multiple retrieval strategies and an exclusion/filtering layer that constrains results based on detected user intent (e.g. franchise exclusions, age-appropriateness, store availability).

## Techniques & Tech Stack
- Sentence embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) for both product and product-relationship representations
- Dual embedding design: separate vector spaces for individual products and for product-pair relationships
- Cosine-similarity semantic search (pandas/NumPy/scikit-learn) as a baseline retrieval method
- ChromaDB vector store with HNSW indexing as an alternative, more scalable retrieval backend
- Co-occurrence matrix analysis to derive relationship strength/"bought together" scores between products, used as retrieval metadata and ranking signal
- LLM-based query analysis (OpenAI GPT-4o-mini) for exclusion detection (franchise/category exclusions), plus a lightweight regex-based fallback detector
- Structured evaluation harness generating comparative metrics across models (exported to spreadsheet)

## Approach (high level)
Raw catalog data is processed into two RAG-ready JSON artifacts: one holding product records with metadata and embeddings, another holding pairwise product relationships with co-occurrence scores. Four RAG variants are implemented and compared: cosine-similarity retrieval, cosine-similarity retrieval with exclusion/age filtering, ChromaDB retrieval, and ChromaDB retrieval with exclusion/age filtering. Each pipeline follows a retrieve → augment → generate flow: relevant products and relationships are retrieved (optionally filtered by detected exclusions or constraints), assembled into a context prompt, and passed to an LLM to produce the final recommendation. A dedicated evaluation notebook runs a fixed battery of test queries (out-of-scope detection, age filtering, store constraints, interest matching, exclusion handling, multi-product scenarios) across all four architectures and reports latency, token usage, retrieval counts, and response quality.

Further detail on the original assessment brief is in `challenge_duarte_lopes/README_CHALLENGE.md`.

## Skills Demonstrated
- RAG system design and architectural comparison (embedding-based vs. vector-database retrieval)
- Vector store integration and indexing strategy (ChromaDB/HNSW)
- Prompt-driven intent detection and business-rule filtering layered on top of semantic search
- Quantitative evaluation methodology for LLM-based systems
- End-to-end data pipeline construction from raw catalog data to retrieval-ready embeddings
