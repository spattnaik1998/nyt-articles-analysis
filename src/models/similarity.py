"""
Similarity Search and Recommendation Module

This module provides functions for finding similar articles and making
recommendations based on semantic embeddings.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
import logging

# Try to import FAISS for efficient similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using NumPy for similarity (slower). Install with: pip install faiss-cpu")

from src.models.embeddings import (
    extract_embeddings_batch,
    get_device,
    load_embeddings,
    TRANSFORMERS_AVAILABLE
)

logger = logging.getLogger(__name__)


def compute_cosine_similarity_numpy(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarity using NumPy (slower but always available).

    Args:
        query_embedding (np.ndarray): Query embedding vector (1D or 2D)
        embeddings (np.ndarray): Database embeddings (2D)
        top_k (int): Number of top results to return

    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices and similarity scores of top-k results
    """
    # Ensure query is 2D
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    # Normalize embeddings
    query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-9)
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    # Compute cosine similarity
    similarities = np.dot(query_norm, embeddings_norm.T).squeeze()

    # Get top-k
    top_k = min(top_k, len(similarities))
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_scores = similarities[top_indices]

    return top_indices, top_scores


def compute_cosine_similarity_faiss(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarity using FAISS (faster for large datasets).

    Args:
        query_embedding (np.ndarray): Query embedding vector (1D or 2D)
        embeddings (np.ndarray): Database embeddings (2D)
        top_k (int): Number of top results to return

    Returns:
        Tuple[np.ndarray, np.ndarray]: Indices and similarity scores of top-k results
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available. Install with: pip install faiss-cpu")

    # Ensure query is 2D and float32
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    query_embedding = query_embedding.astype('float32')
    embeddings_db = embeddings.astype('float32')

    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    faiss.normalize_L2(embeddings_db)

    # Build index
    dimension = embeddings_db.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (dot product) after normalization = cosine similarity
    index.add(embeddings_db)

    # Search
    top_k = min(top_k, embeddings_db.shape[0])
    similarities, indices = index.search(query_embedding, top_k)

    return indices.squeeze(), similarities.squeeze()


def recommend_by_embedding(
    query_text: str,
    embeddings_path: str = 'data/embeddings.npy',
    id_map_csv: str = 'data/embeddings_mapping.csv',
    articles_df_path: Optional[str] = None,
    top_k: int = 5,
    embed_model: str = 'vinai/bertweet-base',
    max_length: int = 128,
    pooling: str = 'cls',
    use_faiss: bool = True
) -> pd.DataFrame:
    """
    Recommend articles similar to a query text using embeddings.

    This function:
    1. Loads pre-computed embeddings and ID mapping
    2. Computes embedding for the query text
    3. Finds top-k most similar articles
    4. Returns results with IDs, scores, and metadata

    Args:
        query_text (str): Query text to find similar articles
        embeddings_path (str): Path to embeddings .npy file
        id_map_csv (str): Path to ID mapping CSV
        articles_df_path (str, optional): Path to articles DataFrame for metadata
        top_k (int): Number of recommendations (default: 5)
        embed_model (str): Model name for query embedding (default: vinai/bertweet-base)
        max_length (int): Max sequence length (default: 128)
        pooling (str): Pooling strategy (default: 'cls')
        use_faiss (bool): Use FAISS if available (default: True)

    Returns:
        pd.DataFrame: Recommendations with columns:
            - _id: Article ID
            - similarity: Similarity score
            - headline: Article headline (if articles_df provided)
            - abstract: Article abstract (if articles_df provided)

    Example:
        >>> results = recommend_by_embedding(
        ...     "stock market crash economy",
        ...     top_k=5
        ... )
        >>> print(results[['_id', 'similarity', 'headline']])
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library required. Install with: pip install transformers torch")

    # Load pre-computed embeddings and mapping
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings, id_mapping = load_embeddings(embeddings_path, id_map_csv)

    # Load transformer model for query embedding
    logger.info(f"Loading model: {embed_model}")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(embed_model)
    model = AutoModel.from_pretrained(embed_model)
    device = get_device()
    model.to(device)
    model.eval()

    # Compute query embedding
    logger.info(f"Computing query embedding for: '{query_text[:50]}...'")
    query_embedding = extract_embeddings_batch(
        [query_text],
        tokenizer,
        model,
        device,
        max_length=max_length,
        pooling=pooling
    )[0]  # Get first (and only) embedding

    # Compute similarities
    if use_faiss and FAISS_AVAILABLE:
        logger.info("Using FAISS for similarity search")
        top_indices, top_scores = compute_cosine_similarity_faiss(
            query_embedding, embeddings, top_k
        )
    else:
        logger.info("Using NumPy for similarity search")
        top_indices, top_scores = compute_cosine_similarity_numpy(
            query_embedding, embeddings, top_k
        )

    # Build results DataFrame
    results = pd.DataFrame({
        '_id': id_mapping.iloc[top_indices]['_id'].values,
        'similarity': top_scores,
        'index': top_indices
    })

    # Add metadata if articles DataFrame provided
    if articles_df_path:
        logger.info(f"Loading article metadata from {articles_df_path}")
        try:
            # Try Parquet first
            if articles_df_path.endswith('.parquet'):
                articles_df = pd.read_parquet(articles_df_path)
            else:
                articles_df = pd.read_csv(articles_df_path)

            # Merge metadata
            results = results.merge(
                articles_df[['_id', 'headline', 'abstract', 'pub_date', 'section_name']],
                on='_id',
                how='left'
            )

            # Truncate abstract for display
            if 'abstract' in results.columns:
                results['abstract_snippet'] = results['abstract'].str[:200] + '...'

        except Exception as e:
            logger.warning(f"Could not load article metadata: {e}")

    return results


def recommend_books(
    query_text: str,
    books_df_path: str,
    book_embeddings_path: str = 'data/book_embeddings.npy',
    book_mapping_path: str = 'data/book_embeddings_mapping.csv',
    top_k: int = 5,
    embed_model: str = 'vinai/bertweet-base',
    use_faiss: bool = True
) -> pd.DataFrame:
    """
    Recommend books based on a query text using embeddings.

    This function mirrors the book recommendation approach from the notebook:
    1. Loads book embeddings
    2. Computes query embedding
    3. Finds similar books
    4. Returns book title, author, and metadata

    Args:
        query_text (str): Query text describing desired book
        books_df_path (str): Path to books DataFrame (with book_title, author_name)
        book_embeddings_path (str): Path to book embeddings .npy
        book_mapping_path (str): Path to book ID mapping CSV
        top_k (int): Number of recommendations (default: 5)
        embed_model (str): Model name (default: vinai/bertweet-base)
        use_faiss (bool): Use FAISS if available (default: True)

    Returns:
        pd.DataFrame: Book recommendations with columns:
            - _id: Article ID
            - similarity: Similarity score
            - book_title: Extracted book title
            - author_name: Extracted author name
            - headline: Article headline
            - abstract: Article abstract

    Example:
        >>> results = recommend_books(
        ...     "historical fiction world war",
        ...     books_df_path='data/books_2001.parquet',
        ...     top_k=5
        ... )
        >>> print(results[['book_title', 'author_name', 'similarity']])
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers library required")

    # Load book embeddings and mapping
    logger.info(f"Loading book embeddings from {book_embeddings_path}")
    book_embeddings, book_mapping = load_embeddings(book_embeddings_path, book_mapping_path)

    # Load transformer model
    logger.info(f"Loading model: {embed_model}")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(embed_model)
    model = AutoModel.from_pretrained(embed_model)
    device = get_device()
    model.to(device)
    model.eval()

    # Compute query embedding
    logger.info(f"Computing query embedding for: '{query_text[:50]}...'")
    query_embedding = extract_embeddings_batch(
        [query_text],
        tokenizer,
        model,
        device,
        max_length=128,
        pooling='cls'
    )[0]

    # Compute similarities
    if use_faiss and FAISS_AVAILABLE:
        logger.info("Using FAISS for book similarity search")
        top_indices, top_scores = compute_cosine_similarity_faiss(
            query_embedding, book_embeddings, top_k
        )
    else:
        logger.info("Using NumPy for book similarity search")
        top_indices, top_scores = compute_cosine_similarity_numpy(
            query_embedding, book_embeddings, top_k
        )

    # Build results
    results = pd.DataFrame({
        '_id': book_mapping.iloc[top_indices]['_id'].values,
        'similarity': top_scores,
        'index': top_indices
    })

    # Load books DataFrame with metadata
    logger.info(f"Loading book metadata from {books_df_path}")
    try:
        if books_df_path.endswith('.parquet'):
            books_df = pd.read_parquet(books_df_path)
        else:
            books_df = pd.read_csv(books_df_path)

        # Merge book metadata
        merge_cols = ['_id']
        if 'book_title' in books_df.columns:
            merge_cols.append('book_title')
        if 'author_name' in books_df.columns:
            merge_cols.append('author_name')
        if 'headline' in books_df.columns:
            merge_cols.append('headline')
        if 'abstract' in books_df.columns:
            merge_cols.append('abstract')
        if 'pub_date' in books_df.columns:
            merge_cols.append('pub_date')

        results = results.merge(
            books_df[merge_cols],
            on='_id',
            how='left'
        )

    except Exception as e:
        logger.warning(f"Could not load book metadata: {e}")

    return results


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = 'flat',
    nlist: int = 100
) -> 'faiss.Index':
    """
    Build a FAISS index for efficient similarity search.

    Args:
        embeddings (np.ndarray): Embeddings to index
        index_type (str): Index type - 'flat' (exact) or 'ivf' (approximate)
        nlist (int): Number of clusters for IVF index

    Returns:
        faiss.Index: Built FAISS index

    Example:
        >>> index = build_faiss_index(embeddings, index_type='flat')
        >>> distances, indices = index.search(query, k=5)
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available")

    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    if index_type == 'flat':
        # Exact search (slower but accurate)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

    elif index_type == 'ivf':
        # Approximate search (faster for large datasets)
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 10  # Number of clusters to search

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    logger.info(f"Built FAISS {index_type} index with {index.ntotal} vectors")
    return index


def save_faiss_index(index: 'faiss.Index', path: str):
    """Save FAISS index to disk"""
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available")

    faiss.write_index(index, path)
    logger.info(f"Saved FAISS index to {path}")


def load_faiss_index(path: str) -> 'faiss.Index':
    """Load FAISS index from disk"""
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available")

    index = faiss.read_index(path)
    logger.info(f"Loaded FAISS index from {path}")
    return index


if __name__ == "__main__":
    # Example usage
    print("Similarity Search Module")
    print("=" * 60)

    # Check FAISS availability
    if FAISS_AVAILABLE:
        print("✓ FAISS available for fast similarity search")
    else:
        print("⚠ FAISS not available. Using NumPy (slower)")
        print("  Install with: pip install faiss-cpu")

    # Example 1: Simple similarity computation
    print("\n\nExample 1: Cosine Similarity Computation")
    print("-" * 60)

    # Create sample embeddings
    np.random.seed(42)
    query_emb = np.random.randn(768)
    db_embs = np.random.randn(100, 768)

    # NumPy version
    indices_np, scores_np = compute_cosine_similarity_numpy(query_emb, db_embs, top_k=5)
    print(f"NumPy - Top 5 indices: {indices_np}")
    print(f"NumPy - Top 5 scores: {scores_np}")

    # FAISS version (if available)
    if FAISS_AVAILABLE:
        indices_faiss, scores_faiss = compute_cosine_similarity_faiss(query_emb, db_embs, top_k=5)
        print(f"FAISS - Top 5 indices: {indices_faiss}")
        print(f"FAISS - Top 5 scores: {scores_faiss}")

    # Example 2: Building FAISS index
    if FAISS_AVAILABLE:
        print("\n\nExample 2: Building FAISS Index")
        print("-" * 60)

        index = build_faiss_index(db_embs, index_type='flat')
        print(f"Index type: {type(index)}")
        print(f"Index size: {index.ntotal} vectors")

    print("\n\nFor full examples, see:")
    print("  - examples/recommend_articles.py")
    print("  - examples/recommend_books.py")
