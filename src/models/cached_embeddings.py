"""
Cached query embedding functions.

Wraps the embedding generation functions with caching to avoid recomputing
embeddings for repeated queries.

This module provides drop-in replacements for single-query embedding functions
with transparent caching.
"""

import logging
import time
import numpy as np
import torch
from typing import Optional, Tuple

from src.models.embedding_cache import cache_query_embedding, get_global_cache
from src.models.embeddings import get_device

logger = logging.getLogger(__name__)


def encode_query_cached(
    query: str,
    tokenizer,
    model,
    max_length: int = 128,
    pooling: str = 'cls',
    use_cache: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Encode a query string to embedding with caching.

    This is a cached wrapper around tokenizer + model inference.
    Repeated queries will return cached embeddings without model computation.

    Args:
        query (str): Query text (e.g., "artificial intelligence")
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model (BERTweet, etc.)
        max_length (int): Max sequence length (default: 128)
        pooling (str): 'cls' or 'mean' pooling (default: 'cls')
        use_cache (bool): Whether to use cache (default: True)

    Returns:
        Tuple[np.ndarray, dict]:
            - embedding: Query embedding (768D for BERTweet)
            - metadata: Dict with 'cached', 'latency_ms', cache stats

    Example:
        >>> from transformers import AutoTokenizer, AutoModel
        >>> tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
        >>> model = AutoModel.from_pretrained("vinai/bertweet-base")
        >>>
        >>> # First call (computes, caches)
        >>> emb1, meta1 = encode_query_cached("test query", tokenizer, model)
        >>> print(meta1['latency_ms'])  # ~100-150ms
        >>>
        >>> # Second call (returns from cache)
        >>> emb2, meta2 = encode_query_cached("test query", tokenizer, model)
        >>> print(meta2['latency_ms'])  # ~1-2ms
    """

    def _encode_impl(q: str) -> np.ndarray:
        """Inner function that performs actual encoding (called if not cached)."""
        device = get_device()

        encoded = tokenizer(
            [q],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)

        # Extract embedding based on pooling strategy
        if pooling == 'cls':
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].astype(np.float32)
        elif pooling == 'mean':
            attention_mask = encoded['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).cpu().numpy()[0].astype(np.float32)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

        return embedding

    # Use cache wrapper
    embedding, metadata = cache_query_embedding(
        query,
        _encode_impl,
        use_cache=use_cache
    )

    return embedding, metadata


def get_query_embedding(
    query: str,
    tokenizer,
    model,
    max_length: int = 128,
    pooling: str = 'cls',
    use_cache: bool = True
) -> np.ndarray:
    """
    Get query embedding with caching (simple interface, returns embedding only).

    This is a convenience wrapper that returns just the embedding without metadata.
    Use encode_query_cached() if you need timing information.

    Args:
        query (str): Query text
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        max_length (int): Max sequence length (default: 128)
        pooling (str): 'cls' or 'mean' pooling (default: 'cls')
        use_cache (bool): Whether to use cache (default: True)

    Returns:
        np.ndarray: Query embedding (768D for BERTweet)

    Example:
        >>> emb = get_query_embedding("test query", tokenizer, model)
    """
    embedding, _ = encode_query_cached(
        query,
        tokenizer,
        model,
        max_length=max_length,
        pooling=pooling,
        use_cache=use_cache
    )
    return embedding


def log_cache_stats() -> None:
    """Log current cache statistics to logger."""
    cache = get_global_cache()
    stats = cache.get_stats()

    logger.info("=" * 60)
    logger.info("QUERY EMBEDDING CACHE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"  Cache hits:     {stats['hits']}")
    logger.info(f"  Cache misses:   {stats['misses']}")
    logger.info(f"  Total requests: {stats['total_requests']}")
    logger.info(f"  Hit rate:       {stats['hit_rate_pct']}")
    logger.info(f"  Cache size:     {stats['cache_size']}/{stats['cache_maxsize']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # Demo usage
    print("Cached Query Embeddings Demo")
    print("=" * 60)

    # This would require actual models loaded, so just show the API
    print("""
Usage:
    from src.models.cached_embeddings import encode_query_cached, log_cache_stats
    from transformers import AutoTokenizer, AutoModel

    # Setup
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    model = AutoModel.from_pretrained("vinai/bertweet-base")

    # First call (computes, caches)
    query = "artificial intelligence"
    emb1, meta1 = encode_query_cached(query, tokenizer, model)
    print(f"First call: {meta1['latency_ms']:.1f}ms (cached={meta1['cached']})")

    # Second call (from cache)
    emb2, meta2 = encode_query_cached(query, tokenizer, model)
    print(f"Second call: {meta2['latency_ms']:.1f}ms (cached={meta2['cached']})")

    # Same query, normalized different way
    emb3, meta3 = encode_query_cached("Artificial Intelligence", tokenizer, model)
    print(f"Third call (different case): {meta3['latency_ms']:.1f}ms (cached={meta3['cached']})")

    # View cache statistics
    log_cache_stats()
    """)
