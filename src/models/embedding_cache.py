"""
Query Embedding Cache - Two-tier caching for embedding queries.

Provides:
- In-process TTL cache for frequently requested queries
- Thread-safe access using threading.Lock
- Normalized query key generation
- Cache hit/miss logging
- Latency tracking and reporting
"""

import logging
import time
import threading
from typing import Optional, Tuple
import numpy as np

try:
    from cachetools import TTLCache
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    logging.warning("cachetools not available. Install with: pip install cachetools")

logger = logging.getLogger(__name__)


class QueryEmbeddingCache:
    """
    Thread-safe cache for query embeddings using TTL (time-to-live).

    Features:
    - Normalized query keys (lowercase, strip whitespace)
    - Automatic expiration (default: 1 hour)
    - Cache statistics (hits, misses, size)
    - Thread-safe using locks
    - Optional: enforce maximum query length
    """

    def __init__(self, maxsize: int = 1000, ttl: int = 3600, max_query_length: int = 256):
        """
        Initialize the cache.

        Args:
            maxsize (int): Maximum number of cached queries (default: 1000)
            ttl (int): Time-to-live in seconds (default: 3600 = 1 hour)
            max_query_length (int): Maximum query length before truncation (default: 256)
        """
        if not CACHETOOLS_AVAILABLE:
            raise ImportError("cachetools required. Install with: pip install cachetools")

        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.max_query_length = max_query_length
        self.lock = threading.RLock()  # Reentrant lock for thread safety

        # Statistics
        self.hits = 0
        self.misses = 0

        logger.info(
            f"QueryEmbeddingCache initialized: maxsize={maxsize}, ttl={ttl}s, max_query_length={max_query_length}"
        )

    @staticmethod
    def normalize_query(query: str, max_length: int = 256) -> str:
        """
        Normalize query string for use as cache key.

        Performs:
        - Lowercase conversion
        - Strip leading/trailing whitespace
        - Collapse multiple spaces to single space
        - Truncate to max_length

        Args:
            query (str): Raw query string
            max_length (int): Maximum length after normalization

        Returns:
            str: Normalized query string suitable for cache key
        """
        # Lowercase and strip
        normalized = query.lower().strip()

        # Collapse multiple spaces to single space
        normalized = " ".join(normalized.split())

        # Truncate to max length
        if len(normalized) > max_length:
            normalized = normalized[:max_length].strip()

        return normalized

    def get(self, query: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache if available.

        Args:
            query (str): Query string (will be normalized)

        Returns:
            np.ndarray or None: Cached embedding if found, None otherwise
        """
        normalized_key = self.normalize_query(query, self.max_query_length)

        with self.lock:
            if normalized_key in self.cache:
                self.hits += 1
                embedding = self.cache[normalized_key]
                logger.debug(
                    f"[CACHE HIT] Query: {query[:50]}... → Key: {normalized_key[:50]}... "
                    f"(hits={self.hits}, misses={self.misses})"
                )
                return embedding
            else:
                self.misses += 1
                logger.debug(
                    f"[CACHE MISS] Query: {query[:50]}... → Key: {normalized_key[:50]}... "
                    f"(hits={self.hits}, misses={self.misses})"
                )
                return None

    def put(self, query: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.

        Args:
            query (str): Query string (will be normalized)
            embedding (np.ndarray): Embedding vector to cache (typically 768D)
        """
        normalized_key = self.normalize_query(query, self.max_query_length)

        with self.lock:
            self.cache[normalized_key] = embedding
            logger.debug(
                f"[CACHE PUT] Key: {normalized_key[:50]}... → Size: {embedding.shape}, "
                f"Cache size: {len(self.cache)}/{self.cache.maxsize}"
            )

    def get_or_put(self, query: str, embedding: np.ndarray) -> bool:
        """
        Atomically check cache and store if not present.

        Args:
            query (str): Query string
            embedding (np.ndarray): Embedding to store if not cached

        Returns:
            bool: True if already in cache (hit), False if newly added (miss)
        """
        normalized_key = self.normalize_query(query, self.max_query_length)

        with self.lock:
            if normalized_key in self.cache:
                self.hits += 1
                logger.debug(f"[CACHE HIT] {normalized_key[:50]}...")
                return True
            else:
                self.cache[normalized_key] = embedding
                self.misses += 1
                logger.debug(f"[CACHE MISS] {normalized_key[:50]}... → Stored")
                return False

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        with self.lock:
            self.cache.clear()
            old_hits = self.hits
            old_misses = self.misses
            self.hits = 0
            self.misses = 0
            logger.info(
                f"Cache cleared. Previous stats: {old_hits} hits, {old_misses} misses"
            )

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            dict: Statistics including hits, misses, hit rate, size
        """
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total,
                "hit_rate": hit_rate,
                "hit_rate_pct": f"{hit_rate * 100:.1f}%",
                "cache_size": len(self.cache),
                "cache_maxsize": self.cache.maxsize,
                "ttl_seconds": self.cache.timer.ttl if hasattr(self.cache, 'timer') else 'N/A',
            }

    def get_stats_str(self) -> str:
        """Get formatted statistics string."""
        stats = self.get_stats()
        return (
            f"Cache Stats: {stats['hits']} hits, {stats['misses']} misses, "
            f"Hit rate: {stats['hit_rate_pct']}, Size: {stats['cache_size']}/{stats['cache_maxsize']}"
        )

    def __repr__(self) -> str:
        return f"QueryEmbeddingCache(size={len(self.cache)}/{self.cache.maxsize}, {self.get_stats_str()})"


# Global cache instance
_embedding_cache: Optional[QueryEmbeddingCache] = None
_cache_lock = threading.Lock()


def get_global_cache() -> QueryEmbeddingCache:
    """
    Get or create global embedding cache instance.

    Uses lazy initialization and thread-safe singleton pattern.

    Returns:
        QueryEmbeddingCache: Global cache instance
    """
    global _embedding_cache

    if _embedding_cache is None:
        with _cache_lock:
            if _embedding_cache is None:
                _embedding_cache = QueryEmbeddingCache(maxsize=1000, ttl=3600)
                logger.info("Global embedding cache initialized")

    return _embedding_cache


def cache_query_embedding(
    query: str,
    embedding_fn,
    *args,
    use_cache: bool = True,
    **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Get query embedding with two-tier caching: Redis (L1) -> memory (L2).

    Lookup order:
      1. Redis cache        — distributed, 24 h TTL
      2. In-memory TTLCache — per-process, 1 h TTL (fallback when Redis down)
      3. Compute fresh; store in both tiers

    Args:
        query (str): Query text to embed
        embedding_fn: Callable that accepts *query* as first positional arg
                      and returns np.ndarray
        *args: Extra positional args forwarded to embedding_fn
        use_cache (bool): Set False to bypass cache entirely (default: True)
        **kwargs: Extra keyword args forwarded to embedding_fn

    Returns:
        Tuple[np.ndarray, dict]: (embedding, metadata)
            metadata keys:
              cached             bool
              cache_tier         'redis' | 'memory' | None
              latency_ms         float   — wall time including cache lookup
              compute_latency_ms float   — model inference time (miss only)
              cache_stats        dict    — in-memory cache counters
    """
    t_start   = time.time()
    mem_cache = get_global_cache()

    if use_cache:
        # ── Tier 1: Redis (L1) ────────────────────────────────────────────
        try:
            from src.models.redis_cache import get_redis_cache
            redis_emb = get_redis_cache().get_embedding(query)
            if redis_emb is not None:
                latency_ms = (time.time() - t_start) * 1000
                mem_cache.put(query, redis_emb)   # backfill memory tier
                logger.info(
                    "[CACHE HIT tier=redis]  %.2fms  %s",
                    latency_ms, mem_cache.get_stats_str(),
                )
                return redis_emb, {
                    "cached":      True,
                    "cache_tier":  "redis",
                    "latency_ms":  latency_ms,
                    "cache_stats": mem_cache.get_stats(),
                }
        except Exception as exc:
            logger.debug("[CACHE] Redis embedding lookup skipped: %s", exc)

        # ── Tier 2: In-memory TTLCache (L2) ───────────────────────────────
        mem_emb = mem_cache.get(query)
        if mem_emb is not None:
            latency_ms = (time.time() - t_start) * 1000
            logger.info(
                "[CACHE HIT tier=memory] %.2fms  %s",
                latency_ms, mem_cache.get_stats_str(),
            )
            return mem_emb, {
                "cached":      True,
                "cache_tier":  "memory",
                "latency_ms":  latency_ms,
                "cache_stats": mem_cache.get_stats(),
            }

    # ── Cache miss — compute ──────────────────────────────────────────────
    t_compute  = time.time()
    embedding  = embedding_fn(query, *args, **kwargs)
    compute_ms = (time.time() - t_compute) * 1000

    if use_cache:
        mem_cache.put(query, embedding)
        try:
            from src.models.redis_cache import get_redis_cache
            get_redis_cache().set_embedding(query, embedding)
        except Exception as exc:
            logger.debug("[CACHE] Redis embedding write skipped: %s", exc)

    latency_ms = (time.time() - t_start) * 1000
    logger.info(
        "[CACHE MISS] computed %.2fms  total %.2fms  %s",
        compute_ms, latency_ms, mem_cache.get_stats_str(),
    )
    return embedding, {
        "cached":             False,
        "cache_tier":         None,
        "latency_ms":         latency_ms,
        "compute_latency_ms": compute_ms,
        "cache_stats":        mem_cache.get_stats(),
    }


def reset_global_cache() -> None:
    """Reset global cache (useful for testing)."""
    global _embedding_cache

    cache = get_global_cache()
    cache.clear()
    logger.info("Global embedding cache reset")


if __name__ == "__main__":
    # Demo usage
    print("Query Embedding Cache Demo")
    print("=" * 60)

    # Create cache
    cache = QueryEmbeddingCache(maxsize=100, ttl=3600)

    # Test normalization
    print("\n1. Query Normalization:")
    test_queries = [
        "Hello World",
        "  hello   world  ",
        "HELLO WORLD",
        "hello world " * 100,  # Long query
    ]

    for q in test_queries:
        normalized = cache.normalize_query(q)
        print(f"  {repr(q)[:50]:<50} → {repr(normalized)[:50]}")

    # Test cache operations
    print("\n2. Cache Operations:")
    dummy_embedding = np.random.randn(768).astype(np.float32)

    print(f"  Initial stats: {cache.get_stats_str()}")

    # First get (miss)
    result = cache.get("test query")
    print(f"  After first get: {cache.get_stats_str()}")

    # Store
    cache.put("test query", dummy_embedding)
    print(f"  After put: {cache.get_stats_str()}")

    # Second get (hit)
    result = cache.get("test query")
    print(f"  After second get (hit): {cache.get_stats_str()}")

    # Normalized key should also hit
    result = cache.get("TEST QUERY")
    print(f"  After normalized query (hit): {cache.get_stats_str()}")

    # Final stats
    print(f"\n3. Final Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
