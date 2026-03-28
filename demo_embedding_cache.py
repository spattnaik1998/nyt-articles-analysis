#!/usr/bin/env python3
"""
Demonstration of Query Embedding Cache.

This script demonstrates the two-tier caching system for query embeddings:
1. Shows cache hit/miss behavior
2. Shows performance improvements
3. Shows concurrent access
4. Shows statistics tracking
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.embedding_cache import QueryEmbeddingCache, get_global_cache, reset_global_cache


def demo_basic_caching():
    """Demonstrate basic cache hit/miss behavior."""
    print("\n" + "=" * 70)
    print("DEMO 1: BASIC CACHE HIT/MISS BEHAVIOR")
    print("=" * 70)

    cache = QueryEmbeddingCache(maxsize=100, ttl=3600)
    dummy_embedding = np.random.randn(768).astype(np.float32)

    # Scenario 1: Cache miss
    print("\n[1] First query (cache miss):")
    result = cache.get("artificial intelligence")
    print(f"    Result: {result} (None expected)")
    print(f"    Stats: {cache.get_stats_str()}")

    # Store the embedding
    print("\n[2] Store embedding in cache:")
    cache.put("artificial intelligence", dummy_embedding)
    print(f"    Stored 768D embedding")
    print(f"    Stats: {cache.get_stats_str()}")

    # Scenario 2: Cache hit
    print("\n[3] Same query again (cache hit):")
    result = cache.get("artificial intelligence")
    print(f"    Result: Found (shape={result.shape})")
    print(f"    Stats: {cache.get_stats_str()}")

    # Scenario 3: Normalized key hit
    print("\n[4] Query with different case (normalized key hit):")
    result = cache.get("ARTIFICIAL INTELLIGENCE")
    print(f"    Result: Found (normalized key matched)")
    print(f"    Stats: {cache.get_stats_str()}")

    # Scenario 4: Different query (miss)
    print("\n[5] Different query (cache miss):")
    result = cache.get("machine learning")
    print(f"    Result: {result} (None expected)")
    print(f"    Stats: {cache.get_stats_str()}")


def demo_query_normalization():
    """Demonstrate query normalization."""
    print("\n" + "=" * 70)
    print("DEMO 2: QUERY NORMALIZATION")
    print("=" * 70)

    cache = QueryEmbeddingCache()

    test_queries = [
        "Hello World",
        "  hello   world  ",
        "HELLO WORLD",
        "HeLLo WoRLd",
        "hello  world  " * 50,  # Very long query
    ]

    print("\nQuery normalization examples:")
    for i, query in enumerate(test_queries, 1):
        normalized = cache.normalize_query(query, max_length=256)
        print(f"\n  [{i}] Input ({len(query)} chars): {repr(query[:60])}")
        print(f"      Normalized ({len(normalized)} chars): {repr(normalized[:60])}")


def demo_cache_efficiency():
    """Demonstrate cache efficiency with repeated queries."""
    print("\n" + "=" * 70)
    print("DEMO 3: CACHE EFFICIENCY SIMULATION")
    print("=" * 70)

    reset_global_cache()
    cache = get_global_cache()

    # Simulate query embedding function (mock)
    def simulate_embedding_computation():
        """Simulate 100ms query embedding computation."""
        time.sleep(0.1)
        return np.random.randn(768).astype(np.float32)

    queries = ["artificial intelligence", "machine learning", "deep learning"]
    repetitions = 3

    total_time_without_cache = 0
    total_time_with_cache = 0

    print(f"\nSimulating {len(queries)} unique queries, {repetitions} repetitions each\n")

    # Warm up cache
    for query in queries:
        embedding = simulate_embedding_computation()
        cache.put(query, embedding)

    print("Cache warmed. Now simulating lookups:")

    # Simulate with cache
    for rep in range(repetitions):
        print(f"\n  Repetition {rep + 1}:")
        rep_time = 0
        for query in queries:
            t_start = time.time()
            result = cache.get(query)
            t_query = (time.time() - t_start) * 1000
            rep_time += t_query
            status = "HIT" if result is not None else "MISS"
            print(f"    {query:30} → {status:4} ({t_query:6.2f}ms)")
        total_time_with_cache += rep_time
        print(f"    Repetition time: {rep_time:.2f}ms")

    # Estimate without cache
    time_per_query = 100  # 100ms for embedding computation
    total_time_without_cache = len(queries) * repetitions * time_per_query

    print(f"\n  Summary:")
    print(f"    With cache:    {total_time_with_cache:.2f}ms")
    print(f"    Without cache: {total_time_without_cache:.2f}ms (estimated)")
    print(f"    Speedup:       {total_time_without_cache / total_time_with_cache:.1f}x")


def demo_hit_rate_tracking():
    """Demonstrate cache statistics and hit rate tracking."""
    print("\n" + "=" * 70)
    print("DEMO 4: CACHE STATISTICS & HIT RATE")
    print("=" * 70)

    reset_global_cache()
    cache = get_global_cache()
    dummy_embedding = np.random.randn(768).astype(np.float32)

    # Create a realistic query pattern
    queries = [
        ("machine learning", 3),     # 3 occurrences
        ("deep learning", 2),         # 2 occurrences
        ("artificial intelligence", 5),  # 5 occurrences
        ("neural networks", 2),      # 2 occurrences
        ("nlp", 4),                  # 4 occurrences
    ]

    print("\nSimulating realistic query pattern:")
    for query, count in queries:
        print(f"  - '{query}': {count} times")

    print("\nProcessing queries:")
    for query, count in queries:
        cache.put(query, dummy_embedding)  # First access is a "miss"
        print(f"  [{query:30}] stored")

        # Subsequent accesses are "hits"
        for i in range(count - 1):
            result = cache.get(query)
            assert result is not None, f"Cache miss for {query}"

    print("\nCache Statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if key == 'hit_rate':
            print(f"  {key:20}: {value:.2%}")
        else:
            print(f"  {key:20}: {value}")


def demo_concurrent_access():
    """Demonstrate thread-safe concurrent access."""
    print("\n" + "=" * 70)
    print("DEMO 5: CONCURRENT ACCESS")
    print("=" * 70)

    import threading

    reset_global_cache()
    cache = get_global_cache()
    dummy_embedding = np.random.randn(768).astype(np.float32)

    # Pre-populate cache
    for i in range(5):
        cache.put(f"query_{i}", dummy_embedding)

    results = []
    errors = []

    def worker(worker_id, iterations):
        """Worker thread that reads from cache."""
        try:
            for i in range(iterations):
                query = f"query_{i % 5}"
                result = cache.get(query)
                if result is not None:
                    results.append((worker_id, i))
                time.sleep(0.001)  # Simulate some work
        except Exception as e:
            errors.append((worker_id, e))

    print(f"\nSpawning 10 threads, each doing 20 cache lookups")
    threads = [
        threading.Thread(target=worker, args=(i, 20))
        for i in range(10)
    ]

    t_start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    t_elapsed = time.time() - t_start

    print(f"\nResults:")
    print(f"  Total lookups: {len(results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {t_elapsed * 1000:.2f}ms")
    print(f"  Throughput: {len(results) / t_elapsed:.0f} lookups/sec")

    if errors:
        print(f"\nErrors occurred:")
        for worker_id, error in errors:
            print(f"  Worker {worker_id}: {error}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "Query Embedding Cache - Demonstration".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_basic_caching()
        demo_query_normalization()
        demo_cache_efficiency()
        demo_hit_rate_tracking()
        demo_concurrent_access()

        print("\n" + "=" * 70)
        print("✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print("\nKey Takeaways:")
        print("  1. Cache hit/miss behavior is transparent")
        print("  2. Query normalization ensures consistent cache keys")
        print("  3. Cache provides significant latency savings (100ms → <1ms)")
        print("  4. Hit rates improve with repeated queries")
        print("  5. Thread-safe access for concurrent requests")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
