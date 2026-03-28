"""
Distributed Redis Cache Layer

Two-tier caching architecture:
  L1: Redis   — distributed, shared across workers, survives restarts
  L2: Memory  — in-process TTLCache fallback when Redis is unavailable

Cache namespaces and TTLs:
  nyt:emb:v1:{sha256}   — query embeddings     (TTL: 24 h  / 86 400 s)
  nyt:res:v1:{sha256}   — search results        (TTL: 30 min / 1 800 s)

Key strategy
  SHA-256 of the concatenated, normalised inputs guarantees:
  - Fixed-length keys regardless of query size
  - No special characters that require quoting in redis-cli
  - Collision resistance (2^256 key space)

Serialisation
  Embeddings : pickle (numpy float32 arrays; written and read by this
               process only — no untrusted pickle input)
  Results    : JSON  (plain dicts/lists/scalars — human-readable in redis-cli)

Fallback circuit-breaker
  Any RedisError sets _available=False and starts a 60-second back-off
  timer before the next reconnect attempt.  Memory-only mode is
  transparent to callers.

Metrics
  Per-tier (redis / memory) hit and miss counters with thread-safe access.
  Exposed via get_metrics() and logged on every access at DEBUG level,
  with a periodic INFO summary.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL constants
# ---------------------------------------------------------------------------

EMBEDDING_TTL: int = 86_400   # 24 hours
RESULTS_TTL:   int = 1_800    # 30 minutes

# Key version — bump to invalidate all cached entries after a model update
_KEY_VERSION   = "v1"
_EMBED_NS      = f"nyt:emb:{_KEY_VERSION}"
_RESULTS_NS    = f"nyt:res:{_KEY_VERSION}"

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class _TierCounters:
    hits:   int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


class CacheMetrics:
    """Thread-safe per-tier hit / miss counters."""

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self.redis  = _TierCounters()
        self.memory = _TierCounters()

    def record(self, tier: str, hit: bool) -> None:
        with self._lock:
            counters: _TierCounters = getattr(self, tier)
            if hit:
                counters.hits += 1
            else:
                counters.misses += 1

    def snapshot(self) -> dict:
        with self._lock:
            total      = self.redis.total  + self.memory.total
            total_hits = self.redis.hits   + self.memory.hits
            return {
                "redis": {
                    "hits":     self.redis.hits,
                    "misses":   self.redis.misses,
                    "hit_rate": round(self.redis.hit_rate, 4),
                },
                "memory": {
                    "hits":     self.memory.hits,
                    "misses":   self.memory.misses,
                    "hit_rate": round(self.memory.hit_rate, 4),
                },
                "overall": {
                    "hits":     total_hits,
                    "total":    total,
                    "hit_rate": round(total_hits / total if total else 0.0, 4),
                },
            }


# ---------------------------------------------------------------------------
# Redis cache manager
# ---------------------------------------------------------------------------

class RedisCacheManager:
    """
    Two-tier cache: Redis (L1) -> in-memory TTLCache (L2 fallback).

    All public methods are thread-safe and never raise — errors are logged
    and execution falls through to the next tier or returns None.
    """

    # Reconnect back-off: do not hammer a down Redis server
    _RECONNECT_INTERVAL = 60.0   # seconds

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        embedding_ttl: int = EMBEDDING_TTL,
        results_ttl:   int = RESULTS_TTL,
        pool_max_connections: int = 20,
        socket_timeout: float = 0.5,      # keep fallback fast
        connect_timeout: float = 1.0,
        memory_maxsize: int = 1_000,
    ) -> None:
        self.redis_url     = redis_url
        self.embedding_ttl = embedding_ttl
        self.results_ttl   = results_ttl
        self.metrics       = CacheMetrics()

        # Connection state
        self._redis            = None   # redis.Redis | None
        self._redis_available  = False
        self._connect_lock     = threading.Lock()
        self._last_attempt     = 0.0
        self._pool_kwargs      = dict(
            max_connections        = pool_max_connections,
            socket_timeout         = socket_timeout,
            socket_connect_timeout = connect_timeout,
            decode_responses       = False,   # we handle bytes ourselves
        )

        # In-memory fallback caches
        self._mem_lock = threading.RLock()
        try:
            from cachetools import TTLCache
            self._mem_embeddings: dict = TTLCache(maxsize=memory_maxsize,       ttl=embedding_ttl)
            self._mem_results:    dict = TTLCache(maxsize=memory_maxsize // 2,  ttl=results_ttl)
            logger.debug("[CACHE] In-memory TTLCache fallback ready")
        except ImportError:
            self._mem_embeddings = {}
            self._mem_results    = {}
            logger.warning("[CACHE] cachetools not installed; using plain dict as fallback")

        # Attempt to connect immediately (non-blocking on failure)
        self._try_connect()

    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------

    def _try_connect(self) -> bool:
        """
        Attempt Redis connection if the back-off window has elapsed.
        Returns True if Redis is available after the attempt.
        """
        now = time.monotonic()
        if now - self._last_attempt < self._RECONNECT_INTERVAL:
            return self._redis_available

        with self._connect_lock:
            # Double-checked inside lock
            if time.monotonic() - self._last_attempt < self._RECONNECT_INTERVAL:
                return self._redis_available

            self._last_attempt = time.monotonic()
            try:
                import redis as redis_lib

                pool = redis_lib.ConnectionPool.from_url(
                    self.redis_url, **self._pool_kwargs
                )
                client = redis_lib.Redis(connection_pool=pool)
                client.ping()                    # raises on failure
                self._redis           = client
                self._redis_available = True
                logger.info("[CACHE] Redis connected: %s", self.redis_url)

            except ImportError:
                logger.warning(
                    "[CACHE] redis package not installed — memory-only mode. "
                    "pip install 'redis>=5.0.0'"
                )
                self._redis_available = False

            except Exception as exc:
                logger.warning(
                    "[CACHE] Redis unavailable (%s) — memory-only fallback active", exc
                )
                self._redis_available = False

        return self._redis_available

    @property
    def is_available(self) -> bool:
        """True if Redis is reachable; re-attempts connection after back-off."""
        if self._redis_available:
            return True
        return self._try_connect()

    def _mark_down(self, exc: Exception) -> None:
        """Called on any RedisError during a real operation."""
        if self._redis_available:
            logger.warning("[CACHE] Redis error — switching to memory-only: %s", exc)
            self._redis_available = False
            self._last_attempt    = time.monotonic()   # reset back-off timer

    # -------------------------------------------------------------------------
    # Key generation
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize(query: str) -> str:
        """Lowercase, collapse whitespace, truncate to 256 chars."""
        return " ".join(query.lower().strip().split())[:256]

    @staticmethod
    def _sha256(*parts: str) -> str:
        joined = "|".join(parts)
        return hashlib.sha256(joined.encode()).hexdigest()

    def _embed_key(self, query: str) -> str:
        return f"{_EMBED_NS}:{self._sha256(self._normalize(query))}"

    def _results_key(self, query: str, **params: Any) -> str:
        """
        Build a results cache key that encodes the query and all search params.
        params dict is serialised deterministically (sorted keys).
        """
        params_str = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return f"{_RESULTS_NS}:{self._sha256(self._normalize(query), params_str)}"

    # -------------------------------------------------------------------------
    # Embedding cache
    # -------------------------------------------------------------------------

    def get_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Return cached embedding for *query* or None on miss.
        Checks Redis (L1) then in-memory TTLCache (L2).
        """
        key = self._embed_key(query)

        # L1: Redis
        if self.is_available:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    self.metrics.record("redis", hit=True)
                    logger.debug("[CACHE] emb  redis  HIT  key=…%s", key[-8:])
                    return pickle.loads(raw)   # noqa: S301 — self-written data only
                self.metrics.record("redis", hit=False)
                logger.debug("[CACHE] emb  redis  MISS key=…%s", key[-8:])
            except Exception as exc:
                self._mark_down(exc)

        # L2: memory
        with self._mem_lock:
            val = self._mem_embeddings.get(key)
        if val is not None:
            self.metrics.record("memory", hit=True)
            logger.debug("[CACHE] emb  memory HIT  key=…%s", key[-8:])
            return val
        self.metrics.record("memory", hit=False)
        return None

    def set_embedding(self, query: str, embedding: np.ndarray) -> None:
        """Store *embedding* in both Redis and the in-memory fallback."""
        key = self._embed_key(query)
        raw = pickle.dumps(embedding.astype(np.float32), protocol=4)

        # Write to Redis
        if self.is_available:
            try:
                self._redis.setex(key, self.embedding_ttl, raw)
                logger.debug("[CACHE] emb  redis  SET  key=…%s ttl=%ds",
                             key[-8:], self.embedding_ttl)
            except Exception as exc:
                self._mark_down(exc)

        # Always write to memory fallback
        with self._mem_lock:
            self._mem_embeddings[key] = embedding

    # -------------------------------------------------------------------------
    # Results cache
    # -------------------------------------------------------------------------

    def get_results(self, query: str, **params: Any) -> Optional[list]:
        """
        Return cached search results or None on miss.
        *params* must match exactly what was passed to set_results().
        """
        key = self._results_key(query, **params)

        # L1: Redis
        if self.is_available:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    self.metrics.record("redis", hit=True)
                    logger.debug("[CACHE] res  redis  HIT  key=…%s", key[-8:])
                    return json.loads(raw.decode())
                self.metrics.record("redis", hit=False)
                logger.debug("[CACHE] res  redis  MISS key=…%s", key[-8:])
            except Exception as exc:
                self._mark_down(exc)

        # L2: memory
        with self._mem_lock:
            val = self._mem_results.get(key)
        if val is not None:
            self.metrics.record("memory", hit=True)
            logger.debug("[CACHE] res  memory HIT  key=…%s", key[-8:])
            return val
        self.metrics.record("memory", hit=False)
        return None

    def set_results(self, query: str, results: list, **params: Any) -> None:
        """Store *results* in both Redis and the in-memory fallback."""
        key = self._results_key(query, **params)
        raw = json.dumps(results).encode()

        if self.is_available:
            try:
                self._redis.setex(key, self.results_ttl, raw)
                logger.debug("[CACHE] res  redis  SET  key=…%s ttl=%ds n=%d",
                             key[-8:], self.results_ttl, len(results))
            except Exception as exc:
                self._mark_down(exc)

        with self._mem_lock:
            self._mem_results[key] = results

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """Return hit-rate metrics for both tiers plus connection status."""
        snap = self.metrics.snapshot()
        snap["redis_connected"] = self._redis_available
        snap["redis_url"]       = self.redis_url
        snap["embedding_ttl_s"] = self.embedding_ttl
        snap["results_ttl_s"]   = self.results_ttl
        return snap

    def log_metrics(self) -> None:
        """Emit a single INFO log line with all cache metrics."""
        m = self.get_metrics()
        o = m["overall"]
        logger.info(
            "[CACHE] status=%-4s  overall=%.1f%%  "
            "redis_hits=%d redis_misses=%d  "
            "memory_hits=%d memory_misses=%d",
            "UP" if m["redis_connected"] else "DOWN",
            o["hit_rate"] * 100,
            m["redis"]["hits"],  m["redis"]["misses"],
            m["memory"]["hits"], m["memory"]["misses"],
        )

    def ping(self) -> dict:
        """Health-check: ping Redis and return latency."""
        if self.is_available:
            try:
                t0 = time.monotonic()
                self._redis.ping()
                latency_ms = round((time.monotonic() - t0) * 1000, 2)
                return {"status": "ok", "latency_ms": latency_ms,
                        "url": self.redis_url}
            except Exception as exc:
                return {"status": "error", "error": str(exc)}
        return {
            "status": "unavailable",
            "reason": "Redis not connected — in-memory fallback active",
            "url":    self.redis_url,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: Optional[RedisCacheManager] = None
_manager_lock = threading.Lock()


def get_redis_cache() -> RedisCacheManager:
    """Return the shared RedisCacheManager, creating it on first call."""
    global _manager
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is None:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            _manager  = RedisCacheManager(
                redis_url            = redis_url,
                pool_max_connections = int(os.environ.get("REDIS_POOL_MAX_CONNECTIONS", "20")),
                socket_timeout       = float(os.environ.get("REDIS_SOCKET_TIMEOUT", "0.5")),
                connect_timeout      = float(os.environ.get("REDIS_CONNECT_TIMEOUT", "1.0")),
            )
    return _manager


def init_redis_cache(redis_url: str) -> RedisCacheManager:
    """
    Initialise (or re-initialise) the singleton with a specific URL.
    Call this from the FastAPI startup event before handling any requests.
    """
    global _manager
    with _manager_lock:
        _manager = RedisCacheManager(redis_url=redis_url)
    return _manager
