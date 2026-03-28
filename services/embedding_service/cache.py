"""
Two-tier embedding cache for the embedding microservice.

L1: Redis  — TTL 24 h, shared across workers / restarts
L2: Memory — in-process TTLCache fallback (cachetools) when Redis is down

This module is intentionally self-contained (no imports from src/) so the
service can be deployed independently.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_TTL  = 86_400   # 24 hours
_KEY_PREFIX    = "svc:emb:v1"
_RECONNECT_SEC = 60.0


class EmbeddingCache:
    """
    Redis (L1) → memory TTLCache (L2) two-tier embedding cache.

    All methods are thread-safe and never raise — errors are logged
    and execution falls through to the next tier or returns None.
    """

    def __init__(
        self,
        redis_url:    str   = "redis://localhost:6379/0",
        ttl:          int   = EMBEDDING_TTL,
        memory_size:  int   = 2_000,
        socket_timeout: float = 0.5,
        connect_timeout: float = 1.0,
    ) -> None:
        self.ttl   = ttl
        self._lock = threading.RLock()

        # In-memory fallback
        try:
            from cachetools import TTLCache
            self._mem: dict = TTLCache(maxsize=memory_size, ttl=ttl)
        except ImportError:
            self._mem = {}
            logger.warning("[CACHE] cachetools missing; using plain dict fallback")

        # Redis state
        self._redis           = None
        self._redis_url       = redis_url
        self._socket_timeout  = socket_timeout
        self._connect_timeout = connect_timeout
        self._available       = False
        self._last_attempt    = 0.0
        self._conn_lock       = threading.Lock()

        # Hit / miss counters
        self._redis_hits    = 0
        self._redis_misses  = 0
        self._memory_hits   = 0
        self._memory_misses = 0

        self._try_connect()

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    def _try_connect(self) -> bool:
        now = time.monotonic()
        if now - self._last_attempt < _RECONNECT_SEC:
            return self._available
        with self._conn_lock:
            if time.monotonic() - self._last_attempt < _RECONNECT_SEC:
                return self._available
            self._last_attempt = time.monotonic()
            try:
                import redis as _r
                pool   = _r.ConnectionPool.from_url(
                    self._redis_url,
                    max_connections        = 10,
                    socket_timeout         = self._socket_timeout,
                    socket_connect_timeout = self._connect_timeout,
                    decode_responses       = False,
                )
                client = _r.Redis(connection_pool=pool)
                client.ping()
                self._redis     = client
                self._available = True
                logger.info("[CACHE] Redis connected: %s", self._redis_url)
            except ImportError:
                logger.warning("[CACHE] redis not installed; memory-only mode")
                self._available = False
            except Exception as exc:
                logger.warning("[CACHE] Redis unavailable (%s); memory fallback active", exc)
                self._available = False
        return self._available

    @property
    def is_available(self) -> bool:
        if self._available:
            return True
        return self._try_connect()

    def _mark_down(self, exc: Exception) -> None:
        if self._available:
            logger.warning("[CACHE] Redis error — memory-only: %s", exc)
            self._available    = False
            self._last_attempt = time.monotonic()

    # -------------------------------------------------------------------------
    # Key
    # -------------------------------------------------------------------------

    @staticmethod
    def _key(query: str) -> str:
        norm = " ".join(query.lower().strip().split())[:256]
        h    = hashlib.sha256(norm.encode()).hexdigest()
        return f"{_KEY_PREFIX}:{h}"

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get(self, query: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Return ``(embedding, tier)`` where tier is ``"redis"`` or ``"memory"``,
        or ``(None, None)`` on a complete miss.
        """
        key = self._key(query)

        # L1: Redis
        if self.is_available:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    with self._lock:
                        self._redis_hits += 1
                    logger.debug("[CACHE] HIT  redis  key=...%s", key[-8:])
                    emb = pickle.loads(raw)   # noqa: S301 — self-written only
                    # Backfill memory
                    with self._lock:
                        self._mem[key] = emb
                    return emb, "redis"
                with self._lock:
                    self._redis_misses += 1
            except Exception as exc:
                self._mark_down(exc)

        # L2: Memory
        with self._lock:
            val = self._mem.get(key)
            if val is not None:
                self._memory_hits += 1
                logger.debug("[CACHE] HIT  memory key=...%s", key[-8:])
                return val, "memory"
            self._memory_misses += 1

        return None, None

    def set(self, query: str, embedding: np.ndarray) -> None:
        """Store *embedding* in Redis and in-memory fallback."""
        key = self._key(query)
        raw = pickle.dumps(embedding.astype(np.float32), protocol=4)

        if self.is_available:
            try:
                self._redis.setex(key, self.ttl, raw)
                logger.debug("[CACHE] SET  redis  key=...%s ttl=%ds", key[-8:], self.ttl)
            except Exception as exc:
                self._mark_down(exc)

        with self._lock:
            self._mem[key] = embedding

    def metrics(self) -> dict:
        with self._lock:
            rh, rm = self._redis_hits, self._redis_misses
            mh, mm = self._memory_hits, self._memory_misses
        total      = rh + rm + mh + mm
        total_hits = rh + mh
        return {
            "redis":  {"hits": rh, "misses": rm,
                       "hit_rate": round(rh / (rh + rm) if rh + rm else 0, 4)},
            "memory": {"hits": mh, "misses": mm,
                       "hit_rate": round(mh / (mh + mm) if mh + mm else 0, 4)},
            "overall": {"hits": total_hits, "total": total,
                        "hit_rate": round(total_hits / total if total else 0, 4)},
            "redis_connected": self._available,
        }
