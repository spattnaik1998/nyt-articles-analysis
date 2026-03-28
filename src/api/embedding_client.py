"""
Async Embedding Service Client
================================

Calls the standalone embedding microservice via HTTP.
Implements exponential-backoff retry and transparent local fallback so the
main app degrades gracefully if the service is unavailable.

Retry schedule (3 attempts total):
  Attempt 1: immediate
  Attempt 2: wait 100 ms
  Attempt 3: wait 200 ms
  → give up → local fallback

Environment variables
---------------------
EMBEDDING_SERVICE_URL   Base URL of the embedding service
                        (default: http://localhost:8001)
EMBEDDING_SERVICE_TIMEOUT
                        Per-request timeout in seconds (default: 5.0)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------

_MAX_RETRIES   = 3
_RETRY_BACKOFF = [0.0, 0.1, 0.2]   # seconds before each attempt
_RETRYABLE_EXCEPTIONS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class EmbeddingServiceClient:
    """
    Async HTTP client for the embedding microservice.

    - Uses a single ``httpx.AsyncClient`` with a connection pool (keep-alive).
    - On connection failure or timeout: retries with exponential back-off.
    - After all retries exhausted: returns ``None`` so the caller can fall back
      to local inference.
    - Tracks consecutive failures; logs a single WARNING when the service goes
      down, and a single INFO when it recovers.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        timeout:  float = 5.0,
    ) -> None:
        self.base_url  = base_url.rstrip("/")
        self._timeout  = httpx.Timeout(timeout, connect=2.0)
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock() if False else threading.Lock()  # created lazily per-loop
        self._consecutive_failures = 0
        self._service_up           = True   # optimistic initial state

    def _get_client(self) -> httpx.AsyncClient:
        """Return (or lazily create) the shared async client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url = self.base_url,
                timeout  = self._timeout,
                limits   = httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def embed(
        self,
        query:      str,
        max_length: int = 128,
        pooling:    str = "cls",
    ) -> Optional[np.ndarray]:
        """
        Request an embedding for *query* from the service.

        Returns a float32 ndarray on success, or ``None`` if the service is
        unreachable after all retries (triggers local fallback in caller).
        """
        client  = self._get_client()
        payload = {"query": query, "max_length": max_length, "pooling": pooling}

        for attempt, delay in enumerate(_RETRY_BACKOFF):
            if delay:
                await asyncio.sleep(delay)

            try:
                t0       = time.monotonic()
                response = await client.post("/embed", json=payload)
                response.raise_for_status()
                data       = response.json()
                latency_ms = round((time.monotonic() - t0) * 1000, 2)

                if self._consecutive_failures:
                    logger.info(
                        "[EMB_CLIENT] Service recovered after %d failure(s)",
                        self._consecutive_failures,
                    )
                self._consecutive_failures = 0
                self._service_up           = True

                logger.debug(
                    "[EMB_CLIENT] ok  tier=%-6s  svc_ms=%.1f  query=%r",
                    data.get("cache_tier") or "compute",
                    latency_ms,
                    query[:60],
                )
                return np.array(data["embedding"], dtype=np.float32)

            except _RETRYABLE_EXCEPTIONS as exc:
                logger.debug(
                    "[EMB_CLIENT] attempt %d/%d failed (%s): %s",
                    attempt + 1, _MAX_RETRIES, type(exc).__name__, exc,
                )
                continue

            except httpx.HTTPStatusError as exc:
                # 5xx — retryable; 4xx — not retryable
                if exc.response.status_code >= 500:
                    logger.debug("[EMB_CLIENT] attempt %d/%d — HTTP %d",
                                 attempt + 1, _MAX_RETRIES, exc.response.status_code)
                    continue
                logger.warning("[EMB_CLIENT] HTTP %d — not retrying",
                               exc.response.status_code)
                break

            except Exception as exc:
                logger.warning("[EMB_CLIENT] unexpected error: %s", exc)
                break

        # All attempts exhausted
        self._consecutive_failures += 1
        if self._service_up:
            logger.warning(
                "[EMB_CLIENT] Embedding service unavailable at %s "
                "(failure #%d) — activating local fallback",
                self.base_url,
                self._consecutive_failures,
            )
            self._service_up = False

        return None   # caller will use local fallback

    async def health(self) -> dict:
        """Ping the service health endpoint; returns status dict."""
        try:
            response = await self._get_client().get("/health", timeout=2.0)
            return response.json()
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc)}

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client_instance: Optional[EmbeddingServiceClient] = None
_instance_lock   = threading.Lock()


def get_embedding_client() -> EmbeddingServiceClient:
    """Return the shared EmbeddingServiceClient, creating it on first call."""
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    with _instance_lock:
        if _client_instance is None:
            url     = os.environ.get("EMBEDDING_SERVICE_URL", "http://localhost:8001")
            timeout = float(os.environ.get("EMBEDDING_SERVICE_TIMEOUT", "5.0"))
            _client_instance = EmbeddingServiceClient(base_url=url, timeout=timeout)
            logger.info("[EMB_CLIENT] Initialised → %s", url)
    return _client_instance


def init_embedding_client(base_url: str, timeout: float = 5.0) -> EmbeddingServiceClient:
    """Initialise the singleton with explicit settings (call from startup)."""
    global _client_instance
    with _instance_lock:
        _client_instance = EmbeddingServiceClient(base_url=base_url, timeout=timeout)
    return _client_instance
