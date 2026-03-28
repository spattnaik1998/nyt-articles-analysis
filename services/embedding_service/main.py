"""
Embedding Microservice
======================

Standalone FastAPI service that owns the BERTweet model and exposes
a single HTTP endpoint for query embedding.

Endpoints
---------
POST /embed         Encode a query; returns a 768-D float32 vector.
GET  /health        Liveness + readiness check.
GET  /metrics       Cache hit rates and request counters.

Environment variables
---------------------
EMBEDDING_MODEL_NAME  HuggingFace model (default: vinai/bertweet-base)
REDIS_URL             Redis URL (default: redis://localhost:6379/0)
PORT                  Listen port   (default: 8001)

Run locally
-----------
    uvicorn services.embedding_service.main:app --port 8001 --reload
"""

from __future__ import annotations

import logging
import os
import time
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.embedding_service.model import (
    MODEL_NAME,
    encode,
    get_model,
    is_loaded,
    warmup,
)
from services.embedding_service.cache import EmbeddingCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("embedding_service")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_cache: EmbeddingCache = EmbeddingCache(
    redis_url=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
)

# Request counters (thread-safe via lock)
_stats_lock    = threading.Lock()
_total_requests = 0
_total_latency  = 0.0   # ms


def _record(latency_ms: float) -> None:
    global _total_requests, _total_latency
    with _stats_lock:
        _total_requests += 1
        _total_latency  += latency_ms


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Embedding service starting — pre-loading model …")
    warmup()
    logger.info("Embedding service ready  (model=%s)", MODEL_NAME)
    yield
    logger.info("Embedding service shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NYT Embedding Service",
    description="BERTweet query embedding microservice with Redis caching",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    query:      str = Field(..., min_length=1, max_length=2048,
                            description="Query text to embed")
    max_length: int = Field(128, ge=16, le=512,
                            description="Tokeniser truncation length")
    pooling:    str = Field("cls", pattern="^(cls|mean)$",
                            description="Pooling strategy: 'cls' or 'mean'")


class EmbedResponse(BaseModel):
    embedding:  list[float]
    dim:        int
    cached:     bool
    cache_tier: Optional[str]
    latency_ms: float
    model:      str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/embed", response_model=EmbedResponse, summary="Encode a query")
async def embed_query(req: EmbedRequest) -> EmbedResponse:
    """
    Return a 768-D float32 embedding for *query*.

    Cache lookup order: Redis (24 h TTL) → in-memory fallback → compute.
    """
    t0 = time.time()

    # ── 1. Cache lookup ───────────────────────────────────────────────────
    cached_emb, tier = _cache.get(req.query)
    if cached_emb is not None:
        latency_ms = round((time.time() - t0) * 1000, 2)
        _record(latency_ms)
        logger.info("[EMBED] cache HIT  tier=%-6s %.2fms  query=%r",
                    tier, latency_ms, req.query[:60])
        return EmbedResponse(
            embedding  = cached_emb.tolist(),
            dim        = len(cached_emb),
            cached     = True,
            cache_tier = tier,
            latency_ms = latency_ms,
            model      = MODEL_NAME,
        )

    # ── 2. Compute ────────────────────────────────────────────────────────
    try:
        tokenizer, model = get_model()
        emb = encode(req.query, tokenizer, model,
                     max_length=req.max_length, pooling=req.pooling)
    except RuntimeError as exc:
        logger.error("[EMBED] model error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("[EMBED] unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Embedding failed")

    # ── 3. Cache write ────────────────────────────────────────────────────
    _cache.set(req.query, emb)

    latency_ms = round((time.time() - t0) * 1000, 2)
    _record(latency_ms)
    logger.info("[EMBED] cache MISS compute=%.2fms  query=%r", latency_ms, req.query[:60])

    return EmbedResponse(
        embedding  = emb.tolist(),
        dim        = len(emb),
        cached     = False,
        cache_tier = None,
        latency_ms = latency_ms,
        model      = MODEL_NAME,
    )


@app.get("/health", summary="Liveness / readiness check")
async def health():
    """Returns 200 when the model is loaded and ready."""
    model_ready = is_loaded()
    cache_info  = _cache.metrics()
    status      = "ok" if model_ready else "loading"

    return {
        "status":        status,
        "model":         MODEL_NAME,
        "model_loaded":  model_ready,
        "cache":         cache_info,
    }


@app.get("/metrics", summary="Cache and throughput metrics")
async def metrics():
    """Aggregate counters: requests served, cache hit rate, average latency."""
    with _stats_lock:
        total   = _total_requests
        avg_lat = round(_total_latency / total if total else 0.0, 2)

    cache_info = _cache.metrics()
    return {
        "requests_total":   total,
        "avg_latency_ms":   avg_lat,
        "model":            MODEL_NAME,
        "cache":            cache_info,
    }
