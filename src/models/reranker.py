"""
Cross-Encoder Re-Ranking Module

Loads ms-marco-MiniLM-L-6-v2 once (lazy, singleton) and exposes a single
`rerank()` function that takes FAISS candidates and returns them sorted by
cross-encoder relevance score.

Design decisions
----------------
- Lazy model load: zero startup cost when reranking is disabled.
- Singleton: one model instance reused across all requests.
- Batch inference: all (query, passage) pairs scored in a single forward pass
  (no Python loop overhead between pairs).
- Passage text = headline + " " + abstract, truncated to 512 tokens by the
  tokenizer automatically.
- Thread-safe: model.predict() is read-only; the singleton reference is set
  once before any request arrives.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Singleton cross-encoder
# ---------------------------------------------------------------------------

_model = None          # sentence_transformers.CrossEncoder
_model_lock = threading.Lock()
_load_error: Optional[str] = None   # cached error message if load failed


def _get_model():
    """Return the shared CrossEncoder, loading it on first call."""
    global _model, _load_error

    if _model is not None:
        return _model
    if _load_error:
        raise RuntimeError(_load_error)

    with _model_lock:
        # Double-checked locking: another thread may have loaded it while we
        # waited on the lock.
        if _model is not None:
            return _model
        if _load_error:
            raise RuntimeError(_load_error)

        try:
            from sentence_transformers import CrossEncoder
            t0 = time.time()
            logger.info("[RERANK] Loading %s …", MODEL_NAME)
            _model = CrossEncoder(MODEL_NAME, max_length=512)
            logger.info("[RERANK] Model loaded in %.1fs", time.time() - t0)
        except ImportError:
            _load_error = (
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise RuntimeError(_load_error)
        except Exception as exc:
            _load_error = f"CrossEncoder load failed: {exc}"
            raise RuntimeError(_load_error)

    return _model


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """One re-ranked candidate."""
    _id: str
    rerank_score: float
    faiss_rank: int          # original rank from FAISS / hybrid fusion
    rerank_rank: int         # rank after cross-encoder
    score_delta: float       # rerank_score − (normalised faiss score)
    # original fields carried through unchanged
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 10,
    passage_fields: tuple[str, ...] = ("headline", "abstract"),
) -> tuple[list[dict], dict]:
    """
    Re-rank *candidates* with a cross-encoder and return the top-k.

    Parameters
    ----------
    query:
        The user's query string.
    candidates:
        List of result dicts from FAISS / hybrid fusion.
        Each dict must contain at least ``_id``; passage text is assembled
        from ``passage_fields``.
    top_k:
        Number of results to return after re-ranking.
    passage_fields:
        Keys whose values are concatenated to form the passage text.

    Returns
    -------
    (reranked_results, timing_dict)
        ``reranked_results`` is a list of the original dicts, augmented with:
          - ``rerank_score``   float   raw cross-encoder logit
          - ``rerank_rank``    int     1-based rank after re-ranking
          - ``faiss_rank``     int     rank before re-ranking (original ``rank`` key)
        ``timing_dict`` contains:
          - ``n_candidates``   number of candidates scored
          - ``batch_ms``       cross-encoder forward-pass time
          - ``total_ms``       total rerank() wall time
    """
    t0 = time.time()
    timing: dict = {}

    if not candidates:
        return [], {"n_candidates": 0, "batch_ms": 0.0, "total_ms": 0.0}

    n = len(candidates)
    timing["n_candidates"] = n

    # ── Build (query, passage) pairs ─────────────────────────────────────
    pairs: list[tuple[str, str]] = []
    for c in candidates:
        passage = " ".join(
            str(c.get(f, "")) for f in passage_fields if c.get(f, "")
        ).strip()
        if not passage:
            passage = str(c.get("_id", ""))   # last-resort non-empty string
        pairs.append((query, passage))

    # ── Batch inference ──────────────────────────────────────────────────
    model = _get_model()
    t_batch = time.time()
    scores = model.predict(pairs, show_progress_bar=False)
    timing["batch_ms"] = round((time.time() - t_batch) * 1000, 2)

    # ── Sort descending by cross-encoder score ───────────────────────────
    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    # ── Assemble output ──────────────────────────────────────────────────
    output: list[dict] = []
    for new_rank, (score, original) in enumerate(scored[:top_k], start=1):
        result = dict(original)           # shallow copy; don't mutate caller's list
        result["rerank_score"] = round(float(score), 4)
        result["faiss_rank"]   = original.get("rank", 0)
        result["rerank_rank"]  = new_rank
        result["rank"]         = new_rank  # overwrite so consumers see final order
        output.append(result)

    timing["total_ms"] = round((time.time() - t0) * 1000, 2)

    logger.info(
        "[RERANK] %d -> %d  batch=%.1fms total=%.1fms  query=%r",
        n,
        len(output),
        timing["batch_ms"],
        timing["total_ms"],
        query[:80],
    )

    return output, timing


def is_available() -> bool:
    """Return True if sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def warmup() -> None:
    """
    Pre-load the model so the first real request is fast.
    Call this from the FastAPI startup event.
    """
    if not is_available():
        logger.warning(
            "[RERANK] sentence-transformers not installed; re-ranking disabled. "
            "pip install sentence-transformers"
        )
        return
    try:
        _get_model()
    except Exception as exc:
        logger.warning("[RERANK] Warmup failed: %s", exc)
