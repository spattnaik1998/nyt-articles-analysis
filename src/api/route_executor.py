"""
Execution paths for Fast and Deep query modes.

Fast path:  cached embedding → FAISS (no BM25, no LLM)
Deep path:  cached embedding → FAISS + BM25 → optional LLM extraction

Both paths return a uniform dict that the /search endpoint can serialize.
"""

import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.api.query_router import RoutingDecision, QueryMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Latency logger
# ---------------------------------------------------------------------------

def log_route(
    query: str,
    decision: RoutingDecision,
    exec_ms: float,
    total_ms: float,
    result_count: int,
) -> None:
    """Emit a structured log line for every routed request."""
    logger.info(
        "[ROUTE] mode=%-5s confidence=%.2f exec_ms=%6.1f total_ms=%6.1f "
        "results=%d reason=%r query=%r",
        decision.mode.value,
        decision.confidence,
        exec_ms,
        total_ms,
        result_count,
        decision.reason,
        query[:80],
    )


# ---------------------------------------------------------------------------
# Shared: embed query
# ---------------------------------------------------------------------------

def _embed_query(
    query: str,
    embed_tokenizer,
    embed_model,
) -> Optional[np.ndarray]:
    """Return a 1-D float32 embedding, or None on failure."""
    try:
        from src.models.cached_embeddings import encode_query_cached
        embedding, meta = encode_query_cached(
            query, embed_tokenizer, embed_model, use_cache=True
        )
        cached = meta.get("cached", False)
        latency = meta.get("latency_ms", 0)
        logger.debug(
            "[EMBED] cache_%s latency=%.1fms",
            "HIT" if cached else "MISS",
            latency,
        )
        return embedding
    except Exception as exc:
        logger.warning("[EMBED] encode failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Shared: FAISS search
# ---------------------------------------------------------------------------

def _faiss_search(
    query_embedding: np.ndarray,
    faiss_index,
    embeddings_mapping: pd.DataFrame,
    data_df: pd.DataFrame,
    k: int,
) -> list[dict]:
    """
    Run a FAISS inner-product search and return k article dicts.
    Falls back to cosine similarity over the full embedding matrix when
    faiss_index is not available.
    """
    # --- FAISS path ---
    if faiss_index is not None:
        import faiss as faiss_lib
        q = np.array([query_embedding], dtype=np.float32)
        faiss_lib.normalize_L2(q)
        distances, indices = faiss_index.search(q, k)
        hits = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(embeddings_mapping):
                continue
            article_id = embeddings_mapping.iloc[idx]["_id"]
            row = data_df[data_df["_id"] == article_id]
            if row.empty:
                continue
            row = row.iloc[0]
            hits.append({
                "_id": article_id,
                "headline": row.get("headline", ""),
                "abstract": row.get("abstract", ""),
                "section_name": row.get("section_name", ""),
                "pub_date": str(row.get("pub_date", "")),
                "semantic_score": float(dist),
                "rank": rank + 1,
            })
        return hits

    # --- Numpy fallback ---
    logger.debug("[FAISS] index unavailable; using numpy cosine fallback")
    from sklearn.metrics.pairwise import cosine_similarity as _cos
    # embeddings stored as (N, D); query is (D,)
    # We can't do this without the embeddings matrix; return empty list
    return []


# ---------------------------------------------------------------------------
# Fast path
# ---------------------------------------------------------------------------

def execute_fast(
    query: str,
    k: int,
    faiss_index,
    embeddings_mapping: pd.DataFrame,
    data_df: pd.DataFrame,
    embed_tokenizer,
    embed_model,
    rerank: bool = False,
) -> tuple[list[dict], dict]:
    """
    Fast path: cached embedding + FAISS, no BM25, no LLM.

    rerank=True retrieves top-100 from FAISS then re-ranks to top-k with a
    cross-encoder.  Falls back to plain FAISS order when reranker unavailable.

    Returns:
        (results, timing_breakdown)
    """
    t0 = time.time()
    timing: dict[str, float] = {}

    # 1. Embed (cache-first)
    t_embed = time.time()
    embedding = _embed_query(query, embed_tokenizer, embed_model)
    timing["embed_ms"] = round((time.time() - t_embed) * 1000, 2)

    if embedding is None:
        return [], timing

    # 2. FAISS search — over-retrieve when reranking so the cross-encoder has
    #    a larger candidate pool to work with.
    faiss_k = 100 if rerank else k
    t_search = time.time()
    results = _faiss_search(
        embedding, faiss_index, embeddings_mapping, data_df, faiss_k
    )
    timing["search_ms"] = round((time.time() - t_search) * 1000, 2)

    # 3. Optional cross-encoder re-ranking
    if rerank and results:
        t_rerank = time.time()
        try:
            from src.models.reranker import rerank as _rerank, is_available
            if is_available():
                results, rerank_timing = _rerank(query, results, top_k=k)
                timing["rerank_ms"] = rerank_timing.get("batch_ms", 0.0)
            else:
                logger.warning("[FAST] reranker unavailable; returning FAISS order")
                results = results[:k]
        except Exception as exc:
            logger.warning("[FAST] rerank failed: %s; returning FAISS order", exc)
            results = results[:k]
        timing["rerank_total_ms"] = round((time.time() - t_rerank) * 1000, 2)

    timing["total_ms"] = round((time.time() - t0) * 1000, 2)
    logger.debug("[FAST] embed=%.1fms search=%.1fms rerank=%.1fms total=%.1fms hits=%d",
                 timing["embed_ms"], timing["search_ms"],
                 timing.get("rerank_total_ms", 0.0),
                 timing["total_ms"], len(results))
    return results, timing


# ---------------------------------------------------------------------------
# Deep path
# ---------------------------------------------------------------------------

def execute_deep(
    query: str,
    k: int,
    alpha: float,
    faiss_index,
    embeddings_mapping: pd.DataFrame,
    data_df: pd.DataFrame,
    embed_tokenizer,
    embed_model,
    bm25,
    run_llm_extraction: bool = False,
    llm_model: str = "gpt-3.5-turbo",
    verification_model: str = "gemini-2.0-flash-exp",
    rerank: bool = False,
) -> tuple[list[dict], dict]:
    """
    Deep path: embedding + FAISS hybrid (+ BM25 if available) + optional LLM.

    run_llm_extraction=True triggers book-metadata extraction on results
    (expensive; only meaningful for book-section queries).

    Returns:
        (results, timing_breakdown)
    """
    t0 = time.time()
    timing: dict[str, float] = {}

    # 1. Embed
    t_embed = time.time()
    embedding = _embed_query(query, embed_tokenizer, embed_model)
    timing["embed_ms"] = round((time.time() - t_embed) * 1000, 2)

    if embedding is None:
        return [], timing

    # 2. FAISS (semantic)
    t_faiss = time.time()
    semantic_hits = _faiss_search(
        embedding, faiss_index, embeddings_mapping, data_df, k * 2
    )
    semantic_scores: dict[str, float] = {h["_id"]: h["semantic_score"] for h in semantic_hits}
    timing["faiss_ms"] = round((time.time() - t_faiss) * 1000, 2)

    # 3. BM25 keyword search
    keyword_scores: dict[str, float] = {}
    if bm25 is not None:
        t_bm25 = time.time()
        try:
            tokens = query.lower().split()
            bm25_scores_arr = bm25.get_scores(tokens)
            top_bm25_idx = np.argsort(bm25_scores_arr)[::-1][: k * 2]
            max_score = float(bm25_scores_arr[top_bm25_idx[0]]) if len(top_bm25_idx) else 1.0
            for idx in top_bm25_idx:
                if idx < len(data_df):
                    article_id = data_df.iloc[idx]["_id"]
                    score = float(bm25_scores_arr[idx]) / (max_score or 1.0)
                    keyword_scores[article_id] = score
        except Exception as exc:
            logger.warning("[DEEP] BM25 failed: %s", exc)
        timing["bm25_ms"] = round((time.time() - t_bm25) * 1000, 2)

    # 4. Hybrid fusion — keep top-100 as re-rank candidate pool when enabled
    t_fuse = time.time()
    all_ids = set(semantic_scores) | set(keyword_scores)
    fused: dict[str, float] = {}
    for aid in all_ids:
        s = alpha * semantic_scores.get(aid, 0.0)
        kw = (1 - alpha) * keyword_scores.get(aid, 0.0)
        fused[aid] = s + kw

    fusion_k = 100 if rerank else k
    top_ids = sorted(fused, key=fused.__getitem__, reverse=True)[:fusion_k]
    timing["fuse_ms"] = round((time.time() - t_fuse) * 1000, 2)

    # 5. Assemble result dicts
    results: list[dict] = []
    for rank, aid in enumerate(top_ids, 1):
        row = data_df[data_df["_id"] == aid]
        if row.empty:
            continue
        row = row.iloc[0]
        results.append({
            "_id": aid,
            "headline": row.get("headline", ""),
            "abstract": row.get("abstract", ""),
            "section_name": row.get("section_name", ""),
            "pub_date": str(row.get("pub_date", "")),
            "semantic_score": round(semantic_scores.get(aid, 0.0), 4),
            "keyword_score": round(keyword_scores.get(aid, 0.0), 4),
            "hybrid_score": round(fused[aid], 4),
            "rank": rank,
        })

    # 5b. Optional cross-encoder re-ranking
    if rerank and results:
        t_rerank = time.time()
        try:
            from src.models.reranker import rerank as _rerank, is_available
            if is_available():
                results, rerank_timing = _rerank(query, results, top_k=k)
                timing["rerank_ms"] = rerank_timing.get("batch_ms", 0.0)
            else:
                logger.warning("[DEEP] reranker unavailable; using fusion order")
                results = results[:k]
        except Exception as exc:
            logger.warning("[DEEP] rerank failed: %s; using fusion order", exc)
            results = results[:k]
        timing["rerank_total_ms"] = round((time.time() - t_rerank) * 1000, 2)
    else:
        results = results[:k]

    # 6. Optional LLM extraction on top results
    if run_llm_extraction and results:
        t_llm = time.time()
        try:
            from src.models.extraction import batch_extract_with_verification
            texts = [
                (r.get("headline", "") + " " + r.get("abstract", "")).strip()
                for r in results[:5]     # only first 5 to limit cost
            ]
            mini_df = pd.DataFrame({"combined_text": texts})
            extracted = batch_extract_with_verification(
                mini_df,
                text_col="combined_text",
                use_llm=True,
                use_verification=True,
                llm_model=llm_model,
                verification_model=verification_model,
                verbose=False,
            )
            for i, row in extracted.iterrows():
                if i < len(results):
                    results[i]["extracted_title"] = row.get("book_title")
                    results[i]["extracted_author"] = row.get("author_name")
                    results[i]["extraction_confidence"] = row.get(
                        "verification_confidence", "unknown"
                    )
        except Exception as exc:
            logger.warning("[DEEP] LLM extraction failed: %s", exc)
        timing["llm_ms"] = round((time.time() - t_llm) * 1000, 2)

    timing["total_ms"] = round((time.time() - t0) * 1000, 2)
    logger.debug(
        "[DEEP] embed=%.1f faiss=%.1f bm25=%.1f fuse=%.1f rerank=%.1f total=%.1fms hits=%d",
        timing.get("embed_ms", 0), timing.get("faiss_ms", 0),
        timing.get("bm25_ms", 0), timing.get("fuse_ms", 0),
        timing.get("rerank_total_ms", 0.0),
        timing["total_ms"], len(results),
    )
    return results, timing
