"""
Celery task definitions for long-running NYT analytics jobs.

Task catalogue
--------------
run_topic_modeling      LDA topic modelling on a year+section slice
run_sentiment_analysis  Multi-model sentiment inference on a year+section slice

Retry policy (both tasks)
--------------------------
Max 3 retries with exponential back-off: 10 s → 20 s → 40 s.
Any unhandled exception triggers an automatic retry.

Progress reporting
------------------
Tasks call self.update_state(state='PROGRESS', meta={…}) so callers polling
GET /jobs/{job_id} receive live progress percentage and stage labels.

Data access
-----------
Each worker process loads the parquet file once into a module-level singleton
(_data_df).  Subsequent tasks reuse the cached DataFrame — no repeated I/O.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from celery import Task
from celery.exceptions import MaxRetriesExceededError
from celery.utils.log import get_task_logger

from src.worker.celery_app import celery_app

logger = get_task_logger(__name__)

# ---------------------------------------------------------------------------
# Per-worker data singleton
# ---------------------------------------------------------------------------

_data_df:   Optional[pd.DataFrame] = None
_data_lock: threading.Lock          = threading.Lock()


def _load_data() -> pd.DataFrame:
    """Load the best available parquet file once per worker process."""
    global _data_df
    if _data_df is not None:
        return _data_df

    with _data_lock:
        if _data_df is not None:
            return _data_df

        candidates = [
            ("data/preprocessed_21m.parquet",  "21M"),
            ("data/preprocessed_500K.parquet", "500K"),
            ("data/preprocessed.parquet",      "20K"),
        ]
        for path, label in candidates:
            if Path(path).exists():
                logger.info("[WORKER] Loading %s parquet …", label)
                df = pd.read_parquet(path)
                if "pub_date" in df.columns and "year" not in df.columns:
                    df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
                    df["year"]     = df["pub_date"].dt.year
                _data_df = df
                logger.info("[WORKER] %d articles loaded", len(df))
                return _data_df

        raise RuntimeError(
            "No preprocessed parquet file found. "
            "Expected one of: data/preprocessed*.parquet"
        )


# ---------------------------------------------------------------------------
# Helper: unified progress meta shape
# ---------------------------------------------------------------------------

def _progress(stage: str, pct: float, extra: Optional[dict] = None) -> dict:
    meta: dict = {"stage": stage, "progress": round(pct, 2)}
    if extra:
        meta.update(extra)
    return meta


# ---------------------------------------------------------------------------
# Task: Topic modelling
# ---------------------------------------------------------------------------

@celery_app.task(
    bind            = True,
    name            = "src.worker.tasks.run_topic_modeling",
    max_retries     = 3,
    default_retry_delay = 10,
    acks_late       = True,
    track_started   = True,
)
def run_topic_modeling(
    self: Task,
    year:       int,
    section:    str,
    model:      str = "lda",
    num_topics: int = 10,
) -> dict:
    """
    Fit an LDA topic model on articles matching *year* and *section*.

    Returns
    -------
    dict with keys: topics, article_count, model, year, section, duration_s
    """
    t0 = time.time()
    job_id = self.request.id

    try:
        # ── Stage 1: load & filter ────────────────────────────────────────
        self.update_state(state="PROGRESS",
                          meta=_progress("loading_data", 0.05))
        df = _load_data()

        filtered = df[
            (df["year"] == year) &
            (df["section_name"].str.lower() == section.lower())
        ]

        if len(filtered) == 0:
            raise ValueError(
                f"No articles found for year={year}, section={section!r}"
            )

        logger.info("[TOPIC] job=%s  articles=%d", job_id, len(filtered))

        # ── Stage 2: vectorise ────────────────────────────────────────────
        self.update_state(state="PROGRESS",
                          meta=_progress("vectorising", 0.25,
                                         {"article_count": len(filtered)}))

        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        texts = filtered["cleaned_text"].fillna("").tolist()
        vectorizer = CountVectorizer(
            max_features = 1_000,
            stop_words   = "english",
            max_df       = 0.8,
            min_df       = 2,
        )
        dtm = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # ── Stage 3: fit LDA ──────────────────────────────────────────────
        self.update_state(state="PROGRESS",
                          meta=_progress("fitting_lda", 0.50))

        lda = LatentDirichletAllocation(
            n_components    = num_topics,
            random_state    = 42,
            max_iter        = 10,
            learning_method = "online",
            n_jobs          = -1,
        )
        lda.fit(dtm)

        # ── Stage 4: extract topics ───────────────────────────────────────
        self.update_state(state="PROGRESS",
                          meta=_progress("extracting_topics", 0.85))

        from src.models.topic_models import generate_topic_description
        topics = []
        for idx, component in enumerate(lda.components_):
            top_idx   = component.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_idx]
            desc      = generate_topic_description(top_words, idx)
            topics.append({
                "topic_id":   idx,
                "description": desc,
                "topic_name": desc,
                "keywords":   ", ".join(top_words[:5]),
                "words":      top_words,
            })

        duration = round(time.time() - t0, 2)
        logger.info("[TOPIC] job=%s  done in %.1fs  topics=%d",
                    job_id, duration, len(topics))

        return {
            "topics":        topics,
            "article_count": len(filtered),
            "model":         model,
            "year":          year,
            "section":       section,
            "duration_s":    duration,
        }

    except ValueError as exc:
        # Validation errors — do not retry
        logger.warning("[TOPIC] job=%s  validation error: %s", job_id, exc)
        raise

    except Exception as exc:
        logger.warning(
            "[TOPIC] job=%s  error (attempt %d/%d): %s",
            job_id,
            self.request.retries + 1,
            self.max_retries + 1,
            exc,
        )
        try:
            # Exponential back-off: 10 s, 20 s, 40 s
            delay = 10 * (2 ** self.request.retries)
            raise self.retry(exc=exc, countdown=delay)
        except MaxRetriesExceededError:
            logger.error("[TOPIC] job=%s  max retries exceeded", job_id)
            raise


# ---------------------------------------------------------------------------
# Task: Sentiment analysis
# ---------------------------------------------------------------------------

@celery_app.task(
    bind            = True,
    name            = "src.worker.tasks.run_sentiment_analysis",
    max_retries     = 3,
    default_retry_delay = 10,
    acks_late       = True,
    track_started   = True,
)
def run_sentiment_analysis(
    self:        Task,
    year:        Optional[int]  = None,
    section:     Optional[str]  = None,
    sample_size: int             = 500,
    models:      Optional[list]  = None,
) -> dict:
    """
    Run multi-model sentiment inference on articles matching the given filters.

    Parameters
    ----------
    year, section : Optional filters; None means all articles.
    sample_size   : Cap the number of articles processed (default 500).
    models        : List of model keys to use; None → auto-select by section.

    Returns
    -------
    dict with keys: models_data, article_count, filters, duration_s
    """
    t0     = time.time()
    job_id = self.request.id

    try:
        # ── Stage 1: load & filter ────────────────────────────────────────
        self.update_state(state="PROGRESS",
                          meta=_progress("loading_data", 0.05))
        df      = _load_data()
        filters = {}

        if year:
            df = df[df["year"] == year]
            filters["year"] = year
        if section:
            df = df[df["section_name"].str.lower() == section.lower()]
            filters["section"] = section

        if len(df) == 0:
            raise ValueError("No articles found with given filters")

        # ── Stage 2: sample ───────────────────────────────────────────────
        self.update_state(state="PROGRESS",
                          meta=_progress("sampling", 0.15,
                                         {"article_count": len(df)}))
        sample = df.head(sample_size).copy()
        logger.info("[SENTIMENT] job=%s  sample=%d/%d",
                    job_id, len(sample), len(df))

        # ── Stage 3: model selection ──────────────────────────────────────
        self.update_state(state="PROGRESS",
                          meta=_progress("selecting_models", 0.20))

        from src.models.sentiment import (
            batch_infer,
            MODEL_REGISTRY,
            generate_sentiment_pie_chart,
            get_recommended_models_for_section,
        )

        if models is None and section:
            models = get_recommended_models_for_section(section)

        # ── Stage 4: inference ────────────────────────────────────────────
        self.update_state(state="PROGRESS",
                          meta=_progress("running_inference", 0.30))

        result_df = batch_infer(
            sample,
            text_col             = "cleaned_text",
            models               = models,
            auto_select_models   = (models is None),
            batch_size           = 16,
            verbose              = False,
        )

        # ── Stage 5: aggregate results ────────────────────────────────────
        self.update_state(state="PROGRESS",
                          meta=_progress("aggregating", 0.90))

        models_data: dict = {}
        sentiment_cols = [c for c in result_df.columns if c.endswith("_label")]

        for col in sentiment_cols:
            model_name = col.replace("_label", "")
            score_col  = f"{model_name}_score"
            if score_col not in result_df.columns:
                continue

            label_dist = result_df[col].value_counts().to_dict()
            pie_chart  = generate_sentiment_pie_chart(
                label_distribution = label_dist,
                model_name         = model_name,
                total_count        = len(result_df),
            )
            models_data[model_name] = {
                "total_classified":    len(result_df),
                "average_confidence":  float(result_df[score_col].mean()),
                "label_distribution":  label_dist,
                "pie_chart":           pie_chart,
                "description":         MODEL_REGISTRY.get(model_name, {}).get("description", ""),
            }

        duration = round(time.time() - t0, 2)
        logger.info("[SENTIMENT] job=%s  done in %.1fs  models=%d",
                    job_id, duration, len(models_data))

        return {
            "models_data":   models_data,
            "article_count": len(sample),
            "filters":       filters,
            "duration_s":    duration,
        }

    except ValueError as exc:
        logger.warning("[SENTIMENT] job=%s  validation error: %s", job_id, exc)
        raise

    except Exception as exc:
        logger.warning(
            "[SENTIMENT] job=%s  error (attempt %d/%d): %s",
            job_id,
            self.request.retries + 1,
            self.max_retries + 1,
            exc,
        )
        try:
            delay = 10 * (2 ** self.request.retries)
            raise self.retry(exc=exc, countdown=delay)
        except MaxRetriesExceededError:
            logger.error("[SENTIMENT] job=%s  max retries exceeded", job_id)
            raise
