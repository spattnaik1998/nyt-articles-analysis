"""
Job status abstraction over Celery's result backend.

Translates raw Celery AsyncResult states into a clean, API-friendly dict
that the /jobs/{job_id} endpoint returns to callers.

Celery states → API states
--------------------------
PENDING   → queued      (task submitted but not picked up yet)
STARTED   → running     (worker has begun processing)
PROGRESS  → running     (task emitting progress updates)
SUCCESS   → completed   (task finished; result attached)
FAILURE   → failed      (task raised an unhandled exception)
RETRY     → retrying    (worker is waiting before the next attempt)
REVOKED   → cancelled   (task was cancelled manually)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Map Celery state strings to cleaner API states
_STATE_MAP = {
    "PENDING":  "queued",
    "RECEIVED": "queued",
    "STARTED":  "running",
    "PROGRESS": "running",
    "SUCCESS":  "completed",
    "FAILURE":  "failed",
    "RETRY":    "retrying",
    "REVOKED":  "cancelled",
}


def get_job_status(job_id: str) -> dict:
    """
    Return a unified job-status dict for the given Celery task ID.

    Always returns a dict — never raises.  If Celery / Redis is unavailable
    the status will be ``"queued"`` (PENDING is the default for unknown IDs).

    Keys returned
    -------------
    job_id       str
    status       queued | running | completed | failed | retrying | cancelled
    progress     float 0.0–1.0  (None when not available)
    stage        str   human-readable stage label (None when not available)
    result       dict  task output (only when status == completed)
    error        str   exception message (only when status == failed)
    """
    try:
        from src.worker.celery_app import celery_app
        result = celery_app.AsyncResult(job_id)
        state  = result.state
    except Exception as exc:
        logger.warning("[JOB_STORE] could not fetch %s: %s", job_id, exc)
        return {
            "job_id":   job_id,
            "status":   "queued",
            "progress": None,
            "stage":    None,
            "result":   None,
            "error":    None,
        }

    api_status = _STATE_MAP.get(state, "queued")
    progress:  Optional[float] = None
    stage:     Optional[str]   = None
    task_result: Optional[Any] = None
    error:     Optional[str]   = None

    if state == "PROGRESS":
        meta     = result.info or {}
        progress = meta.get("progress")
        stage    = meta.get("stage")

    elif state == "SUCCESS":
        progress     = 1.0
        task_result  = result.result

    elif state == "FAILURE":
        exc_obj = result.result
        error   = str(exc_obj) if exc_obj is not None else "Unknown error"
        # Traceback available via result.traceback but not exposed to clients

    elif state == "RETRY":
        meta  = result.info or {}
        stage = meta.get("stage")

    return {
        "job_id":   job_id,
        "status":   api_status,
        "progress": progress,
        "stage":    stage,
        "result":   task_result,
        "error":    error,
    }


def cancel_job(job_id: str) -> bool:
    """
    Revoke (cancel) a queued or running task.

    Returns True if the revocation was sent, False on error.
    Running tasks are signalled with SIGTERM via terminate=True.
    """
    try:
        from src.worker.celery_app import celery_app
        celery_app.control.revoke(job_id, terminate=True)
        logger.info("[JOB_STORE] revoked %s", job_id)
        return True
    except Exception as exc:
        logger.warning("[JOB_STORE] revoke %s failed: %s", job_id, exc)
        return False
