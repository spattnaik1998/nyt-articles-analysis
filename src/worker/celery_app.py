"""
Celery application factory.

Broker  : Redis  (shared with the caching layer)
Backend : Redis  (stores task results / status for up to 1 hour)

Queues
------
default    General-purpose tasks
topic      Topic-modelling jobs  (CPU-intensive, can be scaled independently)
sentiment  Sentiment analysis jobs (GPU-friendly workers when available)

Run a worker locally
--------------------
    # All queues
    celery -A src.worker.celery_app worker --loglevel=info

    # Topic queue only
    celery -A src.worker.celery_app worker -Q topic --loglevel=info --concurrency=2

    # Sentiment queue only (GPU workers)
    celery -A src.worker.celery_app worker -Q sentiment --loglevel=info --concurrency=1
"""

import os
from celery import Celery

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "nyt_worker",
    broker  = REDIS_URL,
    backend = REDIS_URL,
    include = ["src.worker.tasks"],
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

celery_app.conf.update(
    # Serialisation — JSON only (no pickle) for safety
    task_serializer   = "json",
    result_serializer = "json",
    accept_content    = ["json"],

    # Result TTL: keep results 1 h so callers can poll at leisure
    result_expires = 3_600,

    # Reliability
    task_track_started    = True,   # STARTED state emitted when worker picks up
    task_acks_late        = True,   # ack only after completion → safe on crash
    worker_prefetch_multiplier = 1, # one in-flight task per worker slot

    # Routing: dedicated queues per job family
    task_routes = {
        "src.worker.tasks.run_topic_modeling":    {"queue": "topic"},
        "src.worker.tasks.run_sentiment_analysis": {"queue": "sentiment"},
    },

    # Default retry policy applied via @celery_app.task(…) defaults below
    task_default_retry_delay = 10,   # seconds before first retry
)
