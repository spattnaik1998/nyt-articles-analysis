"""
Instrumentation middleware for FastAPI endpoints.

Adds structured timing logs to key operations without modifying endpoint logic.
"""

import time
import json
import logging
import uuid
from typing import Any, Dict, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

# Configure structured logging
class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""

    def format(self, record):
        if isinstance(record.msg, dict):
            return json.dumps(record.msg, default=str)
        return super().format(record)


def setup_structured_logging(log_file: str = "logs/latency.jsonl"):
    """Setup structured logging to file."""
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class LatencyContext:
    """Context for tracking latencies within a request."""

    def __init__(self, request_id: str, endpoint: str):
        self.request_id = request_id
        self.endpoint = endpoint
        self.start_time = time.time()
        self.stages: Dict[str, float] = {}

    def record_stage(self, stage_name: str):
        """Context manager for recording stage latency."""
        class StageTimer:
            def __init__(self, ctx, name):
                self.ctx = ctx
                self.name = name
                self.start = None

            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                self.ctx.stages[self.name] = (time.time() - self.start) * 1000

        return StageTimer(self, stage_name)

    def finalize(self) -> Dict:
        """Get final latency summary."""
        total_ms = (time.time() - self.start_time) * 1000
        return {
            'request_id': self.request_id,
            'endpoint': self.endpoint,
            'timestamp': datetime.utcnow().isoformat(),
            'total_latency_ms': total_ms,
            'stages': self.stages,
            'breakdown_pct': {
                stage: f"{(latency/total_ms)*100:.1f}%"
                for stage, latency in self.stages.items()
            }
        }


def log_search_latency(query: str, k: int, alpha: float, stages: Dict[str, float], total_ms: float):
    """Log search operation details."""
    log_entry = {
        'operation': 'search',
        'query': query[:100],  # Truncate long queries
        'k': k,
        'alpha': alpha,
        'total_latency_ms': total_ms,
        'stages': stages,
        'timestamp': datetime.utcnow().isoformat()
    }
    logger.info(log_entry)


def log_sentiment_latency(year: Optional[int], section: Optional[str], stages: Dict[str, float], total_ms: float, models_used: list):
    """Log sentiment analysis operation details."""
    log_entry = {
        'operation': 'sentiment_analysis',
        'year': year,
        'section': section,
        'models_used': models_used,
        'total_latency_ms': total_ms,
        'stages': stages,
        'timestamp': datetime.utcnow().isoformat()
    }
    logger.info(log_entry)


def log_extraction_latency(year: int, section: str, article_count: int, stages: Dict[str, float], total_ms: float):
    """Log book extraction operation details."""
    log_entry = {
        'operation': 'book_extraction',
        'year': year,
        'section': section,
        'articles_processed': article_count,
        'total_latency_ms': total_ms,
        'stages': stages,
        'timestamp': datetime.utcnow().isoformat()
    }
    logger.info(log_entry)


def generate_latency_summary(measurements: Dict[str, list]) -> str:
    """Generate a summary of latency measurements."""
    from statistics import mean, median, stdev

    summary = "=== LATENCY SUMMARY ===\n\n"

    for stage, times in measurements.items():
        if not times:
            continue

        times_sorted = sorted(times)
        n = len(times)

        summary += f"{stage}:\n"
        summary += f"  Count:   {n}\n"
        summary += f"  Mean:    {mean(times):.2f}ms\n"
        summary += f"  Median:  {median(times):.2f}ms\n"
        summary += f"  P95:     {times_sorted[int(n*0.95)]:.2f}ms\n"
        summary += f"  P99:     {times_sorted[int(n*0.99)]:.2f}ms\n"
        summary += f"  Min/Max: {min(times):.2f}ms / {max(times):.2f}ms\n"
        if n > 1:
            summary += f"  StdDev:  {stdev(times):.2f}ms\n"
        summary += "\n"

    return summary


# Global measurements store
_measurements: Dict[str, list] = {}


def record_measurement(stage: str, latency_ms: float):
    """Record a latency measurement."""
    if stage not in _measurements:
        _measurements[stage] = []
    _measurements[stage].append(latency_ms)


def get_measurements_summary() -> str:
    """Get a formatted summary of all measurements."""
    return generate_latency_summary(_measurements)


def clear_measurements():
    """Clear all measurements."""
    global _measurements
    _measurements.clear()


if __name__ == "__main__":
    # Setup and demo
    setup_structured_logging()

    print("Instrumentation middleware initialized")
    print("Logs will be written to logs/latency.jsonl")
