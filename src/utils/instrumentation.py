"""
Instrumentation utilities for latency measurement and structured logging.

Provides:
- Timing decorators for functions
- Context managers for measuring code blocks
- Structured logging with JSON output
- Latency collection for benchmarking
"""

import time
import json
import logging
from functools import wraps
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

# Thread-local storage for request context
_request_context = threading.local()

# Global latency store (per-request)
_latency_store: Dict[str, Dict[str, Any]] = {}


class LatencyTracker:
    """Collects and tracks latencies for benchmarking."""

    def __init__(self):
        self.measurements: Dict[str, list] = {}
        self.lock = threading.Lock()

    def record(self, stage: str, latency_ms: float, metadata: Optional[Dict] = None):
        """Record a latency measurement."""
        with self.lock:
            if stage not in self.measurements:
                self.measurements[stage] = []
            self.measurements[stage].append({
                'latency_ms': latency_ms,
                'metadata': metadata or {}
            })

    def get_stats(self, stage: str) -> Optional[Dict]:
        """Get statistics for a stage."""
        with self.lock:
            if stage not in self.measurements or not self.measurements[stage]:
                return None

            latencies = [m['latency_ms'] for m in self.measurements[stage]]
            latencies.sort()
            n = len(latencies)

            return {
                'count': n,
                'mean_ms': sum(latencies) / n,
                'p50_ms': latencies[n // 2],
                'p95_ms': latencies[int(n * 0.95)],
                'p99_ms': latencies[int(n * 0.99)],
                'min_ms': min(latencies),
                'max_ms': max(latencies),
            }

    def clear(self):
        """Clear all measurements."""
        with self.lock:
            self.measurements.clear()


# Global tracker instance
tracker = LatencyTracker()


def time_block(stage_name: str, metadata: Optional[Dict] = None) -> contextmanager:
    """
    Context manager to measure execution time of a code block.

    Usage:
        with time_block("query_encoding") as timer:
            # code here
            pass
        print(f"Took {timer.elapsed_ms}ms")
    """
    @contextmanager
    def _timer():
        class Timer:
            def __init__(self):
                self.start_time = None
                self.elapsed_ms = 0

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, *args):
                self.elapsed_ms = (time.time() - self.start_time) * 1000
                tracker.record(stage_name, self.elapsed_ms, metadata)

        yield Timer().__enter__()

    return _timer()


def time_function(stage_name: Optional[str] = None):
    """
    Decorator to measure execution time of a function.

    Usage:
        @time_function("query_embedding")
        def encode_query(text):
            ...
    """
    def decorator(func: Callable) -> Callable:
        stage = stage_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.time() - start) * 1000
                tracker.record(stage, elapsed_ms)
                # Also log
                logger.info(f"[TIMING] {stage}: {elapsed_ms:.2f}ms")

        return wrapper
    return decorator


def log_structured(event: str, data: Dict[str, Any], level: str = "INFO"):
    """
    Log structured data as JSON.

    Usage:
        log_structured("search_completed", {
            "query": "artificial intelligence",
            "k": 5,
            "latency_ms": 250.5,
            "results": 5
        })
    """
    log_entry = {
        'timestamp': time.time(),
        'event': event,
        **data
    }
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(json.dumps(log_entry))


def get_request_context() -> Dict[str, Any]:
    """Get the current request context (timing measurements)."""
    if not hasattr(_request_context, 'data'):
        _request_context.data = {}
    return _request_context.data


def set_request_context(**kwargs):
    """Set values in the request context."""
    ctx = get_request_context()
    ctx.update(kwargs)


def clear_request_context():
    """Clear the current request context."""
    if hasattr(_request_context, 'data'):
        del _request_context.data


@contextmanager
def measure_request(request_id: str):
    """
    Context manager for a full request.

    Usage:
        with measure_request("search_001") as ctx:
            # Your request handling code
            pass
        print(ctx)  # Returns accumulated timings
    """
    clear_request_context()
    set_request_context(request_id=request_id, stages={})

    start = time.time()

    class RequestContext:
        def __init__(self):
            self.request_id = request_id
            self.start_time = start
            self.stages = {}

        def add_stage(self, stage_name: str, latency_ms: float):
            self.stages[stage_name] = latency_ms

        def total_ms(self):
            return (time.time() - self.start_time) * 1000

        def to_dict(self):
            return {
                'request_id': self.request_id,
                'total_ms': self.total_ms(),
                'stages': self.stages
            }

    ctx = RequestContext()
    try:
        yield ctx
    finally:
        clear_request_context()


# Convenience decorators for common operations

def measure_embedding_time(func: Callable) -> Callable:
    """Decorator for embedding functions."""
    return time_function("embedding_generation")(func)


def measure_search_time(func: Callable) -> Callable:
    """Decorator for search functions."""
    return time_function("search")(func)


def measure_inference_time(func: Callable) -> Callable:
    """Decorator for model inference."""
    return time_function("model_inference")(func)


if __name__ == "__main__":
    # Demo usage
    print("Instrumentation utilities demo")
    print("=" * 60)

    # Test 1: Time block
    print("\n1. Time block context manager:")
    with time_block("test_block", {"batch_size": 32}):
        time.sleep(0.1)

    # Test 2: Time function decorator
    print("\n2. Function decorator:")
    @time_function("example_func")
    def example():
        time.sleep(0.05)

    example()

    # Test 3: Structured logging
    print("\n3. Structured logging:")
    log_structured("test_event", {
        "value": 42,
        "status": "success"
    })

    # Test 4: Request context
    print("\n4. Request measurement:")
    with measure_request("test_001") as ctx:
        with time_block("stage_1"):
            time.sleep(0.05)
        with time_block("stage_2"):
            time.sleep(0.03)
        print(f"Total: {ctx.total_ms():.2f}ms")

    # Test 5: Get stats
    print("\n5. Statistics:")
    stats = tracker.get_stats("stage_1")
    print(f"stage_1 stats: {stats}")
