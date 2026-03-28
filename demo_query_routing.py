#!/usr/bin/env python3
"""
Query Routing Layer — Demo & Correctness Check

Runs the router classifier over a test suite and shows routing decisions
without touching the database or any model. Safe to run offline.

Usage:
    python demo_query_routing.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.api.query_router import QueryRouter, QueryMode


# ── Test cases ──────────────────────────────────────────────────────────────
# (query, expected_mode, note)
TEST_CASES = [
    # --- Fast expected ---
    ("business news 2020",              QueryMode.FAST, "short section query"),
    ("technology articles",             QueryMode.FAST, "two-word section lookup"),
    ("sports",                          QueryMode.FAST, "single word"),
    ("NYT politics",                    QueryMode.FAST, "brand + section"),
    ("climate 2022",                    QueryMode.FAST, "topic + year"),
    ("healthcare policy",               QueryMode.FAST, "two-word concept"),

    # --- Deep expected ---
    ("extract book title and author from this review",
     QueryMode.DEEP, "extraction keyword"),
    ("verify the author of the novel mentioned in this article",
     QueryMode.DEEP, "verify keyword"),
    ("who wrote the book reviewed in the New York Times in 2019?",
     QueryMode.DEEP, "question + proper noun + year"),
    ("summarize the key arguments made by economists about inflation",
     QueryMode.DEEP, "summarize keyword + complex"),
    ("compare reporting on climate change in Business versus World sections",
     QueryMode.DEEP, "compare keyword + complex connective"),
    ("analyze sentiment of political articles published before the 2020 election",
     QueryMode.DEEP, "analyze keyword"),
    ("\"The Great Gatsby\" author verification",
     QueryMode.DEEP, "quoted title"),
    ("recommend articles similar to those about Federal Reserve policy changes",
     QueryMode.DEEP, "recommend keyword"),
]


def run_demo():
    router = QueryRouter()

    print("\n" + "=" * 78)
    print("  Query Routing Layer — Classification Demo")
    print("=" * 78)

    passed = 0
    failed = 0

    for query, expected, note in TEST_CASES:
        decision = router.classify(query)
        ok = decision.mode == expected
        symbol = "[PASS]" if ok else "[FAIL]"
        if ok:
            passed += 1
        else:
            failed += 1

        tag = f"{decision.mode.value:5s} (conf={decision.confidence:.2f})"
        print(f"{symbol}  {tag}  {query[:55]:<55s}  # {note}")
        if not ok:
            print(f"         expected={expected.value}  reason: {decision.reason}")

    print()
    print(f"Results: {passed}/{len(TEST_CASES)} passed"
          + (f"  ({failed} failed)" if failed else "  (all correct)"))
    print("=" * 78)

    # ── Feature inspection for a few queries ────────────────────────────
    print("\nFeature breakdown for selected queries:")
    samples = [
        "technology articles",
        "extract book title and author from this review",
        "compare reporting on climate change in Business versus World sections",
    ]
    for q in samples:
        d = router.classify(q)
        print(f"\n  Query: {q!r}")
        print(f"  Mode:  {d.mode.value}  conf={d.confidence}  ({d.latency_ms:.2f} ms)")
        print(f"  Reason: {d.reason}")
        feats = d.features
        print(f"  Features: words={feats['word_count']}  "
              f"deep_kw={feats['deep_keyword_count']}  "
              f"complexity={feats['complexity_pattern_hits']}  "
              f"quoted={feats['has_quoted_phrase']}")

    # ── Override demo ────────────────────────────────────────────────────
    print("\n\nOverride demo (user forces fast/deep):")
    from src.api.query_router import get_router
    r = get_router()

    for q, ov in [
        ("climate change analysis", "fast"),
        ("latest news",            "deep"),
    ]:
        d = r.resolve(q, override=ov)
        print(f"  query={q!r}  override={ov!r} -> mode={d.mode.value}  reason={d.reason!r}")

    print()
    return failed == 0


if __name__ == "__main__":
    ok = run_demo()
    sys.exit(0 if ok else 1)
