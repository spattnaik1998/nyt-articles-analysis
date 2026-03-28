"""
Query Routing Layer - Fast vs Deep Mode Selection

Classifies incoming queries as FAST or DEEP based on:
  - Query length and structure
  - Keyword signals (extraction/verification intent → deep)
  - Complexity heuristics (word count, punctuation, question type)

Fast Mode:  cached embedding + FAISS only, ~20-100ms
Deep Mode:  embedding + BM25 + LLM extraction + Gemini verification, 2-15s
"""

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------

class QueryMode(str, Enum):
    FAST = "fast"
    DEEP = "deep"
    AUTO = "auto"          # resolved before execution


@dataclass
class RoutingDecision:
    mode: QueryMode
    confidence: float          # 0.0 – 1.0
    reason: str
    features: dict = field(default_factory=dict)
    latency_ms: float = 0.0    # time taken to classify


@dataclass
class RouteResult:
    """Wrapper returned by every execution path."""
    mode_used: QueryMode
    results: list
    routing_decision: RoutingDecision
    execution_latency_ms: float
    total_latency_ms: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

# Words that strongly suggest the user wants LLM-powered extraction/analysis
_DEEP_KEYWORDS = frozenset({
    "extract", "extraction", "verify", "verification", "summarize", "summary",
    "analyze", "analysis", "explain", "compare", "recommend", "suggest",
    "review", "evaluate", "assess", "who wrote", "written by", "author of",
    "book by", "novel by", "published", "title of",
})

# Connectives and comparative phrases → richer intent
_COMPLEXITY_PATTERNS = [
    re.compile(r"\b(and|or|but|because|since|although|however|therefore)\b", re.I),
    re.compile(r"\b(what is|who is|how does|why did|when did|where is)\b", re.I),
    re.compile(r"\b(compared to|similar to|different from|as well as)\b", re.I),
    re.compile(r'["\']', ),                     # quoted strings → title lookup
    re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"),# two-word proper noun
]

_FAST_SECTION_TERMS = frozenset({
    "business", "politics", "sports", "world", "technology",
    "science", "health", "arts", "books", "travel",
})


def _extract_features(query: str) -> dict:
    q = query.strip()
    words = q.split()
    lower = q.lower()

    deep_kw_hits = [w for w in _DEEP_KEYWORDS if w in lower]
    complexity_pattern_hits = sum(1 for p in _COMPLEXITY_PATTERNS if p.search(q))
    has_section_term = any(t in lower for t in _FAST_SECTION_TERMS)

    return {
        "char_len": len(q),
        "word_count": len(words),
        "has_question_mark": "?" in q,
        "deep_keyword_hits": deep_kw_hits,
        "deep_keyword_count": len(deep_kw_hits),
        "complexity_pattern_hits": complexity_pattern_hits,
        "has_quoted_phrase": bool(re.search(r'["\']', q)),
        "has_section_term": has_section_term,
        "has_proper_nouns": bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', q)),
    }


# ---------------------------------------------------------------------------
# Core router
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Classify a query as FAST or DEEP.

    Scoring (each rule contributes a signed delta; sum >= threshold → DEEP):

    +2  any deep keyword present
    +1  per additional deep keyword (max +3)
    +1  two or more complexity patterns matched
    +1  query contains a question mark with >8 words
    +1  contains quoted phrase (exact title lookup)
    -1  query ≤ 3 words (almost certainly a simple lookup)
    -1  query contains a section term only (e.g. "business news 2020")

    Threshold: score >= 2 → DEEP, else FAST
    """

    def __init__(self, deep_threshold: int = 2):
        self.deep_threshold = deep_threshold

    def classify(self, query: str) -> RoutingDecision:
        t0 = time.time()
        features = _extract_features(query)
        score = 0
        reason_parts = []

        # --- deep keyword signal ---
        if features["deep_keyword_count"] > 0:
            score += 2
            reason_parts.append(
                f"deep keywords ({', '.join(features['deep_keyword_hits'][:3])})"
            )
            bonus = min(features["deep_keyword_count"] - 1, 3)
            score += bonus

        # --- complexity patterns ---
        if features["complexity_pattern_hits"] >= 2:
            score += 1
            reason_parts.append("complex connectives/patterns")

        # --- question with substance ---
        if features["has_question_mark"] and features["word_count"] > 8:
            score += 1
            reason_parts.append("detailed question")

        # --- quoted phrase (often title lookup) ---
        if features["has_quoted_phrase"]:
            score += 1
            reason_parts.append("quoted phrase")

        # --- simplicity discounts ---
        if features["word_count"] <= 3:
            score -= 1
            reason_parts.append("very short query")

        if features["has_section_term"] and features["word_count"] <= 5:
            score -= 1
            reason_parts.append("section-only query")

        # --- decision ---
        if score >= self.deep_threshold:
            mode = QueryMode.DEEP
            confidence = min(0.5 + score * 0.1, 0.99)
            reason = "Deep mode: " + "; ".join(reason_parts) if reason_parts else "Deep mode: score threshold met"
        else:
            mode = QueryMode.FAST
            confidence = min(0.5 + abs(score - self.deep_threshold) * 0.15, 0.99)
            reason = "Fast mode: " + ("; ".join(reason_parts) if reason_parts else "simple keyword query")

        latency_ms = (time.time() - t0) * 1000

        return RoutingDecision(
            mode=mode,
            confidence=round(confidence, 3),
            reason=reason,
            features=features,
            latency_ms=round(latency_ms, 3),
        )

    def resolve(
        self,
        query: str,
        override: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Return a RoutingDecision, respecting an explicit override.

        Args:
            query:    Raw query string.
            override: "fast" | "deep" | None  (None → auto-classify).
        """
        if override in ("fast", "deep"):
            mode = QueryMode(override)
            return RoutingDecision(
                mode=mode,
                confidence=1.0,
                reason=f"User override -> {override}",
                features={},
                latency_ms=0.0,
            )

        decision = self.classify(query)

        logger.info(
            "[ROUTER] query=%r → mode=%s confidence=%.2f reason=%r latency=%.2fms",
            query[:80],
            decision.mode.value,
            decision.confidence,
            decision.reason,
            decision.latency_ms,
        )
        return decision


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_router: Optional[QueryRouter] = None


def get_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router
