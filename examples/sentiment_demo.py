#!/usr/bin/env python
"""
Sentiment Analysis Quick Demo

This script demonstrates basic sentiment classification with sample texts.

Usage:
    python examples/sentiment_demo.py
    python examples/sentiment_demo.py --model distilroberta
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sentiment import (
    classify_sentiment,
    list_available_models,
    get_model_info
)


# Sample texts covering different sentiments
SAMPLE_TEXTS = {
    'positive': [
        "Stock market rallies to record highs as economic growth accelerates",
        "Corporate earnings exceed expectations driving investor confidence",
        "Economy shows robust growth with strong employment gains",
    ],
    'negative': [
        "Market crashes amid recession fears and economic uncertainty",
        "Trade tensions escalate threatening global economic stability",
        "Financial crisis deepens as markets tumble worldwide",
    ],
    'neutral': [
        "Federal Reserve maintains interest rates at current levels",
        "Economic indicators show mixed signals for growth outlook",
        "Stock markets close with modest changes in key indices",
    ]
}


def main():
    """Run sentiment classification demo."""
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis Quick Demo'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='finbert',
        choices=list_available_models(),
        help='Model to use (default: finbert)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Sentiment Analysis Demo")
    print("=" * 70)

    # Show model info
    info = get_model_info(args.model)
    print(f"\nUsing Model: {args.model}")
    print(f"Description: {info['description']}")
    print(f"HuggingFace: {info['model_name']}")
    print(f"Labels: {', '.join(info['labels'])}")

    print("\n" + "=" * 70)
    print("Classifying Sample Texts")
    print("=" * 70)

    # Classify each category
    for expected_sentiment, texts in SAMPLE_TEXTS.items():
        print(f"\n{expected_sentiment.upper()} Examples:")
        print("-" * 70)

        for i, text in enumerate(texts, 1):
            # Classify
            result = classify_sentiment(text, model_key=args.model)

            # Display
            print(f"\n{i}. Text: {text[:60]}...")
            print(f"   Prediction: {result['label']:10s} (confidence: {result['score']:.3f})")

            # Show if matches expected
            matches = result['label'].lower() == expected_sentiment.lower()
            match_symbol = "✓" if matches else "✗"
            print(f"   Expected: {expected_sentiment:10s} {match_symbol}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Testing Batch Classification")
    print("=" * 70)

    # Collect all texts
    all_texts = []
    for texts in SAMPLE_TEXTS.values():
        all_texts.extend(texts)

    print(f"\nClassifying {len(all_texts)} texts in batch...")

    # Batch classify
    results = classify_sentiment(all_texts, model_key=args.model, batch_size=8)

    # Show distribution
    label_counts = {}
    for result in results:
        label = result['label']
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"\nLabel Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        print(f"  {label:15s}: {count:2d} ({pct:5.1f}%)")

    # Average confidence
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"\nAverage Confidence: {avg_score:.3f}")

    # Next steps
    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("""
Try different models:
  python examples/sentiment_demo.py --model finbert
  python examples/sentiment_demo.py --model distilroberta
  python examples/sentiment_demo.py --model finbert_tone

Run full analysis with report:
  python examples/sentiment_report.py --sample 50

Analyze your own data:
  python examples/sentiment_report.py --input data/preprocessed.parquet

For documentation:
  See docs/sentiment_analysis.md
    """)


if __name__ == "__main__":
    main()
