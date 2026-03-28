#!/usr/bin/env python3
"""
Server-Sent Events (SSE) Streaming Demo

Demonstrates real-time token streaming for LLM responses.
Shows how to consume and process SSE events from the streaming endpoints.

Usage:
    python sse_streaming_demo.py
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.streaming_extraction import (
    stream_extract_article,
    stream_extract_batch
)


def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title):
    """Print section divider."""
    print(f"\n--- {title} ---\n")


async def demo_1_single_article():
    """Demo 1: Stream extraction for a single article."""
    print_header("DEMO 1: SINGLE ARTICLE STREAMING")

    article_text = """
    Stephen King's latest novel "Holly" has received widespread critical acclaim.
    The book follows detective Holly Gibney on a haunting investigation.
    King masterfully weaves suspense and emotional depth throughout the narrative.
    """

    print_section("Streaming extraction...")
    print("Text:", article_text[:100] + "...")
    print("\nTokens:")

    token_count = 0
    first_token_time = None
    start_time = time.time()

    async for event in stream_extract_article(article_text, article_id="demo_1"):
        if event['type'] == 'token':
            token_count += 1
            token = event['data']['token']
            elapsed = event['data']['elapsed_ms']

            # Track first token
            if token_count == 1:
                first_token_time = elapsed
                print(f"  First token received in {first_token_time:.0f}ms ✓")

            print(f"{token}", end='', flush=True)

        elif event['type'] == 'complete':
            total_ms = event['data']['total_ms']
            tokens = event['data']['tokens']
            speed = event['data']['tokens_per_sec']

            print("\n")
            print_section("Results")
            print(f"Total time:        {total_ms:.1f}ms")
            print(f"Tokens:            {tokens}")
            print(f"Speed:             {speed:.1f} tokens/sec")
            print(f"First token:       {first_token_time:.0f}ms" if first_token_time else "N/A")

        elif event['type'] == 'error':
            print(f"\n✗ Error: {event['data']['message']}")

    print("\n✓ Demo 1 complete")


async def demo_2_parsing():
    """Demo 2: Parse streaming response."""
    print_header("DEMO 2: PARSING STREAMING RESPONSE")

    article_text = """
    "The Great Gatsby" by F. Scott Fitzgerald remains a masterpiece of American literature.
    Fitzgerald's prose style and character development continue to captivate readers worldwide.
    """

    print_section("Streaming extraction with parsing...")
    print("Text:", article_text[:80] + "...")

    buffer = ""
    token_count = 0
    start_time = time.time()

    async for event in stream_extract_article(article_text, article_id="demo_2"):
        if event['type'] == 'token':
            token = event['data']['token']
            token_count += 1
            buffer += token

            if token_count <= 20:
                print(f"{token}", end='', flush=True)

        elif event['type'] == 'complete':
            print(f"...[ {token_count} more tokens ]...\n")
            print_section("Parsed Response")

            # Try to parse as JSON
            try:
                import json
                parsed = json.loads(buffer)

                for key, value in parsed.items():
                    if value and value != "null":
                        print(f"{key:20s}: {value}")

            except json.JSONDecodeError:
                print("Response (raw):")
                print(buffer[:200] + ("..." if len(buffer) > 200 else ""))

    print("\n✓ Demo 2 complete")


async def demo_3_batch_streaming():
    """Demo 3: Batch streaming with multiple articles."""
    print_header("DEMO 3: BATCH STREAMING")

    articles = [
        {
            'id': 'article_1',
            'text': 'Stephen King wrote "Holly" about detective Holly Gibney solving a mystery.'
        },
        {
            'id': 'article_2',
            'text': 'J.K. Rowling continues the Harry Potter universe with new stories.'
        },
        {
            'id': 'article_3',
            'text': 'Margaret Atwood releases a sequel to "The Handmaid\'s Tale".'
        }
    ]

    print_section(f"Streaming extraction for {len(articles)} articles...")

    article_results = {}
    current_article = None

    async for event in stream_extract_batch(articles):
        event_type = event['type']

        if event_type == 'batch_start':
            data = event['data']
            print(f"Starting batch: {data['total_articles']} articles")

        elif event_type == 'article_start':
            data = event['data']
            current_article = data['article_id']
            article_num = data['article_number']
            print(f"\n[Article {article_num}] {current_article}")
            article_results[current_article] = ''

        elif event_type == 'token':
            if current_article:
                token = event['data']['token']
                article_results[current_article] += token
                print(token, end='', flush=True)

        elif event_type == 'article_end':
            print()

        elif event_type == 'batch_end':
            data = event['data']
            print(f"\n\nBatch complete: {data['processed_articles']} articles processed")

        elif event_type == 'error':
            print(f"\n✗ Error: {event['data']['message']}")

    print_section("Results Summary")
    for article_id, response in article_results.items():
        token_count = len(response.split())
        print(f"{article_id:20s}: {token_count} tokens")

    print("\n✓ Demo 3 complete")


async def demo_4_performance_tracking():
    """Demo 4: Track performance metrics during streaming."""
    print_header("DEMO 4: PERFORMANCE METRICS")

    article_text = """
    "To Kill a Mockingbird" by Harper Lee is a profound exploration of racial injustice
    in the American South. The novel remains relevant and impactful to contemporary readers.
    Lee's character development and narrative technique are remarkable achievements in literature.
    """

    print_section("Streaming with metrics...")

    metrics = {
        'tokens': 0,
        'first_token_time': None,
        'start_time': time.time(),
        'token_times': []
    }

    print("Tokens arriving:")

    async for event in stream_extract_article(article_text, article_id="demo_4"):
        if event['type'] == 'token':
            data = event['data']
            token = data['token']
            elapsed = data['elapsed_ms']

            metrics['tokens'] += 1
            metrics['token_times'].append(elapsed)

            if metrics['tokens'] == 1:
                metrics['first_token_time'] = elapsed

            print(f"{token}", end='', flush=True)

        elif event['type'] == 'complete':
            data = event['data']
            print("\n")
            print_section("Performance Metrics")

            total_ms = data['total_ms']
            tokens = data['tokens']
            tokens_per_sec = data['tokens_per_sec']

            print(f"First token latency: {metrics['first_token_time']:.0f}ms")
            print(f"Total tokens:        {tokens}")
            print(f"Total time:          {total_ms:.0f}ms")
            print(f"Speed:               {tokens_per_sec:.1f} tokens/sec")
            print(f"Avg per token:       {total_ms / tokens:.1f}ms" if tokens > 0 else "N/A")

            # Calculate token time distribution
            if metrics['token_times']:
                times = metrics['token_times']
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)

                print(f"\nToken timing distribution:")
                print(f"  Min:  {min_time:.0f}ms")
                print(f"  Avg:  {avg_time:.0f}ms")
                print(f"  Max:  {max_time:.0f}ms")

    print("\n✓ Demo 4 complete")


async def demo_5_error_handling():
    """Demo 5: Error handling during streaming."""
    print_header("DEMO 5: ERROR HANDLING")

    print_section("Testing with empty text (will demonstrate fallback)...")

    # Empty text should still work (falls back to regex)
    article_text = ""

    token_count = 0
    async for event in stream_extract_article(article_text, article_id="demo_5"):
        if event['type'] == 'token':
            token_count += 1
            print(f"Token {token_count}: {event['data']['token'][:50]}...")

        elif event['type'] == 'complete':
            print(f"\nCompleted with {token_count} tokens")

        elif event['type'] == 'error':
            print(f"Error caught: {event['data']['message']}")
            print("✓ Error handling works correctly")

    print("\n✓ Demo 5 complete")


async def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "Server-Sent Events (SSE) Streaming - Demonstration".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        # Run demos
        await demo_1_single_article()
        await demo_2_parsing()
        await demo_3_batch_streaming()
        await demo_4_performance_tracking()
        await demo_5_error_handling()

        # Summary
        print_header("DEMONSTRATIONS COMPLETE")
        print("""
✓ Single article streaming with token-by-token output
✓ Parsing and processing streamed JSON responses
✓ Batch processing multiple articles in sequence
✓ Performance metrics tracking during streaming
✓ Error handling and fallback mechanisms

Key Takeaways:
1. Streaming provides immediate feedback (no waiting for full response)
2. First token appears in ~150-300ms (network latency dependent)
3. Tokens arrive at 20-50 tokens/sec (OpenAI streaming speed)
4. Frontend can render tokens faster than they arrive
5. Error handling ensures robustness

Frontend Implementation:
- Open: http://localhost:8000/streaming-extraction.html
- Enter text and click "Stream Extraction"
- Watch tokens appear in real-time with progress metrics

API Endpoints:
- POST /extract/stream?text=...
- POST /books/extract/stream?year=...&section=...&limit=...
        """)

        print("\n" + "=" * 70)
        print("✓ All demonstrations completed successfully!")
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
