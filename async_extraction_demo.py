#!/usr/bin/env python3
"""
Async Extraction Demo - Demonstrates async vs sync performance.

Shows:
1. Async single extraction
2. Async batch extraction with concurrency
3. Latency comparison (async vs sync)
4. Timing breakdown for each pipeline stage
"""

import asyncio
import time
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.async_extraction import (
    async_extract_and_verify,
    async_batch_extract_and_verify,
    run_async_extraction,
)
from src.models.extraction import extract_and_verify_book_meta


# Test data
TEST_TEXTS = [
    "Review of '1984' by George Orwell - a dystopian novel",
    "Jane Austen's 'Pride and Prejudice' is a classic romance",
    "The Great Gatsby by F. Scott Fitzgerald explores the American dream",
    "'To Kill a Mockingbird' by Harper Lee remains influential",
    "George R.R. Martin's 'A Game of Thrones' spans multiple perspectives",
    "Stephen King's 'The Shining' is a horror masterpiece",
    "'The Catcher in the Rye' by J.D. Salinger features Holden Caulfield",
    "Haruki Murakami's 'Norwegian Wood' is a popular romance novel",
    "'1Q84' is another notable work by Murakami",
    "Isaac Asimov wrote 'Foundation', a science fiction epic",
]


async def demo_single_async_extraction():
    """Demonstrate single async extraction."""
    print("\n" + "=" * 70)
    print("DEMO 1: SINGLE ASYNC EXTRACTION")
    print("=" * 70)

    text = TEST_TEXTS[0]
    print(f"\nText: {text}")

    print("\nExtracting with async pipeline...")
    t_start = time.time()

    verified, extraction, timing = await async_extract_and_verify(
        text,
        use_llm=False,  # Skip LLM (no API keys) to demo structure
        use_verification=False
    )

    t_total = (time.time() - t_start) * 1000

    print(f"\nResults:")
    if extraction.success:
        print(f"  Extraction: {extraction.method} (success)")
        print(f"  Title: {extraction.book_meta.book_title}")
        print(f"  Author: {extraction.book_meta.author_name}")
    else:
        print(f"  Extraction failed: {extraction.error}")

    print(f"\nTiming:")
    for key, value in timing.items():
        print(f"  {key:20s}: {value:8.2f}ms")


async def demo_batch_async_extraction():
    """Demonstrate batch async extraction with concurrency."""
    print("\n" + "=" * 70)
    print("DEMO 2: BATCH ASYNC EXTRACTION")
    print("=" * 70)

    print(f"\nProcessing {len(TEST_TEXTS)} texts asynchronously...")
    print(f"Max concurrent: 3")

    t_start = time.time()

    results, stats = await async_batch_extract_and_verify(
        TEST_TEXTS,
        use_llm=False,  # Skip LLM (no API keys) to demo structure
        use_verification=False,
        max_concurrent=3,
        verbose=True
    )

    t_total = (time.time() - t_start) * 1000

    print(f"\nResults Summary:")
    for i, (verified, extraction, timing) in enumerate(results[:3]):
        print(f"\n  [{i+1}] {TEST_TEXTS[i][:50]}...")
        if extraction.success:
            print(f"      Title: {extraction.book_meta.book_title}")
            print(f"      Author: {extraction.book_meta.author_name}")
            print(f"      Time: {timing['total_ms']:.1f}ms")
        else:
            print(f"      Failed: {extraction.error}")
            print(f"      Time: {timing.get('total_ms', 'N/A')}ms")

    print(f"\nBatch Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            if 'rate' in key:
                print(f"  {key:25s}: {value:8.1%}")
            elif 'ms' in key:
                print(f"  {key:25s}: {value:8.1f}ms")
            else:
                print(f"  {key:25s}: {value:8.2f}")
        else:
            print(f"  {key:25s}: {value}")


async def demo_timing_breakdown():
    """Demonstrate timing breakdown for pipeline stages."""
    print("\n" + "=" * 70)
    print("DEMO 3: TIMING BREAKDOWN")
    print("=" * 70)

    # Sample a few texts to show stage breakdown
    sample_texts = TEST_TEXTS[:3]

    print(f"\nProcessing {len(sample_texts)} texts with detailed timing...")

    results_with_timing = []

    for i, text in enumerate(sample_texts):
        print(f"\n  [{i+1}] Processing: {text[:40]}...")
        t_start = time.time()

        verified, extraction, timing = await async_extract_and_verify(
            text,
            use_llm=False,
            use_verification=False
        )

        results_with_timing.append((text, timing))

        print(f"      Extraction:  {timing.get('extraction_ms', 0):6.1f}ms")
        print(f"      Verification: {timing.get('verification_ms', 0):6.1f}ms")
        print(f"      Total:       {timing['total_ms']:6.1f}ms")

    # Summary
    print(f"\nTiming Summary (across {len(sample_texts)} texts):")
    all_extraction_times = [t[1]['extraction_ms'] for t in results_with_timing]
    all_verification_times = [t[1]['verification_ms'] for t in results_with_timing]
    all_total_times = [t[1]['total_ms'] for t in results_with_timing]

    if all_extraction_times:
        print(f"  Extraction:")
        print(f"    Mean:   {sum(all_extraction_times) / len(all_extraction_times):.1f}ms")
        print(f"    Min:    {min(all_extraction_times):.1f}ms")
        print(f"    Max:    {max(all_extraction_times):.1f}ms")

    if all_total_times:
        print(f"  Total:")
        print(f"    Mean:   {sum(all_total_times) / len(all_total_times):.1f}ms")
        print(f"    Min:    {min(all_total_times):.1f}ms")
        print(f"    Max:    {max(all_total_times):.1f}ms")


async def demo_concurrency_comparison():
    """Compare different concurrency levels."""
    print("\n" + "=" * 70)
    print("DEMO 4: CONCURRENCY COMPARISON")
    print("=" * 70)

    test_data = TEST_TEXTS[:5]
    concurrency_levels = [1, 2, 5]

    print(f"\nProcessing {len(test_data)} texts with different concurrency levels...\n")

    for max_concurrent in concurrency_levels:
        print(f"  Max concurrent: {max_concurrent}")
        t_start = time.time()

        results, stats = await async_batch_extract_and_verify(
            test_data,
            use_llm=False,
            use_verification=False,
            max_concurrent=max_concurrent,
            verbose=False
        )

        t_total = (time.time() - t_start) * 1000
        print(f"    Total time: {t_total:.1f}ms")
        print(f"    Avg/text:   {stats['avg_time_per_text_ms']:.1f}ms")
        print()


async def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "Async LLM Extraction - Demonstration".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        await demo_single_async_extraction()
        await demo_batch_async_extraction()
        await demo_timing_breakdown()
        await demo_concurrency_comparison()

        print("\n" + "=" * 70)
        print("✓ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 70)

        print("\nKey Benefits of Async Approach:")
        print("  1. Non-blocking I/O - Efficient concurrency without threads")
        print("  2. Reduced context switching overhead")
        print("  3. Better scalability for high-concurrency scenarios")
        print("  4. Simplified error handling with asyncio")
        print("  5. Natural async/await syntax")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
