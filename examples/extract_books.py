#!/usr/bin/env python
"""
Book Metadata Extraction Example

This script extracts book titles and author names from book review articles
using LLM-based extraction with regex fallbacks.

Usage:
    # Basic usage with sample data
    python examples/extract_books.py

    # Extract from preprocessed data
    python examples/extract_books.py --input data/preprocessed.parquet --section Books

    # Use regex only (no LLM)
    python examples/extract_books.py --no-llm

    # Custom sample size
    python examples/extract_books.py --sample 100

    # Save results
    python examples/extract_books.py --output data/extractions --save-by-year
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.extraction import (
    batch_extract,
    get_extraction_stats,
    filter_successful_extractions
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run book metadata extraction."""
    parser = argparse.ArgumentParser(
        description='Book Metadata Extraction - Extract titles and authors from reviews',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from sample data (regex only)
  python examples/extract_books.py --no-llm

  # Extract from preprocessed Books articles
  python examples/extract_books.py --input data/preprocessed.parquet --section Books

  # Use LLM extraction (requires OPENAI_API_KEY)
  export OPENAI_API_KEY=sk-...
  python examples/extract_books.py --input data/preprocessed.parquet --section Books

  # Custom sample and output
  python examples/extract_books.py --input data/preprocessed.parquet --sample 50 --output data/books
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input CSV or Parquet file'
    )

    parser.add_argument(
        '--text-col',
        type=str,
        default='combined_text',
        help='Column containing text to extract from (default: combined_text)'
    )

    parser.add_argument(
        '--section',
        type=str,
        help='Filter to specific section (e.g., Books)'
    )

    parser.add_argument(
        '--year',
        type=int,
        help='Filter to specific year'
    )

    parser.add_argument(
        '--sample', '-s',
        type=int,
        help='Limit to N random articles'
    )

    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM extraction (requires OPENAI_API_KEY)'
    )

    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Use regex only (no LLM)'
    )

    parser.add_argument(
        '--llm-model',
        type=str,
        default='gpt-3.5-turbo',
        help='OpenAI model to use (default: gpt-3.5-turbo)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/extractions',
        help='Output directory (default: data/extractions)'
    )

    parser.add_argument(
        '--save-by-year',
        action='store_true',
        help='Save separate files per year'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10)'
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Book Metadata Extraction")
    logger.info("=" * 70)

    # Determine LLM usage
    use_llm = False
    if args.use_llm and not args.no_llm:
        # Check for API key
        if os.getenv('OPENAI_API_KEY'):
            use_llm = True
            logger.info(f"Using LLM extraction with {args.llm_model}")
        else:
            logger.warning("OPENAI_API_KEY not set. Falling back to regex only.")
            logger.info("Set API key with: export OPENAI_API_KEY=sk-...")
    else:
        logger.info("Using regex extraction only")

    # Load data
    logger.info(f"\n[1/3] Loading data...")

    if args.input:
        input_path = Path(args.input)

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            logger.info("\nTo generate preprocessed data, run:")
            logger.info("  python scripts/preprocess_sample.py --output data/preprocessed.parquet")
            return 1

        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            return 1

        logger.info(f"✓ Loaded {len(df):,} articles from {input_path}")

    else:
        # Use sample data for demonstration
        logger.info("No input file specified. Creating sample dataset...")

        sample_data = {
            'combined_text': [
                'Review of "The Great Gatsby" by F. Scott Fitzgerald',
                '"1984" by George Orwell explores themes of totalitarianism',
                'Harper Lee\'s "To Kill a Mockingbird" remains a classic',
                'A new biography of Ernest Hemingway, "Papa" by John Smith',
                '"The Catcher in the Rye" by J.D. Salinger',
                'George Orwell\'s dystopian novel "Animal Farm"',
                'Review: "Brave New World" by Aldous Huxley',
                '"Pride and Prejudice," by Jane Austen, published in 1813',
                'The latest novel from Margaret Atwood, "The Testaments"',
                '"The Lord of the Rings" trilogy by J.R.R. Tolkien'
            ],
            'section_name': ['Books'] * 10,
            'pub_date': pd.to_datetime([
                '2020-01-15', '2020-03-20', '2020-06-10', '2020-09-05', '2021-02-14',
                '2021-05-22', '2021-08-30', '2021-11-12', '2022-03-08', '2022-07-19'
            ]),
            'headline': [f'Book Review {i+1}' for i in range(10)]
        }

        df = pd.DataFrame(sample_data)
        logger.info(f"✓ Created sample dataset with {len(df)} book reviews")

    # Validate text column
    if args.text_col not in df.columns:
        logger.error(f"Text column '{args.text_col}' not found in DataFrame")
        logger.info(f"Available columns: {', '.join(df.columns)}")
        return 1

    # Filter by section
    if args.section:
        logger.info(f"\n[2/3] Filtering to section: {args.section}")

        if 'section_name' not in df.columns:
            logger.warning("section_name column not found, skipping filter")
        else:
            before_count = len(df)
            df = df[df['section_name'] == args.section].copy()
            logger.info(f"✓ Filtered to {len(df):,} articles (from {before_count:,})")

            if len(df) == 0:
                logger.error(f"No articles found for section '{args.section}'")
                return 1

    # Filter by year
    if args.year:
        logger.info(f"\n[2/3] Filtering to year: {args.year}")

        if 'pub_date' not in df.columns:
            logger.warning("pub_date column not found, skipping filter")
        else:
            # Ensure pub_date is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['pub_date']):
                df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

            before_count = len(df)
            df = df[df['pub_date'].dt.year == args.year].copy()
            logger.info(f"✓ Filtered to {len(df):,} articles (from {before_count:,})")

    # Sample if requested
    if args.sample and len(df) > args.sample:
        logger.info(f"\n[2/3] Sampling {args.sample:,} random articles...")
        df = df.sample(n=args.sample, random_state=42)
        logger.info(f"✓ Sampled {len(df):,} articles")

    if len(df) == 0:
        logger.error("No articles to process after filtering")
        return 1

    # Run extraction
    logger.info(f"\n[3/3] Extracting book metadata...")
    logger.info(f"Processing {len(df):,} articles")

    try:
        result_df = batch_extract(
            df,
            text_col=args.text_col,
            use_llm=use_llm,
            llm_model=args.llm_model,
            max_workers=args.max_workers,
            output_dir=args.output,
            save_by_year=args.save_by_year,
            verbose=True
        )

        # Show sample results
        logger.info(f"\n" + "=" * 70)
        logger.info("Sample Extraction Results (first 5)")
        logger.info("=" * 70)

        successful = filter_successful_extractions(result_df)

        if len(successful) > 0:
            display_df = successful.head(5)[[
                'book_title',
                'author_name',
                'extraction_method'
            ]].copy()

            print("\n" + display_df.to_string(index=False))
        else:
            logger.warning("No successful extractions to display")

        # Show failed examples if any
        failed = result_df[result_df['extraction_success'] == False]

        if len(failed) > 0:
            logger.info(f"\n" + "=" * 70)
            logger.info(f"Failed Extractions ({len(failed)} total, showing first 3)")
            logger.info("=" * 70)

            for idx, row in failed.head(3).iterrows():
                text_preview = row[args.text_col][:100] if args.text_col in row else "N/A"
                logger.info(f"\nRow {idx}:")
                logger.info(f"  Text: {text_preview}...")

        # Get statistics
        stats = get_extraction_stats(result_df)

        logger.info(f"\n" + "=" * 70)
        logger.info("Final Statistics")
        logger.info("=" * 70)
        logger.info(f"\nTotal articles: {stats['total']:,}")
        logger.info(f"Successful: {stats['successful']:,} ({stats['success_rate']:.2%})")
        logger.info(f"Failed: {stats['failed']:,} ({(1-stats['success_rate']):.2%})")

        logger.info(f"\nMethod breakdown:")
        for method, count in sorted(stats['method_breakdown'].items(), key=lambda x: -x[1]):
            pct = count / stats['total'] * 100
            logger.info(f"  {method:20s}: {count:6,} ({pct:5.1f}%)")

        # Check if we achieved ~99.9% success rate
        if stats['success_rate'] >= 0.999:
            logger.info(f"\n✓ Excellent! Achieved {stats['success_rate']:.3%} success rate (target: ~99.9%)")
        elif stats['success_rate'] >= 0.95:
            logger.info(f"\n✓ Good! Achieved {stats['success_rate']:.1%} success rate")
        else:
            logger.info(f"\n⚠ Success rate: {stats['success_rate']:.1%} (may need improvement)")

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Next steps
    logger.info("\n" + "=" * 70)
    logger.info("Next Steps")
    logger.info("=" * 70)

    logger.info(f"""
1. Review results:
   - Results saved to: {args.output}/
   - Check books_YYYY.parquet files

2. Analyze successful extractions:
   import pandas as pd
   df = pd.read_parquet('{args.output}/books_2020.parquet')
   successful = df[df['extraction_success'] == True]
   print(successful[['book_title', 'author_name']])

3. Use extracted metadata:
   - Generate book embeddings
   - Build book recommendation system
   - Analyze author popularity over time

4. Improve extraction:
   - Use LLM for higher accuracy (set OPENAI_API_KEY)
   - Add custom regex patterns for your data
   - Review failed cases and refine patterns
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
