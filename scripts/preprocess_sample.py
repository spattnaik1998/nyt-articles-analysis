#!/usr/bin/env python
"""
Preprocess NYT Sample Data

This script loads a sample of NYT articles from CSV, applies text preprocessing
(combine + clean), and saves the result to a Parquet file for efficient storage.

Usage:
    python scripts/preprocess_sample.py --input data/nyt-metadata.csv --output data/preprocessed.parquet --sample 1000
    python scripts/preprocess_sample.py --input data/nyt-metadata.csv --output data/preprocessed.parquet --all
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.load_nyt import load_nyt_csv
from src.preprocess.text import preprocess_dataframe, get_preprocessing_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function for preprocessing NYT sample data."""
    parser = argparse.ArgumentParser(
        description='Preprocess NYT article data: combine text and clean',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process 1000 random articles (default)
  python scripts/preprocess_sample.py --input data/nyt-metadata.csv

  # Process 5000 articles
  python scripts/preprocess_sample.py --input data/nyt-metadata.csv --sample 5000

  # Process all articles
  python scripts/preprocess_sample.py --input data/nyt-metadata.csv --all

  # Specify custom output path
  python scripts/preprocess_sample.py --input data/nyt-metadata.csv --output data/my_output.parquet

  # Filter by year and section
  python scripts/preprocess_sample.py --input data/nyt-metadata.csv --year 2001 --section World
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/nyt-metadata.csv',
        help='Path to input CSV file (default: data/nyt-metadata.csv)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/preprocessed.parquet',
        help='Path to output Parquet file (default: data/preprocessed.parquet)'
    )

    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=1000,
        help='Number of articles to sample (default: 1000, use --all for full dataset)'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Process all articles (ignore --sample)'
    )

    parser.add_argument(
        '--year', '-y',
        type=int,
        help='Filter by publication year (optional)'
    )

    parser.add_argument(
        '--section',
        type=str,
        help='Filter by section name (optional, e.g., "World", "Business Day")'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    parser.add_argument(
        '--no-clean',
        action='store_true',
        help='Skip cleaning step (only combine text)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("NYT Article Preprocessing Pipeline")
    logger.info("="*60)

    # Step 1: Load data
    logger.info(f"\n[1/4] Loading data from: {input_path}")
    try:
        df = load_nyt_csv(str(input_path), verbose=args.verbose)
        logger.info(f"✓ Loaded {len(df):,} articles")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    # Step 2: Apply filters
    logger.info(f"\n[2/4] Applying filters...")
    initial_count = len(df)

    if args.year:
        df = df[df['pub_date'].dt.year == args.year]
        logger.info(f"  Filtered by year {args.year}: {len(df):,} articles")

    if args.section:
        df = df[df['section_name'] == args.section]
        logger.info(f"  Filtered by section '{args.section}': {len(df):,} articles")

    if len(df) == 0:
        logger.error("No articles remaining after filtering!")
        sys.exit(1)

    # Step 3: Sample data
    logger.info(f"\n[3/4] Sampling data...")
    if args.all:
        logger.info(f"  Processing ALL {len(df):,} articles")
    else:
        sample_size = min(args.sample, len(df))
        df = df.sample(n=sample_size, random_state=args.random_seed)
        logger.info(f"  Sampled {sample_size:,} articles (seed={args.random_seed})")

    # Step 4: Preprocess
    logger.info(f"\n[4/4] Preprocessing text...")
    try:
        df_processed = preprocess_dataframe(
            df,
            combine=True,
            clean=not args.no_clean
        )
        logger.info(f"✓ Preprocessing complete")

        # Get and display stats
        if not args.no_clean:
            stats = get_preprocessing_stats(df_processed)
            logger.info(f"\nPreprocessing Statistics:")
            logger.info(f"  Total documents: {stats.get('total_documents', 0):,}")
            logger.info(f"  Avg original words: {stats.get('avg_original_words', 0):.1f}")
            logger.info(f"  Avg cleaned words: {stats.get('avg_cleaned_words', 0):.1f}")
            logger.info(f"  Avg word reduction: {stats.get('avg_word_reduction', 0):.1f} ({stats.get('avg_reduction_pct', 0):.1f}%)")
            logger.info(f"  Empty after cleaning: {stats.get('empty_after_cleaning', 0):,}")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

    # Step 5: Save to Parquet
    logger.info(f"\nSaving to: {output_path}")
    try:
        df_processed.to_parquet(output_path, index=False, compression='snappy')
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"✓ Saved {len(df_processed):,} articles to {output_path}")
        logger.info(f"  File size: {file_size:.2f} MB")
    except Exception as e:
        logger.error(f"Failed to save Parquet: {e}")
        sys.exit(1)

    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Articles processed: {len(df_processed):,}")
    logger.info(f"Columns: {list(df_processed.columns)}")

    logger.info("\n✓ Preprocessing pipeline complete!")
    logger.info("\nNext steps:")
    logger.info(f"  - Load preprocessed data: pd.read_parquet('{output_path}')")
    logger.info(f"  - Inspect data: python -c \"import pandas as pd; df = pd.read_parquet('{output_path}'); print(df.head())\"")


if __name__ == "__main__":
    main()
