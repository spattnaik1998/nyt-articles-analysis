"""
Chunked preprocessing of NYT articles for large-scale datasets (21M corpus).

Reads CSV in chunks, applies cleaning and filtering, writes to parquet incrementally.
Reuses src.preprocess.text.clean_text for consistency.
"""

import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from typing import Optional, Dict
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess.text import clean_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_chunk(
    df: pd.DataFrame,
    text_col: str = 'combined_text',
    min_word_count: int = 10
) -> pd.DataFrame:
    """
    Process a single chunk: combine text, clean, filter by word count.

    Args:
        df: Raw articles DataFrame
        text_col: Column name for combined text output
        min_word_count: Minimum words to keep article

    Returns:
        Processed DataFrame
    """
    df = df.copy()

    # Column mapping
    if 'headline' not in df.columns:
        df['headline'] = ''
    if 'abstract' not in df.columns:
        df['abstract'] = ''
    if 'lead_paragraph' not in df.columns:
        df['lead_paragraph'] = ''
    if 'body' not in df.columns:
        df['body'] = ''

    # Combine text: headline × 2 + abstract + body[:500]
    headline = df['headline'].fillna('').astype(str)
    abstract = df['abstract'].fillna('').astype(str)
    body = df['body'].fillna('').astype(str).str[:500]

    df[text_col] = headline + ' ' + headline + ' ' + abstract + ' ' + body

    # Ensure pub_date is datetime
    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

    # Clean text
    df['cleaned_text'] = df[text_col].apply(clean_text)

    # Compute word count
    df['word_count'] = df['cleaned_text'].str.split().str.len()

    # Filter: keep articles with >= min_word_count words
    df = df[df['word_count'] >= min_word_count].copy()

    # Select relevant columns
    keep_cols = [
        '_id', 'headline', 'abstract', 'body', 'section_name',
        'pub_date', 'word_count', 'combined_text', 'cleaned_text'
    ]
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols]

    return df


def preprocess_21m(
    input_file: str,
    output_file: str,
    chunk_size: int = 500000,
    min_word_count: int = 10,
    verbose: bool = True
):
    """
    Stream-process a large CSV into a parquet file with chunked reading and writing.

    Args:
        input_file: Path to input CSV
        output_file: Path to output parquet
        chunk_size: Rows per chunk (default 500K)
        min_word_count: Minimum words to keep article
        verbose: Show progress
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track statistics
    stats = {
        'total_raw': 0,
        'total_kept': 0,
        'total_filtered': 0,
        'year_dist': defaultdict(int),
        'section_dist': defaultdict(int),
    }

    writer = None
    schema = None
    chunk_num = 0

    try:
        logger.info(f"Starting preprocessing: {input_file}")
        logger.info(f"Chunk size: {chunk_size:,} rows")
        logger.info(f"Min word count filter: {min_word_count} words")

        # Read CSV in chunks
        for chunk_df in pd.read_csv(input_path, chunksize=chunk_size, dtype={'_id': str}):
            chunk_num += 1
            stats['total_raw'] += len(chunk_df)

            if verbose:
                logger.info(f"Processing chunk {chunk_num} ({len(chunk_df):,} rows)...")

            # Process chunk
            processed = process_chunk(chunk_df, min_word_count=min_word_count)
            stats['total_kept'] += len(processed)
            stats['total_filtered'] += len(chunk_df) - len(processed)

            # Update year and section distributions
            if 'pub_date' in processed.columns:
                years = pd.to_datetime(processed['pub_date']).dt.year
                for year in years.dropna():
                    stats['year_dist'][int(year)] += 1

            if 'section_name' in processed.columns:
                for section in processed['section_name'].dropna():
                    stats['section_dist'][str(section)] += 1

            # Convert to PyArrow Table
            if len(processed) > 0:
                table = pa.Table.from_pandas(processed, preserve_index=False)

                # Initialize writer on first chunk
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(output_path, schema)

                # Append chunk to parquet
                writer.write_table(table)

        # Close writer
        if writer is not None:
            writer.close()

        # Log results
        logger.info("=" * 70)
        logger.info(f"✓ Preprocessing complete!")
        logger.info(f"  Total raw rows: {stats['total_raw']:,}")
        logger.info(f"  Total kept: {stats['total_kept']:,}")
        logger.info(f"  Total filtered (<{min_word_count} words): {stats['total_filtered']:,}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Output size: {output_path.stat().st_size / (1024**3):.2f} GB")

        logger.info("\nYear Distribution (top 10):")
        for year, count in sorted(stats['year_dist'].items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {year}: {count:,}")

        logger.info("\nSection Distribution (top 10):")
        for section, count in sorted(stats['section_dist'].items(), key=lambda x: -x[1])[:10]:
            logger.info(f"  {section}: {count:,}")

        # Save metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.csv"
        metadata = pd.DataFrame({
            'key': ['total_raw', 'total_kept', 'filtered_count', 'output_file'],
            'value': [
                stats['total_raw'],
                stats['total_kept'],
                stats['total_filtered'],
                str(output_path)
            ]
        })
        metadata.to_csv(metadata_path, index=False)
        logger.info(f"✓ Metadata saved to {metadata_path}")

    except Exception as e:
        logger.error(f"❌ Error during preprocessing: {e}")
        if writer is not None:
            writer.close()
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess large NYT CSV datasets into chunked parquet files"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output parquet file path'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500000,
        help='Chunk size for reading CSV (default: 500000)'
    )
    parser.add_argument(
        '--min-word-count',
        type=int,
        default=10,
        help='Minimum words to keep article (default: 10)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress logging'
    )

    args = parser.parse_args()

    preprocess_21m(
        input_file=args.input,
        output_file=args.output,
        chunk_size=args.chunk_size,
        min_word_count=args.min_word_count,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
