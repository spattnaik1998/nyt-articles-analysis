"""
NYT Article CSV Data Loader

This module provides functions to load and clean NYT article metadata from CSV files.
"""

import pandas as pd
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_nyt_csv(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load and clean NYT article metadata from a CSV file.

    This function robustly loads the NYT metadata CSV, performs essential cleaning,
    and returns a DataFrame ready for analysis.

    Args:
        path (str): Path to the nyt-metadata.csv file
        verbose (bool): Whether to print progress and statistics (default: True)

    Returns:
        pd.DataFrame: Cleaned DataFrame with essential columns and no critical nulls

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If the CSV is empty or missing required columns
        pd.errors.ParserError: If the CSV cannot be parsed

    Example:
        >>> df = load_nyt_csv('data/nyt-metadata.csv')
        Loading NYT metadata from: data/nyt-metadata.csv
        Successfully loaded 2,000,000 rows
        After cleaning: 1,999,995 rows
        Date range: 2000-01-01 to 2025-12-27
    """
    # Essential columns to keep
    essential_columns = [
        '_id', 'pub_date', 'headline', 'web_url', 'abstract', 'lead_paragraph',
        'section_name', 'subsection_name', 'byline', 'document_type',
        'type_of_material', 'word_count', 'keywords'
    ]

    # Text columns that should be filled with empty strings if missing
    text_columns = [
        'abstract', 'lead_paragraph', 'section_name', 'subsection_name',
        'byline', 'type_of_material', 'keywords', 'document_type'
    ]

    # Critical columns that must not be null
    critical_columns = ['_id', 'pub_date', 'headline', 'web_url', 'word_count']

    try:
        if verbose:
            logger.info(f"Loading NYT metadata from: {path}")

        # Load CSV with robust error handling
        df = pd.read_csv(
            path,
            low_memory=False,  # Avoid mixed type inference warnings
            na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None'],
            keep_default_na=True
        )

        initial_row_count = len(df)
        if verbose:
            logger.info(f"Successfully loaded {initial_row_count:,} rows")

        if initial_row_count == 0:
            raise ValueError("CSV file is empty")

        # Check if all essential columns exist
        missing_columns = set(essential_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )

        # Select only essential columns
        df = df[essential_columns].copy()

        # Convert pub_date to datetime
        df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

        # Drop rows where pub_date conversion failed
        null_dates_before = df['pub_date'].isnull().sum()
        if null_dates_before > 0 and verbose:
            logger.warning(f"Found {null_dates_before:,} rows with invalid dates")

        # Drop rows where critical columns are missing
        df = df.dropna(subset=critical_columns)

        rows_dropped = initial_row_count - len(df)
        if verbose and rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped:,} rows with missing critical values")

        # Fill missing values in text columns with empty strings
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')

        # Convert word_count to integer (it might be float after loading)
        df['word_count'] = df['word_count'].astype(int)

        final_row_count = len(df)
        if verbose:
            logger.info(f"After cleaning: {final_row_count:,} rows")

        # Calculate and display date range
        if not df.empty and 'pub_date' in df.columns:
            min_date = df['pub_date'].min()
            max_date = df['pub_date'].max()
            duration = max_date - min_date

            if verbose:
                logger.info(f"Date range: {min_date.date()} to {max_date.date()}")
                logger.info(f"Duration: {duration.days:,} days (~{duration.days/365.25:.1f} years)")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {e}")
        raise


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of the loaded NYT data.

    Args:
        df (pd.DataFrame): DataFrame returned by load_nyt_csv

    Returns:
        dict: Summary statistics including row count, date range, sections, etc.

    Example:
        >>> df = load_nyt_csv('data/nyt-metadata.csv')
        >>> summary = get_data_summary(df)
        >>> print(summary['row_count'])
        1999995
    """
    summary = {
        'row_count': len(df),
        'columns': list(df.columns),
        'date_range': {
            'min': df['pub_date'].min() if not df.empty else None,
            'max': df['pub_date'].max() if not df.empty else None,
        },
        'top_sections': df['section_name'].value_counts().head(10).to_dict() if not df.empty else {},
        'document_types': df['document_type'].value_counts().to_dict() if not df.empty else {},
        'null_counts': df.isnull().sum().to_dict(),
    }

    return summary


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.load_nyt <path_to_csv>")
        print("Example: python -m src.ingest.load_nyt data/nyt-metadata.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = load_nyt_csv(csv_path)

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)

    summary = get_data_summary(df)
    print(f"\nTotal articles: {summary['row_count']:,}")
    print(f"\nDate range:")
    print(f"  Earliest: {summary['date_range']['min']}")
    print(f"  Latest: {summary['date_range']['max']}")

    print(f"\nTop 10 sections:")
    for section, count in list(summary['top_sections'].items())[:10]:
        print(f"  {section}: {count:,}")

    print(f"\nDocument types:")
    for doc_type, count in list(summary['document_types'].items())[:5]:
        print(f"  {doc_type}: {count:,}")

    print(f"\nFirst 5 rows:")
    print(df.head())
