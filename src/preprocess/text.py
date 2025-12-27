"""
Text Preprocessing Utilities for NYT Articles

This module provides functions for combining and cleaning text from NYT articles,
including handling malformed data and removing stopwords.
"""

import re
import ast
import pandas as pd
import numpy as np
from typing import Union, Optional
import logging

# NLTK imports with lazy loading
try:
    import nltk
    from nltk.corpus import stopwords

    # Try to use stopwords, download if not available
    try:
        STOP_WORDS = set(stopwords.words('english'))
    except LookupError:
        logging.warning("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords', quiet=True)
        STOP_WORDS = set(stopwords.words('english'))
except ImportError:
    logging.warning("NLTK not available. Stopword removal will be disabled.")
    STOP_WORDS = set()

logger = logging.getLogger(__name__)


def safe_extract_from_dict_string(value: Union[str, dict]) -> str:
    """
    Safely extract text from a value that might be a dict or dict-like string.

    Some NYT data has headlines stored as dict strings like:
    "{'main': 'Actual Headline', 'kicker': None, ...}"

    This function extracts the 'main' field if possible, otherwise returns the value as-is.

    Args:
        value: Input value (string, dict, or other type)

    Returns:
        str: Extracted text or original value as string

    Example:
        >>> safe_extract_from_dict_string("{'main': 'Test', 'kicker': None}")
        'Test'
        >>> safe_extract_from_dict_string("Regular headline")
        'Regular headline'
    """
    if pd.isna(value) or value is None:
        return ""

    # If already a dict, extract 'main'
    if isinstance(value, dict):
        return str(value.get('main', ''))

    # Convert to string
    value_str = str(value)

    # Check if it looks like a dict string
    if value_str.strip().startswith('{') and 'main' in value_str:
        try:
            # Use ast.literal_eval for safe evaluation
            parsed = ast.literal_eval(value_str)
            if isinstance(parsed, dict) and 'main' in parsed:
                return str(parsed.get('main', ''))
        except (ValueError, SyntaxError):
            # If parsing fails, return the original string
            pass

    return value_str


def combine_text(df: pd.DataFrame,
                 headline_col: str = 'headline',
                 abstract_col: str = 'abstract',
                 lead_col: str = 'lead_paragraph',
                 output_col: str = 'combined_text') -> pd.DataFrame:
    """
    Combine headline, abstract, and lead_paragraph into a single text column.

    Handles dict-like string artifacts in headlines by extracting the 'main' field.
    Handles NaN values and concatenation artifacts like 'nan nan nan'.

    Args:
        df (pd.DataFrame): Input DataFrame with article columns
        headline_col (str): Name of headline column (default: 'headline')
        abstract_col (str): Name of abstract column (default: 'abstract')
        lead_col (str): Name of lead paragraph column (default: 'lead_paragraph')
        output_col (str): Name of output column (default: 'combined_text')

    Returns:
        pd.DataFrame: DataFrame with new combined_text column added

    Example:
        >>> df = pd.DataFrame({
        ...     'headline': ['Test', "{'main': 'Real Headline'}"],
        ...     'abstract': ['Abstract 1', 'Abstract 2'],
        ...     'lead_paragraph': ['Lead 1', 'Lead 2']
        ... })
        >>> df_combined = combine_text(df)
        >>> df_combined['combined_text'].iloc[1]
        'Real Headline Abstract 2 Lead 2'
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Check if required columns exist
    required_cols = [headline_col, abstract_col, lead_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Extract headline (handle dict strings)
    df['_headline_clean'] = df[headline_col].apply(safe_extract_from_dict_string)

    # Convert abstract and lead to string, handling NaN
    df['_abstract_clean'] = df[abstract_col].fillna('').astype(str)
    df['_lead_clean'] = df[lead_col].fillna('').astype(str)

    # Combine the three columns
    df[output_col] = (
        df['_headline_clean'].astype(str) + ' ' +
        df['_abstract_clean'] + ' ' +
        df['_lead_clean']
    )

    # Remove 'nan' artifacts that result from string conversion
    df[output_col] = df[output_col].str.replace(r'\bnan\b', '', regex=True, flags=re.IGNORECASE)

    # Clean up multiple spaces
    df[output_col] = df[output_col].str.replace(r'\s+', ' ', regex=True).str.strip()

    # Drop temporary columns
    df = df.drop(columns=['_headline_clean', '_abstract_clean', '_lead_clean'])

    logger.info(f"Created '{output_col}' column by combining {headline_col}, {abstract_col}, and {lead_col}")

    return df


def clean_text(text: Union[str, float, None],
               lowercase: bool = True,
               remove_punctuation: bool = True,
               remove_numbers: bool = True,
               remove_stopwords: bool = True,
               min_word_length: int = 1) -> str:
    """
    Clean and normalize text for NLP processing.

    Performs the following operations:
    1. Converts to lowercase (optional)
    2. Removes punctuation (optional)
    3. Removes numbers (optional)
    4. Removes 'nan' artifacts
    5. Removes stopwords (optional)
    6. Removes extra whitespace

    Args:
        text: Input text to clean
        lowercase (bool): Convert to lowercase (default: True)
        remove_punctuation (bool): Remove punctuation (default: True)
        remove_numbers (bool): Remove numbers (default: True)
        remove_stopwords (bool): Remove English stopwords (default: True)
        min_word_length (int): Minimum word length to keep (default: 1)

    Returns:
        str: Cleaned text

    Example:
        >>> clean_text("The Stock Market fell 5% today!")
        'stock market fell today'
        >>> clean_text("Test123 with numbers", remove_numbers=False)
        'test123 numbers'
    """
    # Handle NaN and None
    if pd.isna(text) or text is None:
        return ""

    # Convert to string
    text = str(text)

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove 'nan' artifacts (common from pandas string conversion)
    text = re.sub(r'\bnan\b', '', text, flags=re.IGNORECASE)

    # Remove punctuation (keep letters, numbers, and spaces)
    if remove_punctuation:
        # Keep unicode letters for international text
        text = re.sub(r'[^\w\s]', ' ', text)

    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    if remove_stopwords and STOP_WORDS:
        words = text.split()
        words = [w for w in words if w not in STOP_WORDS and len(w) >= min_word_length]
        text = ' '.join(words)
    elif min_word_length > 1:
        # Apply min_word_length filter even without stopword removal
        words = text.split()
        words = [w for w in words if len(w) >= min_word_length]
        text = ' '.join(words)

    return text


def preprocess_dataframe(df: pd.DataFrame,
                        combine: bool = True,
                        clean: bool = True,
                        text_column: str = 'combined_text',
                        output_column: str = 'cleaned_text') -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to a DataFrame.

    Combines text columns and applies cleaning in one operation.

    Args:
        df (pd.DataFrame): Input DataFrame
        combine (bool): Whether to combine headline/abstract/lead (default: True)
        clean (bool): Whether to clean the text (default: True)
        text_column (str): Column to clean (default: 'combined_text')
        output_column (str): Name for cleaned output (default: 'cleaned_text')

    Returns:
        pd.DataFrame: DataFrame with preprocessed text columns

    Example:
        >>> df = pd.DataFrame({
        ...     'headline': ['Breaking News!'],
        ...     'abstract': ['This is big.'],
        ...     'lead_paragraph': ['Today in Washington...']
        ... })
        >>> df_processed = preprocess_dataframe(df)
        >>> 'combined_text' in df_processed.columns
        True
        >>> 'cleaned_text' in df_processed.columns
        True
    """
    df = df.copy()

    # Step 1: Combine text if requested
    if combine:
        if not all(col in df.columns for col in ['headline', 'abstract', 'lead_paragraph']):
            logger.warning("Missing columns for text combination. Skipping combine step.")
        else:
            df = combine_text(df)
            logger.info(f"Combined text into '{text_column}' column")

    # Step 2: Clean text if requested
    if clean:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Run combine step first or provide valid text_column.")

        logger.info(f"Cleaning text in '{text_column}' column...")
        df[output_column] = df[text_column].apply(clean_text)
        logger.info(f"Created '{output_column}' column with cleaned text")

    return df


def get_preprocessing_stats(df: pd.DataFrame,
                           original_col: str = 'combined_text',
                           cleaned_col: str = 'cleaned_text') -> dict:
    """
    Get statistics about the preprocessing results.

    Args:
        df (pd.DataFrame): Preprocessed DataFrame
        original_col (str): Original text column
        cleaned_col (str): Cleaned text column

    Returns:
        dict: Statistics including word counts, length reduction, etc.
    """
    if original_col not in df.columns or cleaned_col not in df.columns:
        return {}

    # Handle empty DataFrame
    if len(df) == 0:
        return {
            'total_documents': 0,
            'avg_original_words': 0,
            'avg_cleaned_words': 0,
            'avg_word_reduction': 0,
            'avg_reduction_pct': 0,
            'empty_after_cleaning': 0,
        }

    # Calculate statistics
    original_words = df[original_col].str.split().str.len()
    cleaned_words = df[cleaned_col].str.split().str.len()

    stats = {
        'total_documents': len(df),
        'avg_original_words': original_words.mean(),
        'avg_cleaned_words': cleaned_words.mean(),
        'avg_word_reduction': (original_words - cleaned_words).mean(),
        'avg_reduction_pct': ((original_words - cleaned_words) / original_words * 100).mean(),
        'empty_after_cleaning': (cleaned_words == 0).sum(),
    }

    return stats


if __name__ == "__main__":
    # Example usage
    print("Text Preprocessing Module")
    print("=" * 60)

    # Example 1: Clean a single text
    sample_text = "The Stock Market fell 5% today! This is breaking news."
    cleaned = clean_text(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Cleaned:  {cleaned}")

    # Example 2: Process a DataFrame
    sample_df = pd.DataFrame({
        'headline': ['Breaking: Markets Crash!', "{'main': 'Election Results'}"],
        'abstract': ['Markets fell today', 'Final results are in'],
        'lead_paragraph': ['Trading was volatile...', 'After counting votes...']
    })

    print("\n\nDataFrame Processing:")
    print("Original DataFrame:")
    print(sample_df)

    processed_df = preprocess_dataframe(sample_df)
    print("\n\nProcessed DataFrame:")
    print(processed_df[['combined_text', 'cleaned_text']])

    # Stats
    stats = get_preprocessing_stats(processed_df)
    print("\n\nPreprocessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
