"""
Tests for Book Metadata Extraction Module

Tests cover:
- Pydantic models
- Regex extraction
- LLM extraction (if available)
- Batch extraction
- Edge cases
"""

import pytest
import pandas as pd
import os
from pathlib import Path
import tempfile
import shutil

from src.models.extraction import (
    BookMeta,
    ExtractionResult,
    extract_with_regex,
    extract_book_meta,
    batch_extract,
    get_extraction_stats,
    filter_successful_extractions,
    INSTRUCTOR_AVAILABLE
)


# Sample texts for testing
SAMPLE_TEXTS = {
    'pattern_1': '"The Great Gatsby" by F. Scott Fitzgerald',
    'pattern_2': 'To Kill a Mockingbird by Harper Lee',
    'pattern_3': 'George Orwell\'s "1984"',
    'pattern_4': '"Pride and Prejudice," by Jane Austen',
    'pattern_6': 'Review of "Animal Farm" by George Orwell',
    'pattern_7': 'Ernest Hemingway, "The Old Man and the Sea"',
}


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataframe():
    """Provide sample DataFrame for testing."""
    return pd.DataFrame({
        'text': list(SAMPLE_TEXTS.values()),
        'pub_date': pd.to_datetime(['2020-01-01', '2020-06-01', '2021-01-01',
                                    '2021-06-01', '2022-01-01', '2022-06-01'])
    })


# ============================================================================
# Pydantic Model Tests
# ============================================================================

def test_book_meta_valid():
    """Test creating valid BookMeta."""
    book = BookMeta(
        book_title="1984",
        author_name="George Orwell"
    )

    assert book.book_title == "1984"
    assert book.author_name == "George Orwell"


def test_book_meta_strips_whitespace():
    """Test that BookMeta strips whitespace."""
    book = BookMeta(
        book_title="  1984  ",
        author_name="  George Orwell  "
    )

    assert book.book_title == "1984"
    assert book.author_name == "George Orwell"


def test_book_meta_cleans_title_quotes():
    """Test that BookMeta removes quotes from title."""
    book = BookMeta(
        book_title='"The Great Gatsby"',
        author_name="F. Scott Fitzgerald"
    )

    assert book.book_title == "The Great Gatsby"


def test_book_meta_cleans_author_by_prefix():
    """Test that BookMeta removes 'by ' prefix from author."""
    book = BookMeta(
        book_title="1984",
        author_name="by George Orwell"
    )

    assert book.author_name == "George Orwell"


def test_book_meta_empty_title_raises_error():
    """Test that empty title raises validation error."""
    with pytest.raises(ValueError):
        BookMeta(book_title="", author_name="Author")


def test_book_meta_empty_author_raises_error():
    """Test that empty author raises validation error."""
    with pytest.raises(ValueError):
        BookMeta(book_title="Title", author_name="")


# ============================================================================
# Regex Extraction Tests
# ============================================================================

def test_extract_with_regex_pattern_1():
    """Test regex extraction with pattern 1: "Title" by Author."""
    text = '"The Great Gatsby" by F. Scott Fitzgerald'
    title, author, pattern = extract_with_regex(text)

    assert title == "The Great Gatsby"
    assert author == "F. Scott Fitzgerald"
    assert pattern == "regex_pattern_1"


def test_extract_with_regex_pattern_2():
    """Test regex extraction with pattern 2: Title by Author."""
    text = 'To Kill a Mockingbird by Harper Lee'
    title, author, pattern = extract_with_regex(text)

    assert title == "To Kill a Mockingbird"
    assert author == "Harper Lee"
    assert pattern is not None


def test_extract_with_regex_pattern_3():
    """Test regex extraction with pattern 3: Author's "Title"."""
    text = 'George Orwell\'s "1984"'
    title, author, pattern = extract_with_regex(text)

    assert title == "1984"
    assert author == "George Orwell"
    assert pattern == "regex_pattern_3"


def test_extract_with_regex_pattern_6():
    """Test regex extraction with pattern 6: Review of "Title" by Author."""
    text = 'Review of "Animal Farm" by George Orwell'
    title, author, pattern = extract_with_regex(text)

    assert title == "Animal Farm"
    assert author == "George Orwell"
    assert pattern == "regex_pattern_6"


def test_extract_with_regex_no_match():
    """Test regex extraction with no matching pattern."""
    text = 'This is just random text with no book information'
    title, author, pattern = extract_with_regex(text)

    assert title is None
    assert author is None
    assert pattern is None


def test_extract_with_regex_multiple_patterns():
    """Test that regex tries multiple patterns."""
    # Should match even with extra text
    text = 'In this review, we discuss "Brave New World" by Aldous Huxley and its themes'
    title, author, pattern = extract_with_regex(text)

    assert title is not None
    assert author is not None
    assert "Brave" in title or "New World" in title


def test_extract_with_regex_case_insensitive():
    """Test that regex is case insensitive for 'by'."""
    text = '"The Catcher in the Rye" BY J.D. Salinger'
    title, author, pattern = extract_with_regex(text)

    assert title is not None
    assert author is not None


# ============================================================================
# Extract Book Meta Tests
# ============================================================================

def test_extract_book_meta_with_regex():
    """Test extract_book_meta using regex (no LLM)."""
    text = '"1984" by George Orwell'

    result = extract_book_meta(text, use_llm=False)

    assert result.success is True
    assert result.book_meta is not None
    assert result.book_meta.book_title == "1984"
    assert result.book_meta.author_name == "George Orwell"
    assert result.method.startswith('regex_pattern_')


def test_extract_book_meta_empty_text():
    """Test extract_book_meta with empty text."""
    result = extract_book_meta("", use_llm=False)

    assert result.success is False
    assert result.method == 'failed'
    assert result.error == 'Empty text'


def test_extract_book_meta_no_match():
    """Test extract_book_meta with text that doesn't match any pattern."""
    text = 'This is just random text'

    result = extract_book_meta(text, use_llm=False)

    assert result.success is False
    assert result.method == 'failed'


def test_extract_book_meta_all_sample_texts():
    """Test extract_book_meta on all sample texts."""
    for name, text in SAMPLE_TEXTS.items():
        result = extract_book_meta(text, use_llm=False)

        assert result.success is True, f"Failed on {name}: {text}"
        assert result.book_meta is not None
        assert len(result.book_meta.book_title) > 0
        assert len(result.book_meta.author_name) > 0


# ============================================================================
# Batch Extraction Tests
# ============================================================================

def test_batch_extract_basic(sample_dataframe, temp_output_dir):
    """Test basic batch extraction."""
    result_df = batch_extract(
        sample_dataframe,
        text_col='text',
        use_llm=False,
        output_dir=None,  # Don't save
        verbose=False
    )

    # Check columns added
    assert 'book_title' in result_df.columns
    assert 'author_name' in result_df.columns
    assert 'extraction_method' in result_df.columns
    assert 'extraction_success' in result_df.columns

    # Check values
    assert len(result_df) == len(sample_dataframe)

    # At least some should succeed
    assert result_df['extraction_success'].sum() > 0


def test_batch_extract_all_succeed(sample_dataframe):
    """Test that batch extraction succeeds on all sample texts."""
    result_df = batch_extract(
        sample_dataframe,
        text_col='text',
        use_llm=False,
        output_dir=None,
        verbose=False
    )

    # All should succeed (our sample texts are well-formatted)
    assert result_df['extraction_success'].all()


def test_batch_extract_saves_output(sample_dataframe, temp_output_dir):
    """Test that batch extraction saves output files."""
    result_df = batch_extract(
        sample_dataframe,
        text_col='text',
        use_llm=False,
        output_dir=temp_output_dir,
        save_by_year=False,
        verbose=False
    )

    # Check file created
    output_file = Path(temp_output_dir) / 'books_all.parquet'
    assert output_file.exists()

    # Verify content
    loaded_df = pd.read_parquet(output_file)
    assert len(loaded_df) == len(result_df)


def test_batch_extract_saves_by_year(sample_dataframe, temp_output_dir):
    """Test that batch extraction saves separate files per year."""
    result_df = batch_extract(
        sample_dataframe,
        text_col='text',
        use_llm=False,
        output_dir=temp_output_dir,
        save_by_year=True,
        verbose=False
    )

    # Check that year files created
    output_path = Path(temp_output_dir)
    year_files = list(output_path.glob('books_*.parquet'))

    assert len(year_files) > 0

    # Should have files for 2020, 2021, 2022
    years_found = set()
    for f in year_files:
        year = int(f.stem.split('_')[1])
        years_found.add(year)

    assert 2020 in years_found
    assert 2021 in years_found
    assert 2022 in years_found


def test_batch_extract_invalid_text_column(sample_dataframe):
    """Test batch extraction with invalid text column raises error."""
    with pytest.raises(ValueError, match="Column .* not found"):
        batch_extract(
            sample_dataframe,
            text_col='nonexistent_column',
            use_llm=False,
            output_dir=None,
            verbose=False
        )


def test_batch_extract_preserves_original_columns(sample_dataframe):
    """Test that batch extraction preserves original DataFrame columns."""
    original_cols = set(sample_dataframe.columns)

    result_df = batch_extract(
        sample_dataframe,
        text_col='text',
        use_llm=False,
        output_dir=None,
        verbose=False
    )

    # Original columns should still be present
    for col in original_cols:
        assert col in result_df.columns


def test_batch_extract_with_nan_values():
    """Test batch extraction with NaN values."""
    df = pd.DataFrame({
        'text': ['"1984" by George Orwell', None, 'Harper Lee\'s "To Kill a Mockingbird"', ''],
        'pub_date': pd.to_datetime(['2020-01-01', '2020-06-01', '2021-01-01', '2021-06-01'])
    })

    result_df = batch_extract(
        df,
        text_col='text',
        use_llm=False,
        output_dir=None,
        verbose=False
    )

    # Should handle NaN/empty gracefully
    assert len(result_df) == len(df)

    # First should succeed, second and fourth should fail
    assert result_df.iloc[0]['extraction_success'] is True
    assert result_df.iloc[1]['extraction_success'] is False


def test_batch_extract_parallel_processing(sample_dataframe):
    """Test that batch extraction uses parallel processing."""
    # Should not raise any errors with multiple workers
    result_df = batch_extract(
        sample_dataframe,
        text_col='text',
        use_llm=False,
        max_workers=4,
        output_dir=None,
        verbose=False
    )

    assert len(result_df) == len(sample_dataframe)


# ============================================================================
# Utility Function Tests
# ============================================================================

def test_get_extraction_stats():
    """Test get_extraction_stats function."""
    df = pd.DataFrame({
        'extraction_success': [True, True, False, True],
        'extraction_method': ['llm', 'regex_pattern_1', 'failed', 'regex_pattern_2']
    })

    stats = get_extraction_stats(df)

    assert stats['total'] == 4
    assert stats['successful'] == 3
    assert stats['failed'] == 1
    assert stats['success_rate'] == 0.75
    assert 'method_breakdown' in stats


def test_get_extraction_stats_no_results_column():
    """Test get_extraction_stats with missing column raises error."""
    df = pd.DataFrame({'some_column': [1, 2, 3]})

    with pytest.raises(ValueError, match="does not contain extraction results"):
        get_extraction_stats(df)


def test_filter_successful_extractions():
    """Test filter_successful_extractions function."""
    df = pd.DataFrame({
        'text': ['A', 'B', 'C', 'D'],
        'extraction_success': [True, False, True, False]
    })

    filtered = filter_successful_extractions(df)

    assert len(filtered) == 2
    assert filtered['extraction_success'].all()


def test_filter_successful_extractions_no_column():
    """Test filter_successful_extractions with missing column raises error."""
    df = pd.DataFrame({'some_column': [1, 2, 3]})

    with pytest.raises(ValueError, match="does not contain extraction results"):
        filter_successful_extractions(df)


# ============================================================================
# Edge Cases
# ============================================================================

def test_extract_book_meta_special_characters():
    """Test extraction with special characters in title."""
    text = '"To Kill a Mockingbird: A Novel" by Harper Lee'

    result = extract_book_meta(text, use_llm=False)

    assert result.success is True
    assert "Mockingbird" in result.book_meta.book_title


def test_extract_book_meta_multiple_authors():
    """Test extraction with multiple authors."""
    text = '"Book Title" by Author One and Author Two'

    result = extract_book_meta(text, use_llm=False)

    # Should still extract something
    if result.success:
        assert result.book_meta.book_title is not None


def test_batch_extract_single_row():
    """Test batch extraction with single row."""
    df = pd.DataFrame({
        'text': ['"1984" by George Orwell'],
        'pub_date': pd.to_datetime(['2020-01-01'])
    })

    result_df = batch_extract(
        df,
        text_col='text',
        use_llm=False,
        output_dir=None,
        verbose=False
    )

    assert len(result_df) == 1
    assert result_df.iloc[0]['extraction_success'] is True


def test_batch_extract_large_text():
    """Test extraction from very long text."""
    long_text = 'This is a very long review. ' * 100 + '"1984" by George Orwell'

    df = pd.DataFrame({
        'text': [long_text],
        'pub_date': pd.to_datetime(['2020-01-01'])
    })

    result_df = batch_extract(
        df,
        text_col='text',
        use_llm=False,
        output_dir=None,
        verbose=False
    )

    # Should still find the book info
    assert result_df.iloc[0]['extraction_success'] is True


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow(sample_dataframe, temp_output_dir):
    """Test complete extraction workflow."""
    # Step 1: Batch extract
    result_df = batch_extract(
        sample_dataframe,
        text_col='text',
        use_llm=False,
        output_dir=temp_output_dir,
        save_by_year=True,
        verbose=False
    )

    # Step 2: Get statistics
    stats = get_extraction_stats(result_df)

    assert stats['total'] == len(sample_dataframe)
    assert stats['success_rate'] > 0.9  # Should be > 90%

    # Step 3: Filter successful
    successful = filter_successful_extractions(result_df)

    assert len(successful) == stats['successful']

    # Step 4: Verify saved files
    year_files = list(Path(temp_output_dir).glob('books_*.parquet'))
    assert len(year_files) > 0

    # Step 5: Load and verify
    for year_file in year_files:
        loaded_df = pd.read_parquet(year_file)
        assert 'book_title' in loaded_df.columns
        assert 'author_name' in loaded_df.columns


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
