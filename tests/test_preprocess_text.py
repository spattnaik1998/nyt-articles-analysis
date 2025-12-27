"""
Unit tests for text preprocessing functions
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocess.text import (
    safe_extract_from_dict_string,
    combine_text,
    clean_text,
    preprocess_dataframe,
    get_preprocessing_stats
)


class TestSafeExtractFromDictString:
    """Test suite for safe_extract_from_dict_string function"""

    def test_extract_from_dict_string(self):
        """Test extraction from dict-like string"""
        input_str = "{'main': 'Test Headline', 'kicker': None}"
        result = safe_extract_from_dict_string(input_str)
        assert result == 'Test Headline'

    def test_extract_from_actual_dict(self):
        """Test extraction from actual dict object"""
        input_dict = {'main': 'Real Dict Headline', 'kicker': 'Breaking'}
        result = safe_extract_from_dict_string(input_dict)
        assert result == 'Real Dict Headline'

    def test_regular_string_passthrough(self):
        """Test that regular strings pass through unchanged"""
        input_str = "Regular headline text"
        result = safe_extract_from_dict_string(input_str)
        assert result == input_str

    def test_handle_none(self):
        """Test handling of None value"""
        result = safe_extract_from_dict_string(None)
        assert result == ""

    def test_handle_nan(self):
        """Test handling of NaN value"""
        result = safe_extract_from_dict_string(np.nan)
        assert result == ""

    def test_malformed_dict_string(self):
        """Test handling of malformed dict string"""
        input_str = "{'main': 'Test', 'broken"
        result = safe_extract_from_dict_string(input_str)
        # Should return original string if parsing fails
        assert result == input_str

    def test_dict_without_main_key(self):
        """Test dict without 'main' key"""
        input_dict = {'title': 'Test', 'subtitle': 'Sub'}
        result = safe_extract_from_dict_string(input_dict)
        assert result == ""

    def test_numeric_value(self):
        """Test numeric value conversion"""
        result = safe_extract_from_dict_string(12345)
        assert result == "12345"


class TestCombineText:
    """Test suite for combine_text function"""

    def test_basic_combination(self):
        """Test basic text combination"""
        df = pd.DataFrame({
            'headline': ['Test Headline'],
            'abstract': ['Test abstract'],
            'lead_paragraph': ['Test lead']
        })
        result = combine_text(df)
        assert 'combined_text' in result.columns
        assert result['combined_text'].iloc[0] == 'Test Headline Test abstract Test lead'

    def test_handle_nan_values(self):
        """Test handling of NaN values"""
        df = pd.DataFrame({
            'headline': ['Test'],
            'abstract': [np.nan],
            'lead_paragraph': ['Lead']
        })
        result = combine_text(df)
        combined = result['combined_text'].iloc[0]
        assert 'Test' in combined
        assert 'Lead' in combined
        # Should not contain 'nan' string
        assert 'nan' not in combined.lower()

    def test_handle_dict_string_headline(self):
        """Test handling of dict-like headline"""
        df = pd.DataFrame({
            'headline': ["{'main': 'Real Headline', 'kicker': None}"],
            'abstract': ['Abstract text'],
            'lead_paragraph': ['Lead text']
        })
        result = combine_text(df)
        assert 'Real Headline' in result['combined_text'].iloc[0]

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError"""
        df = pd.DataFrame({
            'headline': ['Test'],
            'abstract': ['Test']
            # Missing lead_paragraph
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            combine_text(df)

    def test_custom_column_names(self):
        """Test using custom column names"""
        df = pd.DataFrame({
            'title': ['Test Title'],
            'summary': ['Test summary'],
            'intro': ['Test intro']
        })
        result = combine_text(
            df,
            headline_col='title',
            abstract_col='summary',
            lead_col='intro',
            output_col='full_text'
        )
        assert 'full_text' in result.columns
        assert 'Test Title Test summary Test intro' == result['full_text'].iloc[0]

    def test_remove_multiple_spaces(self):
        """Test that multiple spaces are cleaned up"""
        df = pd.DataFrame({
            'headline': ['Test'],
            'abstract': ['  Multiple   spaces  '],
            'lead_paragraph': ['Lead']
        })
        result = combine_text(df)
        # Should not have multiple consecutive spaces
        assert '  ' not in result['combined_text'].iloc[0]

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame({
            'headline': [],
            'abstract': [],
            'lead_paragraph': []
        })
        result = combine_text(df)
        assert len(result) == 0
        assert 'combined_text' in result.columns


class TestCleanText:
    """Test suite for clean_text function"""

    def test_lowercase_conversion(self):
        """Test lowercase conversion"""
        result = clean_text("THIS IS UPPERCASE")
        assert result == result.lower()
        assert 'uppercase' in result

    def test_remove_punctuation(self):
        """Test punctuation removal"""
        result = clean_text("Hello, world! How are you?")
        assert ',' not in result
        assert '!' not in result
        assert '?' not in result

    def test_remove_numbers(self):
        """Test number removal"""
        result = clean_text("There are 123 items and 456 more")
        assert '123' not in result
        assert '456' not in result

    def test_remove_stopwords(self):
        """Test stopword removal"""
        result = clean_text("The quick brown fox jumps over the lazy dog")
        # Common stopwords should be removed
        assert 'the' not in result
        assert 'over' not in result
        # Content words should remain
        assert 'quick' in result
        assert 'brown' in result
        assert 'fox' in result

    def test_handle_none(self):
        """Test handling of None"""
        result = clean_text(None)
        assert result == ""

    def test_handle_nan(self):
        """Test handling of NaN"""
        result = clean_text(np.nan)
        assert result == ""

    def test_remove_nan_artifacts(self):
        """Test removal of 'nan' string artifacts"""
        result = clean_text("This has nan in the text")
        assert 'nan' not in result

    def test_no_lowercase(self):
        """Test with lowercase disabled"""
        result = clean_text("UPPER Text", lowercase=False)
        assert 'UPPER' in result

    def test_keep_punctuation(self):
        """Test with punctuation removal disabled"""
        result = clean_text("Hello, world!", remove_punctuation=False, remove_stopwords=False)
        # Punctuation might still be modified, but should preserve more
        assert 'hello' in result.lower()

    def test_keep_numbers(self):
        """Test with number removal disabled"""
        result = clean_text("Test 123", remove_numbers=False)
        assert '123' in result

    def test_no_stopword_removal(self):
        """Test with stopword removal disabled"""
        result = clean_text("the quick fox", remove_stopwords=False)
        assert 'the' in result

    def test_min_word_length(self):
        """Test minimum word length filter"""
        result = clean_text("a big cat is here", min_word_length=3, remove_stopwords=False)
        words = result.split()
        # Words shorter than 3 chars should be removed
        assert 'a' not in words
        assert 'is' not in words
        # Longer words should remain
        assert 'big' in words
        assert 'cat' in words

    def test_multiple_spaces_removed(self):
        """Test that multiple spaces are normalized"""
        result = clean_text("Too    many     spaces")
        assert '  ' not in result

    def test_empty_string(self):
        """Test empty string input"""
        result = clean_text("")
        assert result == ""

    def test_only_punctuation(self):
        """Test input with only punctuation"""
        result = clean_text("!@#$%^&*()")
        assert result == ""


class TestPreprocessDataframe:
    """Test suite for preprocess_dataframe function"""

    def test_full_pipeline(self):
        """Test complete preprocessing pipeline"""
        df = pd.DataFrame({
            'headline': ['Breaking News!'],
            'abstract': ['This is important.'],
            'lead_paragraph': ['In today\'s news...']
        })
        result = preprocess_dataframe(df)

        assert 'combined_text' in result.columns
        assert 'cleaned_text' in result.columns
        assert len(result) == 1

    def test_combine_only(self):
        """Test with only combine step"""
        df = pd.DataFrame({
            'headline': ['Test'],
            'abstract': ['Abstract'],
            'lead_paragraph': ['Lead']
        })
        result = preprocess_dataframe(df, combine=True, clean=False)

        assert 'combined_text' in result.columns
        assert 'cleaned_text' not in result.columns

    def test_clean_only(self):
        """Test with only clean step (requires existing combined_text)"""
        df = pd.DataFrame({
            'combined_text': ['Test Text With Punctuation!']
        })
        result = preprocess_dataframe(df, combine=False, clean=True, text_column='combined_text')

        assert 'cleaned_text' in result.columns
        assert '!' not in result['cleaned_text'].iloc[0]

    def test_missing_text_column_raises_error(self):
        """Test that missing text column raises error"""
        df = pd.DataFrame({'other_column': ['test']})

        with pytest.raises(ValueError, match="not found"):
            preprocess_dataframe(df, combine=False, clean=True)

    def test_custom_column_names(self):
        """Test with custom output column name"""
        df = pd.DataFrame({
            'headline': ['Test'],
            'abstract': ['Abstract'],
            'lead_paragraph': ['Lead']
        })
        result = preprocess_dataframe(df, output_column='processed')

        assert 'processed' in result.columns


class TestGetPreprocessingStats:
    """Test suite for get_preprocessing_stats function"""

    def test_basic_stats(self):
        """Test basic statistics generation"""
        df = pd.DataFrame({
            'combined_text': ['The quick brown fox jumps', 'Short text'],
            'cleaned_text': ['quick brown fox jumps', 'short text']
        })
        stats = get_preprocessing_stats(df)

        assert 'total_documents' in stats
        assert 'avg_original_words' in stats
        assert 'avg_cleaned_words' in stats
        assert 'avg_word_reduction' in stats
        assert stats['total_documents'] == 2

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame({
            'combined_text': [],
            'cleaned_text': []
        })
        stats = get_preprocessing_stats(df)

        assert stats['total_documents'] == 0

    def test_missing_columns(self):
        """Test with missing required columns"""
        df = pd.DataFrame({'other': ['test']})
        stats = get_preprocessing_stats(df)

        # Should return empty dict
        assert stats == {}

    def test_word_reduction_calculation(self):
        """Test word reduction calculation"""
        df = pd.DataFrame({
            'combined_text': ['The quick brown fox'],  # 4 words
            'cleaned_text': ['quick brown fox']  # 3 words (removed 'the')
        })
        stats = get_preprocessing_stats(df)

        assert stats['avg_original_words'] == 4
        assert stats['avg_cleaned_words'] == 3
        assert stats['avg_word_reduction'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
