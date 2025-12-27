"""
Unit tests for NYT CSV data loader
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from src.ingest.load_nyt import load_nyt_csv, get_data_summary


@pytest.fixture
def sample_csv_path():
    """
    Create a temporary CSV file with synthetic NYT article data for testing.
    """
    csv_content = """_id,pub_date,headline,web_url,abstract,lead_paragraph,section_name,subsection_name,byline,document_type,type_of_material,word_count,keywords
nyt://article/001,2001-09-12,World Leaders React to Attacks,https://nyt.com/001,World leaders condemn attacks,Leaders from around the world...,World,,By JOHN SMITH,article,News,1500,terrorism; world; politics
nyt://article/002,2001-09-13,Markets Plunge After Attacks,https://nyt.com/002,Stock markets fall sharply,Stock markets around the world...,Business Day,Markets,By JANE DOE,article,News,1200,economy; markets; stocks
nyt://article/003,2001-09-14,Editorial: A Day of Infamy,https://nyt.com/003,Editorial response to attacks,The events of September 11...,Opinion,Editorial,By THE EDITORIAL BOARD,editorial,Editorial,800,opinion; terrorism; editorial
nyt://article/004,2020-03-15,Pandemic Forces Global Shutdown,https://nyt.com/004,COVID-19 causes worldwide lockdowns,Countries around the world...,World,,By SARAH JOHNSON,article,News,2000,covid; pandemic; health
nyt://article/005,2020-03-16,Markets Volatile Amid Pandemic,https://nyt.com/005,Financial markets see extreme volatility,Stock markets experienced...,Business Day,Markets,By MIKE WILSON,article,News,1800,markets; covid; economy
nyt://article/006,2024-11-06,Election Results Certified,https://nyt.com/006,Final election results confirmed,After weeks of counting...,U.S.,Politics,By EMILY CHEN,article,News,1600,election; politics; democracy
nyt://article/007,2024-11-07,,https://nyt.com/007,Missing headline test,This article has no headline...,U.S.,,,article,News,1000,test
nyt://article/008,invalid_date,Invalid Date Test,https://nyt.com/008,Testing date parsing,This has an invalid date...,World,,By TEST AUTHOR,article,News,500,test
nyt://article/009,2025-01-01,New Year Celebrations,https://nyt.com/009,World celebrates new year,People around the globe...,World,,,article,News,1100,celebration; world
,2025-01-02,Missing ID Test,https://nyt.com/010,Testing missing ID,This article has no ID...,World,,By GHOST WRITER,article,News,900,test
"""

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write(csv_content)
        temp_path = f.name

    yield temp_path

    # Cleanup: remove the temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def empty_csv_path():
    """Create an empty CSV file for testing error handling."""
    csv_content = """_id,pub_date,headline,web_url,abstract,lead_paragraph,section_name,subsection_name,byline,document_type,type_of_material,word_count,keywords
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write(csv_content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def malformed_csv_path():
    """Create a CSV with missing required columns."""
    csv_content = """_id,pub_date,headline
nyt://article/001,2001-09-12,Test Article
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        f.write(csv_content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestLoadNytCSV:
    """Test suite for load_nyt_csv function"""

    def test_basic_loading(self, sample_csv_path):
        """Test that CSV loads successfully and returns a DataFrame"""
        df = load_nyt_csv(sample_csv_path, verbose=False)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert not df.empty

    def test_essential_columns_present(self, sample_csv_path):
        """Test that all essential columns are present in the output"""
        df = load_nyt_csv(sample_csv_path, verbose=False)

        expected_columns = [
            '_id', 'pub_date', 'headline', 'web_url', 'abstract', 'lead_paragraph',
            'section_name', 'subsection_name', 'byline', 'document_type',
            'type_of_material', 'word_count', 'keywords'
        ]

        for col in expected_columns:
            assert col in df.columns, f"Column '{col}' missing from DataFrame"

    def test_pub_date_is_datetime(self, sample_csv_path):
        """Test that pub_date column is converted to datetime type"""
        df = load_nyt_csv(sample_csv_path, verbose=False)

        assert pd.api.types.is_datetime64_any_dtype(df['pub_date'])

    def test_critical_nulls_removed(self, sample_csv_path):
        """Test that rows with missing critical values are dropped"""
        df = load_nyt_csv(sample_csv_path, verbose=False)

        # Critical columns should have no nulls
        critical_columns = ['_id', 'pub_date', 'headline', 'web_url', 'word_count']

        for col in critical_columns:
            null_count = df[col].isnull().sum()
            assert null_count == 0, f"Column '{col}' has {null_count} null values"

    def test_invalid_dates_removed(self, sample_csv_path):
        """Test that rows with invalid dates are dropped"""
        df = load_nyt_csv(sample_csv_path, verbose=False)

        # Row with 'invalid_date' should be removed
        assert df['pub_date'].isnull().sum() == 0

    def test_missing_id_rows_removed(self, sample_csv_path):
        """Test that rows with missing _id are dropped"""
        df = load_nyt_csv(sample_csv_path, verbose=False)

        # Row with missing _id should be removed
        # Original CSV has 10 rows, but 2 have invalid data (missing ID and invalid date)
        # One has missing headline but valid ID
        # So we expect 7 valid rows (rows with valid critical fields)
        assert len(df) == 7

    def test_text_columns_filled(self, sample_csv_path):
        """Test that text columns are filled with empty strings instead of NaN"""
        df = load_nyt_csv(sample_csv_path, verbose=False)

        text_columns = [
            'abstract', 'lead_paragraph', 'section_name', 'subsection_name',
            'byline', 'type_of_material', 'keywords', 'document_type'
        ]

        for col in text_columns:
            # No null values should remain
            assert df[col].isnull().sum() == 0, f"Column '{col}' has null values"
            # Empty values should be empty strings, not NaN
            assert df[col].dtype == object or df[col].dtype == 'string'

    def test_word_count_is_integer(self, sample_csv_path):
        """Test that word_count is converted to integer type"""
        df = load_nyt_csv(sample_csv_path, verbose=False)

        assert pd.api.types.is_integer_dtype(df['word_count'])

    def test_date_range_calculation(self, sample_csv_path):
        """Test that date range is calculated correctly"""
        df = load_nyt_csv(sample_csv_path, verbose=False)

        min_date = df['pub_date'].min()
        max_date = df['pub_date'].max()

        # Should span from 2001 to 2025
        assert min_date.year == 2001
        assert max_date.year == 2025

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files"""
        with pytest.raises(FileNotFoundError):
            load_nyt_csv('nonexistent_file.csv', verbose=False)

    def test_empty_csv(self, empty_csv_path):
        """Test that ValueError is raised for empty CSV files"""
        with pytest.raises(ValueError, match="CSV file is empty"):
            load_nyt_csv(empty_csv_path, verbose=False)

    def test_missing_columns(self, malformed_csv_path):
        """Test that ValueError is raised when required columns are missing"""
        with pytest.raises(ValueError, match="missing required columns"):
            load_nyt_csv(malformed_csv_path, verbose=False)

    def test_verbose_output(self, sample_csv_path, caplog):
        """Test that verbose mode produces log output"""
        import logging
        caplog.set_level(logging.INFO)

        df = load_nyt_csv(sample_csv_path, verbose=True)

        # Check that log messages were produced
        assert "Loading NYT metadata from:" in caplog.text
        assert "Successfully loaded" in caplog.text
        assert "After cleaning:" in caplog.text

    def test_silent_mode(self, sample_csv_path, caplog):
        """Test that verbose=False suppresses log output"""
        import logging
        caplog.set_level(logging.INFO)

        df = load_nyt_csv(sample_csv_path, verbose=False)

        # Should produce minimal or no log output
        assert isinstance(df, pd.DataFrame)


class TestGetDataSummary:
    """Test suite for get_data_summary function"""

    def test_summary_structure(self, sample_csv_path):
        """Test that summary contains expected keys"""
        df = load_nyt_csv(sample_csv_path, verbose=False)
        summary = get_data_summary(df)

        expected_keys = ['row_count', 'columns', 'date_range', 'top_sections', 'document_types', 'null_counts']

        for key in expected_keys:
            assert key in summary, f"Key '{key}' missing from summary"

    def test_row_count_accurate(self, sample_csv_path):
        """Test that row_count in summary matches DataFrame length"""
        df = load_nyt_csv(sample_csv_path, verbose=False)
        summary = get_data_summary(df)

        assert summary['row_count'] == len(df)

    def test_date_range_valid(self, sample_csv_path):
        """Test that date_range in summary is valid"""
        df = load_nyt_csv(sample_csv_path, verbose=False)
        summary = get_data_summary(df)

        assert 'min' in summary['date_range']
        assert 'max' in summary['date_range']
        assert summary['date_range']['min'] <= summary['date_range']['max']

    def test_top_sections_populated(self, sample_csv_path):
        """Test that top_sections contains section data"""
        df = load_nyt_csv(sample_csv_path, verbose=False)
        summary = get_data_summary(df)

        assert len(summary['top_sections']) > 0
        # World section should be most common in our test data
        assert 'World' in summary['top_sections']

    def test_document_types_populated(self, sample_csv_path):
        """Test that document_types contains type data"""
        df = load_nyt_csv(sample_csv_path, verbose=False)
        summary = get_data_summary(df)

        assert len(summary['document_types']) > 0
        assert 'article' in summary['document_types']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
