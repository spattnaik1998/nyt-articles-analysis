"""
Unit tests for embeddings module
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil

from src.models.embeddings import (
    get_device,
    extract_embeddings_batch,
    build_bertweet_embeddings,
    load_embeddings,
    get_embedding_by_id,
    TRANSFORMERS_AVAILABLE
)


# Skip all tests if transformers not available
pytestmark = pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE,
    reason="transformers library not installed"
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        '_id': ['nyt://article/001', 'nyt://article/002', 'nyt://article/003'],
        'combined_text': [
            'This is the first test article about politics',
            'Second article discusses economic trends',
            'Third article covers sports and entertainment'
        ],
        'cleaned_text': [
            'first test article politics',
            'second article discusses economic trends',
            'third article covers sports entertainment'
        ]
    })


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestGetDevice:
    """Test suite for device detection"""

    def test_get_device_returns_valid_device(self):
        """Test that get_device returns a torch device"""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']

    def test_device_is_accessible(self):
        """Test that the returned device is actually usable"""
        device = get_device()
        # Create a simple tensor on the device
        tensor = torch.zeros(1).to(device)
        assert tensor.device.type == device.type


@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestExtractEmbeddingsBatch:
    """Test suite for batch embedding extraction"""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Load a small test model"""
        from transformers import AutoTokenizer, AutoModel

        # Use a small model for faster testing
        model_name = 'prajjwal1/bert-tiny'  # Very small BERT for testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device('cpu')  # Always use CPU for tests
        model.to(device)
        model.eval()

        return tokenizer, model, device

    def test_extract_batch_cls_pooling(self, model_and_tokenizer):
        """Test CLS token pooling"""
        tokenizer, model, device = model_and_tokenizer
        texts = ['Test sentence one', 'Test sentence two']

        embeddings = extract_embeddings_batch(
            texts, tokenizer, model, device, pooling='cls'
        )

        assert embeddings.shape[0] == 2  # 2 texts
        assert embeddings.shape[1] > 0  # Has embedding dimension
        assert isinstance(embeddings, np.ndarray)

    def test_extract_batch_mean_pooling(self, model_and_tokenizer):
        """Test mean pooling"""
        tokenizer, model, device = model_and_tokenizer
        texts = ['Test sentence']

        embeddings = extract_embeddings_batch(
            texts, tokenizer, model, device, pooling='mean'
        )

        assert embeddings.shape[0] == 1
        assert isinstance(embeddings, np.ndarray)

    def test_invalid_pooling_raises_error(self, model_and_tokenizer):
        """Test that invalid pooling strategy raises error"""
        tokenizer, model, device = model_and_tokenizer

        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            extract_embeddings_batch(
                ['test'], tokenizer, model, device, pooling='invalid'
            )

    def test_empty_text_batch(self, model_and_tokenizer):
        """Test handling of empty text"""
        tokenizer, model, device = model_and_tokenizer
        texts = ['', 'valid text', '']

        embeddings = extract_embeddings_batch(
            texts, tokenizer, model, device
        )

        assert embeddings.shape[0] == 3
        # All embeddings should be valid (not NaN)
        assert not np.isnan(embeddings).any()


@pytest.mark.slow  # Mark as slow test
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="transformers not installed")
class TestBuildBertweetEmbeddings:
    """Test suite for build_bertweet_embeddings function"""

    def test_basic_embedding_generation(self, sample_df, temp_output_dir):
        """Test basic embedding generation"""
        embeddings, mapping = build_bertweet_embeddings(
            sample_df,
            text_col='cleaned_text',
            model_name='prajjwal1/bert-tiny',  # Small model for testing
            batch_size=2,
            output_dir=temp_output_dir,
            save_embeddings=False,
            verbose=False
        )

        assert embeddings.shape[0] == 3  # 3 articles
        assert embeddings.shape[1] > 0  # Has embedding dimension
        assert len(mapping) == 3
        assert list(mapping.columns) == ['_id', 'index']

    def test_sample_limit(self, sample_df, temp_output_dir):
        """Test sample limit parameter"""
        embeddings, mapping = build_bertweet_embeddings(
            sample_df,
            text_col='cleaned_text',
            model_name='prajjwal1/bert-tiny',
            sample_limit=2,
            output_dir=temp_output_dir,
            save_embeddings=False,
            verbose=False
        )

        assert embeddings.shape[0] == 2
        assert len(mapping) == 2

    def test_save_to_disk(self, sample_df, temp_output_dir):
        """Test saving embeddings and mapping to disk"""
        embeddings, mapping = build_bertweet_embeddings(
            sample_df,
            text_col='cleaned_text',
            model_name='prajjwal1/bert-tiny',
            output_dir=temp_output_dir,
            save_embeddings=True,
            verbose=False
        )

        # Check files were created
        embeddings_file = Path(temp_output_dir) / 'embeddings.npy'
        mapping_file = Path(temp_output_dir) / 'embeddings_mapping.csv'

        assert embeddings_file.exists()
        assert mapping_file.exists()

        # Verify content
        loaded_embeddings = np.load(embeddings_file)
        assert np.allclose(loaded_embeddings, embeddings)

        loaded_mapping = pd.read_csv(mapping_file)
        assert len(loaded_mapping) == len(mapping)

    def test_missing_text_column_raises_error(self, sample_df, temp_output_dir):
        """Test that missing text column raises ValueError"""
        with pytest.raises(ValueError, match="not found in DataFrame"):
            build_bertweet_embeddings(
                sample_df,
                text_col='nonexistent_column',
                model_name='prajjwal1/bert-tiny',
                output_dir=temp_output_dir,
                save_embeddings=False,
                verbose=False
            )

    def test_different_pooling_strategies(self, sample_df, temp_output_dir):
        """Test both CLS and mean pooling"""
        # CLS pooling
        emb_cls, _ = build_bertweet_embeddings(
            sample_df,
            model_name='prajjwal1/bert-tiny',
            pooling='cls',
            output_dir=temp_output_dir,
            save_embeddings=False,
            verbose=False
        )

        # Mean pooling
        emb_mean, _ = build_bertweet_embeddings(
            sample_df,
            model_name='prajjwal1/bert-tiny',
            pooling='mean',
            output_dir=temp_output_dir,
            save_embeddings=False,
            verbose=False
        )

        # Both should have same shape
        assert emb_cls.shape == emb_mean.shape
        # But different values (pooling strategies differ)
        assert not np.allclose(emb_cls, emb_mean)


class TestLoadEmbeddings:
    """Test suite for loading saved embeddings"""

    def test_load_embeddings(self, sample_df, temp_output_dir):
        """Test loading previously saved embeddings"""
        # First create and save embeddings
        original_emb, original_map = build_bertweet_embeddings(
            sample_df,
            model_name='prajjwal1/bert-tiny',
            output_dir=temp_output_dir,
            save_embeddings=True,
            verbose=False
        )

        # Load them back
        loaded_emb, loaded_map = load_embeddings(
            embeddings_path=str(Path(temp_output_dir) / 'embeddings.npy'),
            mapping_path=str(Path(temp_output_dir) / 'embeddings_mapping.csv')
        )

        # Verify they match
        assert np.allclose(original_emb, loaded_emb)
        assert len(original_map) == len(loaded_map)


class TestGetEmbeddingById:
    """Test suite for retrieving embeddings by article ID"""

    @pytest.fixture
    def embeddings_and_mapping(self, sample_df, temp_output_dir):
        """Create embeddings for testing"""
        embeddings, mapping = build_bertweet_embeddings(
            sample_df,
            model_name='prajjwal1/bert-tiny',
            output_dir=temp_output_dir,
            save_embeddings=False,
            verbose=False
        )
        return embeddings, mapping

    def test_get_existing_embedding(self, embeddings_and_mapping):
        """Test retrieving an existing embedding"""
        embeddings, mapping = embeddings_and_mapping

        emb = get_embedding_by_id('nyt://article/001', embeddings, mapping)

        assert emb is not None
        assert emb.shape == (embeddings.shape[1],)
        # Should match first row
        assert np.allclose(emb, embeddings[0])

    def test_get_nonexistent_embedding(self, embeddings_and_mapping):
        """Test retrieving a non-existent embedding returns None"""
        embeddings, mapping = embeddings_and_mapping

        emb = get_embedding_by_id('nyt://article/999', embeddings, mapping)

        assert emb is None

    def test_get_all_embeddings_by_id(self, embeddings_and_mapping):
        """Test retrieving all embeddings by their IDs"""
        embeddings, mapping = embeddings_and_mapping

        for idx, article_id in enumerate(mapping['_id']):
            emb = get_embedding_by_id(article_id, embeddings, mapping)
            assert emb is not None
            assert np.allclose(emb, embeddings[idx])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
