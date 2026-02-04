"""
Build embeddings and FAISS IVF index for large-scale datasets (21M corpus).

Reads preprocessed parquet in chunks, generates embeddings, writes to memmap,
then builds a persistent FAISS IVF index with checkpointing support.
"""

import sys
import pandas as pd
import numpy as np
import torch
import faiss
from pathlib import Path
from typing import Optional, Tuple
import logging
import argparse
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embeddings import get_device, extract_embeddings_batch
from transformers import AutoTokenizer, AutoModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingBuilder:
    """Build embeddings and FAISS index with checkpointing."""

    def __init__(
        self,
        input_parquet: str,
        output_dir: str,
        batch_size: int = 128,
        chunk_size: int = 100000,
        nlist: Optional[int] = None,
        use_gpu: bool = True,
        model_name: str = 'vinai/bertweet-base',
        max_length: int = 128,
        verbose: bool = True
    ):
        self.input_parquet = Path(input_parquet)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model_name = model_name
        self.max_length = max_length
        self.verbose = verbose

        # Paths
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memmap_path = self.output_dir / "embeddings_21m.memmap"
        self.checkpoint_path = self.output_dir / ".embed_checkpoint"
        self.mapping_path = self.output_dir / "embeddings_21m_mapping.csv"
        self.index_path = self.output_dir / "faiss_index_21m.bin"

        # Load data info first to determine nlist
        self._load_data_info()
        self.total_rows = self.data_rows
        self.nlist = nlist or max(100, int(np.sqrt(self.total_rows)))

        # Device
        self.device = get_device() if self.use_gpu else torch.device('cpu')

        # Model and tokenizer (loaded on demand)
        self.model = None
        self.tokenizer = None

        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Memmap path: {self.memmap_path}")
        logger.info(f"Index path: {self.index_path}")
        logger.info(f"Total rows: {self.total_rows:,}")
        logger.info(f"FAISS nlist: {self.nlist}")

    def _load_data_info(self):
        """Load parquet metadata to determine total rows."""
        parquet_file = pd.read_parquet(self.input_parquet, columns=['_id'])
        self.data_rows = len(parquet_file)
        logger.info(f"Parquet file: {self.input_parquet} ({self.data_rows:,} rows)")

    def _load_model(self):
        """Load BERTweet model and tokenizer."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")

    def _get_checkpoint(self) -> int:
        """Get last completed chunk index from checkpoint."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
                return data.get('last_chunk', -1)
        return -1

    def _save_checkpoint(self, chunk_idx: int):
        """Save checkpoint after completing a chunk."""
        with open(self.checkpoint_path, 'w') as f:
            json.dump({'last_chunk': chunk_idx}, f)

    def _create_memmap(self):
        """Create and initialize memmap file."""
        if self.memmap_path.exists():
            logger.info(f"Memmap already exists: {self.memmap_path}")
            return np.memmap(
                self.memmap_path,
                dtype=np.float32,
                mode='r+',
                shape=(self.total_rows, 768)
            )
        else:
            logger.info(f"Creating memmap: {self.memmap_path} ({self.total_rows:,} x 768)")
            memmap_array = np.memmap(
                self.memmap_path,
                dtype=np.float32,
                mode='w+',
                shape=(self.total_rows, 768)
            )
            memmap_array.flush()
            return memmap_array

    def build_embeddings(self):
        """Generate embeddings and write to memmap."""
        logger.info("=" * 70)
        logger.info("Phase 1: Generate Embeddings")
        logger.info("=" * 70)

        self._load_model()

        # Create memmap
        memmap_array = self._create_memmap()

        # Get checkpoint
        last_chunk = self._get_checkpoint()
        start_chunk = last_chunk + 1

        # Read parquet and process chunks
        df = pd.read_parquet(self.input_parquet, columns=['_id', 'cleaned_text'])
        chunk_indices = []

        for chunk_idx in range(0, len(df), self.chunk_size):
            if chunk_idx // self.chunk_size < start_chunk:
                logger.info(f"Skipping chunk {chunk_idx // self.chunk_size} (already processed)")
                continue

            chunk_end = min(chunk_idx + self.chunk_size, len(df))
            chunk_df = df.iloc[chunk_idx:chunk_end]

            if self.verbose:
                logger.info(f"Processing chunk {chunk_idx // self.chunk_size}: rows {chunk_idx:,}-{chunk_end:,}")

            # Get texts
            texts = chunk_df['cleaned_text'].fillna('').astype(str).tolist()

            # Process in batches
            embeddings_list = []
            n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            for batch_idx in range(n_batches):
                start_text = batch_idx * self.batch_size
                end_text = min((batch_idx + 1) * self.batch_size, len(texts))
                batch_texts = texts[start_text:end_text]

                try:
                    batch_embeddings = extract_embeddings_batch(
                        batch_texts,
                        self.tokenizer,
                        self.model,
                        self.device,
                        max_length=self.max_length,
                        pooling='cls'
                    )
                    embeddings_list.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    embeddings_list.append(np.zeros((len(batch_texts), 768), dtype=np.float32))

            # Combine batch embeddings
            chunk_embeddings = np.vstack(embeddings_list)

            # Write to memmap
            memmap_array[chunk_idx:chunk_end] = chunk_embeddings
            memmap_array.flush()

            chunk_indices.append((chunk_idx, chunk_end))
            self._save_checkpoint(chunk_idx // self.chunk_size)

            if self.verbose:
                logger.info(f"  ✓ Wrote {len(chunk_embeddings):,} embeddings to memmap")

        # Create mapping
        logger.info("Creating ID mapping...")
        mapping_df = pd.DataFrame({
            '_id': df['_id'].values,
            'index': np.arange(len(df))
        })
        mapping_df.to_csv(self.mapping_path, index=False)
        logger.info(f"✓ Saved mapping to {self.mapping_path}")

        logger.info(f"✓ Embeddings complete: {memmap_array.shape}")
        return memmap_array

    def build_faiss_index(self, embeddings: np.ndarray):
        """Build and save FAISS IVF index."""
        logger.info("=" * 70)
        logger.info("Phase 2: Build FAISS IVF Index")
        logger.info("=" * 70)

        n_total = embeddings.shape[0]
        embedding_dim = embeddings.shape[1]

        logger.info(f"Total embeddings: {n_total:,}")
        logger.info(f"Embedding dimension: {embedding_dim}")
        logger.info(f"nlist (clusters): {self.nlist}")

        # Sample training vectors
        n_train = min(200000, n_total // 10)  # 200K or 10% of total
        logger.info(f"Training on {n_train:,} sampled vectors...")

        sample_indices = np.random.choice(n_total, size=n_train, replace=False)
        train_vectors = embeddings[sample_indices].astype(np.float32)

        # Create index
        logger.info("Creating IVFFlat index...")
        quantizer = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, self.nlist)
        index.nprobe = 32  # Probing parameter

        # Train
        logger.info("Training index...")
        index.train(train_vectors)

        # Add vectors in chunks
        logger.info("Adding vectors to index...")
        add_chunk_size = 100000
        for start_idx in tqdm(range(0, n_total, add_chunk_size), desc="Adding vectors"):
            end_idx = min(start_idx + add_chunk_size, n_total)
            vectors = embeddings[start_idx:end_idx].astype(np.float32)
            index.add(vectors)

        # Save index
        logger.info(f"Saving index to {self.index_path}...")
        faiss.write_index(index, str(self.index_path))

        logger.info(f"✓ Index complete: {self.index_path}")
        logger.info(f"  Index file size: {self.index_path.stat().st_size / (1024**3):.2f} GB")

        # Test search
        logger.info("Testing index with sample query...")
        test_query = embeddings[0:1].astype(np.float32)
        distances, indices = index.search(test_query, k=5)
        logger.info(f"  Top-5 indices: {indices[0]}")
        logger.info(f"  Top-5 distances: {distances[0]}")

    def run(self):
        """Run full pipeline: embeddings + FAISS index."""
        try:
            embeddings = self.build_embeddings()
            self.build_faiss_index(embeddings)

            logger.info("=" * 70)
            logger.info("✓ All complete!")
            logger.info(f"  Embeddings: {self.memmap_path}")
            logger.info(f"  Mapping: {self.mapping_path}")
            logger.info(f"  Index: {self.index_path}")
            logger.info("=" * 70)

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Build embeddings and FAISS IVF index for large NYT datasets"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input preprocessed parquet file'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for embeddings and index'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for embedding generation (default: 128)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100000,
        help='Chunk size for parquet reading (default: 100000)'
    )
    parser.add_argument(
        '--nlist',
        type=int,
        default=None,
        help='FAISS nlist parameter (default: sqrt(n_total))'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Use GPU for embeddings'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU (default: auto-detect)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress logging'
    )

    args = parser.parse_args()
    use_gpu = args.gpu and not args.cpu

    builder = EmbeddingBuilder(
        input_parquet=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        nlist=args.nlist,
        use_gpu=use_gpu,
        verbose=not args.quiet
    )
    builder.run()


if __name__ == '__main__':
    main()
