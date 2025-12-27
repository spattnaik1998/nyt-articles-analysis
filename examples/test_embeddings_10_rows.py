#!/usr/bin/env python
"""
Test Embeddings Generation with 10 Sample Rows

This script demonstrates how to generate BERTweet embeddings for a small
sample of 10 articles.

Usage:
    python examples/test_embeddings_10_rows.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.embeddings import build_bertweet_embeddings, get_embedding_by_id


def main():
    """Generate embeddings for 10 sample articles"""

    print("=" * 70)
    print("BERTweet Embeddings Test - 10 Sample Articles")
    print("=" * 70)

    # Create 10 sample articles
    sample_data = {
        '_id': [f'nyt://article/{i:03d}' for i in range(1, 11)],
        'headline': [
            'Breaking News: Markets Surge',
            'World Leaders Meet for Climate Summit',
            'Technology Giants Announce New AI',
            'Sports Championship Final Results',
            'Election Results Certified by Officials',
            'Healthcare Reform Bill Passes Senate',
            'International Trade Agreement Signed',
            'Major Scientific Discovery Announced',
            'Cultural Festival Draws Large Crowds',
            'Economic Report Shows Growth'
        ],
        'combined_text': [
            'stock markets reached new highs today as investors responded positively to economic news',
            'world leaders gathered in geneva to discuss climate change and global warming policies',
            'major technology companies unveiled new artificial intelligence products and services',
            'championship game ended with a dramatic victory for the home team in overtime',
            'election officials confirmed final vote counts after thorough review process',
            'senate voted to approve comprehensive healthcare reform legislation after months of debate',
            'international leaders signed historic trade agreement covering multiple countries',
            'scientists made groundbreaking discovery in field of quantum physics research',
            'annual cultural festival attracted thousands of visitors from around the world',
            'quarterly economic report indicated positive growth across multiple sectors'
        ],
        'cleaned_text': [
            'stock markets reached new highs today investors responded positively economic news',
            'world leaders gathered geneva discuss climate change global warming policies',
            'major technology companies unveiled new artificial intelligence products services',
            'championship game ended dramatic victory home team overtime',
            'election officials confirmed final vote counts thorough review process',
            'senate voted approve comprehensive healthcare reform legislation months debate',
            'international leaders signed historic trade agreement covering multiple countries',
            'scientists made groundbreaking discovery field quantum physics research',
            'annual cultural festival attracted thousands visitors around world',
            'quarterly economic report indicated positive growth multiple sectors'
        ]
    }

    df = pd.DataFrame(sample_data)

    print(f"\nCreated DataFrame with {len(df)} sample articles")
    print("\nSample articles:")
    print(df[['_id', 'headline']].to_string())

    # Generate embeddings
    print("\n" + "=" * 70)
    print("Generating BERTweet Embeddings")
    print("=" * 70)
    print("\nNote: This will download the BERTweet model (~500MB) on first run")
    print("Subsequent runs will use the cached model.\n")

    try:
        embeddings, mapping = build_bertweet_embeddings(
            df,
            text_col='cleaned_text',
            model_name='vinai/bertweet-base',  # Full BERTweet model
            sample_limit=None,  # Process all 10
            batch_size=5,  # Small batch size
            max_length=128,
            pooling='cls',
            output_dir='data',
            save_embeddings=True,  # Save to disk
            verbose=True
        )

        print("\n" + "=" * 70)
        print("Results")
        print("=" * 70)

        print(f"\n✓ Successfully generated embeddings!")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        print(f"  Number of articles: {embeddings.shape[0]}")

        # Show mapping
        print(f"\nID-to-Index Mapping:")
        print(mapping.to_string())

        # Demonstrate embedding retrieval
        print(f"\n" + "=" * 70)
        print("Testing Embedding Retrieval")
        print("=" * 70)

        test_id = 'nyt://article/001'
        embedding = get_embedding_by_id(test_id, embeddings, mapping)

        if embedding is not None:
            print(f"\n✓ Retrieved embedding for '{test_id}'")
            print(f"  Shape: {embedding.shape}")
            print(f"  First 10 values: {embedding[:10]}")
            print(f"  L2 norm: {np.linalg.norm(embedding):.4f}")

        # Compute pairwise similarities
        print(f"\n" + "=" * 70)
        print("Computing Cosine Similarities")
        print("=" * 70)

        from sklearn.metrics.pairwise import cosine_similarity

        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)

        # Find most similar pairs (excluding self-similarity)
        print("\nTop 5 most similar article pairs:")
        for i in range(5):
            # Find max excluding diagonal
            np.fill_diagonal(sim_matrix, -1)
            max_idx = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
            similarity = sim_matrix[max_idx]

            article1_id = mapping.iloc[max_idx[0]]['_id']
            article2_id = mapping.iloc[max_idx[1]]['_id']
            headline1 = df.iloc[max_idx[0]]['headline']
            headline2 = df.iloc[max_idx[1]]['headline']

            print(f"\n{i+1}. Similarity: {similarity:.4f}")
            print(f"   Article 1: {article1_id}")
            print(f"   Headline:  {headline1}")
            print(f"   Article 2: {article2_id}")
            print(f"   Headline:  {headline2}")

            # Mark as processed
            sim_matrix[max_idx] = -1

        # File info
        print(f"\n" + "=" * 70)
        print("Saved Files")
        print("=" * 70)

        embeddings_file = Path('data/embeddings.npy')
        mapping_file = Path('data/embeddings_mapping.csv')

        if embeddings_file.exists():
            size_mb = embeddings_file.stat().st_size / (1024 * 1024)
            print(f"\n✓ data/embeddings.npy ({size_mb:.2f} MB)")

        if mapping_file.exists():
            size_kb = mapping_file.stat().st_size / 1024
            print(f"✓ data/embeddings_mapping.csv ({size_kb:.2f} KB)")

        print(f"\n" + "=" * 70)
        print("Next Steps")
        print("=" * 70)
        print("""
1. Load embeddings:
   from src.models.embeddings import load_embeddings
   embeddings, mapping = load_embeddings()

2. Use for similarity search:
   query_idx = 0
   similarities = cosine_similarity([embeddings[query_idx]], embeddings)[0]
   top_k = similarities.argsort()[-5:][::-1]

3. Build more embeddings:
   python scripts/build_embeddings.py --input data/preprocessed.parquet --sample 1000
        """)

    except Exception as e:
        print(f"\n✗ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    Path('examples').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)

    sys.exit(main())
