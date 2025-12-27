#!/usr/bin/env python
"""
Article Recommendation Example

This script demonstrates how to recommend articles based on a query text
using pre-computed embeddings and cosine similarity.

Usage:
    python examples/recommend_articles.py "stock market economy finance"
    python examples/recommend_articles.py "climate change global warming"
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.similarity import recommend_by_embedding


def main(query_text: str = None):
    """
    Recommend articles based on query text.

    Args:
        query_text: Query string (default: example query)
    """
    print("=" * 70)
    print("Article Recommendation System")
    print("=" * 70)

    # Default query if none provided
    if query_text is None:
        query_text = "stock market crash economic recession financial crisis"
        print(f"\nUsing example query: '{query_text}'")
    else:
        print(f"\nQuery: '{query_text}'")

    # Check if embeddings exist
    embeddings_path = Path('data/embeddings.npy')
    mapping_path = Path('data/embeddings_mapping.csv')

    if not embeddings_path.exists():
        print(f"\n✗ Embeddings not found at {embeddings_path}")
        print("\nPlease generate embeddings first:")
        print("  python examples/test_embeddings_10_rows.py")
        print("  OR")
        print("  python scripts/build_embeddings.py --sample 1000")
        return 1

    print(f"\n✓ Found embeddings at {embeddings_path}")

    # Check for preprocessed data (for metadata)
    preprocessed_path = Path('data/preprocessed.parquet')
    articles_df_path = str(preprocessed_path) if preprocessed_path.exists() else None

    if articles_df_path:
        print(f"✓ Found article metadata at {preprocessed_path}")
    else:
        print(f"⚠ Article metadata not found (headlines won't be shown)")

    # Get recommendations
    print("\n" + "=" * 70)
    print("Finding Similar Articles...")
    print("=" * 70)

    try:
        results = recommend_by_embedding(
            query_text=query_text,
            embeddings_path=str(embeddings_path),
            id_map_csv=str(mapping_path),
            articles_df_path=articles_df_path,
            top_k=10,
            embed_model='vinai/bertweet-base',
            use_faiss=True  # Use FAISS if available
        )

        print(f"\n✓ Found {len(results)} recommendations")

        # Display results
        print("\n" + "=" * 70)
        print("Top Recommendations")
        print("=" * 70)

        for idx, row in results.iterrows():
            print(f"\n{idx + 1}. Article ID: {row['_id']}")
            print(f"   Similarity: {row['similarity']:.4f}")

            if 'headline' in row and pd.notna(row['headline']):
                print(f"   Headline: {row['headline']}")

            if 'section_name' in row and pd.notna(row['section_name']):
                print(f"   Section: {row['section_name']}")

            if 'pub_date' in row and pd.notna(row['pub_date']):
                print(f"   Date: {row['pub_date']}")

            if 'abstract_snippet' in row and pd.notna(row['abstract_snippet']):
                print(f"   Abstract: {row['abstract_snippet']}")

        # Show results DataFrame
        print("\n" + "=" * 70)
        print("Results Summary")
        print("=" * 70)

        display_cols = ['_id', 'similarity']
        if 'headline' in results.columns:
            display_cols.append('headline')
        if 'section_name' in results.columns:
            display_cols.append('section_name')

        print(results[display_cols].to_string(index=False))

        # Save results
        output_path = Path('data/recommendations.csv')
        results.to_csv(output_path, index=False)
        print(f"\n✓ Saved results to {output_path}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Usage tips
    print("\n" + "=" * 70)
    print("Next Steps")
    print("=" * 70)
    print("""
Try different queries:
  python examples/recommend_articles.py "technology artificial intelligence"
  python examples/recommend_articles.py "sports championship victory"
  python examples/recommend_articles.py "election politics government"

Use in Python:
  from src.models.similarity import recommend_by_embedding
  results = recommend_by_embedding("your query here", top_k=5)
  print(results[['_id', 'similarity', 'headline']])
    """)

    return 0


if __name__ == "__main__":
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = None

    sys.exit(main(query))
