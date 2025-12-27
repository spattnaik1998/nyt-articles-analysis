#!/usr/bin/env python
"""
Book Recommendation Example

This script demonstrates book recommendations using embeddings,
mirroring the approach from the original notebook.

Usage:
    python examples/recommend_books.py "historical fiction world war"
    python examples/recommend_books.py "science fiction space exploration"
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.similarity import recommend_books


def main(query_text: str = None):
    """
    Recommend books based on query text.

    Args:
        query_text: Query string describing desired book
    """
    print("=" * 70)
    print("Book Recommendation System")
    print("=" * 70)

    # Default query if none provided
    if query_text is None:
        query_text = "historical fiction about world war two"
        print(f"\nUsing example query: '{query_text}'")
    else:
        print(f"\nQuery: '{query_text}'")

    # Check if book embeddings exist
    book_embeddings_path = Path('data/book_embeddings.npy')
    book_mapping_path = Path('data/book_embeddings_mapping.csv')

    if not book_embeddings_path.exists():
        print(f"\n✗ Book embeddings not found at {book_embeddings_path}")
        print("\nYou need to:")
        print("1. Filter books articles from your dataset:")
        print("   python scripts/preprocess_sample.py --section 'Books' --all")
        print("\n2. Generate embeddings for books:")
        print("   python scripts/build_embeddings.py \\")
        print("     --input data/preprocessed.parquet \\")
        print("     --output-dir data \\")
        print("     --sample 1000")
        print("\n3. Rename outputs:")
        print("   mv data/embeddings.npy data/book_embeddings.npy")
        print("   mv data/embeddings_mapping.csv data/book_embeddings_mapping.csv")
        return 1

    print(f"\n✓ Found book embeddings at {book_embeddings_path}")

    # Check for books DataFrame (with book_title, author_name)
    books_df_path = Path('data/books_with_metadata.parquet')

    if not books_df_path.exists():
        # Try preprocessed.parquet
        books_df_path = Path('data/preprocessed.parquet')

    if not books_df_path.exists():
        print(f"✗ Books DataFrame not found")
        return 1

    print(f"✓ Found books data at {books_df_path}")

    # Get recommendations
    print("\n" + "=" * 70)
    print("Finding Similar Books...")
    print("=" * 70)

    try:
        results = recommend_books(
            query_text=query_text,
            books_df_path=str(books_df_path),
            book_embeddings_path=str(book_embeddings_path),
            book_mapping_path=str(book_mapping_path),
            top_k=10,
            embed_model='vinai/bertweet-base',
            use_faiss=True
        )

        print(f"\n✓ Found {len(results)} book recommendations")

        # Display results
        print("\n" + "=" * 70)
        print("Top Book Recommendations")
        print("=" * 70)

        for idx, row in results.iterrows():
            print(f"\n{idx + 1}. Similarity: {row['similarity']:.4f}")

            if 'book_title' in row and pd.notna(row['book_title']):
                print(f"   Title: {row['book_title']}")
            else:
                print(f"   Title: [Not extracted]")

            if 'author_name' in row and pd.notna(row['author_name']):
                print(f"   Author: {row['author_name']}")
            else:
                print(f"   Author: [Not extracted]")

            if 'headline' in row and pd.notna(row['headline']):
                print(f"   Review Headline: {row['headline']}")

            if 'pub_date' in row and pd.notna(row['pub_date']):
                print(f"   Review Date: {row['pub_date']}")

            if 'abstract' in row and pd.notna(row['abstract']):
                abstract_snippet = str(row['abstract'])[:150] + '...'
                print(f"   Review: {abstract_snippet}")

        # Show results DataFrame
        print("\n" + "=" * 70)
        print("Results Summary")
        print("=" * 70)

        display_cols = ['similarity']
        if 'book_title' in results.columns:
            display_cols.append('book_title')
        if 'author_name' in results.columns:
            display_cols.append('author_name')

        print(results[display_cols].to_string(index=False))

        # Save results
        output_path = Path('data/book_recommendations.csv')
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
  python examples/recommend_books.py "mystery thriller detective"
  python examples/recommend_books.py "romance love story"
  python examples/recommend_books.py "biography memoir politics"

Use in Python:
  from src.models.similarity import recommend_books
  results = recommend_books(
      "science fiction",
      books_df_path='data/books.parquet',
      book_embeddings_path='data/book_embeddings.npy',
      top_k=5
  )
  print(results[['book_title', 'author_name', 'similarity']])
    """)

    return 0


if __name__ == "__main__":
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = None

    sys.exit(main(query))
