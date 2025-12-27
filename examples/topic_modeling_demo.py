#!/usr/bin/env python
"""
Topic Modeling Demo - LDA and BERTopic

This script demonstrates both LDA and BERTopic topic modeling
on sample news articles, following patterns from the notebook.

Usage:
    python examples/topic_modeling_demo.py
    python examples/topic_modeling_demo.py --num-topics 5
"""

import sys
from pathlib import Path
import pandas as pd
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.topic_models import run_lda, run_bertopic


# Sample news articles for demonstration
SAMPLE_ARTICLES = [
    # Economics/Business
    "stock market plunges amid economic recession fears financial crisis deepens",
    "economy growth slows as markets tumble investors worried about downturn",
    "wall street stocks fall sharply trading volatile economic uncertainty grows",
    "financial markets unstable business confidence drops economic indicators weak",
    "market crash fears rise as economy struggles investors flee stocks",

    # Politics
    "president announces new policy congress debates legislation vote expected soon",
    "senate committee hearing political tensions rise lawmakers discuss reform",
    "election campaign heats up candidates debate policy positions voters decide",
    "white house statement political controversy government officials respond",
    "congressional leaders negotiate bipartisan deal political compromise reached",

    # International/War
    "military forces advance troops deployed conflict escalates regional tensions",
    "war continues fighting intensifies civilian casualties mount humanitarian crisis",
    "peace talks stalled diplomatic efforts fail violence spreads across border",
    "armed conflict intensifies soldiers battle insurgents security deteriorates",
    "international intervention military strikes target strategic locations",

    # Technology
    "new technology breakthrough innovation revolutionizes industry artificial intelligence",
    "tech company launches product digital transformation silicon valley startup",
    "software update released technology advances internet platform grows",
    "innovation drives change technology sector booms venture capital invests",
    "artificial intelligence machine learning algorithms improve technology evolves",

    # Climate/Environment
    "climate change global warming temperatures rise environmental crisis worsens",
    "carbon emissions increase pollution levels high environmental damage spreads",
    "renewable energy solar power wind farms reduce greenhouse gases",
    "environmental protection conservation efforts biodiversity endangered species",
    "climate scientists warn rising seas melting ice caps ecosystem collapse",

    # Health
    "medical research breakthrough doctors discover treatment disease cure found",
    "health care reform hospital patients access medicine public health improves",
    "pandemic spreads virus infections rise vaccine development urgent",
    "medical experts recommend prevention measures health officials advise caution",
    "healthcare system strained hospitals overcrowded medical staff exhausted",

    # Sports
    "championship game team wins victory celebration fans cheer players excel",
    "sports tournament final match athletic performance record broken",
    "football season playoffs basketball finals baseball championship series",
    "olympic games athletes compete medals awarded world records set",
    "sports league expansion team performance coach strategy winning streak",

    # Culture/Arts
    "museum exhibition art gallery opens new show cultural event attracts visitors",
    "film festival movie premiere director wins award cinema celebrates",
    "music concert performance artist tours fans enjoy entertainment industry",
    "theater production opens broadway show cultural performance acclaimed",
    "literary prize awarded author book published cultural achievement recognized"
]


def main():
    """Run topic modeling demo on sample articles."""
    parser = argparse.ArgumentParser(
        description='Topic Modeling Demo - LDA and BERTopic'
    )
    parser.add_argument(
        '--num-topics', '-n',
        type=int,
        default=5,
        help='Number of topics to discover (default: 5)'
    )
    parser.add_argument(
        '--lda-only',
        action='store_true',
        help='Run only LDA (skip BERTopic)'
    )
    parser.add_argument(
        '--bertopic-only',
        action='store_true',
        help='Run only BERTopic (skip LDA)'
    )
    parser.add_argument(
        '--save-outputs',
        action='store_true',
        help='Save models and visualizations to data/demo_topics/'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Topic Modeling Demo - LDA and BERTopic")
    print("=" * 70)
    print(f"\nSample articles: {len(SAMPLE_ARTICLES)}")
    print(f"Number of topics: {args.num_topics}")

    # Prepare output directory
    output_dir = None
    if args.save_outputs:
        output_dir = Path('data/demo_topics')
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Outputs will be saved to: {output_dir}")

    # Run LDA
    if not args.bertopic_only:
        print("\n" + "=" * 70)
        print("Running LDA Topic Modeling")
        print("=" * 70)

        try:
            lda_model, lda_topics_df, dictionary, corpus = run_lda(
                SAMPLE_ARTICLES,
                num_topics=args.num_topics,
                no_below=1,  # Low threshold for small sample
                no_above=0.9,  # High threshold for small sample
                passes=10,
                random_state=100,
                workers=2,  # Fewer workers for small dataset
                output_dir=str(output_dir) if output_dir else None,
                save_model=args.save_outputs,
                verbose=True
            )

            print("\n" + "-" * 70)
            print("LDA Topics Discovered")
            print("-" * 70)

            for idx, row in lda_topics_df.iterrows():
                print(f"\nTopic {row['topic_id']}:")
                print(f"  Keywords: {row['keywords']}")
                print(f"  Top 5 words: {', '.join(row['top_10_words'][:5])}")

            # Show dictionary stats
            print(f"\n" + "-" * 70)
            print("LDA Model Statistics")
            print("-" * 70)
            print(f"Dictionary size: {len(dictionary)} unique tokens")
            print(f"Corpus size: {len(corpus)} documents")
            print(f"Topics: {args.num_topics}")

            if args.save_outputs:
                print(f"\n✓ LDA model saved to {output_dir}/lda_model")
                print(f"✓ Dictionary saved to {output_dir}/dictionary.pkl")
                print(f"✓ Topics saved to {output_dir}/lda_topics.csv")

        except Exception as e:
            print(f"\n✗ LDA failed: {e}")
            import traceback
            traceback.print_exc()

    # Run BERTopic
    if not args.lda_only:
        print("\n" + "=" * 70)
        print("Running BERTopic Topic Modeling")
        print("=" * 70)

        try:
            bertopic_model, topic_info = run_bertopic(
                SAMPLE_ARTICLES,
                nr_topics=args.num_topics,
                min_topic_size=3,  # Low threshold for small sample
                random_state=100,
                verbose=False,
                output_dir=str(output_dir) if output_dir else None,
                save_model=args.save_outputs,
                save_intertopic_map=args.save_outputs
            )

            print("\n" + "-" * 70)
            print("BERTopic Topics Discovered")
            print("-" * 70)

            # Show all topics (excluding outliers)
            topics = topic_info[topic_info['Topic'] != -1]

            for idx, row in topics.iterrows():
                print(f"\nTopic {row['Topic']} ({row['Count']} documents):")
                print(f"  Name: {row['Name']}")

                # Show top 5 representative words
                if 'Representation' in row and row['Representation']:
                    top_words = row['Representation'][:5]
                    print(f"  Top words: {', '.join(top_words)}")

            # Show outliers
            outliers = topic_info[topic_info['Topic'] == -1]
            if len(outliers) > 0:
                print(f"\nOutliers (Topic -1): {outliers.iloc[0]['Count']} documents")

            print(f"\n" + "-" * 70)
            print("BERTopic Model Statistics")
            print("-" * 70)
            print(f"Total topics: {len(topics)} (excluding outliers)")
            print(f"Total documents: {len(SAMPLE_ARTICLES)}")
            print(f"Outliers: {outliers.iloc[0]['Count'] if len(outliers) > 0 else 0}")

            if args.save_outputs:
                print(f"\n✓ BERTopic model saved to {output_dir}/bertopic_model")
                print(f"✓ Topic info saved to {output_dir}/bertopic_topics.csv")
                print(f"✓ Intertopic map saved to {output_dir}/intertopic_distance_map.html")

        except Exception as e:
            print(f"\n✗ BERTopic failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("Topic Modeling Demo Complete!")
    print("=" * 70)

    print("\nNext Steps:")
    print("  1. Try different numbers of topics:")
    print("     python examples/topic_modeling_demo.py --num-topics 3")
    print()
    print("  2. Run on real data:")
    print("     python scripts/run_topics_year.py --year 2001 --section World")
    print()
    print("  3. Save outputs for analysis:")
    print("     python examples/topic_modeling_demo.py --save-outputs")
    print()
    print("  4. Compare LDA vs BERTopic:")
    print("     python examples/topic_modeling_demo.py --num-topics 5")
    print()

    if args.save_outputs and output_dir:
        print(f"  5. View intertopic distance map:")
        print(f"     open {output_dir}/intertopic_distance_map.html")
        print()

    print("For full documentation, see: docs/topic_modeling.md")


if __name__ == "__main__":
    main()
