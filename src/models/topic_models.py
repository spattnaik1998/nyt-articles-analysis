"""
Topic Modeling Module - LDA and BERTopic

This module provides reproducible topic modeling functions using:
- LDA (Latent Dirichlet Allocation) via Gensim
- BERTopic for neural topic modeling

Based on patterns from the NYT analysis notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Union
import logging
import pickle

# Gensim imports
try:
    import gensim
    from gensim.corpora import Dictionary
    from gensim.models import LdaMulticore
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logging.warning("Gensim not available. Install with: pip install gensim")

# BERTopic imports
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    logging.warning("BERTopic not available. Install with: pip install bertopic")

logger = logging.getLogger(__name__)


def generate_topic_description(keywords: List[str], topic_id: int, use_natural_language: bool = True) -> str:
    """
    Generate a human-readable topic description from keywords.

    Converts technical keyword lists into natural language descriptions
    suitable for reporters and readers unfamiliar with data science.

    Args:
        keywords (List[str]): List of topic keywords
        topic_id (int): Topic ID number
        use_natural_language (bool): Generate natural language description (default: True)

    Returns:
        str: Human-readable topic description

    Example:
        >>> keywords = ['economy', 'market', 'growth', 'trade', 'business']
        >>> desc = generate_topic_description(keywords, 0)
        >>> print(desc)
        'Economic Growth and Market Trends'
    """
    if not keywords or len(keywords) == 0:
        return f"Topic {topic_id + 1}"

    if not use_natural_language:
        # Simple format: just capitalize and join
        return ", ".join([kw.capitalize() for kw in keywords[:5]])

    # Pattern matching for common themes
    keywords_lower = [kw.lower() for kw in keywords[:10]]
    keywords_str = " ".join(keywords_lower)

    # Economic/Business themes
    if any(kw in keywords_lower for kw in ['economy', 'market', 'business', 'financial', 'trade']):
        if any(kw in keywords_lower for kw in ['growth', 'boom', 'increase']):
            return "Economic Growth and Market Trends"
        elif any(kw in keywords_lower for kw in ['crisis', 'recession', 'decline']):
            return "Economic Challenges and Market Concerns"
        else:
            return "Business and Economic News"

    # Political themes
    elif any(kw in keywords_lower for kw in ['election', 'vote', 'campaign', 'political', 'politics']):
        if 'president' in keywords_lower or 'presidential' in keywords_lower:
            return "Presidential Elections and Campaigns"
        else:
            return "Political News and Elections"

    # Government themes
    elif any(kw in keywords_lower for kw in ['government', 'congress', 'senate', 'house', 'law', 'policy']):
        return "Government Policy and Legislation"

    # International/World themes
    elif any(kw in keywords_lower for kw in ['war', 'conflict', 'military', 'attack', 'troops']):
        return "International Conflicts and Military Affairs"
    elif any(kw in keywords_lower for kw in ['country', 'world', 'foreign', 'international', 'nation']):
        return "International Relations and World News"

    # Health themes
    elif any(kw in keywords_lower for kw in ['health', 'medical', 'disease', 'patient', 'doctor', 'hospital']):
        if any(kw in keywords_lower for kw in ['pandemic', 'virus', 'covid', 'outbreak']):
            return "Public Health and Pandemic Response"
        else:
            return "Healthcare and Medical News"

    # Technology themes
    elif any(kw in keywords_lower for kw in ['technology', 'tech', 'digital', 'internet', 'online', 'computer']):
        if any(kw in keywords_lower for kw in ['ai', 'artificial', 'intelligence', 'machine']):
            return "Artificial Intelligence and Technology"
        else:
            return "Technology and Digital Innovation"

    # Climate/Environment themes
    elif any(kw in keywords_lower for kw in ['climate', 'environment', 'warming', 'carbon', 'emissions']):
        return "Climate Change and Environmental Issues"
    elif any(kw in keywords_lower for kw in ['energy', 'oil', 'gas', 'renewable', 'solar', 'wind']):
        return "Energy and Environmental Policy"

    # Education themes
    elif any(kw in keywords_lower for kw in ['school', 'student', 'education', 'teacher', 'university', 'college']):
        return "Education and Academic Affairs"

    # Arts/Culture themes
    elif any(kw in keywords_lower for kw in ['art', 'music', 'film', 'movie', 'theater', 'culture', 'artist']):
        return "Arts and Cultural Events"
    elif any(kw in keywords_lower for kw in ['book', 'author', 'novel', 'writer', 'literature']):
        return "Books and Literature"

    # Sports themes
    elif any(kw in keywords_lower for kw in ['game', 'team', 'player', 'sport', 'win', 'championship']):
        return "Sports and Athletic Competition"

    # Crime/Justice themes
    elif any(kw in keywords_lower for kw in ['police', 'court', 'judge', 'trial', 'crime', 'justice']):
        return "Crime, Law, and Justice"

    # Social issues
    elif any(kw in keywords_lower for kw in ['family', 'children', 'parent', 'child', 'home']):
        return "Family and Social Issues"

    # Default: create description from top keywords
    else:
        # Capitalize first keyword and create a descriptive title
        if len(keywords) >= 3:
            return f"{keywords[0].capitalize()} and {keywords[1].capitalize()}"
        elif len(keywords) >= 1:
            return f"{keywords[0].capitalize()} Related Topics"
        else:
            return f"Topic {topic_id + 1}"


def tokenize_documents(
    documents: List[str],
    deacc: bool = True
) -> List[List[str]]:
    """
    Tokenize documents using Gensim's simple_preprocess.

    This function applies the same tokenization used in the notebook:
    - Lowercases
    - Removes accents (deacc=True)
    - Splits on whitespace
    - Removes short tokens

    Args:
        documents (List[str]): List of text documents
        deacc (bool): Remove accents (default: True)

    Returns:
        List[List[str]]: List of tokenized documents

    Example:
        >>> docs = ["This is a test document", "Another test"]
        >>> tokens = tokenize_documents(docs)
        >>> print(tokens[0])
        ['this', 'is', 'test', 'document']
    """
    logger.info(f"Tokenizing {len(documents)} documents...")
    tokenized = [simple_preprocess(doc, deacc=deacc) for doc in documents]
    logger.info(f"Tokenization complete")
    return tokenized


def run_lda(
    documents: Union[List[str], List[List[str]]],
    num_topics: int = 10,
    no_below: int = 15,
    no_above: float = 0.5,
    keep_n: int = 100000,
    passes: int = 10,
    random_state: int = 100,
    chunksize: int = 100,
    workers: int = 4,
    per_word_topics: bool = True,
    alpha: str = 'auto',
    eta: str = 'auto',
    output_dir: Optional[str] = None,
    save_model: bool = False,
    verbose: bool = True
) -> Tuple['LdaMulticore', pd.DataFrame, Dictionary, list]:
    """
    Run LDA topic modeling using Gensim.

    This function follows the notebook pattern:
    1. Tokenizes documents (if not already tokenized)
    2. Creates Gensim dictionary
    3. Filters extremes (no_below, no_above, keep_n)
    4. Creates bag-of-words corpus
    5. Trains LDA model with reproducible parameters
    6. Extracts topics

    Args:
        documents (List[str] or List[List[str]]): Documents or pre-tokenized documents
        num_topics (int): Number of topics (default: 10)
        no_below (int): Minimum document frequency (default: 15)
        no_above (float): Maximum document proportion (default: 0.5)
        keep_n (int): Keep only top N tokens (default: 100000)
        passes (int): Number of training passes (default: 10)
        random_state (int): Random seed for reproducibility (default: 100)
        chunksize (int): Chunk size for training (default: 100)
        workers (int): Number of worker threads (default: 4)
        per_word_topics (bool): Compute per-word topic distribution (default: True)
        alpha (str): Document-topic density ('auto' or float)
        eta (str): Topic-word density ('auto' or float)
        output_dir (str, optional): Directory to save outputs
        save_model (bool): Save model to disk (default: False)
        verbose (bool): Print progress (default: True)

    Returns:
        Tuple containing:
            - lda_model: Trained LDA model
            - topics_df: DataFrame with topic keywords and weights
            - dictionary: Gensim dictionary
            - corpus: Bag-of-words corpus

    Example:
        >>> docs = ["market economy trade", "politics election vote"]
        >>> model, topics_df, dictionary, corpus = run_lda(docs, num_topics=5)
        >>> print(topics_df)
    """
    if not GENSIM_AVAILABLE:
        raise ImportError("Gensim required. Install with: pip install gensim")

    # Step 1: Tokenize if needed
    if isinstance(documents[0], str):
        if verbose:
            logger.info("Tokenizing documents...")
        tokenized = tokenize_documents(documents)
    else:
        tokenized = documents
        if verbose:
            logger.info(f"Using pre-tokenized documents")

    if verbose:
        logger.info(f"Processing {len(tokenized)} documents")

    # Step 2: Create dictionary
    if verbose:
        logger.info("Creating Gensim dictionary...")
    dictionary = Dictionary(tokenized)

    if verbose:
        logger.info(f"Initial dictionary size: {len(dictionary)} unique tokens")

    # Step 3: Filter extremes (following notebook pattern)
    dictionary.filter_extremes(
        no_below=no_below,
        no_above=no_above,
        keep_n=keep_n
    )

    if verbose:
        logger.info(f"After filtering: {len(dictionary)} unique tokens")

    # Step 4: Create bag-of-words corpus
    if verbose:
        logger.info("Creating bag-of-words corpus...")
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    if verbose:
        logger.info(f"Corpus created with {len(corpus)} documents")

    # Step 5: Train LDA model
    if verbose:
        logger.info(f"Training LDA model with {num_topics} topics...")
        logger.info(f"Parameters: passes={passes}, random_state={random_state}, workers={workers}")

    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        chunksize=chunksize,
        passes=passes,
        per_word_topics=per_word_topics,
        workers=workers,
        alpha=alpha,
        eta=eta
    )

    if verbose:
        logger.info("LDA training complete")

    # Step 6: Extract topics
    topics_data = []
    for idx in range(num_topics):
        topic_words = lda_model.show_topic(idx, topn=10)
        keywords = ", ".join([word for word, _ in topic_words])
        top_words = [word for word, _ in topic_words]

        # Generate human-readable description
        description = generate_topic_description(top_words, idx)

        topics_data.append({
            'topic_id': idx,
            'topic_name': description,
            'keywords': keywords,
            'top_10_words': top_words,
            'top_10_weights': [weight for _, weight in topic_words]
        })

    topics_df = pd.DataFrame(topics_data)

    # Step 7: Save outputs if requested
    if save_model and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_path / 'lda_model'
        lda_model.save(str(model_path))
        logger.info(f"Saved LDA model to {model_path}")

        # Save dictionary
        dict_path = output_path / 'dictionary.pkl'
        dictionary.save(str(dict_path))
        logger.info(f"Saved dictionary to {dict_path}")

        # Save topics
        topics_path = output_path / 'lda_topics.csv'
        topics_df.to_csv(topics_path, index=False)
        logger.info(f"Saved topics to {topics_path}")

    return lda_model, topics_df, dictionary, corpus


def run_bertopic(
    documents: List[str],
    nr_topics: Optional[int] = 10,
    min_topic_size: Optional[int] = None,
    language: str = 'english',
    seed_topic_list: Optional[List[List[str]]] = None,
    random_state: int = 100,
    verbose: bool = False,
    output_dir: Optional[str] = None,
    save_model: bool = False,
    save_intertopic_map: bool = False,
    intertopic_map_width: int = 1200,
    intertopic_map_height: int = 800
) -> Tuple['BERTopic', pd.DataFrame]:
    """
    Run BERTopic neural topic modeling.

    This function follows the notebook pattern for BERTopic:
    - Initializes BERTopic with reproducible parameters
    - Trains on documents
    - Extracts topic information
    - Optionally saves intertopic distance map

    Args:
        documents (List[str]): List of text documents
        nr_topics (int, optional): Number of topics (None = auto, int = reduce to N)
        min_topic_size (int, optional): Minimum cluster size (default: auto-calculated)
        language (str): Language for models (default: 'english')
        seed_topic_list (List[List[str]], optional): Seed topics for guided modeling
        random_state (int): Random seed (default: 100)
        verbose (bool): Verbose output (default: False)
        output_dir (str, optional): Directory to save outputs
        save_model (bool): Save BERTopic model (default: False)
        save_intertopic_map (bool): Save intertopic distance map HTML (default: False)
        intertopic_map_width (int): Width of intertopic map (default: 1200)
        intertopic_map_height (int): Height of intertopic map (default: 800)

    Returns:
        Tuple containing:
            - bertopic_model: Trained BERTopic model
            - topic_info: DataFrame with topic information (Topic, Count, Name, Representation)

    Example:
        >>> docs = ["market economy", "politics election", "sports game"]
        >>> model, topic_info = run_bertopic(docs, nr_topics=3, min_topic_size=1)
        >>> print(topic_info[['Topic', 'Count', 'Name']])
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError("BERTopic required. Install with: pip install bertopic")

    # Calculate min_topic_size if not provided (following notebook pattern)
    if min_topic_size is None:
        # Dynamic calculation: max(5, 0.5% of documents)
        min_topic_size = max(5, int(len(documents) * 0.005))

    logger.info(f"Running BERTopic on {len(documents)} documents")
    logger.info(f"Parameters: nr_topics={nr_topics}, min_topic_size={min_topic_size}, seed={random_state}")

    # Initialize BERTopic
    bertopic_model = BERTopic(
        language=language,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        seed_topic_list=seed_topic_list,
        verbose=verbose,
        calculate_probabilities=True  # For better topic assignments
    )

    # Fit model
    logger.info("Training BERTopic model...")
    topics, probabilities = bertopic_model.fit_transform(documents)

    logger.info("BERTopic training complete")

    # Get topic information
    topic_info = bertopic_model.get_topic_info()

    # Add human-readable descriptions
    topic_descriptions = []
    for idx, row in topic_info.iterrows():
        topic_id = row['Topic']

        # Get keywords for this topic (BERTopic stores them in 'Representation' or we can extract)
        if topic_id == -1:
            # Outlier topic
            topic_descriptions.append("Miscellaneous/Outliers")
        else:
            # Extract keywords from representation
            topic_words = bertopic_model.get_topic(topic_id)
            if topic_words:
                keywords = [word for word, _ in topic_words[:10]]
                description = generate_topic_description(keywords, topic_id)
                topic_descriptions.append(description)
            else:
                topic_descriptions.append(f"Topic {topic_id + 1}")

    topic_info['Description'] = topic_descriptions

    logger.info(f"Discovered {len(topic_info) - 1} topics (excluding outliers)")  # -1 for topic -1

    # Save outputs if requested
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save topic info
        topics_path = output_path / 'bertopic_topics.csv'
        topic_info.to_csv(topics_path, index=False)
        logger.info(f"Saved topic info to {topics_path}")

        # Save model
        if save_model:
            model_path = output_path / 'bertopic_model'
            try:
                # Try with safetensors serialization (BERTopic >= 0.15)
                bertopic_model.save(str(model_path), serialization='safetensors')
                logger.info(f"Saved BERTopic model to {model_path}")
            except TypeError:
                # Fallback for older BERTopic versions
                logger.warning("safetensors serialization not available, using default")
                bertopic_model.save(str(model_path))
                logger.info(f"Saved BERTopic model to {model_path}")

        # Save intertopic distance map
        if save_intertopic_map:
            try:
                logger.info("Generating intertopic distance map...")
                fig = bertopic_model.visualize_topics(
                    width=intertopic_map_width,
                    height=intertopic_map_height
                )

                map_path = output_path / 'intertopic_distance_map.html'
                fig.write_html(str(map_path))
                logger.info(f"Saved intertopic distance map to {map_path}")

            except Exception as e:
                logger.warning(f"Could not generate intertopic map: {e}")

    return bertopic_model, topic_info


def get_topic_keywords(
    lda_model: 'LdaMulticore',
    topic_id: int,
    topn: int = 10
) -> List[Tuple[str, float]]:
    """
    Get keywords for a specific LDA topic.

    Args:
        lda_model: Trained LDA model
        topic_id (int): Topic ID
        topn (int): Number of top keywords

    Returns:
        List[Tuple[str, float]]: List of (word, weight) tuples
    """
    return lda_model.show_topic(topic_id, topn=topn)


def get_document_topics(
    lda_model: 'LdaMulticore',
    corpus: list,
    minimum_probability: float = 0.0
) -> List[List[Tuple[int, float]]]:
    """
    Get topic distribution for each document.

    Args:
        lda_model: Trained LDA model
        corpus: Bag-of-words corpus
        minimum_probability (float): Minimum topic probability threshold

    Returns:
        List[List[Tuple[int, float]]]: Topic distributions per document
    """
    return [
        lda_model.get_document_topics(doc, minimum_probability=minimum_probability)
        for doc in corpus
    ]


def load_lda_model(model_path: str) -> 'LdaMulticore':
    """Load a saved LDA model."""
    if not GENSIM_AVAILABLE:
        raise ImportError("Gensim required")
    return LdaMulticore.load(model_path)


def load_bertopic_model(model_path: str) -> 'BERTopic':
    """Load a saved BERTopic model."""
    if not BERTOPIC_AVAILABLE:
        raise ImportError("BERTopic required")
    return BERTopic.load(model_path)


if __name__ == "__main__":
    # Example usage
    print("Topic Modeling Module")
    print("=" * 60)

    # Sample documents
    sample_docs = [
        "stock market economy trade financial growth",
        "market economy business financial stocks trading",
        "politics election government vote democracy",
        "political election campaign government voting",
        "sports game team championship victory",
        "sports championship team game player winning",
        "climate change environment global warming",
        "environment climate change global temperature",
        "technology innovation artificial intelligence",
        "technology AI machine learning innovation"
    ]

    print(f"\nSample documents: {len(sample_docs)}")

    # Example 1: LDA
    if GENSIM_AVAILABLE:
        print("\n" + "="*60)
        print("Example 1: LDA Topic Modeling")
        print("="*60)

        lda_model, topics_df, dictionary, corpus = run_lda(
            sample_docs,
            num_topics=3,
            no_below=1,  # Low threshold for small sample
            no_above=0.9,
            passes=10,
            random_state=42,
            verbose=True
        )

        print("\nLDA Topics:")
        for _, row in topics_df.iterrows():
            print(f"\nTopic {row['topic_id']}: {row['keywords']}")

    # Example 2: BERTopic
    if BERTOPIC_AVAILABLE:
        print("\n" + "="*60)
        print("Example 2: BERTopic Topic Modeling")
        print("="*60)

        bertopic_model, topic_info = run_bertopic(
            sample_docs,
            nr_topics=3,
            min_topic_size=2,
            random_state=42,
            verbose=False
        )

        print("\nBERTopic Topics:")
        print(topic_info[['Topic', 'Count', 'Name']].to_string())

    print("\n\nFor full examples, see:")
    print("  - scripts/run_topics_year.py")
    print("  - examples/topic_modeling_demo.py")
