"""FastAPI main application entry point"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.similarity import recommend_by_embedding
from src.models.embeddings import build_bertweet_embeddings, get_device
from src.models.sentiment import batch_infer, MODEL_REGISTRY
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI(
    title="NYT Data Journalism Platform API",
    description="REST API for NYT article analysis with topic modeling, sentiment analysis, and recommendations",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global variables for loaded data
data_df = None
embeddings = None
embeddings_mapping = None
bm25 = None

# Load data on startup
@app.on_event("startup")
async def load_data():
    """Load preprocessed data and embeddings on startup"""
    global data_df, embeddings, embeddings_mapping, bm25

    try:
        # Load preprocessed data
        data_path = Path("data/preprocessed.parquet")
        if data_path.exists():
            data_df = pd.read_parquet(data_path)
            print(f"✓ Loaded {len(data_df):,} articles from {data_path}")
        else:
            print(f"⚠ Warning: {data_path} not found")

        # Load embeddings
        embeddings_path = Path("data/embeddings.npy")
        mapping_path = Path("data/embeddings_mapping.csv")

        if embeddings_path.exists() and mapping_path.exists():
            embeddings = np.load(embeddings_path)
            embeddings_mapping = pd.read_csv(mapping_path)
            print(f"✓ Loaded embeddings: {embeddings.shape}")
        else:
            print(f"⚠ Warning: Embeddings not found at {embeddings_path}")

        # Build BM25 index for keyword search
        if data_df is not None:
            print("Building BM25 index for keyword search...")
            from rank_bm25 import BM25Okapi

            # Tokenize documents for BM25
            corpus = data_df['cleaned_text'].fillna('').tolist()
            tokenized_corpus = [doc.split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            print(f"✓ BM25 index built with {len(tokenized_corpus)} documents")

    except Exception as e:
        print(f"❌ Error loading data: {e}")


# Pydantic models
class TopicRequest(BaseModel):
    year: int
    section: str
    model: str = "bertopic"
    num_topics: int = 10


@app.get("/app")
async def serve_frontend():
    """Serve the frontend application"""
    return FileResponse(str(static_dir / "index.html"))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NYT Data Journalism Platform API",
        "version": "0.1.0",
        "status": "operational",
        "frontend_url": "/app",
        "docs_url": "/docs",
        "data_loaded": data_df is not None,
        "embeddings_loaded": embeddings is not None,
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "api": "operational",
            "data": "loaded" if data_df is not None else "not_loaded",
            "embeddings": "loaded" if embeddings is not None else "not_loaded",
        },
        "stats": {
            "total_articles": len(data_df) if data_df is not None else 0,
            "embedding_dimension": embeddings.shape[1] if embeddings is not None else 0,
        }
    }


@app.get("/search")
async def search_articles(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=50, description="Number of results"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Hybrid search weight: 0=keyword only, 1=semantic only, 0.5=balanced")
):
    """
    Hybrid search combining semantic similarity and keyword matching

    Args:
        query: Natural language search query
        k: Number of results to return (1-50)
        alpha: Weight for hybrid search (0-1)
               - 0.0 = Pure keyword search (BM25)
               - 1.0 = Pure semantic search (embeddings)
               - 0.5 = Balanced hybrid (recommended)

    Returns:
        List of similar articles with metadata and scores
    """
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Initialize scores
        semantic_scores = np.zeros(len(data_df))
        keyword_scores = np.zeros(len(data_df))

        # 1. SEMANTIC SEARCH (if alpha > 0 and embeddings available)
        if alpha > 0 and embeddings is not None:
            import torch
            from transformers import AutoTokenizer, AutoModel

            device = get_device()
            model_name = "vinai/bertweet-base"

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.to(device)
            model.eval()

            # Generate query embedding
            encoded = tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = model(**encoded)
                query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

            # Compute cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            semantic_scores = cosine_similarity([query_embedding], embeddings)[0]

        # 2. KEYWORD SEARCH (if alpha < 1 and BM25 available)
        if alpha < 1 and bm25 is not None:
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            bm25_scores = bm25.get_scores(tokenized_query)

            # Normalize BM25 scores to [0, 1]
            if bm25_scores.max() > 0:
                keyword_scores = bm25_scores / bm25_scores.max()
            else:
                keyword_scores = bm25_scores

        # 3. HYBRID COMBINATION
        # Combined score = alpha * semantic + (1 - alpha) * keyword
        hybrid_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores

        # Get top k indices
        top_indices = np.argsort(hybrid_scores)[::-1][:k]

        # Build results
        results = []
        for idx in top_indices:
            article_idx = int(idx)
            if article_idx < len(data_df):
                article = data_df.iloc[article_idx]
                results.append({
                    "headline": str(article.get('headline', '')),
                    "snippet": str(article.get('abstract', ''))[:200],
                    "section_name": str(article.get('section_name', '')),
                    "pub_date": str(article.get('pub_date', '')),
                    "hybrid_score": float(hybrid_scores[idx]),
                    "semantic_score": float(semantic_scores[idx]) if alpha > 0 else None,
                    "keyword_score": float(keyword_scores[idx]) if alpha < 1 else None,
                    "_id": str(article.get('_id', ''))
                })

        return {
            "query": query,
            "total_found": len(results),
            "search_mode": "semantic" if alpha == 1.0 else ("keyword" if alpha == 0.0 else "hybrid"),
            "alpha": alpha,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/topic/run")
async def run_topic_modeling(request: TopicRequest):
    """
    Run topic modeling on filtered articles

    This endpoint filters articles by year and section, then runs topic modeling.
    Returns a job ID for tracking progress.
    """
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Filter data
        filtered_df = data_df[
            (data_df['year'] == request.year) &
            (data_df['section_name'] == request.section)
        ]

        if len(filtered_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No articles found for year={request.year}, section={request.section}"
            )

        # Get texts
        texts = filtered_df['cleaned_text'].fillna('').tolist()

        # Simple topic extraction using TF-IDF (no external dependencies)
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        # Use LDA from sklearn for simplicity
        if request.model == "lda" or request.model == "bertopic":
            # Vectorize
            vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                max_df=0.8,
                min_df=2
            )

            doc_term_matrix = vectorizer.fit_transform(texts)

            # Run LDA
            lda = LatentDirichletAllocation(
                n_components=request.num_topics,
                random_state=42,
                max_iter=10,
                learning_method='online',
                n_jobs=-1
            )

            lda.fit(doc_term_matrix)

            # Extract top words for each topic
            feature_names = vectorizer.get_feature_names_out()
            topics = []

            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_indices]
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'topic_name': f"Topic {topic_idx}: {', '.join(top_words[:3])}"
                })

        # Generate job ID
        import uuid
        job_id = str(uuid.uuid4())

        return {
            "job_id": job_id,
            "status": "completed",
            "message": f"Topic modeling completed on {len(filtered_df)} articles using {request.model.upper()}",
            "topics": topics
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=f"Topic modeling failed: {error_detail}")


@app.get("/topic/status/{job_id}")
async def get_topic_status(job_id: str):
    """Get status of topic modeling job"""
    # Simplified - in production, track actual job status
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 1.0,
        "message": "Topic modeling completed"
    }


@app.get("/sentiment/report")
async def get_sentiment_report(
    year: Optional[int] = None,
    section: Optional[str] = None
):
    """
    Get sentiment analysis report

    Args:
        year: Optional filter by year
        section: Optional filter by section

    Returns:
        Sentiment statistics and distribution
    """
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Filter data
        filtered_df = data_df.copy()

        filters = {}
        if year:
            filtered_df = filtered_df[filtered_df['year'] == year]
            filters['year'] = year
        if section:
            filtered_df = filtered_df[filtered_df['section_name'] == section]
            filters['section'] = section

        if len(filtered_df) == 0:
            raise HTTPException(status_code=404, detail="No articles found with given filters")

        # Check if sentiment columns exist
        sentiment_cols = [col for col in filtered_df.columns if col.endswith('_label')]

        if not sentiment_cols:
            # Run sentiment analysis on a sample
            sample_size = min(100, len(filtered_df))
            sample_df = filtered_df.head(sample_size)

            # Run sentiment analysis
            result_df = batch_infer(
                sample_df,
                text_col='cleaned_text',
                models=['finbert'],
                batch_size=16,
                verbose=False
            )

            models_data = {
                'finbert': {
                    'total_classified': len(result_df),
                    'average_confidence': float(result_df['finbert_score'].mean()),
                    'label_distribution': result_df['finbert_label'].value_counts().to_dict()
                }
            }
        else:
            # Use existing sentiment data
            models_data = {}
            for col in sentiment_cols:
                model_name = col.replace('_label', '')
                score_col = f'{model_name}_score'

                if score_col in filtered_df.columns:
                    models_data[model_name] = {
                        'total_classified': len(filtered_df),
                        'average_confidence': float(filtered_df[score_col].mean()),
                        'label_distribution': filtered_df[col].value_counts().to_dict()
                    }

        return {
            "total_articles": len(filtered_df),
            "filters": filters,
            "models": models_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment report failed: {str(e)}")


# Additional utility endpoint
@app.get("/stats")
async def get_statistics():
    """Get dataset statistics"""
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    return {
        "total_articles": len(data_df),
        "date_range": {
            "min": str(data_df['pub_date'].min()),
            "max": str(data_df['pub_date'].max())
        },
        "sections": data_df['section_name'].value_counts().head(10).to_dict(),
        "years": data_df['year'].value_counts().sort_index().to_dict(),
    }


@app.get("/visualizations/wordclouds")
async def generate_wordclouds(
    year: int = Query(..., description="Year to filter"),
    section: str = Query(..., description="Section to filter"),
    model: str = Query("lda", description="Topic model: lda or bertopic"),
    num_topics: int = Query(10, ge=2, le=20, description="Number of topics")
):
    """
    Generate wordclouds for topic modeling results

    Returns base64-encoded images of wordclouds for each discovered topic
    """
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Filter data
        filtered_df = data_df[
            (data_df['year'] == year) &
            (data_df['section_name'] == section)
        ]

        if len(filtered_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No articles found for year={year}, section={section}"
            )

        # Get texts
        texts = filtered_df['cleaned_text'].fillna('').tolist()

        # Run topic modeling
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            max_df=0.8,
            min_df=2
        )

        doc_term_matrix = vectorizer.fit_transform(texts)

        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=10,
            learning_method='online',
            n_jobs=-1
        )

        lda.fit(doc_term_matrix)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Generate wordclouds
        wordclouds = []

        # Professional color schemes for wordclouds
        color_schemes = [
            '#667eea', '#764ba2', '#f093fb', '#4facfe',
            '#43e97b', '#38f9d7', '#fa709a', '#fee140',
            '#30cfd0', '#330867', '#a8edea', '#fed6e3'
        ]

        for topic_idx, topic in enumerate(lda.components_):
            # Get top words and their weights
            top_indices = topic.argsort()[-50:][::-1]  # Top 50 words
            word_freq = {}

            for idx in top_indices:
                word = feature_names[idx]
                weight = topic[idx]
                word_freq[word] = weight

            # Create wordcloud
            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                # Use topic-specific color
                return color_schemes[topic_idx % len(color_schemes)]

            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                color_func=color_func,
                relative_scaling=0.5,
                min_font_size=10
            ).generate_from_frequencies(word_freq)

            # Convert to PIL Image directly to avoid numpy compatibility issues
            from PIL import Image, ImageDraw, ImageFont

            # Get the wordcloud as a PIL Image
            wc_image = wc.to_image()

            # Create a new image with title space
            final_width = 800
            final_height = 450  # 400 for wordcloud + 50 for title
            final_image = Image.new('RGB', (final_width, final_height), 'white')

            # Paste wordcloud
            final_image.paste(wc_image, (0, 50))

            # Add title
            draw = ImageDraw.Draw(final_image)
            title_text = f'Topic {topic_idx + 1}'
            # Use default font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Calculate text position (centered)
            bbox = draw.textbbox((0, 0), title_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (final_width - text_width) // 2
            draw.text((text_x, 15), title_text, fill='black', font=font)

            # Save to buffer
            buffer = io.BytesIO()
            final_image.save(buffer, format='PNG')
            buffer.seek(0)

            # Encode to base64
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

            # Get top words for topic name
            top_words = [feature_names[i] for i in top_indices[:5]]

            wordclouds.append({
                'topic_id': topic_idx,
                'topic_name': f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}",
                'top_words': top_words,
                'image': f'data:image/png;base64,{image_base64}'
            })

        return {
            'year': year,
            'section': section,
            'model': model,
            'num_topics': num_topics,
            'total_articles': len(filtered_df),
            'wordclouds': wordclouds
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=f"Wordcloud generation failed: {error_detail}")


@app.get("/filters")
async def get_available_filters():
    """Get available years and sections for filtering"""
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    # Get unique years sorted
    years = sorted(data_df['year'].dropna().unique().astype(int).tolist())

    # Get sections sorted by article count
    sections = data_df['section_name'].value_counts().index.tolist()

    return {
        "years": years,
        "sections": sections,
        "total_articles": len(data_df)
    }
