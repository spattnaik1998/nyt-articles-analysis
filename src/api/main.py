"""FastAPI main application entry point"""

import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.similarity import recommend_by_embedding
from src.models.embeddings import build_bertweet_embeddings, get_device
from src.models.cached_embeddings import encode_query_cached, log_cache_stats
from src.models.sentiment import (
    batch_infer,
    MODEL_REGISTRY,
    generate_sentiment_pie_chart,
    select_model_for_section,
    get_recommended_models_for_section
)
from src.models.extraction import (
    extract_book_meta,
    get_openai_client,
    batch_extract_with_verification,
    extract_and_verify_book_meta
)
from src.models.streaming_extraction import (
    stream_extract_article,
    stream_extract_batch,
    format_sse_event
)
from src.models.topic_models import generate_topic_description
from src.api.query_router import get_router, QueryMode
from src.api.route_executor import execute_fast, execute_deep, log_route
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI(
    title="NYT Article Analytics Platform API",
    description="REST API for NYT article analysis with topic discovery, sentiment analysis, and content recommendations",
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
faiss_index = None
embed_model = None
embed_tokenizer = None

# Load data on startup
@app.on_event("startup")
async def load_data():
    """Load preprocessed data and embeddings on startup"""
    global data_df, embeddings, embeddings_mapping, bm25, faiss_index, embed_model, embed_tokenizer

    try:
        # Load preprocessed data - Priority: 21M > 500K > 20K
        data_path_21m = Path("data/preprocessed_21m.parquet")
        data_path_500k = Path("data/preprocessed_500K.parquet")
        data_path_20k = Path("data/preprocessed.parquet")

        if data_path_21m.exists():
            data_path = data_path_21m
            data_df = pd.read_parquet(data_path)
            print(f"✓ Loaded {len(data_df):,} articles from {data_path} (21M dataset)")
        elif data_path_500k.exists():
            data_path = data_path_500k
            data_df = pd.read_parquet(data_path)
            print(f"✓ Loaded {len(data_df):,} articles from {data_path} (500K dataset)")
        elif data_path_20k.exists():
            data_path = data_path_20k
            data_df = pd.read_parquet(data_path)
            print(f"✓ Loaded {len(data_df):,} articles from {data_path} (20K dataset)")
        else:
            print(f"⚠ Warning: No preprocessed data found")

        # Add year and month columns if they don't exist
        if data_df is not None and 'year' not in data_df.columns:
            data_df['pub_date'] = pd.to_datetime(data_df['pub_date'], errors='coerce')
            data_df['year'] = data_df['pub_date'].dt.year
            data_df['month'] = data_df['pub_date'].dt.month
            print(f"✓ Added year and month columns from pub_date")

        # Load FAISS index if available (21M scale) - Priority: 21M > 500K
        import faiss
        faiss_index_21m = Path("data/faiss_index_21m.bin")
        faiss_index_500k = Path("data/faiss_index_500k.bin")

        if faiss_index_21m.exists():
            faiss_index = faiss.read_index(str(faiss_index_21m))
            print(f"✓ Loaded FAISS index: {faiss_index_21m} (ntotal={faiss_index.ntotal:,})")
        elif faiss_index_500k.exists():
            faiss_index = faiss.read_index(str(faiss_index_500k))
            print(f"✓ Loaded FAISS index: {faiss_index_500k} (ntotal={faiss_index.ntotal:,})")
        else:
            # Fall back to loading embeddings array (older 20K path)
            embeddings_path_500k = Path("data/embeddings_500k.npy")
            mapping_path_500k = Path("data/embeddings_500k_mapping.csv")
            embeddings_path_20k = Path("data/embeddings.npy")
            mapping_path_20k = Path("data/embeddings_mapping.csv")

            if embeddings_path_500k.exists() and mapping_path_500k.exists():
                embeddings_path = embeddings_path_500k
                mapping_path = mapping_path_500k
                embeddings = np.load(embeddings_path)
                embeddings_mapping = pd.read_csv(mapping_path)
                print(f"✓ Loaded embeddings: {embeddings.shape} (500K dataset)")
            elif embeddings_path_20k.exists() and mapping_path_20k.exists():
                embeddings_path = embeddings_path_20k
                mapping_path = mapping_path_20k
                embeddings = np.load(embeddings_path)
                embeddings_mapping = pd.read_csv(mapping_path)
                print(f"✓ Loaded embeddings: {embeddings.shape} (20K dataset)")
            else:
                print(f"⚠ Warning: No embeddings found")

        # Load BERTweet model and tokenizer once (cached globally)
        from transformers import AutoTokenizer, AutoModel
        model_name = "vinai/bertweet-base"
        print(f"Loading BERTweet model: {model_name}...")
        embed_tokenizer = AutoTokenizer.from_pretrained(model_name)
        embed_model = AutoModel.from_pretrained(model_name)
        embed_model.to(get_device())
        embed_model.eval()
        print(f"✓ BERTweet model loaded and cached")

        # Build BM25 index for keyword search (only if < 1M articles)
        if data_df is not None:
            if len(data_df) <= 1_000_000:
                print("Building BM25 index for keyword search...")
                from rank_bm25 import BM25Okapi

                # Tokenize documents for BM25
                corpus = data_df['cleaned_text'].fillna('').tolist()
                tokenized_corpus = [doc.split() for doc in corpus]
                bm25 = BM25Okapi(tokenized_corpus)
                print(f"✓ BM25 index built with {len(tokenized_corpus)} documents")
            else:
                print(f"⚠ Skipping BM25 (>1M articles): search will use semantic-only mode")

        # Pre-load cross-encoder so the first reranked request is fast
        try:
            from src.models.reranker import warmup as reranker_warmup
            reranker_warmup()
        except Exception as warmup_err:
            print(f"⚠ Reranker warmup skipped: {warmup_err}")

        # Initialise Redis distributed cache
        try:
            import os
            from src.models.redis_cache import init_redis_cache
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            rc = init_redis_cache(redis_url)
            ping = rc.ping()
            if ping["status"] == "ok":
                print(f"✓ Redis cache connected: {redis_url} ({ping['latency_ms']} ms)")
            else:
                print(f"⚠ Redis unavailable ({ping.get('reason', ping.get('error', ''))}); "
                      f"in-memory fallback active")
        except Exception as redis_err:
            print(f"⚠ Redis cache init skipped: {redis_err}; in-memory fallback active")

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
        "message": "NYT Article Analytics Platform API",
        "version": "0.1.0",
        "status": "operational",
        "frontend_url": "/app",
        "docs_url": "/docs",
        "data_loaded": data_df is not None,
        "embeddings_loaded": embeddings is not None,
    }


@app.get("/cache/stats")
async def cache_stats():
    """
    Cache hit-rate metrics across both tiers (Redis and in-memory fallback).

    Returns per-tier breakdown: hits, misses, hit_rate
    and overall aggregated stats plus Redis connection status.
    """
    from src.models.redis_cache import get_redis_cache
    from src.models.embedding_cache import get_global_cache
    rc  = get_redis_cache()
    mc  = get_global_cache()
    return {
        "redis_cache":  rc.get_metrics(),
        "memory_cache": mc.get_stats(),
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
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Semantic weight (0=keyword, 1=semantic, 0.5=hybrid)"),
    mode: str = Query("auto", description="Execution mode: 'auto' (router decides), 'fast' (FAISS+cache only), 'deep' (full pipeline)"),
    llm_extract: bool = Query(False, description="Deep mode only: run LLM book-metadata extraction on results"),
    rerank: bool = Query(False, description="Re-rank FAISS candidates with a cross-encoder (ms-marco-MiniLM-L-6-v2); requires sentence-transformers"),
):
    """
    Smart search with automatic Fast / Deep routing.

    **Fast mode** (default for simple queries):
    - Cached BERTweet embedding + FAISS only
    - No BM25, no LLM calls
    - Typical latency: 20–100 ms

    **Deep mode** (triggered for complex/analytical queries):
    - Cached embedding + FAISS + BM25 hybrid fusion
    - Optional LLM book-metadata extraction (`llm_extract=true`)
    - Typical latency: 200 ms – 15 s (with LLM)

    **Auto mode** (default):
    - Router classifies query complexity and selects the appropriate path.
    - Pass `mode=fast` or `mode=deep` to override the router.

    Returns routing decision metadata alongside results.
    """
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    t_total_start = time.time()

    try:
        # ── 1. Routing decision ───────────────────────────────────────────
        router = get_router()
        override = mode if mode in ("fast", "deep") else None
        decision = router.resolve(query, override=override)
        resolved_mode = decision.mode   # QueryMode.FAST or QueryMode.DEEP

        # ── 2. Execute chosen path ────────────────────────────────────────
        t_exec_start = time.time()

        if resolved_mode == QueryMode.FAST:
            results, timing = execute_fast(
                query=query,
                k=k,
                faiss_index=faiss_index,
                embeddings_mapping=embeddings_mapping,
                data_df=data_df,
                embed_tokenizer=embed_tokenizer,
                embed_model=embed_model,
                rerank=rerank,
            )
        else:  # DEEP
            results, timing = execute_deep(
                query=query,
                k=k,
                alpha=alpha,
                faiss_index=faiss_index,
                embeddings_mapping=embeddings_mapping,
                data_df=data_df,
                embed_tokenizer=embed_tokenizer,
                embed_model=embed_model,
                bm25=bm25,
                run_llm_extraction=llm_extract,
                rerank=rerank,
            )

        exec_ms = round((time.time() - t_exec_start) * 1000, 2)
        total_ms = round((time.time() - t_total_start) * 1000, 2)

        # ── 3. Structured latency log ─────────────────────────────────────
        log_route(query, decision, exec_ms, total_ms, len(results))

        # ── 4. Response ───────────────────────────────────────────────────
        return {
            "query": query,
            "total_found": len(results),
            "results": results,
            # routing metadata
            "routing": {
                "mode": resolved_mode.value,
                "mode_source": "user_override" if override else "auto_router",
                "confidence": decision.confidence,
                "reason": decision.reason,
                "routing_latency_ms": decision.latency_ms,
            },
            # performance
            "latency": {
                **timing,
                "exec_ms": exec_ms,
                "total_ms": total_ms,
            },
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


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

                # Generate human-readable description
                description = generate_topic_description(top_words, topic_idx)

                topics.append({
                    'topic_id': topic_idx,
                    'description': description,
                    'keywords': ', '.join(top_words[:5]),
                    'words': top_words,
                    'topic_name': description  # Use description as topic_name
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

            # Intelligently select models based on section
            # If a section is specified, use recommended models for that section
            if section:
                selected_models = get_recommended_models_for_section(section)
            else:
                # Auto-select based on sections in the sample
                selected_models = None  # Let batch_infer auto-select

            # Run sentiment analysis with intelligent model selection
            result_df = batch_infer(
                sample_df,
                text_col='cleaned_text',
                models=selected_models,
                auto_select_models=(selected_models is None),
                batch_size=16,
                verbose=False
            )

            # Generate reports for all models used
            models_data = {}
            sentiment_cols = [col for col in result_df.columns if col.endswith('_label')]

            for col in sentiment_cols:
                model_name = col.replace('_label', '')
                score_col = f'{model_name}_score'

                if score_col in result_df.columns:
                    label_dist = result_df[col].value_counts().to_dict()
                    pie_chart = generate_sentiment_pie_chart(
                        label_distribution=label_dist,
                        model_name=model_name,
                        total_count=len(result_df)
                    )

                    models_data[model_name] = {
                        'total_classified': len(result_df),
                        'average_confidence': float(result_df[score_col].mean()),
                        'label_distribution': label_dist,
                        'pie_chart': pie_chart,
                        'description': MODEL_REGISTRY.get(model_name, {}).get('description', '')
                    }
        else:
            # Use existing sentiment data
            models_data = {}
            for col in sentiment_cols:
                model_name = col.replace('_label', '')
                score_col = f'{model_name}_score'

                if score_col in filtered_df.columns:
                    label_dist = filtered_df[col].value_counts().to_dict()
                    pie_chart = generate_sentiment_pie_chart(
                        label_distribution=label_dist,
                        model_name=model_name,
                        total_count=len(filtered_df)
                    )

                    models_data[model_name] = {
                        'total_classified': len(filtered_df),
                        'average_confidence': float(filtered_df[score_col].mean()),
                        'label_distribution': label_dist,
                        'pie_chart': pie_chart,
                        'description': MODEL_REGISTRY.get(model_name, {}).get('description', '')
                    }

        return {
            "total_articles": len(filtered_df),
            "filters": filters,
            "models": models_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment report failed: {str(e)}")


# Cache monitoring endpoint
@app.get("/cache/stats")
async def get_cache_statistics():
    """Get query embedding cache statistics"""
    from src.models.embedding_cache import get_global_cache

    cache = get_global_cache()
    stats = cache.get_stats()

    return {
        "cache": {
            "type": "QueryEmbeddingCache (TTL)",
            "hits": stats['hits'],
            "misses": stats['misses'],
            "total_requests": stats['total_requests'],
            "hit_rate": stats['hit_rate'],
            "hit_rate_pct": stats['hit_rate_pct'],
            "size": stats['cache_size'],
            "maxsize": stats['cache_maxsize'],
            "ttl_seconds": 3600,  # 1 hour
        }
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear query embedding cache (admin endpoint)"""
    from src.models.embedding_cache import reset_global_cache

    reset_global_cache()

    return {"message": "Cache cleared successfully"}


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

            # Generate human-readable description
            all_top_words = [feature_names[i] for i in top_indices[:10]]
            description = generate_topic_description(all_top_words, topic_idx)

            wordclouds.append({
                'topic_id': topic_idx,
                'topic_name': description,
                'description': description,
                'keywords': ', '.join(top_words),
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

    # Get unique years sorted (extract from pub_date if year column doesn't exist)
    if 'year' in data_df.columns:
        years = sorted(data_df['year'].dropna().unique().astype(int).tolist())
    else:
        # Extract year from pub_date
        pub_dates = pd.to_datetime(data_df['pub_date'], errors='coerce')
        years = sorted(pub_dates.dt.year.dropna().unique().astype(int).tolist())

    # Get sections sorted by article count
    sections = data_df['section_name'].value_counts().index.tolist()

    return {
        "years": years,
        "sections": sections,
        "total_articles": len(data_df)
    }


@app.get("/books/extract")
async def extract_books(
    year: int = Query(..., description="Year to filter"),
    section: str = Query("Books", description="Section to filter (default: Books)"),
    use_verification: bool = Query(True, description="Use web search verification"),
    min_confidence: str = Query("medium", description="Minimum confidence level (high/medium/low)")
):
    """
    Extract and verify book titles and authors from articles

    This endpoint:
    1. Extracts book metadata from articles using OpenAI GPT
    2. Uses Gemini as LLM judge to verify using internal knowledge
    3. Filters out invalid author names (Various, Unknown, Staff, etc.)
    4. Only includes verified, high-confidence results

    Args:
        year: Year to filter articles
        section: Section to filter (default: Books)
        use_verification: Enable web search verification (default: True)
        min_confidence: Minimum confidence level to accept (default: medium)

    Returns:
        Verified books with authors, confidence scores, statistics, and visualizations
    """
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    try:
        # Filter data by year and section
        filtered_df = data_df[
            (data_df['year'] == year) &
            (data_df['section_name'] == section)
        ].copy()

        if len(filtered_df) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No articles found for year={year}, section={section}"
            )

        # Use combined_text or create it
        if 'combined_text' not in filtered_df.columns:
            filtered_df['combined_text'] = (
                filtered_df['headline'].fillna('') + ' ' +
                filtered_df['abstract'].fillna('')
            )

        # Extract and verify books with the new verification pipeline
        # Using GPT-3.5 for extraction, Tavily for search, and Gemini for verification
        result_df = batch_extract_with_verification(
            filtered_df,
            text_col='combined_text',
            use_llm=True,
            use_verification=use_verification,
            llm_model="gpt-3.5-turbo",  # OpenAI for extraction
            verification_model="gemini-2.0-flash-exp",  # Gemini for verification
            use_gemini_for_verification=True,  # Use Gemini instead of OpenAI
            min_confidence_threshold=min_confidence,
            max_workers=2,  # Reduced to avoid Tavily rate limits
            verbose=False  # Disable verbose logging in API
        )

        # Get verified extractions (only those that passed verification)
        verified_df = result_df[result_df['extraction_success'] == True]

        # Statistics
        total = len(result_df)
        verified = len(verified_df)
        verification_rate = verified / total if total > 0 else 0

        # Count corrections
        corrected = verified_df['was_corrected'].sum() if 'was_corrected' in verified_df.columns else 0

        # Confidence breakdown
        confidence_breakdown = {}
        if 'verification_confidence' in result_df.columns:
            confidence_counts = result_df['verification_confidence'].value_counts()
            confidence_breakdown = {
                conf: int(count)
                for conf, count in confidence_counts.items()
            }

        # Top authors (from verified results only)
        top_authors = []
        if len(verified_df) > 0:
            author_counts = verified_df['author_name'].value_counts().head(10)
            top_authors = [
                {"author": author, "count": int(count)}
                for author, count in author_counts.items()
            ]

        # Method breakdown (extraction methods used)
        method_counts = result_df['extraction_method'].value_counts()
        method_breakdown = {
            method: int(count)
            for method, count in method_counts.items()
        }

        # Generate bar chart for top authors
        if len(top_authors) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))

            authors_list = [item['author'] for item in top_authors]
            counts_list = [item['count'] for item in top_authors]

            bars = ax.barh(authors_list[::-1], counts_list[::-1], color='#326891')

            ax.set_xlabel('Number of Books', fontsize=12, weight='bold')
            ax.set_ylabel('Author', fontsize=12, weight='bold')
            ax.set_title(f'Top Authors in {section} - {year}', fontsize=14, weight='bold', pad=20)
            ax.grid(axis='x', alpha=0.3)

            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts_list[::-1])):
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                       f' {count}', va='center', fontsize=10, weight='bold')

            plt.tight_layout()

            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='PNG', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close(fig)

            author_chart = f'data:image/png;base64,{chart_base64}'
        else:
            author_chart = None

        # Sample verified books (first 20)
        sample_books = []
        for _, row in verified_df.head(20).iterrows():
            sample_books.append({
                'title': row['book_title'],
                'author': row['author_name'],
                'headline': row.get('headline', ''),
                'pub_date': str(row.get('pub_date', '')),
                'confidence': row.get('verification_confidence', 'unknown'),
                'was_corrected': bool(row.get('was_corrected', False)),
                'reasoning': row.get('verification_reasoning', '')[:200]  # Truncate reasoning
            })

        return {
            'year': year,
            'section': section,
            'total_articles': total,
            'verified_extractions': verified,
            'verification_rate': float(verification_rate),
            'corrected_count': int(corrected),
            'top_authors': top_authors,
            'confidence_breakdown': confidence_breakdown,
            'method_breakdown': method_breakdown,
            'author_chart': author_chart,
            'sample_books': sample_books,
            'verification_enabled': use_verification,
            'min_confidence_threshold': min_confidence
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=f"Book extraction failed: {error_detail}")


# ============================================================================
# Server-Sent Events (SSE) Streaming Endpoints
# ============================================================================

@app.post("/books/extract/stream")
async def extract_books_stream(
    year: int = Query(..., description="Year to filter"),
    section: str = Query("Books", description="Section to filter"),
    limit: int = Query(10, description="Number of articles to process")
):
    """
    Stream book extraction results as Server-Sent Events.

    Returns tokens progressively as they're generated by OpenAI.
    Provides real-time progress updates with first token < 500ms.

    Args:
        year: Year to filter articles
        section: Section to filter
        limit: Max articles to process (limits API costs)

    Returns:
        StreamingResponse with SSE events
    """
    if data_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    async def event_generator():
        """Generate SSE events for streaming extraction."""
        try:
            # Filter data
            filtered_df = data_df[
                (data_df['year'] == year) &
                (data_df['section_name'] == section)
            ].copy()

            if len(filtered_df) == 0:
                yield format_sse_event("error", {
                    "message": f"No articles found for year={year}, section={section}"
                })
                return

            # Limit articles to avoid excessive API calls
            filtered_df = filtered_df.head(limit)

            # Prepare articles for streaming
            articles = []
            for idx, row in filtered_df.iterrows():
                article_text = (
                    str(row.get('headline', '')) + ' ' +
                    str(row.get('abstract', ''))
                )
                articles.append({
                    'id': f"article_{idx}",
                    'text': article_text
                })

            # Stream extraction for all articles
            async for event in stream_extract_batch(articles):
                yield format_sse_event(event['type'], event['data'])

        except Exception as e:
            logger.error(f"Error in streaming extraction: {e}")
            yield format_sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.post("/extract/stream")
async def extract_single_stream(
    text: str = Query(..., description="Text to extract from")
):
    """
    Stream extraction for a single text as Server-Sent Events.

    Real-time token streaming with progress indicators.

    Args:
        text: Article text to extract from

    Returns:
        StreamingResponse with SSE events
    """
    async def event_generator():
        """Generate SSE events for single extraction."""
        try:
            async for event in stream_extract_article(text):
                yield format_sse_event(event['type'], event['data'])

        except Exception as e:
            logger.error(f"Error in streaming extraction: {e}")
            yield format_sse_event("error", {"message": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
