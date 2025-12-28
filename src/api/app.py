"""
FastAPI Application - NYT Data Journalism Platform

This module provides REST API endpoints for:
- Similarity search
- Topic modeling (async background jobs)
- Sentiment analysis reports
"""

import os
import sys
import uuid
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.similarity import recommend_by_embedding
from src.models.topic_models import run_bertopic, run_lda
from src.models.sentiment import batch_infer, model_comparison_report

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NYT Data Journalism Platform API",
    description="API for similarity search, topic modeling, and sentiment analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage (in-memory for MVP - use Redis/database in production)
JOBS: Dict[str, Dict[str, Any]] = {}
JOB_RESULTS_DIR = Path("data/api_results")
JOB_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Pydantic Models
# ============================================================================

class SearchResponse(BaseModel):
    """Response model for similarity search."""
    query: str
    top_k: int
    results: List[Dict[str, Any]]
    total_found: int


class TopicJobRequest(BaseModel):
    """Request model for topic modeling job."""
    year: int = Field(..., description="Year to analyze", ge=2000, le=2030)
    section: str = Field(..., description="Section to analyze (e.g., 'World', 'Business')")
    model: str = Field(default="bertopic", description="Model to use: 'bertopic' or 'lda'")
    num_topics: int = Field(default=10, description="Number of topics", ge=2, le=50)
    input_file: Optional[str] = Field(default="data/preprocessed_500K.parquet", description="Input data file")


class TopicJobResponse(BaseModel):
    """Response model for topic job submission."""
    job_id: str
    status: str
    message: str
    created_at: str


class JobStatus(BaseModel):
    """Model for job status."""
    job_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    result_path: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_job_status(job_id: str) -> Optional[JobStatus]:
    """Get status of a background job."""
    if job_id not in JOBS:
        return None

    job_data = JOBS[job_id]
    return JobStatus(**job_data)


def update_job_status(
    job_id: str,
    status: str,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    result_path: Optional[str] = None,
    error: Optional[str] = None
):
    """Update job status."""
    if job_id in JOBS:
        JOBS[job_id]["status"] = status
        if progress is not None:
            JOBS[job_id]["progress"] = progress
        if message is not None:
            JOBS[job_id]["message"] = message
        if result_path is not None:
            JOBS[job_id]["result_path"] = result_path
        if error is not None:
            JOBS[job_id]["error"] = error
        if status in ["completed", "failed"]:
            JOBS[job_id]["completed_at"] = datetime.now().isoformat()


def run_topic_modeling_job(job_id: str, request: TopicJobRequest):
    """Background task for topic modeling."""
    try:
        logger.info(f"Starting topic modeling job {job_id}")
        update_job_status(job_id, "running", progress=0.1, message="Loading data...")

        # Load data
        input_path = Path(request.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {request.input_file}")

        df = pd.read_parquet(input_path)
        logger.info(f"Loaded {len(df):,} articles")

        update_job_status(job_id, "running", progress=0.2, message="Filtering data...")

        # Ensure pub_date is datetime
        if 'pub_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['pub_date']):
            df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')

        # Filter by year and section
        filtered_df = df[
            (df['pub_date'].dt.year == request.year) &
            (df['section_name'] == request.section)
        ].copy()

        if len(filtered_df) == 0:
            raise ValueError(f"No articles found for year={request.year}, section={request.section}")

        logger.info(f"Filtered to {len(filtered_df):,} articles")

        # Prepare documents
        text_col = 'cleaned_text' if 'cleaned_text' in filtered_df.columns else 'combined_text'
        documents = filtered_df[text_col].fillna('').astype(str).tolist()
        documents = [doc for doc in documents if doc.strip()]

        update_job_status(job_id, "running", progress=0.3, message=f"Running {request.model} on {len(documents)} documents...")

        # Create output directory
        output_dir = JOB_RESULTS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run topic modeling
        if request.model.lower() == "bertopic":
            logger.info("Running BERTopic...")
            min_topic_size = max(5, int(len(documents) * 0.005))

            bertopic_model, topic_info = run_bertopic(
                documents,
                nr_topics=request.num_topics,
                min_topic_size=min_topic_size,
                random_state=100,
                output_dir=str(output_dir),
                save_model=True,
                save_intertopic_map=True,
                verbose=False
            )

            result_file = "bertopic_topics.csv"

        elif request.model.lower() == "lda":
            logger.info("Running LDA...")

            lda_model, topics_df, dictionary, corpus = run_lda(
                documents,
                num_topics=request.num_topics,
                no_below=15,
                no_above=0.5,
                passes=10,
                random_state=100,
                output_dir=str(output_dir),
                save_model=True,
                verbose=False
            )

            result_file = "lda_topics.csv"

        else:
            raise ValueError(f"Unknown model: {request.model}")

        update_job_status(job_id, "running", progress=0.9, message="Finalizing results...")

        # Save job metadata
        metadata = {
            "job_id": job_id,
            "year": request.year,
            "section": request.section,
            "model": request.model,
            "num_topics": request.num_topics,
            "documents_processed": len(documents),
            "result_file": result_file,
            "created_at": JOBS[job_id]["created_at"],
            "completed_at": datetime.now().isoformat()
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Mark as complete
        result_path = str(output_dir / result_file)
        update_job_status(
            job_id,
            "completed",
            progress=1.0,
            message="Topic modeling completed successfully",
            result_path=result_path
        )

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        update_job_status(
            job_id,
            "failed",
            error=str(e),
            message=f"Job failed: {str(e)}"
        )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "NYT Data Journalism Platform API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search?query=...&k=5",
            "topic_run": "POST /topic/run",
            "topic_status": "/topic/status/{job_id}",
            "topic_result": "/topic/result/{job_id}",
            "sentiment_report": "/sentiment/report?year=..."
        },
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/search", response_model=SearchResponse)
async def search_articles(
    query: str = Query(..., description="Search query text"),
    k: int = Query(5, ge=1, le=100, description="Number of results to return"),
    embeddings_path: str = Query("data/embeddings_500k.npy", description="Path to embeddings file"),
    mapping_path: str = Query("data/embeddings_500k_mapping.csv", description="Path to mapping file"),
    articles_path: Optional[str] = Query(None, description="Path to articles DataFrame")
):
    """
    Search for similar articles using embedding similarity.

    Args:
        query: Search query text
        k: Number of results to return
        embeddings_path: Path to embeddings .npy file
        mapping_path: Path to ID mapping CSV
        articles_path: Optional path to articles DataFrame for metadata

    Returns:
        SearchResponse with top-k similar articles
    """
    try:
        logger.info(f"Search query: '{query}' (k={k})")

        # Check if files exist
        if not Path(embeddings_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Embeddings file not found: {embeddings_path}"
            )

        if not Path(mapping_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Mapping file not found: {mapping_path}"
            )

        # Perform search
        results_df = recommend_by_embedding(
            query_text=query,
            embeddings_path=embeddings_path,
            id_map_csv=mapping_path,
            articles_df_path=articles_path,
            top_k=k,
            use_faiss=True
        )

        # Convert to dict
        results = results_df.to_dict('records')

        return SearchResponse(
            query=query,
            top_k=k,
            results=results,
            total_found=len(results)
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/topic/run", response_model=TopicJobResponse)
async def run_topic_job(
    request: TopicJobRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit a topic modeling job to run in the background.

    Args:
        request: Topic job configuration
        background_tasks: FastAPI background tasks

    Returns:
        TopicJobResponse with job ID and status
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create job record
        job_data = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0.0,
            "message": "Job submitted",
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "result_path": None,
            "error": None,
            "config": request.dict()
        }

        JOBS[job_id] = job_data

        # Add background task
        background_tasks.add_task(run_topic_modeling_job, job_id, request)

        logger.info(f"Created topic modeling job {job_id}")

        return TopicJobResponse(
            job_id=job_id,
            status="pending",
            message="Topic modeling job submitted successfully",
            created_at=job_data["created_at"]
        )

    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@app.get("/topic/status/{job_id}", response_model=JobStatus)
async def get_topic_job_status(job_id: str):
    """
    Get status of a topic modeling job.

    Args:
        job_id: Job ID

    Returns:
        JobStatus with current status and progress
    """
    status = get_job_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return status


@app.get("/topic/result/{job_id}")
async def get_topic_result(job_id: str):
    """
    Get results of a completed topic modeling job.

    Args:
        job_id: Job ID

    Returns:
        CSV file with topic results or JSON with topic data
    """
    status = get_job_status(job_id)

    if status is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    if status.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Current status: {status.status}"
        )

    if not status.result_path or not Path(status.result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    # Return CSV file
    return FileResponse(
        path=status.result_path,
        media_type='text/csv',
        filename=f"topics_{job_id}.csv"
    )


@app.get("/sentiment/report")
async def get_sentiment_report(
    year: Optional[int] = Query(None, description="Year to filter"),
    section: Optional[str] = Query(None, description="Section to filter"),
    results_path: str = Query("data/sentiment_results.csv", description="Path to sentiment results")
):
    """
    Get precomputed sentiment analysis report.

    Args:
        year: Optional year filter
        section: Optional section filter
        results_path: Path to sentiment results CSV

    Returns:
        Sentiment report with statistics and distributions
    """
    try:
        results_file = Path(results_path)

        if not results_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Sentiment results file not found: {results_path}"
            )

        # Load results
        df = pd.read_csv(results_file)

        # Apply filters
        if year is not None:
            if 'pub_date' in df.columns:
                df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
                df = df[df['pub_date'].dt.year == year]

        if section is not None:
            if 'section_name' in df.columns:
                df = df[df['section_name'] == section]

        if len(df) == 0:
            return {
                "message": "No data found for the specified filters",
                "filters": {"year": year, "section": section},
                "total_articles": 0
            }

        # Calculate statistics
        stats = {
            "total_articles": len(df),
            "filters": {
                "year": year,
                "section": section
            },
            "models": {}
        }

        # Get model columns
        label_cols = [col for col in df.columns if col.endswith('_label')]

        for label_col in label_cols:
            model_name = label_col.replace('_label', '')
            score_col = f'{model_name}_score'

            if score_col in df.columns:
                label_dist = df[label_col].value_counts().to_dict()
                avg_score = df[score_col].mean()

                stats["models"][model_name] = {
                    "label_distribution": label_dist,
                    "average_confidence": float(avg_score),
                    "total_classified": len(df[label_col].dropna())
                }

        return stats

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate sentiment report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@app.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of jobs to return")
):
    """
    List all jobs with optional status filter.

    Args:
        status: Optional status filter (pending, running, completed, failed)
        limit: Maximum number of jobs to return

    Returns:
        List of jobs with their status
    """
    jobs = list(JOBS.values())

    # Filter by status if provided
    if status:
        jobs = [job for job in jobs if job["status"] == status]

    # Sort by creation time (most recent first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)

    # Limit results
    jobs = jobs[:limit]

    return {
        "total": len(JOBS),
        "filtered": len(jobs),
        "jobs": jobs
    }


# ============================================================================
# Static Files & Frontend
# ============================================================================

# Mount static files directory if it exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
