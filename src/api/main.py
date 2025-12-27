"""FastAPI main application entry point"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NYT Data Journalism Platform API",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "api": "operational",
            "database": "not_configured",
            "models": "not_loaded",
        },
    }


# Future endpoints will be added here:
# - /api/v1/search - Embedding-based article search
# - /api/v1/topics - Topic modeling endpoints
# - /api/v1/sentiment - Sentiment analysis endpoints
# - /api/v1/recommendations - Article recommendation endpoints
