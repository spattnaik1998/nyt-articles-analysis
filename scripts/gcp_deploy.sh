#!/bin/bash
#
# GCP Deployment: Build Docker image and deploy to Cloud Run
#
# Usage:
#   bash gcp_deploy.sh --project my-project --region us-central1 --build-only
#   bash gcp_deploy.sh --project my-project --region us-central1 --deploy
#

set -e

# Defaults
ACTION="deploy"
PROJECT_ID=""
REGION="us-central1"
SERVICE_NAME="nyt-api"
IMAGE_NAME="nyt-api"
MEMORY="8Gi"
MIN_INSTANCES="1"
MAX_INSTANCES="10"
TIMEOUT="3600"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --region)
            REGION="$2"
            shift 2
            ;;
        --service)
            SERVICE_NAME="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --min-instances)
            MIN_INSTANCES="$2"
            shift 2
            ;;
        --max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        --build-only)
            ACTION="build"
            shift
            ;;
        --deploy)
            ACTION="deploy"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate project
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project)
    if [ -z "$PROJECT_ID" ]; then
        echo "❌ No GCP project specified and none configured. Use --project or set default project."
        exit 1
    fi
fi

REGISTRY="gcr.io"
IMAGE_URI="${REGISTRY}/${PROJECT_ID}/${IMAGE_NAME}:latest"

echo "GCP Deployment Configuration:"
echo "  Action: $ACTION"
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Service: $SERVICE_NAME"
echo "  Image: $IMAGE_URI"
echo "  Memory: $MEMORY"
echo "  Min instances: $MIN_INSTANCES"
echo "  Max instances: $MAX_INSTANCES"
echo ""

# Function: Build Docker image
build_image() {
    echo "Building Docker image..."

    if [ ! -f "Dockerfile" ]; then
        echo "❌ Dockerfile not found in current directory"
        exit 1
    fi

    docker build -t "$IMAGE_URI" .

    echo "✓ Image built: $IMAGE_URI"
}

# Function: Push to Container Registry
push_image() {
    echo "Configuring Docker authentication with gcloud..."
    gcloud auth configure-docker "$REGISTRY"

    echo "Pushing image to GCR..."
    docker push "$IMAGE_URI"

    echo "✓ Image pushed to $IMAGE_URI"
}

# Function: Deploy to Cloud Run
deploy_service() {
    echo "Deploying to Cloud Run..."

    gcloud run deploy "$SERVICE_NAME" \
        --project="$PROJECT_ID" \
        --image="$IMAGE_URI" \
        --region="$REGION" \
        --memory="$MEMORY" \
        --cpu="2" \
        --min-instances="$MIN_INSTANCES" \
        --max-instances="$MAX_INSTANCES" \
        --timeout="$TIMEOUT" \
        --allow-unauthenticated \
        --set-env-vars="OPENAI_API_KEY=${OPENAI_API_KEY},GEMINI_API_KEY=${GEMINI_API_KEY},TAVILY_API_KEY=${TAVILY_API_KEY}" \
        --set-cloudsql-instances="" \
        --command="/bin/sh" \
        --args="-c,mkdir -p /app/data && gsutil -m cp -r gs://${PROJECT_ID}-nyt-data/data/* /app/data/ 2>/dev/null || true; uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

    # Get service URL
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --format='value(status.url)')

    echo "✓ Service deployed!"
    echo "  URL: $SERVICE_URL"
    echo "  API Docs: ${SERVICE_URL}/docs"
    echo "  Frontend: ${SERVICE_URL}/app"
}

# Execute action
case "$ACTION" in
    build)
        build_image
        echo ""
        echo "Next, run: docker push $IMAGE_URI"
        ;;
    deploy)
        build_image
        push_image
        deploy_service
        ;;
    *)
        echo "Unknown action: $ACTION"
        exit 1
        ;;
esac

echo "Done!"
