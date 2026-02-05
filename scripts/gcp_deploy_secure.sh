#!/bin/bash
#
# GCP Secure Deployment: Build Docker image and deploy to Cloud Run with Secret Manager
#
# This script:
# 1. Creates/updates secrets in GCP Secret Manager
# 2. Builds and pushes Docker image to GCR
# 3. Deploys to Cloud Run with secret references
#
# Usage:
#   # First time setup - create secrets
#   bash gcp_deploy_secure.sh \
#     --project Sarthak-ai-aura \
#     --region us-central1 \
#     --openai-key "sk-proj-..." \
#     --gemini-key "..." \
#     --tavily-key "..." \
#     --setup-secrets
#
#   # Deploy with existing secrets
#   bash gcp_deploy_secure.sh \
#     --project Sarthak-ai-aura \
#     --region us-central1 \
#     --deploy
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
SETUP_SECRETS=false

# Secret keys
OPENAI_API_KEY=""
GEMINI_API_KEY=""
TAVILY_API_KEY=""

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
        --openai-key)
            OPENAI_API_KEY="$2"
            shift 2
            ;;
        --gemini-key)
            GEMINI_API_KEY="$2"
            shift 2
            ;;
        --tavily-key)
            TAVILY_API_KEY="$2"
            shift 2
            ;;
        --setup-secrets)
            SETUP_SECRETS=true
            shift
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

echo "=========================================="
echo "GCP Secure Deployment Configuration"
echo "=========================================="
echo "  Action: $ACTION"
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Service: $SERVICE_NAME"
echo "  Image: $IMAGE_URI"
echo "  Memory: $MEMORY"
echo "  Setup Secrets: $SETUP_SECRETS"
echo ""

# Function: Create/Update secrets in Secret Manager
setup_secrets() {
    echo "Setting up GCP Secret Manager..."

    if [ -z "$OPENAI_API_KEY" ] || [ -z "$GEMINI_API_KEY" ] || [ -z "$TAVILY_API_KEY" ]; then
        echo "❌ Missing API keys. Provide --openai-key, --gemini-key, --tavily-key"
        exit 1
    fi

    echo "Creating secrets in Secret Manager..."

    # Create OPENAI_API_KEY secret
    if gcloud secrets describe nyt-openai-api-key --project="$PROJECT_ID" &>/dev/null; then
        echo "  Updating nyt-openai-api-key..."
        echo -n "$OPENAI_API_KEY" | gcloud secrets versions add nyt-openai-api-key --data-file=- --project="$PROJECT_ID"
    else
        echo "  Creating nyt-openai-api-key..."
        echo -n "$OPENAI_API_KEY" | gcloud secrets create nyt-openai-api-key --data-file=- --project="$PROJECT_ID" --replication-policy="automatic"
    fi

    # Create GEMINI_API_KEY secret
    if gcloud secrets describe nyt-gemini-api-key --project="$PROJECT_ID" &>/dev/null; then
        echo "  Updating nyt-gemini-api-key..."
        echo -n "$GEMINI_API_KEY" | gcloud secrets versions add nyt-gemini-api-key --data-file=- --project="$PROJECT_ID"
    else
        echo "  Creating nyt-gemini-api-key..."
        echo -n "$GEMINI_API_KEY" | gcloud secrets create nyt-gemini-api-key --data-file=- --project="$PROJECT_ID" --replication-policy="automatic"
    fi

    # Create TAVILY_API_KEY secret
    if gcloud secrets describe nyt-tavily-api-key --project="$PROJECT_ID" &>/dev/null; then
        echo "  Updating nyt-tavily-api-key..."
        echo -n "$TAVILY_API_KEY" | gcloud secrets versions add nyt-tavily-api-key --data-file=- --project="$PROJECT_ID"
    else
        echo "  Creating nyt-tavily-api-key..."
        echo -n "$TAVILY_API_KEY" | gcloud secrets create nyt-tavily-api-key --data-file=- --project="$PROJECT_ID" --replication-policy="automatic"
    fi

    echo "✓ Secrets created/updated in Secret Manager"
}

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
    gcloud auth configure-docker "$REGISTRY" --quiet

    echo "Pushing image to GCR..."
    docker push "$IMAGE_URI"

    echo "✓ Image pushed to $IMAGE_URI"
}

# Function: Grant Cloud Run service account access to secrets
grant_secret_access() {
    echo "Granting Cloud Run service account access to secrets..."

    # Get the default Cloud Run service account
    SA_EMAIL="${PROJECT_ID}@appspot.gserviceaccount.com"

    for SECRET_NAME in nyt-openai-api-key nyt-gemini-api-key nyt-tavily-api-key; do
        echo "  Granting access to $SECRET_NAME..."
        gcloud secrets add-iam-policy-binding "$SECRET_NAME" \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="roles/secretmanager.secretAccessor" \
            --project="$PROJECT_ID" \
            --quiet 2>/dev/null || echo "    (Already granted or service account not ready yet)"
    done

    echo "✓ Secret access configured"
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
        --set-secrets="OPENAI_API_KEY=nyt-openai-api-key:latest,GEMINI_API_KEY=nyt-gemini-api-key:latest,TAVILY_API_KEY=nyt-tavily-api-key:latest" \
        --command="/bin/sh" \
        --args="-c,mkdir -p /app/data && (gsutil -m cp -r gs://${PROJECT_ID}-nyt-data/data/* /app/data/ 2>/dev/null || echo 'Note: Data bucket not available, using local data if present'); uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2" \
        --quiet

    # Get service URL
    SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --format='value(status.url)')

    echo "✓ Service deployed!"
    echo ""
    echo "=========================================="
    echo "Deployment Complete"
    echo "=========================================="
    echo "  Service URL: $SERVICE_URL"
    echo "  API Docs:    ${SERVICE_URL}/docs"
    echo "  Frontend:    ${SERVICE_URL}/app"
    echo "  Health:      ${SERVICE_URL}/health"
    echo ""
}

# Execute action
echo ""
case "$ACTION" in
    build)
        build_image
        echo ""
        echo "Next, run: docker push $IMAGE_URI"
        ;;
    deploy)
        if [ "$SETUP_SECRETS" = true ]; then
            setup_secrets
        fi
        build_image
        push_image
        grant_secret_access
        deploy_service
        ;;
    *)
        echo "Unknown action: $ACTION"
        exit 1
        ;;
esac

echo "Done!"
