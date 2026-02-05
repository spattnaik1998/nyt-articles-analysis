#!/bin/bash
#
# Complete Deployment Script for NYT Analytics Platform
#
# This script handles the entire deployment in one command:
# 1. Configures GCP
# 2. Initializes GCP resources
# 3. Stores secrets in Secret Manager
# 4. Uploads data to Cloud Storage
# 5. Deploys application to Cloud Run
# 6. Displays final service URL
#
# Usage:
#   bash scripts/deploy_complete.sh \
#     --openai-key "sk-proj-YOUR-KEY" \
#     --gemini-key "YOUR-GEMINI-KEY" \
#     --tavily-key "YOUR-TAVILY-KEY"
#

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="Sarthak-ai-aura"
REGION="us-central1"
SERVICE_NAME="nyt-api"

# API Keys (will be provided by user)
OPENAI_API_KEY=""
GEMINI_API_KEY=""
TAVILY_API_KEY=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            echo "Usage:"
            echo "  bash scripts/deploy_complete.sh \\"
            echo "    --openai-key 'sk-proj-YOUR-KEY' \\"
            echo "    --gemini-key 'YOUR-GEMINI-KEY' \\"
            echo "    --tavily-key 'YOUR-TAVILY-KEY'"
            exit 1
            ;;
    esac
done

# Validate API keys
if [ -z "$OPENAI_API_KEY" ] || [ -z "$GEMINI_API_KEY" ] || [ -z "$TAVILY_API_KEY" ]; then
    echo -e "${RED}❌ ERROR: All API keys are required${NC}"
    echo ""
    echo "Usage:"
    echo "  bash scripts/deploy_complete.sh \\"
    echo "    --openai-key 'sk-proj-YOUR-KEY' \\"
    echo "    --gemini-key 'YOUR-GEMINI-KEY' \\"
    echo "    --tavily-key 'YOUR-TAVILY-KEY'"
    exit 1
fi

echo -e "${BLUE}"
echo "=========================================="
echo "NYT Analytics Platform - Complete Deployment"
echo "=========================================="
echo -e "${NC}"
echo ""
echo "Configuration:"
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Service: $SERVICE_NAME"
echo ""
echo "This will:"
echo "  1. Configure GCP project"
echo "  2. Initialize GCP resources (APIs, storage, IAM)"
echo "  3. Create secrets in Secret Manager"
echo "  4. Upload data to Cloud Storage"
echo "  5. Build and deploy Docker image"
echo "  6. Show your live API URL"
echo ""
echo -e "${YELLOW}Press Enter to continue, or Ctrl+C to cancel...${NC}"
read

# ============================================================
# STEP 1: Configure GCP
# ============================================================
echo ""
echo -e "${BLUE}[STEP 1/6] Configuring GCP project...${NC}"
gcloud config set project "$PROJECT_ID" --quiet
gcloud config set compute/region "$REGION" --quiet
echo -e "${GREEN}✓ GCP configured${NC}"

# ============================================================
# STEP 2: Initialize GCP Resources
# ============================================================
echo ""
echo -e "${BLUE}[STEP 2/6] Initializing GCP resources...${NC}"
echo "  Enabling APIs..."
APIs=(
    "run.googleapis.com"
    "cloudbuild.googleapis.com"
    "containerregistry.googleapis.com"
    "secretmanager.googleapis.com"
    "storage-api.googleapis.com"
)

for api in "${APIs[@]}"; do
    gcloud services enable "$api" --quiet 2>/dev/null || true
done

echo "  Creating storage bucket..."
BUCKET="${PROJECT_ID}-nyt-data"
gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://${BUCKET}" 2>/dev/null || echo "    Bucket already exists"

echo "  Creating secret placeholders..."
for secret in nyt-openai-api-key nyt-gemini-api-key nyt-tavily-api-key; do
    gcloud secrets describe "$secret" --project="$PROJECT_ID" &>/dev/null || \
        echo "placeholder" | gcloud secrets create "$secret" \
            --data-file=- \
            --project="$PROJECT_ID" \
            --replication-policy="automatic" \
            --quiet 2>/dev/null
done

echo -e "${GREEN}✓ GCP resources ready${NC}"

# ============================================================
# STEP 3: Create Secrets in Secret Manager
# ============================================================
echo ""
echo -e "${BLUE}[STEP 3/6] Storing secrets in Secret Manager...${NC}"

echo "  Storing OpenAI API key..."
echo -n "$OPENAI_API_KEY" | gcloud secrets versions add nyt-openai-api-key --data-file=- --project="$PROJECT_ID" --quiet

echo "  Storing Gemini API key..."
echo -n "$GEMINI_API_KEY" | gcloud secrets versions add nyt-gemini-api-key --data-file=- --project="$PROJECT_ID" --quiet

echo "  Storing Tavily API key..."
echo -n "$TAVILY_API_KEY" | gcloud secrets versions add nyt-tavily-api-key --data-file=- --project="$PROJECT_ID" --quiet

echo "  Configuring IAM permissions..."
SA_EMAIL="${PROJECT_ID}@appspot.gserviceaccount.com"
for secret in nyt-openai-api-key nyt-gemini-api-key nyt-tavily-api-key; do
    gcloud secrets add-iam-policy-binding "$secret" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="roles/secretmanager.secretAccessor" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null || true
done

echo -e "${GREEN}✓ Secrets stored securely${NC}"

# ============================================================
# STEP 4: Upload Data to Cloud Storage
# ============================================================
echo ""
echo -e "${BLUE}[STEP 4/6] Uploading data to Cloud Storage...${NC}"
echo "  This may take a few minutes..."

# Check if data files exist
if [ -f "data/embeddings_500k.npy" ]; then
    gsutil -m cp -r data/* "gs://${BUCKET}/data/" 2>/dev/null || true
    echo -e "${GREEN}✓ Data uploaded${NC}"
else
    echo -e "${YELLOW}⚠ Warning: data/embeddings_500k.npy not found${NC}"
    echo "  The app will still deploy but may not have data loaded"
fi

# ============================================================
# STEP 5: Deploy to Cloud Run
# ============================================================
echo ""
echo -e "${BLUE}[STEP 5/6] Building and deploying application...${NC}"
echo "  This may take 5-10 minutes..."

bash scripts/gcp_deploy_secure.sh \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --deploy

echo -e "${GREEN}✓ Deployment complete${NC}"

# ============================================================
# STEP 6: Display Results
# ============================================================
echo ""
echo -e "${BLUE}[STEP 6/6] Getting service details...${NC}"

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --format='value(status.url)')

echo ""
echo -e "${GREEN}=========================================="
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo "==========================================${NC}"
echo ""
echo -e "${BLUE}Your API is live at:${NC}"
echo -e "${GREEN}  ${SERVICE_URL}${NC}"
echo ""
echo -e "${BLUE}Access your application:${NC}"
echo "  Frontend:     ${SERVICE_URL}/app"
echo "  API Docs:     ${SERVICE_URL}/docs"
echo "  Health Check: ${SERVICE_URL}/health"
echo ""
echo -e "${BLUE}Test the deployment:${NC}"
echo "  curl ${SERVICE_URL}/health"
echo ""
echo -e "${BLUE}View logs:${NC}"
echo "  gcloud run logs read $SERVICE_NAME --project $PROJECT_ID --region $REGION --follow"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Open ${SERVICE_URL}/app in your browser"
echo "  2. Try searching for articles"
echo "  3. Test API endpoints at ${SERVICE_URL}/docs"
echo "  4. Check logs for any issues"
echo ""
echo -e "${GREEN}Happy analyzing! 🚀${NC}"
echo ""
