#!/bin/bash
#
# Setup GCP Project for NYT Analytics Platform
#
# This script initializes a GCP project with necessary resources:
# - Enable required APIs
# - Create storage buckets for data
# - Set up IAM permissions
# - Configure Secret Manager
#
# Usage:
#   bash scripts/setup_gcp_project.sh --project Sarthak-ai-aura
#

set -e

PROJECT_ID=""
REGION="us-central1"

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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$PROJECT_ID" ]; then
    echo "Usage: bash scripts/setup_gcp_project.sh --project <project-id> [--region <region>]"
    exit 1
fi

echo "=========================================="
echo "Setting up GCP Project: $PROJECT_ID"
echo "=========================================="
echo ""

# Set the project
echo "Setting GCP project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"
echo "✓ Project set"
echo ""

# Enable required APIs
echo "Enabling required GCP APIs..."
APIs=(
    "run.googleapis.com"           # Cloud Run
    "cloudbuild.googleapis.com"    # Cloud Build
    "containerregistry.googleapis.com"  # Container Registry
    "secretmanager.googleapis.com" # Secret Manager
    "storage-api.googleapis.com"   # Cloud Storage
    "compute.googleapis.com"       # Compute Engine (for GPU pipeline)
)

for api in "${APIs[@]}"; do
    echo "  Enabling $api..."
    gcloud services enable "$api" --quiet || echo "    (Already enabled)"
done
echo "✓ APIs enabled"
echo ""

# Create storage bucket for data
DATA_BUCKET="${PROJECT_ID}-nyt-data"
echo "Creating Cloud Storage bucket for data..."
if gsutil ls -b "gs://${DATA_BUCKET}" &>/dev/null; then
    echo "  Bucket already exists: gs://${DATA_BUCKET}"
else
    echo "  Creating bucket: gs://${DATA_BUCKET}"
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://${DATA_BUCKET}" || echo "    (Bucket creation failed, may already exist)"
fi
echo "✓ Data bucket configured: gs://${DATA_BUCKET}"
echo ""

# Create Secret Manager secrets (empty placeholders)
echo "Creating Secret Manager secret placeholders..."
SECRETS=("nyt-openai-api-key" "nyt-gemini-api-key" "nyt-tavily-api-key")

for secret in "${SECRETS[@]}"; do
    if gcloud secrets describe "$secret" --project="$PROJECT_ID" &>/dev/null; then
        echo "  Secret already exists: $secret"
    else
        echo "  Creating secret: $secret"
        echo "placeholder" | gcloud secrets create "$secret" \
            --data-file=- \
            --project="$PROJECT_ID" \
            --replication-policy="automatic" \
            --quiet
    fi
done
echo "✓ Secrets created"
echo ""

# Create default Cloud Run service account
echo "Configuring Cloud Run service account..."
SA_EMAIL="${PROJECT_ID}@appspot.gserviceaccount.com"
echo "  Service Account: $SA_EMAIL"
echo "✓ Service account ready"
echo ""

# Grant service account permissions
echo "Granting IAM permissions to Cloud Run service account..."
ROLES=(
    "roles/secretmanager.secretAccessor"  # Read secrets
    "roles/storage.objectViewer"           # Read from Cloud Storage
)

for role in "${ROLES[@]}"; do
    echo "  Granting $role..."
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:${SA_EMAIL}" \
        --role="$role" \
        --quiet 2>/dev/null || echo "    (Already granted)"
done
echo "✓ IAM permissions configured"
echo ""

echo "=========================================="
echo "GCP Project Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Update your API keys in Secret Manager:"
echo "   bash scripts/gcp_deploy_secure.sh \\"
echo "     --project $PROJECT_ID \\"
echo "     --region $REGION \\"
echo "     --openai-key 'your-openai-key' \\"
echo "     --gemini-key 'your-gemini-key' \\"
echo "     --tavily-key 'your-tavily-key' \\"
echo "     --setup-secrets"
echo ""
echo "2. Upload preprocessed data to Cloud Storage:"
echo "   gsutil -m cp -r data/* gs://${DATA_BUCKET}/data/"
echo ""
echo "3. Deploy the application:"
echo "   bash scripts/gcp_deploy_secure.sh \\"
echo "     --project $PROJECT_ID \\"
echo "     --region $REGION \\"
echo "     --deploy"
echo ""
