#!/bin/bash
#
# GCP Pipeline: Set up GPU instance and run the 21M embedding pipeline
#
# Usage:
#   bash gcp_pipeline.sh --create --gpu t4 --project my-project --zone us-central1-a
#   bash gcp_pipeline.sh --run --instance nyt-pipeline-01
#   bash gcp_pipeline.sh --cleanup --instance nyt-pipeline-01
#

set -e

# Defaults
ACTION="run"
INSTANCE_NAME="nyt-pipeline-01"
PROJECT_ID=""
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-8"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT="1"
DISK_SIZE="100"
IMAGE_FAMILY="debian-11"
KAGGLE_JSON=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --create)
            ACTION="create"
            shift
            ;;
        --run)
            ACTION="run"
            shift
            ;;
        --cleanup)
            ACTION="cleanup"
            shift
            ;;
        --instance)
            INSTANCE_NAME="$2"
            shift 2
            ;;
        --project)
            PROJECT_ID="$2"
            shift 2
            ;;
        --zone)
            ZONE="$2"
            shift 2
            ;;
        --machine-type)
            MACHINE_TYPE="$2"
            shift 2
            ;;
        --gpu)
            GPU_TYPE="nvidia-tesla-$2"
            shift 2
            ;;
        --disk-size)
            DISK_SIZE="$2"
            shift 2
            ;;
        --kaggle-json)
            KAGGLE_JSON="$2"
            shift 2
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

echo "GCP Pipeline Configuration:"
echo "  Action: $ACTION"
echo "  Instance: $INSTANCE_NAME"
echo "  Project: $PROJECT_ID"
echo "  Zone: $ZONE"
echo "  Machine: $MACHINE_TYPE + 1× $GPU_TYPE"
echo "  Disk: ${DISK_SIZE}GB"

# Function: Create GCE instance
create_instance() {
    echo "Creating GCE instance..."
    gcloud compute instances create "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --boot-disk-size="${DISK_SIZE}GB" \
        --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
        --image-family="$IMAGE_FAMILY" \
        --image-project="debian-cloud" \
        --maintenance-policy="TERMINATE" \
        --scopes="https://www.googleapis.com/auth/cloud-platform" \
        --tags="nyt-pipeline"

    echo "✓ Instance created: $INSTANCE_NAME"
    echo "Waiting for instance to be ready..."
    sleep 10

    # Install NVIDIA drivers
    echo "Installing NVIDIA drivers and CUDA toolkit..."
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command="curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.sh | bash"

    echo "✓ NVIDIA drivers installed"
}

# Function: Run pipeline on instance
run_pipeline() {
    echo "Setting up and running pipeline on instance..."

    # Copy source code to instance
    echo "Uploading project files..."
    gcloud compute scp \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --recurse \
        "." "$INSTANCE_NAME:~/nyt-project/" \
        --exclude=".git,__pycache__,venv,data,.pytest_cache"

    # Install dependencies and run pipeline
    echo "Running pipeline on instance..."
    gcloud compute ssh "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --command='
        set -e
        cd ~/nyt-project

        # Install Python 3.10 and pip
        echo "Installing Python..."
        sudo apt-get update
        sudo apt-get install -y python3.10 python3.10-venv python3-pip

        # Create virtual environment
        python3.10 -m venv venv
        source venv/bin/activate

        # Install requirements
        echo "Installing Python dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt

        # Download Kaggle dataset (assumes KAGGLE_API_KEY in environment)
        echo "Downloading Kaggle NYT dataset..."
        python -c "import kagglehub; kagglehub.dataset_download(\"thepushkar/nyt-articles\")"

        # Run preprocessing
        echo "Preprocessing 21M articles (chunked)..."
        python scripts/preprocess_21m.py \
            --input ~/kaggle_dataset/nyt_articles.csv \
            --output data/preprocessed_21m.parquet \
            --chunk-size 500000

        # Generate embeddings and FAISS index
        echo "Generating embeddings and building FAISS index..."
        python scripts/build_embeddings_21m.py \
            --input data/preprocessed_21m.parquet \
            --output-dir data \
            --batch-size 128 \
            --chunk-size 100000 \
            --gpu

        echo "✓ Pipeline complete!"
        ls -lh data/
        '

    # Download results from instance
    echo "Downloading results from instance..."
    gcloud compute scp \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --recurse \
        "$INSTANCE_NAME:~/nyt-project/data/" \
        "./data_from_gcp/" \

    echo "✓ Pipeline complete. Results in ./data_from_gcp/"
}

# Function: Cleanup instance
cleanup_instance() {
    echo "Cleaning up instance: $INSTANCE_NAME"
    gcloud compute instances delete "$INSTANCE_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --quiet

    echo "✓ Instance deleted"
}

# Execute action
case "$ACTION" in
    create)
        create_instance
        echo ""
        echo "Next, run: bash gcp_pipeline.sh --run --instance $INSTANCE_NAME --project $PROJECT_ID --zone $ZONE"
        ;;
    run)
        run_pipeline
        echo ""
        echo "Next, clean up with: bash gcp_pipeline.sh --cleanup --instance $INSTANCE_NAME --project $PROJECT_ID --zone $ZONE"
        ;;
    cleanup)
        cleanup_instance
        ;;
    *)
        echo "Unknown action: $ACTION"
        exit 1
        ;;
esac

echo "Done!"
