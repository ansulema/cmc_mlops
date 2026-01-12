#!/bin/bash
# Build .mar archive for TorchServe

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_DIR/models/spam_classifier"
OUTPUT_DIR="$SCRIPT_DIR/model-store"

echo "=== Building TorchServe MAR archive ==="

# Check model exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model not found at $MODEL_DIR"
    echo "Run 'dvc repro' first to train the model"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Install torch-model-archiver if not installed
pip show torch-model-archiver > /dev/null 2>&1 || pip install torch-model-archiver

# Create MAR archive
echo "Creating MAR archive..."
torch-model-archiver \
    --model-name spam_classifier \
    --version 1.0 \
    --handler "$SCRIPT_DIR/handler.py" \
    --extra-files "$MODEL_DIR" \
    --export-path "$OUTPUT_DIR" \
    --force

echo "=== MAR archive created: $OUTPUT_DIR/spam_classifier.mar ==="
echo ""
echo "To build Docker image:"
echo "  cd $SCRIPT_DIR && docker build -t spam-serve:v1 ."
echo ""
echo "To run container:"
echo "  docker run -d -p 8080:8080 -p 8081:8081 spam-serve:v1"
echo ""
echo "To test:"
echo '  curl -X POST http://localhost:8080/predictions/spam_classifier -H "Content-Type: application/json" -d '\''{"text": "Hello world"}'\'''
