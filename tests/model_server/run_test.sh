#!/bin/bash
# Run model server integration test

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="/ttt"
SAMPLE_IMAGE="/ttt/data/tests/test_images/sample.jpg"
MODEL_SERVER_URL=${1:-"http://localhost:8000"}

echo "================================================================="
echo "T4 Model Server Integration Test"
echo "================================================================="
echo "Using model server: $MODEL_SERVER_URL"
echo "Using sample image: $SAMPLE_IMAGE"
echo

# Make sure Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Please install Poetry first."
    exit 1
fi

# Run the test with Poetry
echo "Running test with Poetry..."
cd "$ROOT_DIR"
poetry run python "$SCRIPT_DIR/test_external_model_service.py" --image "$SAMPLE_IMAGE" --server "$MODEL_SERVER_URL"

# Check the result
if [ $? -eq 0 ]; then
    echo
    echo "================================================================="
    echo "✅ Test completed successfully!"
    echo "================================================================="
    exit 0
else
    echo
    echo "================================================================="
    echo "❌ Test failed!"
    echo "================================================================="
    exit 1
fi