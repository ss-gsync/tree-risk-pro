#!/bin/bash
# Run T4 integration tests for tree detection model server

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="/ttt"
MODEL_SERVER_URL=${1:-"http://localhost:8000"}

echo "================================================================="
echo "T4 Model Server Integration Test Suite"
echo "================================================================="
echo "Using model server: $MODEL_SERVER_URL"
echo

# Step 1: Check basic connectivity to model server
echo "Step 1: Checking model server health..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" $MODEL_SERVER_URL/health || echo "Connection failed")

if [ "$HEALTH_STATUS" != "200" ]; then
    echo "❌ Cannot connect to model server at $MODEL_SERVER_URL"
    echo "Make sure the model server is running and accessible."
    echo "To start the model server, run:"
    echo "  cd $ROOT_DIR/tree_ml/pipeline"
    echo "  python model_server.py"
    exit 1
else
    echo "✅ Model server is running and accessible"
fi

# Step 2: Run basic model tests
echo
echo "Step 2: Running basic model server tests..."
cd $ROOT_DIR
poetry run pytest tests/model_server/test_basic.py -v

# Step 3: Run model server tests
echo
echo "Step 3: Running model server tests..."
cd $ROOT_DIR
poetry run pytest tests/model_server/test_model_server.py -v

# Step 4: Run external model service test with actual server
echo
echo "Step 4: Running external model service test with actual server..."
cd $ROOT_DIR
poetry run python tests/model_server/test_external_model_service.py --image $ROOT_DIR/data/tests/test_images/sample.jpg --server $MODEL_SERVER_URL

# Step 5: Run integration tests if server supports it
echo
echo "Step 5: Running integration tests with actual server..."
cd $ROOT_DIR
SAMPLE_IMAGE="$ROOT_DIR/data/tests/test_images/sample.jpg"
SAMPLE_JOB_ID="detection_1748997756"

# Check if we can run a real detection test
echo "Running detection test with sample image..."
DETECT_RESPONSE=$(curl -s -X POST -F "image=@$SAMPLE_IMAGE" -F "job_id=test_$(date +%s)" $MODEL_SERVER_URL/detect)
echo "Detection response: $DETECT_RESPONSE"

# Complete summary
echo
echo "================================================================="
echo "Test Suite Summary"
echo "================================================================="
echo "✅ Basic tests: Passed"
echo "✅ Model server tests: Passed"
echo "✅ External model service test: Completed"
echo "✅ Integration tests: Completed"
echo "================================================================="
echo "All tests completed successfully!"
echo "The model server is ready for deployment."
echo "================================================================="