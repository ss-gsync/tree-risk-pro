#!/bin/bash
# T4 Model Server Launch Script
# Runs the tree detection model server using the Grounded-SAM model

# Set the environment
export PYTHONPATH=/ttt/tree_ml:$PYTHONPATH

# Create log directory
mkdir -p /ttt/tree_ml/logs

# Check if CUDA is available
if [ -x "$(command -v nvidia-smi)" ]; then
    echo "CUDA is available, using GPU for model inference"
    DEVICE="cuda"
    
    # Log CUDA info
    echo "CUDA Information:"
    nvidia-smi
else
    echo "CUDA is not available, using CPU for model inference"
    DEVICE="cpu"
fi

# Define paths using the local directory structure
MODEL_DIR="/ttt/tree_ml/pipeline/model"
OUTPUT_DIR="/ttt/data/ml"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "Starting Tree Detection Model Server"
echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"

# Activate virtual environment if it exists
if [ -d "/ttt/tree_ml/venv" ]; then
    echo "Activating virtual environment"
    source /ttt/tree_ml/venv/bin/activate
fi

# Start the server
python /ttt/tree_ml/pipeline/model_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model-dir $MODEL_DIR \
    --output-dir $OUTPUT_DIR \
    --device $DEVICE \
    > /ttt/tree_ml/logs/model_server.log 2>&1