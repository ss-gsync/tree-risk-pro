#!/bin/bash
# T4 Model Server Launch Script

# Set the environment
export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

# Verify CUDA setup
if [ -x "$(command -v nvidia-smi)" ]; then
    nvidia-smi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0
else
    echo "WARNING: CUDA not found! Running in CPU mode only."
fi

# Check if CUDA extension exists, build if needed
CUDA_EXT_PATH="/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn"
if [ ! -f "$CUDA_EXT_PATH/_C.so" ] && [ -x "$(command -v nvidia-smi)" ]; then
    echo "Building CUDA extension for GroundingDINO..."
    cd $CUDA_EXT_PATH
    python setup.py build install
    cd /ttt/tree_ml
fi

# Create log directory
mkdir -p /ttt/tree_ml/logs

# Start the model server
echo "Starting Model Server..."
python /ttt/tree_ml/pipeline/model_server.py --port 8000 --host 0.0.0.0 \
    --model-dir /ttt/tree_ml/pipeline/model \
    --output-dir /ttt/data/ml \
    --device cuda