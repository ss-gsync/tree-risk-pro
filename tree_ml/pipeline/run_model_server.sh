#!/bin/bash
# T4 Model Server Launch Script

# Activate the Python virtual environment
source /home/ss/tree_ml/bin/activate

# Set the environment
export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

# Set LD_LIBRARY_PATH to include PyTorch libraries
if python -c "import torch" &>/dev/null; then
    PYTORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
    if [ -d "$PYTORCH_PATH/lib" ]; then
        export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:$LD_LIBRARY_PATH
        echo "Added PyTorch libraries to LD_LIBRARY_PATH: $PYTORCH_PATH/lib"
    fi
fi

# Verify CUDA setup
if [ -x "$(command -v nvidia-smi)" ]; then
    nvidia-smi
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=0
else
    echo "WARNING: CUDA not found! Running in CPU mode only."
fi

# Make sure we have NumPy 1.x (PyTorch requirement)
if python -c "import numpy; v=numpy.__version__; exit(0 if int(v.split('.')[0]) <= 1 else 1)" &>/dev/null; then
    echo "NumPy version is compatible with PyTorch"
else
    echo "WARNING: NumPy version may be incompatible with PyTorch. Installing NumPy 1.26.4..."
    pip install numpy==1.26.4
fi

# Build GroundingDINO extension if needed
if ! python -c "from groundingdino import _C" &>/dev/null; then
    echo "Building GroundingDINO CUDA extension..."
    cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
    # Ensure directory structure is correct
    mkdir -p groundingdino
    touch groundingdino/__init__.py
    python setup.py build develop
    cd /ttt/tree_ml
fi

# Create log directory
mkdir -p /ttt/tree_ml/logs
mkdir -p /ttt/data/ml

# Start the model server
echo "Starting Model Server..."
python /ttt/tree_ml/pipeline/model_server.py --port 8000 --host 0.0.0.0 \
    --model-dir /ttt/tree_ml/pipeline/model \
    --output-dir /ttt/data/ml \
    --device cuda