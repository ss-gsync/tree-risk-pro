#!/bin/bash
# T4 Deployment Script
# This script sets up the T4 instance with all required dependencies and starts the model server

set -e

echo "============================================================"
echo "Tree Detection T4 Deployment Script"
echo "============================================================"

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root" 
    exit 1
fi

# Environment setup - use the existing directory structure
BASE_DIR="/ttt"
CODE_DIR="$BASE_DIR/tree_ml"
MODEL_DIR="$CODE_DIR/pipeline/model"
DATA_DIR="$BASE_DIR/data/ml"
LOG_DIR="$CODE_DIR/logs"
SERVICE_NAME="tree-detection"

echo "Setting up directories..."
mkdir -p $LOG_DIR $DATA_DIR

# Update system
echo "Updating system packages..."
apt-get update
apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    nginx \
    unzip

# Install CUDA if not already installed
if [ ! -x "$(command -v nvidia-smi)" ]; then
    echo "Installing CUDA..."
    # Download and install CUDA following NVIDIA instructions
    # This is a simplified version - in production, follow NVIDIA's official guide
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
    source /etc/profile.d/cuda.sh
fi

# Create Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv $CODE_DIR/venv
source $CODE_DIR/venv/bin/activate

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required packages
echo "Installing required Python packages..."
pip install \
    numpy \
    opencv-python \
    pillow \
    fastapi \
    uvicorn \
    python-multipart \
    aiofiles \
    pycocotools \
    matplotlib \
    pyproj \
    scipy

# Ensure the model repos are available in the existing directory structure
echo "Setting up model repositories..."

# GroundingDINO
cd $CODE_DIR/pipeline
if [ ! -d "grounded-sam/GroundingDINO" ]; then
    echo "GroundingDINO not found in the expected location. Initializing submodules..."
    cd $CODE_DIR/pipeline/grounded-sam
    git submodule update --init --recursive
    cd GroundingDINO
    pip install -e .
    cd ../../
else
    echo "GroundingDINO found, installing as development package..."
    cd $CODE_DIR/pipeline/grounded-sam/GroundingDINO
    pip install -e .
    cd ../../
fi

# Segment Anything
if [ ! -d "grounded-sam/segment_anything" ]; then
    echo "Segment Anything not found in the expected location. Initializing submodules..."
    cd $CODE_DIR/pipeline/grounded-sam
    if [ ! -d "segment_anything" ]; then
        git clone https://github.com/facebookresearch/segment-anything.git segment_anything
    fi
    cd segment_anything
    pip install -e .
    cd ../../
else
    echo "Segment Anything found, installing as development package..."
    cd $CODE_DIR/pipeline/grounded-sam/segment_anything
    pip install -e .
    cd ../../
fi

# Check if model weights exist and download if needed
echo "Checking model weights..."

# GroundingDINO weights
if [ ! -f "$MODEL_DIR/groundingdino_swint_ogc.pth" ]; then
    echo "Downloading GroundingDINO weights..."
    mkdir -p $MODEL_DIR
    wget -O $MODEL_DIR/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
else
    echo "GroundingDINO weights already exist at $MODEL_DIR/groundingdino_swint_ogc.pth"
fi

# SAM weights
if [ ! -f "$MODEL_DIR/sam_vit_h_4b8939.pth" ]; then
    echo "Downloading SAM weights..."
    mkdir -p $MODEL_DIR
    wget -O $MODEL_DIR/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo "SAM weights already exist at $MODEL_DIR/sam_vit_h_4b8939.pth"
fi

# Make run_model_server.sh executable
chmod +x $CODE_DIR/pipeline/run_model_server.sh

# Set up systemd service
echo "Setting up systemd service..."
cp $CODE_DIR/pipeline/tree-detection.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable tree-detection

# Set up Nginx as reverse proxy
echo "Setting up Nginx as reverse proxy..."
cat > /etc/nginx/sites-available/tree-detection << EOL
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeout settings
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Serve model results directly
    location /data/ {
        alias $DATA_DIR/;
        autoindex on;
    }
    
    # Larger file upload support
    client_max_body_size 100M;
}
EOL

# Enable the site and restart Nginx
ln -sf /etc/nginx/sites-available/tree-detection /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
systemctl restart nginx

# Start the service
echo "Starting Tree Detection service..."
systemctl start tree-detection

echo "============================================================"
echo "Tree Detection deployment complete!"
echo "Model server is running at: http://localhost:8000"
echo "Nginx proxy is running at: http://localhost:80"
echo "============================================================"
echo "To check service status: systemctl status tree-detection"
echo "To view logs: journalctl -u tree-detection -f"
echo "============================================================"