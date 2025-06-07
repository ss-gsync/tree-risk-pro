# ML Deployment Guide for Tree ML v0.2.3

This document provides detailed instructions for deploying the Tree ML platform with GPU support for ML inference.

## Overview

The v0.2.3 release introduces significant architectural improvements with two deployment options:

1. **Unified Deployment (Recommended)**: Run both dashboard and ML models on a single GPU instance
2. **Split Deployment (Advanced)**: Offload ML tasks to a dedicated T4 GPU server

### Unified Deployment Benefits

- Simplified setup and configuration
- No network latency between components
- Direct access to ML models from dashboard
- Easier maintenance and troubleshooting
- Immediate overlay display and better UX

### T4 Split Deployment Benefits

- Offloading intensive ML tasks to a dedicated GPU server
- Improved inference performance (up to 20x faster)
- Separation of concerns between dashboard and ML processing
- Flexible scaling based on computational needs

## Architecture

```
+---------------------+      HTTP/REST      +---------------------+
|                     |    Detection API    |                     |
|  Dashboard Server   +-------------------->+  T4 Model Server    |
|  (Compute Instance) |                     |  (GPU Instance)     |
|                     |<--------------------+                     |
+---------------------+       Results       +---------------------+
```

## 1. T4 GPU Server Setup

### 1.1. Hardware Requirements

- NVIDIA T4 GPU (or equivalent)
- Minimum 16GB RAM
- At least 100GB storage
- Ubuntu 20.04 LTS or later

### 1.2. CUDA Installation

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-11-8
```

### 1.3. System Dependencies

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-pip
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 nginx

# Install Python package manager
pip install --upgrade pip
pip install poetry
```

### 1.4. Clone Repository

```bash
# Clone repo (adjust URL as needed)
git clone https://github.com/yourusername/tree-ml.git /opt/tree_ml
cd /opt/tree_ml
git checkout main  # Ensure you're on v0.2.3 or later
```

### 1.5. Install Dependencies

```bash
# Install Python dependencies
cd /opt/tree_ml
poetry config virtualenvs.in-project true
poetry install

# Install Pytorch with CUDA support (if not handled by poetry)
poetry run pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 1.6. Download Model Weights

```bash
# Create model directories
mkdir -p /opt/tree_ml/tree_ml/pipeline/model/weights

# Download SAM model weights
wget -O /opt/tree_ml/tree_ml/pipeline/model/weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download DeepForest model weights (if not already included)
# Use the included deepforest weights or download as needed
```

### 1.7. Run Deployment Script

```bash
# Make script executable
chmod +x /opt/tree_ml/tree_ml/pipeline/deploy_t4.sh

# Run deployment script
cd /opt/tree_ml/tree_ml/pipeline
sudo ./deploy_t4.sh
```

The deployment script performs the following actions:
- Creates systemd service for model server
- Configures environment variables
- Sets up logging
- Installs and configures nginx as reverse proxy
- Sets appropriate permissions
- Starts the service

### 1.8. Verify Installation

```bash
# Check service status
sudo systemctl status tree-detection

# View logs
sudo journalctl -u tree-detection -n 100

# Test API endpoint
curl http://localhost:8000/health
```

Expected output:
```json
{
  "status": "ok",
  "models_loaded": true,
  "cuda_available": true,
  "device": "cuda:0",
  "memory_used_gb": 2.1,
  "uptime_seconds": 120
}
```

## 2. Dashboard Server Configuration

### 2.1. Update Environment Configuration

```bash
# Navigate to backend directory
cd /path/to/tree_ml/tree_ml/dashboard/backend

# Create or edit .env file
nano .env
```

Add or modify these lines:
```
USE_EXTERNAL_MODEL_SERVER=true
MODEL_SERVER_URL=http://<T4-INSTANCE-IP>:8000
```

### 2.2. Restart Services

```bash
# If using systemd
sudo systemctl restart tree-dashboard

# If running manually
cd /path/to/tree_ml/tree_ml/dashboard/backend
poetry run python app.py
```

## 3. Detailed Component Overview

### 3.1. Model Server (`model_server.py`)

The model server is a FastAPI application that:
- Loads ML models (DeepForest, SAM) on startup
- Exposes REST API endpoints for detection
- Handles image processing and inference
- Returns detection results in a standardized format

Key endpoints:
- `/health`: Status check and GPU stats
- `/detect`: Primary endpoint for tree detection
- `/models`: Information about loaded models

### 3.2. External Model Service Client (`external_model_service.py`)

This client:
- Connects dashboard to T4 model server
- Maintains API compatibility with local model service
- Handles error conditions gracefully
- Implements health checking and connection management

### 3.3. Service Selection Logic (`__init__.py`)

The unified service selection logic:
- Determines whether to use local or external models
- Maintains backward compatibility
- Provides a consistent interface for all detection services

## 4. Troubleshooting

### 4.1. Model Server Issues

#### CUDA Not Available

If `nvidia-smi` works but CUDA isn't detected in Python:

```bash
# Check CUDA environment
echo $LD_LIBRARY_PATH

# Add CUDA to path if missing
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Model Loading Failures

```bash
# Check model paths
ls -la /opt/tree_ml/tree_ml/pipeline/model/weights/

# Check permissions
sudo chown -R youruser:youruser /opt/tree_ml/tree_ml/pipeline/model/
```

#### Memory Issues

If experiencing out-of-memory errors:

```bash
# Check GPU memory usage
nvidia-smi

# Adjust batch size in model_server.py
# Look for BATCH_SIZE parameter
```

### 4.2. Communication Issues

#### Connection Refused

```bash
# Check if model server is running
sudo systemctl status tree-detection

# Check firewall settings
sudo ufw status
sudo ufw allow 8000/tcp

# Test connection
curl http://localhost:8000/health
```

#### Slow Response Times

```bash
# Check network latency
ping <T4-INSTANCE-IP>

# Check model server load
htop
nvidia-smi -l 1
```

### 4.3. Dashboard Integration Issues

If detection preview doesn't update:

```bash
# Check browser console for errors

# Check backend logs
tail -f /path/to/tree_ml/tree_ml/dashboard/backend/logs/app.log

# Verify model selection
grep -r "get_unified_model_service" /path/to/tree_ml/tree_ml/dashboard/backend/
```

## 5. Performance Optimization

### 5.1. Model Quantization

For improved inference speed:

```python
# In model_server.py
# Find model loading section and add quantization
model = model.half()  # Convert to FP16
```

### 5.2. Batch Processing

For multiple images:

```bash
# Adjust max batch size in model_server.py
# Look for MAX_BATCH_SIZE constant
```

### 5.3. Nginx Configuration

For better throughput, edit `/etc/nginx/sites-available/tree-ml`:

```nginx
# Increase max body size for large images
client_max_body_size 20M;

# Add caching for repeated requests
proxy_cache_path /var/cache/nginx/tree_ml levels=1:2 keys_zone=tree_ml:10m max_size=1g inactive=60m;
```

## 6. Monitoring and Maintenance

### 6.1. Log Rotation

```bash
# Set up log rotation
sudo nano /etc/logrotate.d/tree-ml

# Add configuration
/opt/tree_ml/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
}
```

### 6.2. Service Monitoring

```bash
# Check service health
systemctl status tree-detection

# Set up simple monitoring
echo '*/5 * * * * curl -s http://localhost:8000/health | grep -q "models_loaded\":true" || systemctl restart tree-detection' | sudo tee /etc/cron.d/check-tree-detection
```

### 6.3. Updating Models

To update model weights:

1. Place new weights in `/opt/tree_ml/tree_ml/pipeline/model/weights/`
2. Update configuration in `model_server.py` if needed
3. Restart the service: `sudo systemctl restart tree-detection`

## 7. Security Considerations

### 7.1. API Access Control

By default, the model server has no authentication. For production:

1. Add API key authentication:
   ```python
   # In model_server.py
   # Add header verification
   api_key = request.headers.get("X-API-Key")
   if api_key != os.environ.get("MODEL_SERVER_API_KEY"):
       raise HTTPException(status_code=403, detail="Invalid API key")
   ```

2. Set environment variable on both servers:
   ```bash
   # Add to .env files
   MODEL_SERVER_API_KEY=your-secure-random-key
   ```

### 7.2. HTTPS Configuration

For secure communication:

```bash
# Generate certificates
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/tree-ml.key -out /etc/ssl/certs/tree-ml.crt

# Configure nginx with SSL
# Edit /etc/nginx/sites-available/tree-ml
```

Add to nginx config:
```nginx
server {
    listen 443 ssl;
    server_name your-server-name;
    
    ssl_certificate /etc/ssl/certs/tree-ml.crt;
    ssl_certificate_key /etc/ssl/private/tree-ml.key;
    
    # Rest of config...
}
```

## 8. Testing

### 8.1. Integration Test

```bash
# Run integration test
cd /opt/tree_ml/tests/model_server
chmod +x test_external_model_service.py
./test_external_model_service.py --image /opt/tree_ml/data/tests/test_images/sample.jpg --server http://localhost:8000
```

### 8.2. Load Testing

```bash
# Install wrk for HTTP benchmarking
sudo apt install -y wrk

# Run load test (adjust concurrency as needed)
wrk -t4 -c10 -d30s -s post_image.lua http://localhost:8000/detect
```

Create `post_image.lua`:
```lua
wrk.method = "POST"
wrk.body = "file=@/path/to/test/image.jpg"
wrk.headers["Content-Type"] = "multipart/form-data"
```

## 9. References

- [NVIDIA T4 Documentation](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [DeepForest Documentation](https://deepforest.readthedocs.io/)
- [Segment Anything Model (SAM)](https://segment-anything.com/)

## 10. Appendix: Full Deployment Script

Below is a reference implementation of the `deploy_t4.sh` script:

```bash
#!/bin/bash
set -e

# Configuration
MODEL_SERVER_PORT=8000
SERVICE_NAME="tree-detection"
SERVICE_USER=$(whoami)
REPO_PATH=$(pwd)
LOG_DIR="${REPO_PATH}/logs"
MODEL_DIR="${REPO_PATH}/tree_ml/pipeline/model/weights"

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $MODEL_DIR

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y nginx

echo "Checking CUDA availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA drivers not found. Please install CUDA first."
    exit 1
fi

echo "Checking for model weights..."
if [ ! -f "${MODEL_DIR}/sam_vit_h_4b8939.pth" ]; then
    echo "SAM model weights not found. Downloading..."
    wget -O "${MODEL_DIR}/sam_vit_h_4b8939.pth" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

echo "Creating systemd service..."
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee $SERVICE_FILE > /dev/null << EOF
[Unit]
Description=Tree ML Detection Model Server
After=network.target

[Service]
User=${SERVICE_USER}
WorkingDirectory=${REPO_PATH}
ExecStart=${REPO_PATH}/.venv/bin/python ${REPO_PATH}/tree_ml/pipeline/run_model_server.sh
Restart=on-failure
Environment=PYTHONPATH=${REPO_PATH}
Environment=MODEL_DIR=${MODEL_DIR}
Environment=PORT=${MODEL_SERVER_PORT}
Environment=LOG_DIR=${LOG_DIR}

[Install]
WantedBy=multi-user.target
EOF

echo "Configuring nginx as reverse proxy..."
NGINX_CONF="/etc/nginx/sites-available/${SERVICE_NAME}"

sudo tee $NGINX_CONF > /dev/null << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:${MODEL_SERVER_PORT};
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 180s;
        proxy_send_timeout 180s;
        proxy_read_timeout 180s;
        
        # Buffer settings for large responses
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
        
        # Max body size for image uploads
        client_max_body_size 10M;
    }
}
EOF

# Enable the nginx configuration
sudo ln -sf $NGINX_CONF /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

echo "Starting model server service..."
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME
sudo systemctl start $SERVICE_NAME

echo "Waiting for model server to initialize..."
sleep 10

echo "Checking model server health..."
curl -s http://localhost:${MODEL_SERVER_PORT}/health || {
    echo "ERROR: Model server is not responding. Check logs:"
    echo "sudo journalctl -u ${SERVICE_NAME} -n 50"
    exit 1
}

echo "=================================================="
echo "T4 Model Server Deployment Complete!"
echo "=================================================="
echo "Service: ${SERVICE_NAME}"
echo "Status: $(systemctl is-active ${SERVICE_NAME})"
echo "API URL: http://localhost:${MODEL_SERVER_PORT}"
echo ""
echo "To check service status: sudo systemctl status ${SERVICE_NAME}"
echo "To view logs: sudo journalctl -u ${SERVICE_NAME} -f"
echo "To test API: curl http://localhost:${MODEL_SERVER_PORT}/health"
echo "=================================================="
```

---

For any issues not covered in this document, please consult the source code or contact the development team.