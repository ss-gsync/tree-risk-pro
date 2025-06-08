# Tree ML - Deployment Guide (v0.2.3)

> This is the deployment guide for the Tree ML project, including the dashboard, backend components, and ML model server.

This guide provides instructions for deploying the Tree ML system on a Google Cloud Platform (GCP) T4 GPU instance for unified deployment.

## Deployment Summary

Our production server will be deployed on a T4 GPU instance for unified ML and dashboard operation.

**Current access credentials:**
- Username: `TestAdmin`
- Password: `trp345!`

## v0.2.3 Release Deployment Notes

v0.2.3 includes these key improvements requiring deployment attention:
- Integrated ML Overlay for tree detection visualization
- Enhanced tree detection with DeepForest, SAM, and Gemini API
- T4 GPU integration for dedicated model server capabilities
- Unified deployment architecture (dashboard + ML on same instance)
- Fixed overlay controls (opacity slider, show/hide toggle)
- Improved objects counter that preserves counts between detections
- Fixed detection sidebar with real-time controls
- New T4 status indicator component

## Prerequisites

### Hardware Requirements
- T4 GPU instance (GCP n1-standard-4 with NVIDIA T4)
- At least 4 vCPU, 16 GB memory
- 100+ GB SSD storage
- Ubuntu 20.04 LTS or later

### Software Requirements
- CUDA 11.8+
- Python 3.10+
- Node.js 18+
- Git
- Nginx
- Poetry for Python dependency management

## Required API Keys

**IMPORTANT: You must obtain these API keys before deployment:**

1. **Google Maps API Key**: 
   - Go to https://console.cloud.google.com/
   - Navigate to Google Maps Platform > Credentials
   - Create an API key with Maps JavaScript API access

2. **Google Maps Map ID**:
   - Go to Google Maps Platform > Map Management
   - Create a new Map ID (or use existing one: e2f2ee8160dbcea0)

3. **Gemini API Key**:
   - Go to https://aistudio.google.com/app/apikey
   - Create a new API key for Gemini model access
   - Ensure your account has access to gemini-2.0-flash model

## T4 GPU Manual Deployment

This section provides a detailed, step-by-step guide for manually deploying the application on a T4 GPU instance, ensuring the proper configuration of all components, particularly the Grounded-SAM model.

### T4 GPU Setup Steps

1. **Verify CUDA Installation**:
   ```bash
   # Check if NVIDIA drivers are installed
   nvidia-smi
   
   # Check CUDA compiler version
   nvcc --version
   
   # Set CUDA environment variables
   echo 'export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit' >> ~/.bashrc
   echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Create Python Environment**:
   ```bash
   # Create a virtual environment
   python3 -m venv ~/tree_ml
   source ~/tree_ml/bin/activate
   
   # Install the correct PyTorch version for your CUDA
   # For CUDA 12.x (newer GCP instances):
   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8:
   # pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
   
   # Verify PyTorch CUDA detection
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"
   ```

3. **Clone Repository and Install Dependencies**:
   ```bash
   # Clone the repository
   git clone https://github.com/your-repo/tree-ml.git /ttt/tree_ml
   cd /ttt/tree_ml
   
   # Install Python dependencies
   pip install poetry
   poetry config virtualenvs.in-project true
   poetry install
   
   # Install additional dependencies
   pip install numpy opencv-python matplotlib timm tensorboard transformers pycocotools addict
   ```

4. **Set Up Grounded-SAM Directories and Download Weights**:
   ```bash
   # Create required directories
   mkdir -p /ttt/tree_ml/pipeline/model
   
   # Clone Grounded-Segment-Anything repository
   cd /ttt/tree_ml/pipeline
   git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git grounded-sam
   
   # Download SAM model weights
   wget -O /ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   
   # Download GroundingDINO weights
   wget -O /ttt/tree_ml/pipeline/model/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   ```

5. **Configure Directory Structure and Install Components**:
   ```bash
   # Set PYTHONPATH to include all necessary directories
   export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH
   
   # Ensure the config directory structure is correct
   mkdir -p /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO/groundingdino/config/
   cp /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO/config/GroundingDINO_SwinT_OGC.py /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO/groundingdino/config/
   
   # Install segment_anything
   cd /ttt/tree_ml/pipeline/grounded-sam/segment_anything
   pip install -e .
   
   # Install GroundingDINO with no-build-isolation flag
   cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
   pip install --no-build-isolation -e .
   ```

6. **Build CUDA Extensions**:
   ```bash
   # Set CUDA_HOME and build the extension
   export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
   cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn
   python setup.py build install
   
   # Verify the extension was built successfully
   cd /ttt/tree_ml
   python -c "from groundingdino.models.GroundingDINO.ms_deform_attn import _C; print('CUDA extension loaded successfully')"
   ```

7. **Create Systemd Service**:
   ```bash
   # Create service file
   sudo tee /etc/systemd/system/tree-detection.service > /dev/null << EOL
   [Unit]
   Description=Tree Detection Model Server
   After=network.target
   
   [Service]
   User=root
   WorkingDirectory=/ttt/tree_ml
   ExecStart=/bin/bash /ttt/tree_ml/pipeline/run_model_server.sh
   Restart=always
   RestartSec=10
   Environment=PYTHONUNBUFFERED=1
   Environment=PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything
   Environment=CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
   
   # Limit resource usage
   CPUWeight=90
   IOWeight=90
   MemoryHigh=8G
   MemoryMax=12G
   
   # Security settings
   ProtectSystem=full
   PrivateTmp=true
   NoNewPrivileges=true
   
   [Install]
   WantedBy=multi-user.target
   EOL
   
   # Enable and start the service
   sudo systemctl daemon-reload
   sudo systemctl enable tree-detection
   sudo systemctl start tree-detection
   ```

8. **Configure run_model_server.sh**:
   ```bash
   # Create or edit the run script
   cat > /ttt/tree_ml/pipeline/run_model_server.sh << EOL
   #!/bin/bash
   # T4 Model Server Launch Script
   
   # Set the environment
   export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:\$PYTHONPATH
   export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
   
   # Verify CUDA setup
   if [ -x "\$(command -v nvidia-smi)" ]; then
       nvidia-smi
       export CUDA_DEVICE_ORDER=PCI_BUS_ID
       export CUDA_VISIBLE_DEVICES=0
   else
       echo "WARNING: CUDA not found! Running in CPU mode only."
   fi
   
   # Check if CUDA extension exists, build if needed
   CUDA_EXT_PATH="/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn"
   if [ ! -f "\$CUDA_EXT_PATH/_C.so" ] && [ -x "\$(command -v nvidia-smi)" ]; then
       echo "Building CUDA extension for GroundingDINO..."
       cd \$CUDA_EXT_PATH
       python setup.py build install
       cd /ttt/tree_ml
   fi
   
   # Create log directory
   mkdir -p /ttt/tree_ml/logs
   
   # Start the model server
   echo "Starting Model Server..."
   python /ttt/tree_ml/pipeline/model_server.py --port 8000 --host 0.0.0.0 \\
       --model-dir /ttt/tree_ml/pipeline/model \\
       --output-dir /ttt/data/ml \\
       --device cuda
   EOL
   
   # Make the script executable
   chmod +x /ttt/tree_ml/pipeline/run_model_server.sh
   ```

9. **Configure Nginx for the Dashboard**:
   ```bash
   # Create Nginx configuration
   sudo tee /etc/nginx/sites-available/tree-ml.conf > /dev/null << EOL
   server {
       listen 80;
       server_name _;
       
       # Frontend static files
       location / {
           root /ttt/tree_ml/dashboard/dist;
           try_files \$uri \$uri/ /index.html;
       }
       
       # Backend API
       location /api {
           proxy_pass http://localhost:5000;
           proxy_set_header Host \$host;
           proxy_set_header X-Real-IP \$remote_addr;
       }
       
       # Model Server API
       location /model-api/ {
           proxy_pass http://localhost:8000/;
           proxy_set_header Host \$host;
           proxy_set_header X-Real-IP \$remote_addr;
           proxy_read_timeout 300;
           client_max_body_size 20M;
       }
   }
   EOL
   
   # Enable the site
   sudo ln -sf /etc/nginx/sites-available/tree-ml.conf /etc/nginx/sites-enabled/
   sudo rm -f /etc/nginx/sites-enabled/default
   sudo nginx -t
   sudo systemctl restart nginx
   ```

10. **Verify Installation**:
    ```bash
    # Check service status
    sudo systemctl status tree-detection
    
    # View logs
    sudo journalctl -u tree-detection -n 100
    
    # Test API endpoint
    curl http://localhost:8000/health
    ```

### Common T4 Deployment Errors and Solutions

#### Error: "name '_C' is not defined"

This error occurs when the CUDA extensions for GroundingDINO haven't been properly built or can't be found.

**Solution**:
```bash
# Set CUDA_HOME to the correct location
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

# Make sure PYTHONPATH includes all required directories
export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH

# Build the MS Deform Attention CUDA extension
cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn
python setup.py build install

# Verify the extension was built successfully
python -c "from groundingdino.models.GroundingDINO.ms_deform_attn import _C; print('CUDA extension loaded successfully')"

# Restart the service
sudo systemctl restart tree-detection
```

#### Error: "Numpy is not available"

This indicates an issue with the Python environment where numpy isn't accessible.

**Solution**:
```bash
# Make sure your virtual environment is activated
source ~/tree_ml/bin/activate

# Reinstall numpy
pip install -U numpy

# Check that numpy is installed and accessible
python -c "import numpy; print(numpy.__version__)"

# Make sure model_server.py can find numpy
export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH
```

#### Error: PyTorch CUDA version mismatch

If you get PyTorch CUDA version compatibility errors:

```bash
# For CUDA 12.x (common in newer GCP instances)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (if specifically installed)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"
```

### Verifying T4 Setup

To verify your T4 GPU setup is working correctly:

```bash
# Check CUDA availability
nvidia-smi

# Check if PyTorch can see the GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Test the CUDA extension
python -c "from groundingdino.models.GroundingDINO.ms_deform_attn import _C; print('CUDA extension loaded successfully')"

# Test basic inference
cd /ttt/tree_ml
poetry run python tests/model_server/test_basic.py

# Check model server health
curl http://localhost:8000/health
```

If all these checks pass, your T4 GPU deployment should be working correctly.

## Security Considerations

1. **Authentication**:
   - Update the default credentials after deployment
   - Use a strong password (12+ characters with mixed case, numbers, symbols)

2. **SSL/TLS**:
   - For a production deployment, set up HTTPS with Let's Encrypt:
     ```bash
     sudo apt install certbot python3-certbot-nginx
     sudo certbot --nginx -d yourdomain.com
     ```

3. **Firewall**:
   - Configure firewall to allow only necessary ports:
     ```bash
     sudo ufw allow 22/tcp  # SSH
     sudo ufw allow 80/tcp  # HTTP
     sudo ufw allow 443/tcp # HTTPS
     sudo ufw enable
     ```

4. **Updates**:
   - Update system packages regularly:
     ```bash
     sudo apt update && sudo apt upgrade -y
     ```

## Monitoring and Maintenance

### Log Monitoring

```bash
# Model server logs
sudo journalctl -u tree-detection -f
tail -f /ttt/tree_ml/logs/model_server.log

# GPU monitoring
watch -n 1 nvidia-smi

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Service Management

```bash
# Restart model server
sudo systemctl restart tree-detection

# Check status
sudo systemctl status tree-detection

# View recent logs
sudo journalctl -u tree-detection -n 100
```

### Backup and Recovery

```bash
# Backup model weights
sudo tar -czf /tmp/tree-ml-weights-$(date +%Y%m%d).tar.gz /ttt/tree_ml/pipeline/model

# Backup detection data
sudo tar -czf /tmp/tree-ml-data-$(date +%Y%m%d).tar.gz /ttt/data/ml
```