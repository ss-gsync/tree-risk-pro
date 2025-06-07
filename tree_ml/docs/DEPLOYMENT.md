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

## GPU and CUDA Setup

Before deploying the application, you need to install CUDA and the appropriate NVIDIA drivers on your T4 instance:

```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-11-8

# Install system dependencies for ML pipeline
sudo apt-get install -y build-essential python3-dev python3-pip
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 
sudo apt-get install -y nginx

# Set up CUDA environment variables
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvidia-smi
nvcc --version
```

## Deployment Process

### 1. Kill Existing Processes

Before deployment, ensure all existing processes are properly terminated:

```bash
# Stop the systemd service if it exists
sudo systemctl stop dashboard-backend

# Kill any Python processes related to the application
sudo pkill -f "python.*app.py"
sudo pkill -f "gunicorn.*app:app"

# Kill any Node.js processes that might be running the server
sudo pkill -f "node.*server.js" 

# Kill any processes running on the server ports
sudo fuser -k 5000/tcp  # For the Flask backend
sudo fuser -k 5173/tcp  # For the Vite dev server

# Verify all processes are stopped
ps aux | grep -E "(python.*app|gunicorn|node.*server)"
```

### 2. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-org/tree-ml.git ~/tree-ml
cd ~/tree-ml
```

### 3. Install Python Dependencies with Poetry

```bash
# Install Poetry package manager
pip install --upgrade pip
pip install poetry

# Configure Poetry to create virtual environment in project directory
poetry config virtualenvs.in-project true

# Install Python dependencies
poetry install

# Install PyTorch with CUDA support
poetry run pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Download ML Model Weights

```bash
# Create model directories
mkdir -p tree_ml/pipeline/model/weights

# Download SAM model weights
wget -O tree_ml/pipeline/model/weights/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download GroundingDINO weights
mkdir -p tree_ml/pipeline/grounded-sam/weights
wget -O tree_ml/pipeline/grounded-sam/weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 5. Configure Environment

Create environment files with your API keys:

```bash
# Frontend environment (.env)
cat > .env << EOF
# Important: Empty string is intentional to avoid path duplication
VITE_API_URL=
VITE_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
VITE_GOOGLE_MAPS_MAP_ID=your_map_id
EOF

# Backend environment (backend/.env)
cat > tree_ml/dashboard/backend/.env << EOF
APP_MODE=production
SKIP_AUTH=false
DASHBOARD_USERNAME=TestAdmin
DASHBOARD_PASSWORD=trp345!
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
# Configure for unified deployment (model server on same machine)
USE_EXTERNAL_MODEL_SERVER=false
MODEL_SERVER_URL=http://localhost:8000
EOF
```

**IMPORTANT**: Replace the placeholder values with your actual API keys before proceeding:
- `your_google_maps_api_key` with your Google Maps API key
- `your_map_id` with your Google Maps Map ID
- `your_gemini_api_key` with your Gemini API key

### 6. Test ML Model Server

```bash
# Run the model server test to verify CUDA and model setup
cd ~/tree-ml
poetry run python tests/model_server/test_basic.py

# You should see output indicating CUDA is available and models can be loaded
# The output should include "CUDA is available: True" if properly configured
```

### 7. Build Frontend

```bash
# Install dependencies and build
cd ~/tree-ml/tree_ml/dashboard
npm install
npm run build
```

### 8. Setup Deployment Directory

```bash
# Clear previous deployment to prevent any cached or old files
sudo rm -rf /opt/tree-ml/dist/*
sudo rm -rf /opt/tree-ml/backend/*
sudo rm -f /opt/tree-ml/backend/.env
sudo rm -rf /opt/tree-ml/model-server/*

# Create directory structure
sudo mkdir -p /opt/tree-ml/{backend,dist,model-server}
sudo mkdir -p /opt/tree-ml/backend/{logs,data/temp,data/zarr,data/reports,data/exports,data/ml}
sudo mkdir -p /opt/tree-ml/model-server/{logs,weights}

# Copy files
sudo cp -r ~/tree-ml/tree_ml/dashboard/dist/* /opt/tree-ml/dist/
sudo cp -r ~/tree-ml/tree_ml/dashboard/backend/* /opt/tree-ml/backend/
sudo cp -r ~/tree-ml/tree_ml/pipeline/* /opt/tree-ml/model-server/
sudo cp ~/tree-ml/pyproject.toml /opt/tree-ml/
sudo cp ~/tree-ml/.env /opt/tree-ml/
sudo cp ~/tree-ml/tree_ml/dashboard/backend/.env /opt/tree-ml/backend/

# Copy model weights
sudo cp -r ~/tree-ml/tree_ml/pipeline/model/weights/* /opt/tree-ml/model-server/weights/
sudo cp -r ~/tree-ml/tree_ml/pipeline/grounded-sam/weights/* /opt/tree-ml/model-server/weights/

# Set permissions
sudo chmod -R 755 /opt/tree-ml/backend/logs
sudo chmod -R 755 /opt/tree-ml/backend/data
sudo chmod -R 755 /opt/tree-ml/model-server/logs
sudo chmod -R 755 /opt/tree-ml/model-server/weights
```

**IMPORTANT**: The step to clear previous deployment files is critical to ensure no cached or outdated files remain from previous versions.

### 9. Setup Model Server Service

Create a systemd service file for the model server:

```bash
sudo tee /etc/systemd/system/tree-ml-model-server.service > /dev/null << EOF
[Unit]
Description=Tree ML Model Server
After=network.target

[Service]
User=root
WorkingDirectory=/opt/tree-ml/model-server
ExecStart=/usr/bin/poetry run python model_server.py --port 8000 --host 0.0.0.0
Environment="PYTHONPATH=/opt/tree-ml"
Environment="MODEL_DIR=/opt/tree-ml/model-server/weights"
Environment="LOG_DIR=/opt/tree-ml/model-server/logs"
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable tree-ml-model-server
sudo systemctl start tree-ml-model-server

# Check the status
sudo systemctl status tree-ml-model-server
```

### 10. Install Backend Dependencies

```bash
cd /opt/tree-ml
sudo pip install poetry
sudo poetry install

# Install PyTorch with CUDA support in the deployment environment
sudo poetry run pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
```

### 11. Configure Nginx

```bash
# Generate self-signed certificate if needed
sudo mkdir -p /etc/ssl/private
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/dashboard-selfsigned.key \
    -out /etc/ssl/certs/dashboard-selfsigned.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=your-server-ip"

# Create Nginx configuration for unified deployment
sudo tee /etc/nginx/sites-available/tree-ml.conf > /dev/null << EOF
server {
    listen 80;
    server_name _;
    
    # Redirect all HTTP traffic to HTTPS
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    server_name _;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/dashboard-selfsigned.crt;
    ssl_certificate_key /etc/ssl/private/dashboard-selfsigned.key;
    
    # Frontend static files
    location / {
        root /opt/tree-ml/dist;
        try_files \$uri \$uri/ /index.html;

        # Add cache control for static assets
        location /assets {
            expires 7d;
            add_header Cache-Control "public, max-age=604800";
        }
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 180; # Increased timeout for ML operations
        client_max_body_size 20M; # Allow larger image uploads
    }
    
    # Model Server API - direct access for testing/development
    location /model-api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300; # Longer timeout for ML inference
        client_max_body_size 20M; # Allow larger image uploads
    }
}
EOF

# Replace "your-server-ip" with your actual server IP in the Nginx config
sudo sed -i "s/your-server-ip/$(curl -s ifconfig.me)/g" /etc/nginx/sites-available/tree-ml.conf

sudo ln -sf /etc/nginx/sites-available/tree-ml.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### 12. Create Backend Systemd Service

```bash
# First find the exact path to gunicorn in the Poetry environment
cd /opt/tree-ml
GUNICORN_PATH=$(sudo poetry env info --path)
echo "Poetry virtualenv path: $GUNICORN_PATH"

# Create the systemd service file for the backend
sudo tee /etc/systemd/system/tree-ml-backend.service > /dev/null << EOF
[Unit]
Description=Tree ML Dashboard Backend
After=network.target

[Service]
User=root
WorkingDirectory=/opt/tree-ml/backend
ExecStart=$GUNICORN_PATH/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Environment="DASHBOARD_USERNAME=TestAdmin"
Environment="DASHBOARD_PASSWORD=trp345!"
Environment="APP_MODE=production" 
Environment="GEMINI_API_KEY=$(grep GEMINI_API_KEY /opt/tree-ml/backend/.env | cut -d= -f2)"
Environment="GEMINI_MODEL=$(grep GEMINI_MODEL /opt/tree-ml/backend/.env | cut -d= -f2)"
Environment="USE_EXTERNAL_MODEL_SERVER=false"
Environment="MODEL_SERVER_URL=http://localhost:8000"
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable tree-ml-backend
sudo systemctl start tree-ml-backend
```

### 13. Verify Deployment

```bash
# Check services status
sudo systemctl status tree-ml-backend
sudo systemctl status tree-ml-model-server

# Verify model server is running
curl http://localhost:8000/health

# Verify backend API is running
curl -u TestAdmin:trp345! http://localhost:5000/api/config

# Verify the HTTPS site is working (ignore self-signed certificate warning)
curl https://$(curl -s ifconfig.me)/api/config -k
```

After completing these steps, your Tree ML application should be accessible via HTTPS at your server's IP address.

## Troubleshooting

### T4 and CUDA Issues

If you're experiencing GPU-related issues:

1. **Verify CUDA installation**:
   ```bash
   # Check if NVIDIA drivers are installed correctly
   nvidia-smi
   
   # Check CUDA version
   nvcc --version
   
   # Test CUDA with a simple PyTorch script
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
   ```

2. **Check model server logs for GPU errors**:
   ```bash
   sudo journalctl -u tree-ml-model-server -e
   tail -f /opt/tree-ml/model-server/logs/model_server.log
   ```

3. **Monitor GPU usage during inference**:
   ```bash
   # Install GPU monitoring tool
   sudo apt-get install -y nvidia-cuda-toolkit
   
   # Monitor GPU usage in real-time
   watch -n 1 nvidia-smi
   ```

4. **Common CUDA issues and solutions**:
   - **"CUDA out of memory"**: Reduce batch size or image resolution
   - **"CUDA driver version is insufficient"**: Update NVIDIA drivers
   - **"No CUDA-capable device"**: Check if T4 is recognized by the system

### Complete Rebuild and Cache Clear

If you're experiencing deployment issues:

1. **Stop all services and kill any related processes**:
   ```bash
   # Stop all systemd services
   sudo systemctl stop tree-ml-backend
   sudo systemctl stop tree-ml-model-server
   
   # Kill any Python processes related to the application
   sudo pkill -f "python.*app.py"
   sudo pkill -f "python.*model_server.py"
   sudo pkill -f "gunicorn.*app:app"
   
   # Kill any processes running on the server ports
   sudo fuser -k 5000/tcp  # For the Flask backend
   sudo fuser -k 8000/tcp  # For the ML model server
   
   # Verify all processes are stopped
   ps aux | grep -E "(python.*app|model_server|gunicorn)"
   ```

2. **Create a fresh deployment directory**:
   ```bash
   sudo rm -rf /opt/tree-ml
   sudo mkdir -p /opt/tree-ml
   ```

3. **Get the latest code**:
   ```bash
   cd ~/tree-ml
   git fetch
   git reset --hard origin/main
   ```

4. **Follow the deployment steps** from section 3 onwards.

### CORS or API Connection Issues

If you see CORS errors or API connection problems:

1. **Check your API_BASE_URL in the frontend**:
   
   The key issue with this deployment is ensuring the frontend doesn't try to connect directly to `http://localhost:5000`. 
   
   In the source code:
   ```javascript
   // src/services/api/apiService.js
   const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
   ```

   This is why we set `VITE_API_URL=` (empty string) in the .env file. When built with an empty string, API calls become relative paths (e.g., `/api/config`) that work correctly with the Nginx proxy.

2. **Verify built JavaScript doesn't contain localhost references**:
   ```bash
   cd /opt/tree-ml/dist
   grep -r "localhost:5000" --include="*.js" .
   ```

3. **Check if model server is accessible from backend**:
   ```bash
   # Test if model server is accessible
   curl http://localhost:8000/health
   
   # Test backend configuration
   grep -r "MODEL_SERVER_URL" /opt/tree-ml/backend
   ```

### ML Model Issues

If tree detection is not working properly:

1. **Check model weights**:
   ```bash
   # Verify SAM model weights exist
   ls -la /opt/tree-ml/model-server/weights/sam_vit_h_4b8939.pth
   
   # Verify GroundingDINO weights exist
   ls -la /opt/tree-ml/model-server/weights/groundingdino_swint_ogc.pth
   ```

2. **Test the model server API directly**:
   ```bash
   # Get server status
   curl http://localhost:8000/health
   
   # Get model information
   curl http://localhost:8000/models
   ```

3. **Review model server logs for specific errors**:
   ```bash
   tail -f /opt/tree-ml/model-server/logs/model_server.log
   ```

## Service Management

### Monitoring Logs

```bash
# Backend Flask logs
sudo journalctl -u tree-ml-backend.service -f
tail -f /opt/tree-ml/backend/logs/app.log

# Model server logs
sudo journalctl -u tree-ml-model-server.service -f
tail -f /opt/tree-ml/model-server/logs/model_server.log

# GPU monitoring
watch -n 1 nvidia-smi

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Managing Services

```bash
# Service management
sudo systemctl status tree-ml-backend
sudo systemctl restart tree-ml-backend
sudo systemctl status tree-ml-model-server
sudo systemctl restart tree-ml-model-server

# Nginx service management
sudo systemctl status nginx
sudo systemctl restart nginx

# Check current resource usage
top -u $(whoami)
nvidia-smi

# Monitor disk space
df -h /opt/tree-ml
```

### Backup and Restore

```bash
# Backup the entire application
sudo tar -czf /tmp/tree-ml-backup-$(date +%Y%m%d).tar.gz /opt/tree-ml

# Backup just the data directories
sudo tar -czf /tmp/tree-ml-data-$(date +%Y%m%d).tar.gz /opt/tree-ml/backend/data

# Backup model weights
sudo tar -czf /tmp/tree-ml-weights-$(date +%Y%m%d).tar.gz /opt/tree-ml/model-server/weights

# Restore from backup
sudo tar -xzf /tmp/tree-ml-backup-20250607.tar.gz -C /
sudo systemctl restart tree-ml-backend
sudo systemctl restart tree-ml-model-server
```

## Security Checklist

1. **Authentication**:
   - Update the default credentials after deployment
   - Use a strong password (12+ characters with mixed case, numbers, symbols)
   - Current auth implementation is suitable for beta testing

2. **SSL/TLS**:
   - Our deployment script sets up a self-signed certificate by default
   - For domain-based deployment, use Let's Encrypt:
     ```bash
     sudo apt install certbot python3-certbot-nginx
     sudo certbot --nginx -d yourdomain.com
     sudo systemctl status certbot.timer  # Check auto-renewal status
     ```

3. **Firewall**:
   - Configure firewall to allow only necessary ports:
     ```bash
     sudo ufw allow 22/tcp  # SSH
     sudo ufw allow 80/tcp  # HTTP
     sudo ufw allow 443/tcp # HTTPS
     sudo ufw enable
     sudo ufw status  # Should show only 22, 80, 443
     ```

4. **Updates**:
   - Update system packages monthly:
     ```bash
     sudo apt update && sudo apt upgrade -y
     ```
   - Check for outdated dependencies:
     ```bash
     cd /opt/tree-ml && npm outdated
     cd /opt/tree-ml && poetry show --outdated
     ```

5. **API Security**:
   - Keep API keys secure
   - Gemini API key should be restricted to the production VM
   - Do not commit credentials to git

6. **GPU Security**:
   - Monitor for unauthorized GPU usage:
     ```bash
     # Set up automatic monitoring for suspicious GPU usage
     echo '*/15 * * * * root nvidia-smi -q | grep "Process ID" > /tmp/nvidia-smi-prev.txt && sleep 60 && nvidia-smi -q | grep "Process ID" > /tmp/nvidia-smi-curr.txt && diff /tmp/nvidia-smi-prev.txt /tmp/nvidia-smi-curr.txt || echo "GPU usage changed"' | sudo tee /etc/cron.d/monitor-gpu
     ```