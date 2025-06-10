# Tree ML - Deployment Guide (v0.2.3)

This guide provides instructions for deploying the Tree ML system on a Google Cloud Platform (GCP) T4 GPU instance. The deployment has been thoroughly tested to ensure the GPU-accelerated model server works correctly with the dashboard and backend components.

## Deployment Summary

The production server uses a T4 GPU instance for unified ML processing and dashboard operation.

**Access Credentials:**
- Username: `TestAdmin`
- Password: `trp345!`

## v0.2.3 Release Notes

This version includes the following key improvements:

- Integrated ML Overlay for tree detection visualization
- Enhanced tree detection with DeepForest, SAM, and Gemini API integration
- T4 GPU support for accelerated model processing
- Unified deployment architecture (dashboard + ML on same instance)
- Improved UI components:
  - Fixed overlay controls with real-time opacity adjustment
  - Enhanced objects counter with persistence between detections
  - Streamlined detection sidebar with intuitive controls
  - New T4 status indicator for GPU availability

## System Requirements

### Hardware
- **GPU:** NVIDIA T4 (GCP n1-standard-4 instance recommended)
- **CPU:** Minimum 4 vCPU cores
- **Memory:** 16 GB RAM minimum
- **Storage:** 100+ GB SSD
- **OS:** Ubuntu 20.04 LTS or newer

### Software
- **CUDA:** Version 11.8 or newer
- **Python:** Version 3.10 or newer
- **Node.js:** Version 18 or newer
- **Other Tools:**
  - Git for version control
  - Nginx for web server
  - Poetry for Python dependency management

## API Key Setup

Before deployment, you must obtain the following API keys:

### 1. Google Maps API Key
- Navigate to [Google Cloud Console](https://console.cloud.google.com/)
- Go to Google Maps Platform > Credentials
- Create an API key with Maps JavaScript API access
- Restrict the key to your domains for security

### 2. Google Maps Map ID
- Go to Google Maps Platform > Map Management
- Create a new Map ID or use the existing one (e2f2ee8160dbcea0)
- Configure the Map ID with appropriate styling for satellite imagery

### 3. Gemini API Key
- Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- Create a new API key for Gemini model access
- Verify your account has access to the gemini-2.0-flash model
- Set appropriate quota limits based on expected usage

## Deployment Instructions

This section provides step-by-step instructions for deploying Tree ML on a T4 GPU instance.

### Setup Process Overview

1. **Configure CUDA Environment**

   First, verify your CUDA installation and set up the environment:

   ```bash
   # Verify NVIDIA drivers
   nvidia-smi
   
   # Check CUDA compiler
   nvcc --version
   
   # Locate CUDA installation directory
   # Common locations: /usr/lib/nvidia-cuda-toolkit, /usr/local/cuda, /usr/lib/cuda
   find /usr -name nvcc -type f 2>/dev/null
   
   # Set environment variables (adjust path based on your system)
   echo 'export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit' >> ~/.bashrc
   echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Set Up Python Environment**

   Create a dedicated environment with proper PyTorch configuration:

   ```bash
   # Create and activate virtual environment
   python3 -m venv ~/tree_ml
   source ~/tree_ml/bin/activate
   
   # Install NumPy 1.x (critical for PyTorch compatibility)
   pip install numpy==1.26.4
   
   # Install PyTorch with CUDA support
   # For CUDA 12.x (newer GCP instances):
   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8:
   # pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
   
   # Verify installation
   python -c "import torch, numpy; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}, NumPy: {numpy.__version__}')"
   ```

3. **Install Application**

   Clone the repository and install dependencies:

   ```bash
   # Clone repository
   git clone https://github.com/your-repo/tree-ml.git /ttt/tree_ml
   cd /ttt/tree_ml
   
   # Install Python package manager and dependencies
   pip install poetry
   poetry config virtualenvs.in-project true
   poetry install
   
   # Install additional ML dependencies
   pip install numpy opencv-python matplotlib timm tensorboard transformers pycocotools addict
   ```

4. **Configure ML Model Components**

   Set up the Grounded-SAM model:

   ```bash
   # Create model directory
   mkdir -p /ttt/tree_ml/pipeline/model
   
   # Clone Grounded-Segment-Anything repository
   cd /ttt/tree_ml/pipeline
   git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git grounded-sam
   
   # Download model weights
   wget -O /ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth \
     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
   
   wget -O /ttt/tree_ml/pipeline/model/groundingdino_swint_ogc.pth \
     https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   ```

5. **Configure Python Environment**

   Set up the Python environment for model components:

   ```bash
   # Set PYTHONPATH to include all required directories
   export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:\
   /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH
   
   # Install model components
   cd /ttt/tree_ml/pipeline/grounded-sam
   
   # Install Segment Anything Model
   python -m pip install -e segment_anything
   
   # Install GroundingDINO with build isolation disabled
   pip install --no-build-isolation -e GroundingDINO
   ```

6. **Build GPU Acceleration Components**

   Compile the CUDA extensions for optimal performance:

   ```bash
   # Set CUDA environment variable
   export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
   
   # Ensure correct NumPy version for PyTorch compatibility
   # CRITICAL: NumPy 2.x causes issues with PyTorch
   pip uninstall -y numpy
   pip install numpy==1.26.4
   
   # Prepare GroundingDINO directory
   cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
   mkdir -p groundingdino
   touch groundingdino/__init__.py
   
   # Build CUDA extension
   python setup.py build develop
   
   # Configure library path for PyTorch
   PYTORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
   export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:$LD_LIBRARY_PATH
   
   # Verify extension was built correctly
   cd /ttt/tree_ml
   python -c "from groundingdino import _C; print('CUDA extension loaded successfully')"
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
   
   # Activate the Python virtual environment
   source /home/ss/tree_ml/bin/activate
   
   # Set the environment
   export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:\$PYTHONPATH
   export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
   
   # Set LD_LIBRARY_PATH to include PyTorch libraries
   if python -c "import torch" &>/dev/null; then
       PYTORCH_PATH=\$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
       if [ -d "\$PYTORCH_PATH/lib" ]; then
           export LD_LIBRARY_PATH=\$PYTORCH_PATH/lib:\$LD_LIBRARY_PATH
           echo "Added PyTorch libraries to LD_LIBRARY_PATH: \$PYTORCH_PATH/lib"
       fi
   fi
   
   # Verify CUDA setup
   if [ -x "\$(command -v nvidia-smi)" ]; then
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
       python setup.py build develop
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

9. **Build and Deploy Frontend**:
   ```bash
   # Install Node.js if not already installed
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   
   # Verify Node.js installation
   node -v  # Should be v18.x or later
   npm -v   # Should be 8.x or later
   
   # Navigate to dashboard directory
   cd /ttt/tree_ml/dashboard
   
   # Install dependencies
   npm ci
   
   # Set environment variables for production build
   cat > .env.production << EOL
   VITE_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   VITE_GOOGLE_MAPS_MAP_ID=your_map_id
   VITE_API_BASE_URL=/api
   VITE_MODEL_API_URL=/model-api
   EOL
   
   # Build the frontend
   npm run build
   
   # Check if build was successful
   ls -la dist/
   ```

10. **Setup and Deploy Backend**:
   ```bash
   # Navigate to backend directory
   cd /ttt/tree_ml/dashboard/backend
   
   # Setup Python environment if not using the same as ML model
   source ~/tree_ml/bin/activate
   
   # Install backend dependencies
   pip install fastapi uvicorn python-dotenv
   npm ci  # For Node.js backend components
   
   # Create .env file for backend configuration
   cat > .env << EOL
   DEBUG=False
   SECRET_KEY=your_secret_key_here
   GEMINI_API_KEY=your_gemini_api_key
   MODEL_SERVER_URL=http://localhost:8000
   EOL
   
   # Create backend service file
   sudo tee /etc/systemd/system/tree-backend.service > /dev/null << EOL
   [Unit]
   Description=Tree ML Backend API Service
   After=network.target
   
   [Service]
   User=root
   WorkingDirectory=/ttt/tree_ml/dashboard/backend
   ExecStart=/bin/bash -c "source ~/tree_ml/bin/activate && python server.js"
   Restart=always
   RestartSec=10
   Environment=PYTHONUNBUFFERED=1
   Environment=NODE_ENV=production
   
   # Limit resource usage
   CPUWeight=70
   IOWeight=70
   MemoryHigh=4G
   MemoryMax=6G
   
   # Security settings
   ProtectSystem=full
   PrivateTmp=true
   NoNewPrivileges=true
   
   [Install]
   WantedBy=multi-user.target
   EOL
   
   # Enable and start the backend service
   sudo systemctl daemon-reload
   sudo systemctl enable tree-backend
   sudo systemctl start tree-backend
   
   # Check backend service status
   sudo systemctl status tree-backend
   ```

11. **Configure Nginx for the Dashboard**:
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
           proxy_http_version 1.1;
           proxy_set_header Upgrade \$http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_read_timeout 120s;
       }
       
       # Model Server API
       location /model-api/ {
           proxy_pass http://localhost:8000/;
           proxy_set_header Host \$host;
           proxy_set_header X-Real-IP \$remote_addr;
           proxy_read_timeout 300;
           client_max_body_size 20M;
       }
       
       # Add security headers
       add_header X-Content-Type-Options "nosniff";
       add_header X-Frame-Options "SAMEORIGIN";
       add_header X-XSS-Protection "1; mode=block";
   }
   EOL
   
   # Enable the site
   sudo ln -sf /etc/nginx/sites-available/tree-ml.conf /etc/nginx/sites-enabled/
   sudo rm -f /etc/nginx/sites-enabled/default
   sudo nginx -t
   sudo systemctl restart nginx
   ```

12. **Verify Complete Installation**:
    ```bash
    # Check model server status
    sudo systemctl status tree-detection
    curl http://localhost:8000/status
    
    # Check backend service status
    sudo systemctl status tree-backend
    curl http://localhost:5000/health
    
    # Check if frontend is being served through Nginx
    curl -I http://localhost
    
    # Check CUDA status on the model server
    nvidia-smi
    
    # Test model inference with a sample image
    cd /ttt/tree_ml/pipeline
    curl -X POST -F "image=@/ttt/data/tests/test_images/sample.jpg" -F "job_id=test_deployment" http://localhost:8000/detect
    
    # Check if the result was generated
    ls -la /ttt/data/ml/test_deployment/ml_response/
    
    # Verify the dashboard is accessible in a browser
    echo "Open http://$(hostname -I | awk '{print $1}') in a browser to verify the dashboard"
    sudo journalctl -u tree-detection -n 100
    
    # Test API endpoint
    curl http://localhost:8000/health
    ```

## Model Server Troubleshooting Guide

This section provides a comprehensive troubleshooting guide for the T4 GPU model server, addressing common issues you might encounter during deployment and operation.

### Common GPU Deployment Errors

#### Error: CUDA Extension Not Found
**Error Message:** `"name '_C' is not defined"` or `"No module named 'groundingdino._C'"`

**Cause:** The CUDA extensions for GroundingDINO haven't been properly built or can't be found in the Python path.

**Solution:**
```bash
# 1. Set correct CUDA_HOME
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

# 2. Ensure PYTHONPATH includes all required directories
export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH

# 3. Install compatible NumPy version
pip uninstall -y numpy
pip install numpy==1.26.4

# 4. Create groundingdino directory structure
cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
mkdir -p groundingdino
touch groundingdino/__init__.py

# 5. Build the CUDA extension
python setup.py build develop

# 6. Set LD_LIBRARY_PATH for PyTorch libraries
PYTORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:$LD_LIBRARY_PATH

# 7. Verify the extension was built successfully
python -c "from groundingdino import _C; print('CUDA extension loaded successfully')"

# 8. Restart the service
sudo systemctl restart tree-detection
```

#### Error: HTTP 500 from Model Server
**Error Message:** HTTP 500 errors with empty detail fields when calling the `/detect` endpoint

**Cause:** Type conversion issues between PyTorch tensors and NumPy arrays in the model server code.

**Solution:**

1. First check if the model server is properly initialized:
   ```bash
   curl http://localhost:8000/status
   # Expected: {"status":"ready","initialized":true,"device":"cuda"}
   ```

2. If not initialized, check the model server logs:
   ```bash
   sudo journalctl -u tree-detection -n 200
   ```

3. Fix the type conversion issues in model_server.py:
   ```python
   # Edit /ttt/tree_ml/pipeline/model_server.py
   
   # CHANGE the predict_sam method:
   # FROM: PyTorch tensor approach
   sam_box = torch.tensor(box_pixel, device=self.device)
   sam_result = self.sam_predictor.predict(
       box=sam_box.unsqueeze(0),
       multimask_output=False
   )
   mask = sam_result[0][0].cpu().numpy()
   
   # TO: NumPy array approach (SAM predictor expects numpy)
   sam_box = np.array(box_pixel)  # Convert to numpy array
   sam_result = self.sam_predictor.predict(
       box=sam_box,
       multimask_output=False
   )
   mask = sam_result[0][0]  # Already a numpy array
   ```

4. Improve error handling in the detect endpoint:
   ```python
   # CHANGE the error handling in the detect endpoint:
   # FROM: Raising HTTP exception
   if not result.get("success", False):
       logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
       raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
   
   # TO: Returning detailed JSON response
   if not result.get("success", False):
       error_message = result.get("error", "Unknown error during detection")
       logger.error(f"Processing failed: {error_message}")
       return JSONResponse(
           status_code=500,
           content={"detail": error_message, "job_id": job_id, "status": "failed"}
       )
   ```

5. Restart the model server:
   ```bash
   sudo systemctl restart tree-detection
   ```

#### Error: Meta Tensor Error
**Error Message:** `"Cannot copy out of meta tensor; no data!"`

**Cause:** PyTorch meta tensor issues when loading the model directly on GPU.

**Solution:**

Edit model_server.py to load models on CPU first, then transfer to GPU:

```python
# Find the model loading code in initialize_groundingdino and change:
# FROM: Direct GPU loading
model = build_model(args)
checkpoint = torch.load(weights_path, map_location=self.device)
model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
model = model.to(self.device)

# TO: CPU-first loading approach
# First load the model on CPU to avoid meta tensor issues
with torch.device('cpu'):
    model = build_model(args)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    
# Then move to desired device after initialization
model = model.to(self.device)
```

#### Error: NumPy Compatibility Issues
**Error Message:** `"Numpy is not available"` or `"C++ exception"` with PyTorch tensors

**Cause:** Version incompatibilities between NumPy and PyTorch, particularly when NumPy 2.x is used with PyTorch 2.x (which expects NumPy 1.x).

**Solution:**

1. Fix NumPy version:
   ```bash
   # Install compatible NumPy version
   pip uninstall -y numpy
   pip install numpy==1.26.4
   
   # Verify installation
   python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
   python -c "import torch, numpy; print(f'PyTorch: {torch.__version__}, NumPy: {numpy.__version__}')"
   ```

2. Fix type conversions in model_server.py:
   ```python
   # For tensor to numpy conversion:
   if isinstance(variable, torch.Tensor):
       numpy_variable = variable.cpu().detach().numpy()
   else:
       numpy_variable = variable  # Already a numpy array
   
   # For numpy to tensor conversion:
   if not isinstance(variable, torch.Tensor):
       tensor_variable = torch.tensor(variable, dtype=torch.float32, device=self.device)
   else:
       tensor_variable = variable  # Already a tensor
   ```

#### Error: Missing CUDA Libraries
**Error Message:** `"libc10.so: cannot open shared object file"` or `"libcudart.so.XX.Y not found"`

**Cause:** PyTorch's CUDA libraries can't be found in the system's library path.

**Solution:**

1. Add PyTorch libraries to LD_LIBRARY_PATH:
   ```bash
   # Find PyTorch library directory
   PYTORCH_LIB=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
   
   # Add to current session
   export LD_LIBRARY_PATH=$PYTORCH_LIB/lib:$LD_LIBRARY_PATH
   
   # Add permanently to your environment
   echo "export LD_LIBRARY_PATH=$PYTORCH_LIB/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
   source ~/.bashrc
   ```

2. Update the systemd service configuration:
   ```bash
   # Edit the service configuration
   sudo systemctl edit tree-detection
   
   # Add this line (update the path based on your system):
   [Service]
   Environment=LD_LIBRARY_PATH=/home/yourusername/tree_ml/lib/python3.10/site-packages/torch/lib
   
   # Reload and restart
   sudo systemctl daemon-reload
   sudo systemctl restart tree-detection
   ```

#### Error: Type Confusion Errors
**Error Message:** `"'Tensor' object has no attribute 'astype'"` or `"'numpy.ndarray' object has no attribute 'cpu'"`

**Cause:** Mixing PyTorch tensor methods with NumPy arrays or vice versa.

**Solution:**

1. Add explicit type checking before method calls:
   ```python
   # When converting tensors to numpy:
   if isinstance(variable, torch.Tensor):
       numpy_variable = variable.cpu().detach().numpy()
   else:
       numpy_variable = variable  # Already a numpy array
   
   # When working with SAM predictor:
   # SAM box should be numpy array, not a PyTorch tensor
   sam_box = np.array(box_pixel)  # not torch.tensor()
   
   # And for SAM output, it's already a numpy array
   mask = sam_result[0][0]  # not .cpu().numpy()
   ```

### Advanced Troubleshooting Techniques

If the basic solutions don't resolve your issue, try these advanced troubleshooting steps:

#### 1. Verify GPU Configuration

Check that the GPU is properly recognized and accessible:

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Verify PyTorch can access the GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
  print(f'Device count: {torch.cuda.device_count()}'); \
  print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Monitor GPU usage during startup (run in a separate terminal)
watch -n 0.5 nvidia-smi

# In another terminal, restart the model server
sudo systemctl restart tree-detection
```

#### 2. Diagnostic Script for Model Loading

Create a diagnostic script to isolate model loading issues:

```bash
# Create a diagnostic script
cat > /ttt/tree_ml/pipeline/debug_model.py << 'EOF'
import os
import sys
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Set paths
os.environ["PYTHONPATH"] = "/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:" + os.environ.get("PYTHONPATH", "")

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Try loading SAM model
try:
    print("Loading SAM model...")
    model_type = "vit_h"
    sam_checkpoint = "/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    print("SAM model loaded successfully!")
except Exception as e:
    print(f"Error loading SAM model: {str(e)}")

# Try importing GroundingDINO
try:
    print("Importing GroundingDINO modules...")
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    print("GroundingDINO modules imported successfully!")
    
    # Try loading GroundingDINO model
    print("Loading GroundingDINO model...")
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    weights_path = "/ttt/tree_ml/pipeline/model/groundingdino_swint_ogc.pth"
    
    args = SLConfig.fromfile(config_file)
    model = build_model(args)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    model = model.to(device)
    print("GroundingDINO model loaded successfully!")
except Exception as e:
    print(f"Error with GroundingDINO: {str(e)}")
EOF

# Run the diagnostic script
chmod +x /ttt/tree_ml/pipeline/debug_model.py
cd /ttt/tree_ml
source ~/tree_ml/bin/activate
python /ttt/tree_ml/pipeline/debug_model.py
```

#### 3. Enhanced Logging Configuration

Add detailed logging to model_server.py to diagnose issues:

```bash
# Edit model_server.py to add enhanced logging
cat > /ttt/tree_ml/pipeline/model_server_logging.patch << 'EOF'
import logging
import sys

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/ttt/tree_ml/logs/model_server_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Get logger
logger = logging.getLogger(__name__)

# Add these debug statements in key functions:
# In initialize method:
logger.debug(f"Initializing with device: {self.device}")
logger.debug(f"Model directory: {self.model_dir}")

# In load_model methods:
logger.debug(f"Loading model from: {model_path}")

# In process_image/detect methods:
logger.debug(f"Processing image type: {type(image)}, shape: {getattr(image, 'shape', 'unknown')}")
logger.debug(f"Detection boxes type: {type(boxes)}, count: {len(boxes) if isinstance(boxes, list) else 'n/a'}")
EOF

# Apply enhanced logging to model_server.py
cat /ttt/tree_ml/pipeline/model_server_logging.patch
# Manually add these logging statements to model_server.py
```

#### 4. API Endpoint Testing

Test the API endpoints directly to identify issues:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test status endpoint
curl http://localhost:8000/status

# Test detect endpoint with a sample image (verbose output)
curl -X POST -F "image=@/ttt/data/tests/test_images/sample.jpg" \
  -F "job_id=test123" http://localhost:8000/detect -v

# Check output directory for results
ls -la /ttt/data/ml/test123/ml_response/
```

#### 5. Environment Verification Script

Create a comprehensive script to verify all aspects of your environment:

```bash
# Create environment verification script
cat > /ttt/tree_ml/verify_env.sh << 'EOF'
#!/bin/bash
echo "=== System Environment Verification ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"

echo -e "\n=== Python and pip ==="
which python
python --version
which pip
pip --version

echo -e "\n=== CUDA Installation ==="
which nvcc
nvcc --version
nvidia-smi

echo -e "\n=== Environment Variables ==="
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo -e "\n=== PyTorch Installation ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo -e "\n=== NumPy Installation ==="
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo -e "\n=== Key Directory Structure ==="
echo "Model directory:"
ls -la /ttt/tree_ml/pipeline/model

echo -e "\nGrounded-SAM directory:"
ls -la /ttt/tree_ml/pipeline/grounded-sam

echo -e "\nGroundingDINO directory:"
ls -la /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO

echo -e "\n=== CUDA Extension Verification ==="
cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
echo "Build directory exists: $(if [ -d "build" ]; then echo "Yes"; else echo "No"; fi)"
echo "CUDA extension exists: $(if ls groundingdino/_C*.so 1> /dev/null 2>&1; then echo "Yes"; else echo "No"; fi)"

echo -e "\n=== Service Status ==="
systemctl status tree-detection | grep Active

echo -e "\nEnvironment verification complete"
EOF

# Make script executable and run it
chmod +x /ttt/tree_ml/verify_env.sh
/ttt/tree_ml/verify_env.sh | tee /ttt/tree_ml/logs/env_verification.log
```

### Deployment Verification Checklist

Use this checklist to systematically verify your T4 GPU deployment:

1. **Environment Setup**
   - [ ] CUDA is installed and accessible (`nvidia-smi` works)
   - [ ] PyTorch can see the GPU (`torch.cuda.is_available()` returns `True`)
   - [ ] PYTHONPATH includes all necessary directories
   - [ ] NumPy is version 1.x (compatible with PyTorch)
   - [ ] LD_LIBRARY_PATH includes PyTorch lib directory

2. **Model Weights**
   - [ ] SAM model weights exist at `/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth`
   - [ ] GroundingDINO weights exist at `/ttt/tree_ml/pipeline/model/groundingdino_swint_ogc.pth`

3. **CUDA Extensions**
   - [ ] GroundingDINO extension is built successfully
   - [ ] Extension can be imported (`from groundingdino import _C` works)

4. **Code Fixes**
   - [ ] Tensor/array type conversions are correct in model_server.py
   - [ ] Error handling is improved in the detect endpoint
   - [ ] Models are loaded on CPU first, then moved to GPU

5. **Service Configuration**
   - [ ] Systemd service has correct environment variables
   - [ ] Service has proper working directory and permissions
   - [ ] Service can successfully start and stay running

By following this structured troubleshooting guide, you should be able to resolve most issues with the T4 GPU model server deployment.

### Common Deployment Issues and Solutions

#### Error: "Numpy is not available"

This error occurs due to NumPy 2.x compatibility issues with PyTorch. PyTorch 2.2.0 was compiled with NumPy 1.x but is trying to run with NumPy 2.x.

**Solution**:
```bash
# Make sure your virtual environment is activated
source ~/tree_ml/bin/activate

# Downgrade NumPy to a compatible version (1.x)
pip uninstall -y numpy
pip install numpy==1.26.4

# Verify NumPy version
python -c "import numpy; print(numpy.__version__)"

# Make sure PyTorch can use NumPy without warnings
python -c "import torch, numpy; print(f'NumPy: {numpy.__version__}, PyTorch: {torch.__version__}')"

# Restart the service
sudo systemctl restart tree-detection
```

You may see this warning: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.1.1 as it may crash." This is exactly what's causing the "Numpy is not available" error, and downgrading to NumPy 1.x resolves it.

#### Error: "libc10.so: cannot open shared object file"

This error occurs when PyTorch's CUDA libraries can't be found in the system's library path.

**Solution**:
```bash
# Find the PyTorch library directory
PYTORCH_LIB=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
echo "PyTorch lib directory: $PYTORCH_LIB"

# Add the directory to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTORCH_LIB/lib:$LD_LIBRARY_PATH

# Add it permanently to your environment
echo "export LD_LIBRARY_PATH=$PYTORCH_LIB/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

# Update the systemd service file to include the LD_LIBRARY_PATH
sudo systemctl edit tree-detection
# Add the following lines:
# [Service]
# Environment=LD_LIBRARY_PATH=/home/yourusername/tree_ml/lib/python3.12/site-packages/torch/lib

# Reload and restart the service
sudo systemctl daemon-reload
sudo systemctl restart tree-detection
```

#### Error: "AttributeError: 'NoneType' object has no attribute 'model_dir'"

This error occurs in model_server.py when the model_server variable is None at the time of accessing its attributes.

**Solution**:

Edit the model_server.py file to initialize the model_server in the main() function:

```python
def main():
    """
    Run the model server
    """
    global model_server
    
    # Parse arguments...
    
    # Initialize model server if not already initialized
    if model_server is None:
        model_server = GroundedSAMServer()
    
    # Set properties...
    
    # Initialize model in background
    threading.Thread(target=model_server.initialize).start()
    
    # Run server
    uvicorn.run(app, host=args.host, port=args.port)
```

This ensures the model_server is properly initialized regardless of how it's started.

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
curl http://localhost:8000/status
```

If all these checks pass, your T4 GPU deployment should be working correctly.

### Known Issues and Workarounds

During deployment, you might encounter these issues that we've identified and fixed:

#### 1. GroundingDINO Meta Tensor Errors

If you see errors like "Cannot copy out of meta tensor; no data!" when loading the GroundingDINO model:

```bash
# Edit model_server.py to load models on CPU first, then move to GPU
# First load the model on CPU to avoid meta tensor issues
with torch.device('cpu'):
    model = build_model(args)
    checkpoint = torch.load(grounding_dino_weights_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    
# Then move to desired device after initialization
model = model.to(self.device)
```

#### 2. Tensor vs. NumPy Array Type Errors

If you encounter errors like "'Tensor' object has no attribute 'astype'" or "'numpy.ndarray' object has no attribute 'cpu'", you need to ensure proper type conversion in model_server.py:

```python
# For PyTorch tensor to NumPy conversion
# Convert torch tensor to numpy array for SAM predictor
sam_box = np.array(box_pixel)  # not torch.tensor(box_pixel)

# For handling NumPy array outputs
# Don't call .cpu().numpy() on NumPy arrays
mask = sam_result[0][0]  # not sam_result[0][0].cpu().numpy()
```

#### 3. Model Initialization Status Issues

Sometimes the model may show as not initialized in status checks even when it is. We solved this by:

1. Pre-setting the initialization flag when weights are found
2. Improving the attribute checking logic in status endpoints
3. Adding diagnostic logs during initialization

#### 4. Optional Components

The backend may show warnings about missing database or Gemini connections:

```
Failed to connect to master database: [Errno 111] Connect call failed
Skipping Gemini initialization to avoid event loop conflicts
```

These are non-critical services and the application will still function for tree detection. If you need these services:

- For database: Update the `.env` file in the backend directory with correct connection details
- For Gemini: Add a valid Gemini API key to the `.env` file

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
   - Configure local firewall to allow only necessary ports:
     ```bash
     sudo ufw allow 22/tcp  # SSH
     sudo ufw allow 80/tcp  # HTTP
     sudo ufw allow 443/tcp # HTTPS
     sudo ufw enable
     ```
   
   - Configure GCP firewall rules to allow external access:
     1. Navigate to the GCP Console: https://console.cloud.google.com/
     2. Go to VPC Network > Firewall
     3. Click "CREATE FIREWALL RULE"
     4. Configure the rule:
        - Name: `allow-tree-detection-services`
        - Network: Select your VPC network
        - Priority: 1000 (or your preferred priority)
        - Direction of traffic: Ingress
        - Action on match: Allow
        - Targets: Specified target tags
        - Target tags: `tree-detection-vm` (or use your VM's existing network tags)
        - Source filter: IP ranges
        - Source IP ranges: `0.0.0.0/0` (or restrict to specific IP ranges for better security)
        - Protocols and ports:
          - Check "TCP"
          - Enter ports: `80,5000,8000`
     5. Click "Create"

     6. Assign the network tag to your VM:
        - Go to Compute Engine > VM Instances
        - Click on your VM instance
        - Click "Edit"
        - Under "Network tags", add the tag you specified in the firewall rule (e.g., `tree-detection-vm`)
        - Click "Save"

     After applying these settings, your services should be accessible at:
     - Frontend: http://[VM_EXTERNAL_IP]
     - Backend API: http://[VM_EXTERNAL_IP]:5000
     - Model Server: http://[VM_EXTERNAL_IP]:8000
     
     To verify GCP firewall rules are correctly configured:
     ```bash
     # List all firewall rules affecting your VM
     gcloud compute firewall-rules list --filter="direction=INGRESS AND disabled=false AND (port:80 OR port:5000 OR port:8000)"
     
     # Check if ports are reachable from external machine
     # Replace VM_EXTERNAL_IP with your VM's external IP address
     nc -zv VM_EXTERNAL_IP 80
     nc -zv VM_EXTERNAL_IP 5000
     nc -zv VM_EXTERNAL_IP 8000
     
     # Check service status and interface binding
     sudo netstat -tulpn | grep -E ':(80|5000|8000)'
     # You should see entries with 0.0.0.0:80, 0.0.0.0:5000, and 0.0.0.0:8000
     ```
     
     If you still can't connect after configuring firewall rules:
     1. Verify the VM's network interface is properly configured
     2. Check if the services are running and listening on all interfaces (0.0.0.0)
     3. Ensure no conflicting firewall rules with higher priority are blocking traffic
     4. Try using an SSH tunnel for testing if firewall issues persist:
        ```bash
        # On your local machine:
        ssh -L 8000:localhost:8000 -L 5000:localhost:5000 -L 80:localhost:80 username@VM_EXTERNAL_IP
        ```

4. **Updates**:
   - Update system packages regularly:
     ```bash
     sudo apt update && sudo apt upgrade -y
     ```

## Monitoring and Maintenance

This section covers essential monitoring and maintenance tasks to keep your Tree ML deployment running smoothly.

### Log Monitoring

Use these commands to view real-time logs from different components:

```bash
# Model server logs (real-time)
sudo journalctl -u tree-detection -f

# Model server log file
tail -f /ttt/tree_ml/logs/model_server.log

# GPU monitoring (updates every second)
watch -n 1 nvidia-smi

# Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Nginx error logs
sudo tail -f /var/log/nginx/error.log
```

### Service Management

Manage your Tree ML services with these systemd commands:

```bash
# Restart model server
sudo systemctl restart tree-detection

# Check model server status
sudo systemctl status tree-detection

# View recent model server logs
sudo journalctl -u tree-detection -n 100

# Restart backend service
sudo systemctl restart tree-backend

# Restart web server
sudo systemctl restart nginx

# Check status of all services
sudo systemctl status tree-detection tree-backend nginx
```

### Backup and Recovery

Perform regular backups of critical data:

```bash
# Backup model weights
sudo tar -czf /tmp/tree-ml-weights-$(date +%Y%m%d).tar.gz /ttt/tree_ml/pipeline/model

# Backup detection data
sudo tar -czf /tmp/tree-ml-data-$(date +%Y%m%d).tar.gz /ttt/data/ml

# Backup full application (excluding large directories)
sudo tar -czf /tmp/tree-ml-full-$(date +%Y%m%d).tar.gz \
  --exclude='node_modules' \
  --exclude='.git' \
  --exclude='data/ml' \
  /ttt/tree_ml

# Copy backups to secure storage
# Example: Copy to Google Cloud Storage bucket
# gsutil cp /tmp/tree-ml-*.tar.gz gs://your-backup-bucket/
```

## Updating and Redeploying the Application

This section provides step-by-step instructions for updating your Tree ML deployment with new code.

### 1. Pulling Updates

First, retrieve the latest code from the repository:

```bash
# Navigate to repository directory
cd /ttt/tree_ml

# Backup current code (recommended before updates)
tar -czf /tmp/tree-ml-backup-$(date +%Y%m%d).tar.gz --exclude=node_modules --exclude=.git .

# Pull latest changes
git pull origin main

# Review what changed
git log -p -1
```

### 2. Updating Frontend

Update the dashboard user interface:

```bash
# Navigate to dashboard directory
cd /ttt/tree_ml/dashboard

# Install any new dependencies
npm ci

# Build the frontend
npm run build

# Verify the build was successful
ls -la dist/
```

### 3. Updating Backend

Update the API and server components:

```bash
# Navigate to backend directory
cd /ttt/tree_ml/dashboard/backend

# Activate Python environment
source ~/tree_ml/bin/activate

# Install any new Python dependencies
pip install -r requirements.txt

# Install any new Node.js dependencies
npm ci

# Restart the backend service
sudo systemctl restart tree-backend
```

### 4. Updating ML Model Server

Update the machine learning components:

```bash
# If model weights were updated, download them
cd /ttt/tree_ml/pipeline/model
# Add wget commands for updated model files if necessary

# If CUDA extensions need rebuilding
cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
python setup.py build develop

# Restart the model server service
sudo systemctl restart tree-detection
```

### 5. Complete System Restart

For major updates, perform a full system restart:

```bash
# Stop all services in reverse order of dependencies
sudo systemctl stop tree-detection
sudo systemctl stop tree-backend
sudo systemctl stop nginx

# Clear any cached data (if necessary)
rm -rf /ttt/data/ml/temp/*

# Start services in the correct dependency order
sudo systemctl start tree-detection
sleep 10  # Allow model server to initialize
sudo systemctl start tree-backend
sudo systemctl start nginx

# Verify all services are running properly
sudo systemctl status tree-detection tree-backend nginx
```

### 6. Troubleshooting Update Issues

#### Frontend Build Failures

If the frontend build fails:

```bash
# Get detailed npm errors
cd /ttt/tree_ml/dashboard
npm ci --verbose

# Try clearing npm cache and rebuilding
npm cache clean --force
npm ci
npm run build

# Check for JavaScript errors in source files
npx eslint src/
```

#### Backend Update Issues

If the backend service fails to start:

```bash
# Check for missing Python dependencies
pip list | grep -E 'fastapi|uvicorn|sqlalchemy'

# Try running the server manually to see errors
cd /ttt/tree_ml/dashboard/backend
source ~/tree_ml/bin/activate
python server.js

# Check logs for errors
sudo journalctl -u tree-backend -n 100
```

#### ML Model Server Issues

If the model server fails to start or process requests:

```bash
# Run the server manually to see startup errors
cd /ttt/tree_ml
source ~/tree_ml/bin/activate
bash /ttt/tree_ml/pipeline/run_model_server.sh

# Check systemd service logs
sudo journalctl -u tree-detection -n 100

# Check application logs
cat /ttt/tree_ml/logs/model_server.log

# Verify CUDA is working with PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test if model extensions are properly built
python -c "from groundingdino import _C; print('CUDA extension loaded successfully')"
```