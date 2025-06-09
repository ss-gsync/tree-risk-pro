# Tree ML - Deployment Guide (v0.2.3)

> This is the deployment guide for the Tree ML project, including the dashboard, backend components, and ML model server.

This guide provides instructions for deploying the Tree ML system on a Google Cloud Platform (GCP) T4 GPU instance for unified deployment. It has been thoroughly tested to ensure the GPU-accelerated model server works correctly.

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
   
   # Find the correct CUDA location - it may vary between systems
   # Common locations: /usr/lib/nvidia-cuda-toolkit, /usr/local/cuda, /usr/lib/cuda
   find /usr -name nvcc -type f 2>/dev/null
   
   # Set CUDA environment variables (update path as needed based on your system)
   echo 'export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit' >> ~/.bashrc
   echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Create Python Environment**:
   ```bash
   # Create a virtual environment
   python3 -m venv ~/tree_ml
   source ~/tree_ml/bin/activate
   
   # Install NumPy 1.x first (important for PyTorch compatibility)
   pip install numpy==1.26.4
   
   # Install the correct PyTorch version for your CUDA
   # For CUDA 12.x (newer GCP instances):
   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8:
   # pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
   
   # Verify PyTorch CUDA detection
   python -c "import torch, numpy; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}, NumPy: {numpy.__version__}')"
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

5. **Configure Environment and Install Components**:
   ```bash
   # Set PYTHONPATH to include all necessary directories
   export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH
   
   # Navigate to the grounded-sam directory
   cd /ttt/tree_ml/pipeline/grounded-sam
   
   # Install segment_anything using the correct command
   python -m pip install -e segment_anything
   
   # Install GroundingDINO with no-build-isolation flag
   pip install --no-build-isolation -e GroundingDINO
   ```

6. **Build CUDA Extensions**:
   ```bash
   # Set CUDA_HOME environment variable (update based on your system's CUDA location)
   export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
   
   # Make sure NumPy is the correct version (PyTorch 2.2.0 needs NumPy 1.x)
   # THIS IS CRITICAL: NumPy 2.x will cause compatibility issues with PyTorch
   pip uninstall -y numpy
   pip install numpy==1.26.4
   
   # Set up the groundingdino directory structure
   cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
   mkdir -p groundingdino
   touch groundingdino/__init__.py
   
   # Build the CUDA extension through the GroundingDINO setup.py
   python setup.py build develop
   
   # Make sure PyTorch libraries are in LD_LIBRARY_PATH
   PYTORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
   export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:$LD_LIBRARY_PATH
   
   # Verify the extension was built successfully
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

This section provides comprehensive troubleshooting for the model server, addressing common issues encountered during deployment and operation.

### Common T4 Deployment Errors and Solutions

#### Error: "name '_C' is not defined" or "No module named 'groundingdino._C'"

This error occurs when the CUDA extensions for GroundingDINO haven't been properly built or can't be found.

**Solution**:
```bash
# Set CUDA_HOME to the correct location (update based on your system)
export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit

# Make sure PYTHONPATH includes all required directories
export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH

# Make sure NumPy 1.x is installed (required for PyTorch compatibility)
pip uninstall -y numpy
pip install numpy==1.26.4

# Create and prepare the groundingdino directory
cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
mkdir -p groundingdino
touch groundingdino/__init__.py

# Build the extension
python setup.py build develop

# Set LD_LIBRARY_PATH to include PyTorch libraries
PYTORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:$LD_LIBRARY_PATH

# Verify the extension was built successfully
python -c "from groundingdino import _C; print('CUDA extension loaded successfully')"

# Restart the service
sudo systemctl restart tree-detection
```

#### Error: HTTP 500 from Model Server with Empty Detail

If the model server is returning 500 errors with empty detail fields when calling the `/detect` endpoint:

**Solution**:

1. First check if the model server is properly initialized:
   ```bash
   curl http://localhost:8000/status
   # Should return {"status":"ready","initialized":true,"device":"cuda"}
   ```

2. If not initialized, check logs for initialization errors:
   ```bash
   sudo journalctl -u tree-detection -n 200
   ```

3. Fix type conversion issues in model_server.py:
   ```python
   # Open the model_server.py file
   nano /ttt/tree_ml/pipeline/model_server.py
   
   # Look for the predict_sam method and change:
   # FROM:
   sam_box = torch.tensor(box_pixel, device=self.device)
   sam_result = self.sam_predictor.predict(
       box=sam_box.unsqueeze(0),
       multimask_output=False
   )
   
   # TO:
   # Convert to numpy array for SAM predictor (which expects numpy, not torch tensors)
   sam_box = np.array(box_pixel)
   sam_result = self.sam_predictor.predict(
       box=sam_box,
       multimask_output=False
   )
   
   # AND CHANGE:
   # FROM:
   mask = sam_result[0][0].cpu().numpy()
   
   # TO:
   # SAM returns masks as numpy arrays when using the numpy input
   mask = sam_result[0][0]  # Already a numpy array
   ```

4. Improve error handling in the detect endpoint:
   ```python
   # Find the detect endpoint and update error handling:
   # FROM:
   if not result.get("success", False):
       logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
       raise HTTPException(status_code=500, detail=result.get("error", "Unknown error during detection"))
   
   # TO:
   if not result.get("success", False):
       error_message = result.get("error", "Unknown error during detection")
       logger.error(f"Processing failed: {error_message}")
       return JSONResponse(
           status_code=500,
           content={"detail": error_message, "job_id": job_id, "status": "failed"}
       )
   ```

5. Restart the model server service:
   ```bash
   sudo systemctl restart tree-detection
   ```

#### Error: "Cannot copy out of meta tensor; no data!"

This error occurs when loading the GroundingDINO model on GPU directly without initializing on CPU first.

**Solution**:

Edit `/ttt/tree_ml/pipeline/model_server.py` to load models on CPU first:

```python
# Find the model loading code in initialize_groundingdino and change:
# FROM:
model = build_model(args)
checkpoint = torch.load(weights_path, map_location=self.device)
model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
model = model.to(self.device)

# TO:
# First load the model on CPU to avoid meta tensor issues
with torch.device('cpu'):
    model = build_model(args)
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    
# Then move to desired device after initialization
model = model.to(self.device)
```

#### Error: "Numpy is not available" or "C++ exception" with PyTorch Tensors

These errors occur due to version incompatibilities between NumPy and PyTorch, or type mismatches between PyTorch tensors and NumPy arrays.

**Solution**:

1. Fix NumPy version incompatibility:
   ```bash
   # Ensure you're using NumPy 1.x (PyTorch 2.x requirement)
   pip uninstall -y numpy
   pip install numpy==1.26.4
   
   # Verify compatibility
   python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
   python -c "import torch, numpy; print(f'PyTorch: {torch.__version__}, NumPy: {numpy.__version__}')"
   ```

2. Fix tensor/array type conversions in model_server.py:
   ```python
   # In process_image method, make sure conversion to numpy is correct:
   # Convert tensor outputs to numpy arrays before further processing
   if isinstance(image_tensor, torch.Tensor):
       image_tensor = image_tensor.cpu().numpy()
   
   # When creating tensors from arrays, be explicit:
   tensor_input = torch.tensor(numpy_array, dtype=torch.float32, device=self.device)
   ```

3. Check predict_sam method specifically:
   ```python
   # SAM predictor expects numpy arrays for boxes, not tensors
   # FROM:
   sam_box = torch.tensor(box_pixel, device=self.device)
   # TO:
   sam_box = np.array(box_pixel)
   ```

#### Error: "libc10.so: cannot open shared object file" or "libcudart.so.XX.Y not found"

These errors occur when PyTorch's CUDA libraries can't be found in the system's library path.

**Solution**:

1. Find and set the PyTorch library directory in LD_LIBRARY_PATH:
   ```bash
   # Find the PyTorch library directory
   PYTORCH_LIB=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
   echo "PyTorch lib directory: $PYTORCH_LIB"
   
   # Add to LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=$PYTORCH_LIB/lib:$LD_LIBRARY_PATH
   
   # Add permanently to your environment
   echo "export LD_LIBRARY_PATH=$PYTORCH_LIB/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
   ```

2. Update the systemd service file with the correct path:
   ```bash
   sudo systemctl edit tree-detection
   # Add:
   [Service]
   Environment=LD_LIBRARY_PATH=/path/to/pytorch/lib
   
   # Reload and restart
   sudo systemctl daemon-reload
   sudo systemctl restart tree-detection
   ```

#### Error: "'Tensor' object has no attribute 'astype'" or "'numpy.ndarray' object has no attribute 'cpu'"

These errors occur when you're calling NumPy methods on PyTorch tensors or PyTorch methods on NumPy arrays.

**Solution**:

1. Check the model_server.py for type confusion:
   ```python
   # For tensor to numpy conversion, always use:
   if isinstance(variable, torch.Tensor):
       numpy_variable = variable.cpu().detach().numpy()
   else:
       numpy_variable = variable  # Already a numpy array
   
   # For numpy to tensor conversion:
   if not isinstance(variable, torch.Tensor):
       tensor_variable = torch.tensor(variable, device=self.device)
   else:
       tensor_variable = variable  # Already a tensor
   ```

2. Pay special attention to the SAM predictor which expects numpy arrays:
   ```python
   # SAM box should be numpy array
   sam_box = np.array(box_pixel)
   
   # And the result is already a numpy array, don't call cpu().numpy()
   mask = sam_result[0][0]  # Already a numpy array
   ```

### Advanced Troubleshooting Steps

If you're still encountering issues after trying the solutions above:

#### 1. Check Model Initialization and GPU Usage

```bash
# Verify GPU is working
nvidia-smi

# Check if PyTorch can access the GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Monitor GPU usage during model server startup
watch -n 0.5 nvidia-smi

# In another terminal, restart the model server
sudo systemctl restart tree-detection
```

#### 2. Debug Model Loading Step by Step

Create a diagnostic script to test model loading:

```python
# Save as /ttt/tree_ml/pipeline/debug_model.py
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
```

Run the diagnostic script:
```bash
cd /ttt/tree_ml
source ~/tree_ml/bin/activate
python /ttt/tree_ml/pipeline/debug_model.py
```

#### 3. Increase Logging Detail

Edit model_server.py to add more detailed logging:

```python
# At the top of model_server.py, update logging configuration
import logging
import sys

# Configure root logger
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

# Add more logging throughout the code:
logger.debug(f"Variable type: {type(variable)}")
logger.debug(f"Processing image with shape: {image.shape}")
```

#### 4. Test API Endpoints Directly

Test the API endpoints directly to pinpoint issues:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test status endpoint
curl http://localhost:8000/status

# Test detect endpoint with a sample image
curl -X POST -F "image=@/ttt/data/tests/test_images/sample.jpg" -F "job_id=test123" http://localhost:8000/detect -v

# Check output directory for results
ls -la /ttt/data/ml/test123/ml_response/
```

#### 5. Verify Environment Configuration

Check environment variables and paths:

```bash
# Create a script to verify environment
cat > /ttt/tree_ml/verify_env.sh << 'EOF'
#!/bin/bash
echo "=== Python and pip ==="
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

echo -e "\n=== Directory Structure ==="
ls -la /ttt/tree_ml/pipeline/grounded-sam
ls -la /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
ls -la /ttt/tree_ml/pipeline/model

echo -e "\n=== Extension Building ==="
cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
python -c "import os; print(os.path.exists('build')); print(os.path.exists('groundingdino/_C.so') or os.path.exists('groundingdino/_C.cpython-*.so'))"
EOF

chmod +x /ttt/tree_ml/verify_env.sh
/ttt/tree_ml/verify_env.sh
```

### Final Checklist for Model Server Troubleshooting

Use this checklist to systematically verify all aspects of the model server:

1. **Environment Setup**
   - [ ] CUDA is installed and accessible (nvidia-smi works)
   - [ ] PyTorch can see the GPU (torch.cuda.is_available() returns True)
   - [ ] PYTHONPATH includes all necessary directories
   - [ ] NumPy is version 1.x (compatible with PyTorch)
   - [ ] LD_LIBRARY_PATH includes PyTorch lib directory

2. **Model Weights**
   - [ ] SAM model weights exist at /ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth
   - [ ] GroundingDINO weights exist at /ttt/tree_ml/pipeline/model/groundingdino_swint_ogc.pth

3. **CUDA Extensions**
   - [ ] GroundingDINO extension is built successfully
   - [ ] Extension can be imported (from groundingdino import _C works)

4. **Code Fixes**
   - [ ] Tensor/array type conversions are correct in model_server.py
   - [ ] Error handling is improved in the detect endpoint
   - [ ] Models are loaded on CPU first, then moved to GPU

5. **Service Configuration**
   - [ ] Systemd service has correct environment variables
   - [ ] Service has proper working directory and permissions
   - [ ] Service can successfully start and stay running

By following this troubleshooting guide, you should be able to resolve most issues with the model server and ensure it runs correctly with GPU acceleration.

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

## Updating and Redeploying the Application

When you need to update the application with new code:

### 1. Pulling Updates

```bash
# Navigate to repository directory
cd /ttt/tree_ml

# Backup current code (optional but recommended)
tar -czf /tmp/tree-ml-backup-$(date +%Y%m%d).tar.gz --exclude=node_modules --exclude=.git .

# Pull latest changes
git pull origin main

# Check what changed
git log -p -1
```

### 2. Updating Frontend

```bash
# Navigate to dashboard directory
cd /ttt/tree_ml/dashboard

# Install dependencies (in case there are changes)
npm ci

# Build the frontend
npm run build

# Verify the build
ls -la dist/
```

### 3. Updating Backend

```bash
# Navigate to backend directory
cd /ttt/tree_ml/dashboard/backend

# Install any new dependencies
source ~/tree_ml/bin/activate
pip install -r requirements.txt
npm ci

# Restart the backend service
sudo systemctl restart tree-backend
```

### 4. Updating ML Model Server

```bash
# If model weights were updated, download them
cd /ttt/tree_ml/pipeline/model
# Add wget commands for updated model files if necessary

# Restart the model server service
sudo systemctl restart tree-detection
```

### 5. Complete System Restart

If you need to do a full system restart:

```bash
# Stop all services
sudo systemctl stop tree-detection
sudo systemctl stop tree-backend
sudo systemctl stop nginx

# Clear any cached data (if necessary)
rm -rf /ttt/data/ml/temp/*

# Start services in the right order
sudo systemctl start tree-detection
sudo systemctl start tree-backend
sudo systemctl start nginx

# Verify all services are running
sudo systemctl status tree-detection
sudo systemctl status tree-backend
sudo systemctl status nginx
```

### 6. Troubleshooting Common Update Issues

#### Frontend Build Failures

```bash
# Check for Node.js/npm errors
cd /ttt/tree_ml/dashboard
npm ci --verbose

# Try clearing npm cache
npm cache clean --force
npm ci
npm run build
```

#### Backend Update Issues

```bash
# Check Python dependencies
pip list | grep -E 'fastapi|uvicorn|sqlalchemy'

# Manually restart with verbose output
cd /ttt/tree_ml/dashboard/backend
source ~/tree_ml/bin/activate
python server.js
```

#### ML Model Server Issues

```bash
# Manually start the model server to see errors
cd /ttt/tree_ml
source ~/tree_ml/bin/activate
bash /ttt/tree_ml/pipeline/run_model_server.sh

# Check logs
sudo journalctl -u tree-detection -n 100
cat /ttt/tree_ml/logs/model_server.log
```