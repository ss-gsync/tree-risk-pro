# T4 GPU Tree Detection Model Server Troubleshooting Guide

This document provides solutions for common issues that may arise when deploying the tree detection model server on a T4 GPU instance.

## Model Server Errors

### Error: "Cannot import name '_C' from 'groundingdino.models.GroundingDINO.ms_deform_attn'"

This error occurs when the CUDA extension for GroundingDINO hasn't been properly built.

**Solution:**

1. Ensure CUDA_HOME is correctly set:
   ```bash
   # Find where CUDA is installed
   find /usr -name nvcc -type f
   
   # Set CUDA_HOME to the parent directory of the bin directory containing nvcc
   export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
   ```

2. Make sure NumPy 1.x is installed (PyTorch compatibility):
   ```bash
   pip uninstall -y numpy
   pip install numpy==1.26.4
   ```

3. Create and initialize the groundingdino directory:
   ```bash
   cd /ttt/tree_ml/pipeline/grounded-sam/GroundingDINO
   mkdir -p groundingdino
   touch groundingdino/__init__.py
   ```

4. Build the extension:
   ```bash
   python setup.py build develop
   ```

5. Set LD_LIBRARY_PATH for PyTorch libraries:
   ```bash
   PYTORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
   export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:$LD_LIBRARY_PATH
   ```

6. Verify the extension was built successfully:
   ```bash
   python -c "from groundingdino import _C; print('CUDA extension loaded successfully')"
   ```

### Error: "'Tensor' object has no attribute 'astype'"

This error occurs in model_server.py when passing PyTorch tensors to functions expecting NumPy arrays.

**Solution:**

Modify the model_server.py file to use NumPy arrays with the SAM predictor:

```python
# Change this:
sam_box = torch.tensor(box_pixel, device=self.device)
sam_result = self.sam_predictor.predict(
    box=sam_box.unsqueeze(0),
    multimask_output=False
)

# To this:
sam_box = np.array(box_pixel)
sam_result = self.sam_predictor.predict(
    box=sam_box,
    multimask_output=False
)
```

### Error: "'numpy.ndarray' object has no attribute 'cpu'"

This error occurs when trying to call PyTorch methods on NumPy arrays.

**Solution:**

Modify the model_server.py file to handle NumPy arrays correctly:

```python
# Change this:
mask = sam_result[0][0].cpu().numpy()

# To this:
mask = sam_result[0][0]  # Already a numpy array
```

### Error: "libc10.so: cannot open shared object file"

This error occurs when PyTorch's CUDA libraries can't be found in the system's library path.

**Solution:**

Add the PyTorch library directory to LD_LIBRARY_PATH:

```bash
PYTORCH_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")
export LD_LIBRARY_PATH=$PYTORCH_PATH/lib:$LD_LIBRARY_PATH
```

For persistent configuration, add this to your shell profile or the model server startup script.

## GPU Detection Issues

### Error: "CUDA not available" despite having a GPU

This might happen if PyTorch can't detect the GPU.

**Solution:**

1. Verify the GPU is recognized by the system:
   ```bash
   nvidia-smi
   ```

2. Check CUDA environment variables:
   ```bash
   echo $CUDA_HOME
   echo $CUDA_VISIBLE_DEVICES
   ```

3. Verify PyTorch CUDA detection:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
   ```

4. If PyTorch doesn't detect CUDA, reinstall PyTorch with CUDA support:
   ```bash
   pip uninstall -y torch torchvision
   pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   ```

## Model Server Configuration

### Best Practices

1. Always set these environment variables before starting the model server:
   ```bash
   export PYTHONPATH=/ttt/tree_ml:/ttt/tree_ml/pipeline:/ttt/tree_ml/pipeline/grounded-sam:/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO:/ttt/tree_ml/pipeline/grounded-sam/segment_anything:$PYTHONPATH
   export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
   export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")/lib:$LD_LIBRARY_PATH
   ```

2. Use NumPy 1.x for PyTorch compatibility:
   ```bash
   pip install numpy==1.26.4
   ```

3. Build the model server with proper GPU capabilities:
   ```bash
   python /ttt/tree_ml/pipeline/model_server.py --port 8000 --host 0.0.0.0 --model-dir /ttt/tree_ml/pipeline/model --output-dir /ttt/data/ml --device cuda
   ```

4. Always include full error logs when troubleshooting:
   ```bash
   cat /ttt/tree_ml/logs/model_server.log
   ```

## Testing the Model Server

After fixing issues, test the model server with:

```bash
# Check server status
curl http://localhost:8000/status

# Test detection with a sample image
curl -X POST -F "image=@/ttt/data/tests/test_images/sample.jpg" -F "job_id=test_fix" http://localhost:8000/detect

# Verify the results
ls -la /ttt/data/ml/test_fix/ml_response/
cat /ttt/data/ml/test_fix/ml_response/trees.json
```

A successful detection will return a 200 status code and create detection results in the specified output directory.