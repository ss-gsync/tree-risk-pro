#!/usr/bin/env python3
"""
Diagnostic script to test GPU availability and compatibility
"""

import os
import sys
import numpy as np
import torch
from PIL import Image

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # CUDA is working fine
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Try a simple CUDA operation
    a = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
    b = a * 2.0
    print(f"CUDA computation successful: {b.to('cpu').numpy()}")
else:
    # CUDA error - print diagnostic information
    print(f"CUDA version from PyTorch: {torch.version.cuda}")
    print("\nDiagnosing CUDA issue:")
    
    try:
        # Try to manually initialize CUDA
        print("Attempting manual CUDA initialization...")
        device_count = torch._C._cuda_getDeviceCount()
        print(f"CUDA device count: {device_count}")
    except Exception as e:
        print(f"CUDA initialization error: {str(e)}")
    
    print("\nSystem Environment:")
    for var in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'PATH']:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

# Try to import DeepForest and SAM
try:
    import deepforest
    print(f"\nDeepForest version: {deepforest.__version__}")
    
    try:
        model = deepforest.main.deepforest()
        print("DeepForest model initialized successfully")
    except Exception as e:
        print(f"DeepForest model initialization error: {str(e)}")
    
    # Try creating a simple model on CPU first
    try:
        print("\nCreating small test model on CPU...")
        small_model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU()
        )
        print("Model created on CPU successfully")
        
        # Try moving to CUDA if available
        if torch.cuda.is_available():
            print("Moving model to CUDA...")
            small_model.to("cuda:0")
            print("Model moved to CUDA successfully")
            
            # Test with dummy input
            dummy_input = torch.randn(1, 10, device="cuda:0")
            output = small_model(dummy_input)
            print("Model inference on CUDA successful")
    except Exception as e:
        print(f"Model test error: {str(e)}")
        
except ImportError:
    print("DeepForest not available")

try:
    from segment_anything import sam_model_registry
    print("\nSAM module imported successfully")
except ImportError:
    print("SAM not available")

print("\nGPU diagnosis complete")