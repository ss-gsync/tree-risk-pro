#!/usr/bin/env python3
"""
CUDA Tester - GPU compatibility testing for Tree ML project

This script provides comprehensive testing of CUDA and GPU capabilities
for machine learning models used in the Tree ML project. It verifies:

1. CUDA installation and configuration
2. PyTorch GPU compatibility
3. Basic model operations on GPU
4. Environment configuration

Usage examples:
  # Run basic CUDA tests
  python cuda_tester.py

  # Run with extra diagnostic info
  python cuda_tester.py --verbose

  # Run with additional model tests
  python cuda_tester.py --models
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cuda_tester')

# Default directory for results
RESULTS_DIR = "/ttt/data/tests/ml_test_results/cuda"
os.makedirs(RESULTS_DIR, exist_ok=True)


def check_system_info():
    """Gather system information"""
    import platform
    import subprocess
    
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0]
    }
    
    # Try to get GPU info from nvidia-smi
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False
        )
        if nvidia_smi.returncode == 0:
            gpu_info = nvidia_smi.stdout.strip().split('\n')
            system_info["gpu_info"] = gpu_info
        else:
            system_info["gpu_info"] = "nvidia-smi command failed"
    except Exception as e:
        system_info["gpu_info"] = f"Error running nvidia-smi: {e}"
    
    return system_info


def check_cuda_installation():
    """Check CUDA installation and configuration"""
    cuda_info = {
        "cuda_available": False,
        "cuda_version": "Not available",
        "cudnn_version": "Not available",
        "device_count": 0,
        "devices": []
    }
    
    # Check environment variables
    env_vars = {}
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"]:
        env_vars[var] = os.environ.get(var, "Not set")
    
    cuda_info["environment"] = env_vars
    
    # Try importing torch
    try:
        import torch
        cuda_info["torch_version"] = torch.__version__
        cuda_info["torch_cuda_available"] = torch.cuda.is_available()
        
        if torch.cuda.is_available():
            cuda_info["cuda_available"] = True
            cuda_info["cuda_version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()
            
            # Get device info
            for i in range(torch.cuda.device_count()):
                device_info = {
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i)
                }
                cuda_info["devices"].append(device_info)
        
        # Check cuDNN
        if hasattr(torch.backends, 'cudnn'):
            cuda_info["cudnn_available"] = torch.backends.cudnn.is_available()
            cuda_info["cudnn_enabled"] = torch.backends.cudnn.enabled
            if hasattr(torch.backends.cudnn, 'version'):
                cuda_info["cudnn_version"] = str(torch.backends.cudnn.version())
    except ImportError:
        cuda_info["error"] = "PyTorch not installed"
    except Exception as e:
        cuda_info["error"] = f"Error checking CUDA: {e}"
    
    return cuda_info


def test_cuda_operations():
    """Test basic CUDA tensor operations"""
    if not is_torch_available():
        return {"status": "skipped", "reason": "PyTorch not available"}
    
    import torch
    
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA not available"}
    
    try:
        results = {
            "status": "success",
            "operations_tested": 0,
            "operations_succeeded": 0,
            "details": []
        }
        
        # Test 1: Create tensor on CPU and move to GPU
        try:
            tensor_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
            tensor_gpu = tensor_cpu.cuda()
            cpu_data = tensor_gpu.cpu().tolist()
            
            test_result = {
                "name": "CPU to GPU transfer",
                "success": True,
                "data": cpu_data
            }
            results["operations_tested"] += 1
            results["operations_succeeded"] += 1
        except Exception as e:
            test_result = {
                "name": "CPU to GPU transfer",
                "success": False,
                "error": str(e)
            }
            results["operations_tested"] += 1
        
        results["details"].append(test_result)
        
        # Test 2: Create tensor directly on GPU
        try:
            tensor_gpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device="cuda:0")
            cpu_data = tensor_gpu.cpu().tolist()
            
            test_result = {
                "name": "Direct GPU tensor creation",
                "success": True,
                "data": cpu_data
            }
            results["operations_tested"] += 1
            results["operations_succeeded"] += 1
        except Exception as e:
            test_result = {
                "name": "Direct GPU tensor creation",
                "success": False,
                "error": str(e)
            }
            results["operations_tested"] += 1
        
        results["details"].append(test_result)
        
        # Test 3: Basic arithmetic on GPU
        try:
            a = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
            b = torch.tensor([4.0, 5.0, 6.0], device="cuda:0")
            c = a + b
            d = a * b
            
            test_result = {
                "name": "GPU arithmetic",
                "success": True,
                "addition": c.cpu().tolist(),
                "multiplication": d.cpu().tolist()
            }
            results["operations_tested"] += 1
            results["operations_succeeded"] += 1
        except Exception as e:
            test_result = {
                "name": "GPU arithmetic",
                "success": False,
                "error": str(e)
            }
            results["operations_tested"] += 1
        
        results["details"].append(test_result)
        
        # Test 4: Matrix operations
        try:
            matrix_a = torch.randn(3, 4).cuda()
            matrix_b = torch.randn(4, 5).cuda()
            matrix_c = torch.matmul(matrix_a, matrix_b)
            
            test_result = {
                "name": "GPU matrix operations",
                "success": True,
                "shape": list(matrix_c.shape)
            }
            results["operations_tested"] += 1
            results["operations_succeeded"] += 1
        except Exception as e:
            test_result = {
                "name": "GPU matrix operations",
                "success": False,
                "error": str(e)
            }
            results["operations_tested"] += 1
        
        results["details"].append(test_result)
        
        # Test 5: Memory usage
        try:
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
            
            # Allocate a large tensor
            large_tensor = torch.zeros(1000, 1000, device="cuda:0")
            mem_after = torch.cuda.memory_allocated()
            
            test_result = {
                "name": "GPU memory allocation",
                "success": True,
                "memory_before_mb": mem_before / (1024 * 1024),
                "memory_after_mb": mem_after / (1024 * 1024),
                "difference_mb": (mem_after - mem_before) / (1024 * 1024)
            }
            results["operations_tested"] += 1
            results["operations_succeeded"] += 1
        except Exception as e:
            test_result = {
                "name": "GPU memory allocation",
                "success": False,
                "error": str(e)
            }
            results["operations_tested"] += 1
        
        results["details"].append(test_result)
        
        return results
    
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


def test_simple_model():
    """Test a simple neural network model on GPU"""
    if not is_torch_available():
        return {"status": "skipped", "reason": "PyTorch not available"}
    
    import torch
    
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA not available"}
    
    try:
        # Create a simple model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(10, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, 20),
                    torch.nn.ReLU(),
                    torch.nn.Linear(20, 1)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Initialize model
        model = SimpleModel()
        
        # Test on CPU first
        input_cpu = torch.randn(5, 10)
        output_cpu = model(input_cpu)
        
        # Move to GPU
        model = model.cuda()
        input_gpu = input_cpu.cuda()
        
        # Forward pass on GPU
        output_gpu = model(input_gpu)
        
        # Backward pass (test gradients)
        loss = output_gpu.mean()
        loss.backward()
        
        return {
            "status": "success",
            "cpu_output_shape": list(output_cpu.shape),
            "gpu_output_shape": list(output_gpu.shape),
            "model_on_gpu": next(model.parameters()).is_cuda,
            "has_gradients": all(p.grad is not None for p in model.parameters())
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


def test_ml_models():
    """Test if ML models can be loaded and moved to GPU"""
    results = {
        "deepforest": {"status": "not_tested"},
        "sam": {"status": "not_tested"}
    }
    
    # Test DeepForest
    try:
        import deepforest
        from deepforest import main
        
        model = deepforest.main.deepforest()
        model.use_release()
        
        # Check if model can be moved to GPU
        if is_cuda_available():
            model.model.to("cuda:0")
            results["deepforest"] = {
                "status": "success",
                "loaded": True,
                "on_gpu": next(model.model.parameters()).is_cuda
            }
        else:
            results["deepforest"] = {
                "status": "skipped",
                "reason": "CUDA not available"
            }
    except ImportError:
        results["deepforest"] = {
            "status": "skipped",
            "reason": "DeepForest not installed"
        }
    except Exception as e:
        results["deepforest"] = {
            "status": "failed",
            "error": str(e)
        }
    
    # Test SAM
    try:
        from segment_anything import sam_model_registry
        
        # Find checkpoint
        sam_checkpoints = [
            "/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth",
            "/ttt/tree_ml/pipeline/model/sam_vit_l_0b3195.pth",
            "/ttt/tree_ml/pipeline/model/sam_vit_b_01ec64.pth"
        ]
        
        sam_checkpoint = None
        for path in sam_checkpoints:
            if os.path.exists(path):
                sam_checkpoint = path
                break
        
        if sam_checkpoint:
            if "vit_h" in sam_checkpoint:
                model_type = "vit_h"
            elif "vit_l" in sam_checkpoint:
                model_type = "vit_l"
            elif "vit_b" in sam_checkpoint:
                model_type = "vit_b"
            else:
                model_type = "vit_b"  # default
            
            if is_cuda_available():
                # Load model (might take a while)
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device="cuda:0")
                
                results["sam"] = {
                    "status": "success",
                    "loaded": True,
                    "model_type": model_type,
                    "checkpoint": sam_checkpoint,
                    "on_gpu": next(sam.parameters()).is_cuda
                }
            else:
                results["sam"] = {
                    "status": "skipped",
                    "reason": "CUDA not available",
                    "checkpoint_found": sam_checkpoint
                }
        else:
            results["sam"] = {
                "status": "skipped",
                "reason": "No SAM checkpoint found"
            }
    except ImportError:
        results["sam"] = {
            "status": "skipped",
            "reason": "SAM not installed"
        }
    except Exception as e:
        results["sam"] = {
            "status": "failed",
            "error": str(e)
        }
    
    return results


def is_torch_available():
    """Check if PyTorch is available"""
    try:
        import torch
        return True
    except ImportError:
        return False


def is_cuda_available():
    """Check if CUDA is available through PyTorch"""
    if not is_torch_available():
        return False
    
    import torch
    return torch.cuda.is_available()


def generate_test_report(system_info, cuda_info, operations_results, model_results=None, simple_model_results=None):
    """Generate comprehensive test report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "cuda_info": cuda_info,
        "operations_test": operations_results
    }
    
    if model_results is not None:
        report["ml_models_test"] = model_results
    
    if simple_model_results is not None:
        report["simple_model_test"] = simple_model_results
    
    # Overall test result
    cuda_success = cuda_info.get("cuda_available", False)
    ops_success = operations_results.get("status") == "success" and operations_results.get("operations_succeeded", 0) > 0
    
    report["summary"] = {
        "cuda_available": cuda_success,
        "basic_operations": ops_success,
        "overall_status": "success" if cuda_success and ops_success else "failure"
    }
    
    return report


def main():
    """Main function for CUDA testing"""
    parser = argparse.ArgumentParser(description="CUDA/GPU compatibility tester for Tree ML")
    parser.add_argument("--verbose", action="store_true", help="Show detailed diagnostic information")
    parser.add_argument("--models", action="store_true", help="Test ML models (DeepForest, SAM)")
    parser.add_argument("--nosave", action="store_true", help="Don't save test report to file")
    
    args = parser.parse_args()
    
    # Banner
    logger.info("=" * 60)
    logger.info("CUDA/GPU Compatibility Tester")
    logger.info("=" * 60)
    
    # System info
    logger.info("Gathering system information...")
    system_info = check_system_info()
    if args.verbose:
        logger.info(f"System: {system_info['platform']}")
        logger.info(f"Python: {system_info['python_version']}")
        if "gpu_info" in system_info and isinstance(system_info["gpu_info"], list):
            for i, gpu in enumerate(system_info["gpu_info"]):
                logger.info(f"GPU {i}: {gpu}")
    
    # CUDA installation
    logger.info("Checking CUDA installation...")
    cuda_info = check_cuda_installation()
    
    if cuda_info.get("cuda_available", False):
        logger.info(f"✅ CUDA is available: version {cuda_info['cuda_version']}")
        if "devices" in cuda_info and cuda_info["devices"]:
            for i, device in enumerate(cuda_info["devices"]):
                logger.info(f"  GPU {i}: {device['name']}")
    else:
        logger.warning("❌ CUDA is not available")
        if "error" in cuda_info:
            logger.error(f"Error: {cuda_info['error']}")
    
    if args.verbose:
        logger.info("CUDA environment variables:")
        for var, value in cuda_info.get("environment", {}).items():
            logger.info(f"  {var}: {value}")
    
    # Test CUDA operations
    logger.info("Testing CUDA tensor operations...")
    operations_results = test_cuda_operations()
    
    if operations_results.get("status") == "success":
        success_rate = operations_results.get("operations_succeeded", 0) / max(1, operations_results.get("operations_tested", 1))
        if success_rate == 1.0:
            logger.info(f"✅ All CUDA operations succeeded: {operations_results['operations_succeeded']}/{operations_results['operations_tested']}")
        else:
            logger.warning(f"⚠️ Some CUDA operations failed: {operations_results['operations_succeeded']}/{operations_results['operations_tested']}")
    elif operations_results.get("status") == "skipped":
        logger.warning(f"⚠️ CUDA operations test skipped: {operations_results.get('reason')}")
    else:
        logger.error(f"❌ CUDA operations test failed: {operations_results.get('error', 'Unknown error')}")
    
    # Test simple model
    logger.info("Testing simple neural network on GPU...")
    simple_model_results = test_simple_model()
    
    if simple_model_results.get("status") == "success":
        logger.info("✅ Simple model test successful")
        if args.verbose:
            logger.info(f"  Model on GPU: {simple_model_results.get('model_on_gpu', False)}")
            logger.info(f"  Has gradients: {simple_model_results.get('has_gradients', False)}")
    elif simple_model_results.get("status") == "skipped":
        logger.warning(f"⚠️ Simple model test skipped: {simple_model_results.get('reason')}")
    else:
        logger.error(f"❌ Simple model test failed: {simple_model_results.get('error', 'Unknown error')}")
    
    # Test ML models
    model_results = None
    if args.models:
        logger.info("Testing ML models on GPU...")
        model_results = test_ml_models()
        
        # DeepForest
        df_status = model_results.get("deepforest", {}).get("status", "not_tested")
        if df_status == "success":
            logger.info("✅ DeepForest model test successful")
            if args.verbose:
                logger.info(f"  Model on GPU: {model_results['deepforest'].get('on_gpu', False)}")
        elif df_status == "skipped":
            logger.warning(f"⚠️ DeepForest model test skipped: {model_results['deepforest'].get('reason')}")
        elif df_status == "failed":
            logger.error(f"❌ DeepForest model test failed: {model_results['deepforest'].get('error', 'Unknown error')}")
        
        # SAM
        sam_status = model_results.get("sam", {}).get("status", "not_tested")
        if sam_status == "success":
            logger.info("✅ SAM model test successful")
            if args.verbose:
                logger.info(f"  Model type: {model_results['sam'].get('model_type', 'unknown')}")
                logger.info(f"  Model on GPU: {model_results['sam'].get('on_gpu', False)}")
        elif sam_status == "skipped":
            logger.warning(f"⚠️ SAM model test skipped: {model_results['sam'].get('reason')}")
        elif sam_status == "failed":
            logger.error(f"❌ SAM model test failed: {model_results['sam'].get('error', 'Unknown error')}")
    
    # Generate test report
    report = generate_test_report(
        system_info, 
        cuda_info, 
        operations_results,
        model_results,
        simple_model_results
    )
    
    # Save report
    if not args.nosave:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        report_path = os.path.join(RESULTS_DIR, "cuda_test_report.json")
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Test report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving test report: {e}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("CUDA Test Summary")
    logger.info("=" * 60)
    
    if report["summary"]["overall_status"] == "success":
        logger.info("✅ CUDA is properly configured and working")
    else:
        logger.warning("⚠️ CUDA configuration incomplete or not working")
        
        # Provide troubleshooting help
        if not cuda_info.get("cuda_available", False):
            logger.info("\nTroubleshooting tips:")
            logger.info("1. Check if NVIDIA drivers are installed:")
            logger.info("   nvidia-smi")
            logger.info("2. Check CUDA installation:")
            logger.info("   nvcc --version")
            logger.info("3. Ensure PyTorch was installed with CUDA support:")
            logger.info("   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117")
            logger.info("4. Check environment variables (CUDA_HOME, PATH)")
    
    return 0 if report["summary"]["overall_status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())