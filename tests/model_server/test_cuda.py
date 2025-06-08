#!/usr/bin/env python3
"""
CUDA and Model GPU Usage Tester for Tree ML
===========================================

This script tests GPU compatibility and usage with the Tree ML model server components.
It verifies CUDA availability and attempts to load models on the GPU.
"""

import os
import sys
import json
import logging
import torch
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cuda_model_tester")

# Add necessary paths
REPO_PATH = Path(__file__).parent.parent.parent
MODEL_DIR = REPO_PATH / "tree_ml" / "pipeline" / "model"
PIPELINE_PATH = REPO_PATH / "tree_ml" / "pipeline"

# Add required paths to sys.path
sys.path.append(str(REPO_PATH))
sys.path.append(str(PIPELINE_PATH))
sys.path.append(str(PIPELINE_PATH / "grounded-sam"))
sys.path.append(str(PIPELINE_PATH / "grounded-sam" / "GroundingDINO"))
sys.path.append(str(PIPELINE_PATH / "grounded-sam" / "segment_anything"))

def test_cuda_availability():
    """Test basic CUDA availability and capabilities"""
    logger.info("Testing CUDA availability...")
    
    results = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    }
    
    if results["cuda_available"]:
        logger.info(f"✅ CUDA is available: version {results['cuda_version']}")
        logger.info(f"   Device: {results['device_name']}")
    else:
        logger.error("❌ CUDA is NOT available")
    
    return results

def test_basic_cuda_operations():
    """Test basic CUDA tensor operations"""
    if not torch.cuda.is_available():
        logger.error("Skipping CUDA operations test as CUDA is not available")
        return {"status": "skipped"}
    
    logger.info("Testing basic CUDA operations...")
    results = {"operations": []}
    
    try:
        # Create a CUDA tensor
        device = torch.device("cuda")
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        results["operations"].append({
            "name": "Create CUDA tensor",
            "success": True,
            "device": str(x.device)
        })
        
        # Perform some operations
        y = x * 2
        results["operations"].append({
            "name": "Tensor multiplication",
            "success": True,
            "result": y.cpu().tolist()
        })
        
        # Matrix operation
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # Wait for GPU operation to complete
        end_time = time.time()
        results["operations"].append({
            "name": "Matrix multiplication",
            "success": True,
            "time_taken": end_time - start_time,
            "result_shape": c.shape
        })
        
        logger.info("✅ All basic CUDA operations successful")
        results["status"] = "success"
        
    except Exception as e:
        logger.error(f"❌ CUDA operations test failed: {str(e)}")
        results["status"] = "failed"
        results["error"] = str(e)
    
    return results

def test_model_loading():
    """Test loading models on GPU"""
    if not torch.cuda.is_available():
        logger.error("Skipping model loading test as CUDA is not available")
        return {"status": "skipped"}
    
    logger.info("Testing model loading on GPU...")
    results = {"models": []}
    
    try:
        # Import the required modules
        from groundingdino.util.inference import load_model as load_grounding_dino
        # Try the correct import path for SLConfig
        try:
            from groundingdino.util.slconfig import SLConfig
            logger.info("✅ Successfully imported SLConfig from groundingdino.util.slconfig")
            results["slconfig_import"] = "groundingdino.util.slconfig"
        except ImportError:
            try:
                from groundingdino.config import SLConfig
                logger.info("✅ Successfully imported SLConfig from groundingdino.config")
                results["slconfig_import"] = "groundingdino.config"
            except ImportError:
                logger.error("❌ Failed to import SLConfig from any location")
                results["slconfig_import"] = "failed"
                return results
        
        # Try to import segment_anything
        try:
            from segment_anything import sam_model_registry, SamPredictor
            logger.info("✅ Successfully imported segment_anything")
            results["sam_import"] = "success"
        except ImportError as e:
            logger.error(f"❌ Failed to import segment_anything: {e}")
            results["sam_import"] = "failed"
            return results
            
        # Check model files
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths to model files
        grounding_dino_config_path = PIPELINE_PATH / "grounded-sam" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
        grounding_dino_weights_path = MODEL_DIR / "groundingdino_swint_ogc.pth"
        sam_weights_path = MODEL_DIR / "sam_vit_h_4b8939.pth"
        
        # Check if files exist
        results["model_files"] = {
            "grounding_dino_config": {
                "path": str(grounding_dino_config_path),
                "exists": grounding_dino_config_path.exists()
            },
            "grounding_dino_weights": {
                "path": str(grounding_dino_weights_path),
                "exists": grounding_dino_weights_path.exists()
            },
            "sam_weights": {
                "path": str(sam_weights_path),
                "exists": sam_weights_path.exists()
            }
        }
        
        # Log file status
        for name, info in results["model_files"].items():
            status = "✅ exists" if info["exists"] else "❌ missing"
            logger.info(f"{name}: {status} at {info['path']}")
        
        # Try loading GroundingDINO model
        if results["model_files"]["grounding_dino_config"]["exists"] and \
           results["model_files"]["grounding_dino_weights"]["exists"]:
            logger.info("Attempting to load GroundingDINO model...")
            
            # Record GPU memory before loading
            if torch.cuda.is_available():
                before_mem = torch.cuda.memory_allocated(0) / 1024**2
            
            # Try the original loading method
            try:
                args = SLConfig.fromfile(str(grounding_dino_config_path))
                args.device = device
                grounding_dino = load_grounding_dino(str(grounding_dino_weights_path), args, device)
                is_on_gpu = next(grounding_dino.parameters()).device.type == "cuda"
                
                results["models"].append({
                    "name": "GroundingDINO",
                    "success": True,
                    "method": "original",
                    "device": str(next(grounding_dino.parameters()).device),
                    "on_gpu": is_on_gpu
                })
                
                logger.info(f"✅ GroundingDINO loaded successfully on {next(grounding_dino.parameters()).device}")
                
            except Exception as e:
                logger.warning(f"Original loading method failed: {e}")
                
                # Try alternative loading method
                try:
                    logger.info("Using alternative config loading method...")
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("config", str(grounding_dino_config_path))
                    config = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config)
                    
                    # Create args from config
                    class Args:
                        def __init__(self, **kwargs):
                            self.__dict__.update(kwargs)
                    
                    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
                    args = Args(**config_dict)
                    args.device = device
                    
                    # Load model with alternative method
                    from groundingdino.models import build_model
                    from groundingdino.util.utils import clean_state_dict
                    
                    model = build_model(args)
                    checkpoint = torch.load(str(grounding_dino_weights_path), map_location=device)
                    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
                    model.eval()
                    model = model.to(device)
                    
                    is_on_gpu = next(model.parameters()).device.type == "cuda"
                    
                    results["models"].append({
                        "name": "GroundingDINO",
                        "success": True,
                        "method": "alternative",
                        "device": str(next(model.parameters()).device),
                        "on_gpu": is_on_gpu
                    })
                    
                    logger.info(f"✅ GroundingDINO loaded with alternative method on {next(model.parameters()).device}")
                    
                except Exception as e2:
                    logger.error(f"❌ Both loading methods failed for GroundingDINO: {e2}")
                    results["models"].append({
                        "name": "GroundingDINO",
                        "success": False,
                        "error": str(e2)
                    })
        
        # Try loading SAM model
        if results["model_files"]["sam_weights"]["exists"]:
            logger.info("Attempting to load SAM model...")
            try:
                sam = sam_model_registry["vit_h"](checkpoint=str(sam_weights_path))
                sam.to(device=device)
                
                # Check if model is on GPU
                is_on_gpu = next(sam.parameters()).device.type == "cuda"
                
                results["models"].append({
                    "name": "SAM",
                    "success": True,
                    "device": str(next(sam.parameters()).device),
                    "on_gpu": is_on_gpu
                })
                
                logger.info(f"✅ SAM model loaded successfully on {next(sam.parameters()).device}")
                
                # Create predictor
                sam_predictor = SamPredictor(sam)
                logger.info("✅ SAM predictor created successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load SAM model: {e}")
                results["models"].append({
                    "name": "SAM",
                    "success": False,
                    "error": str(e)
                })
        
        # Record GPU memory after loading models
        if torch.cuda.is_available():
            after_mem = torch.cuda.memory_allocated(0) / 1024**2
            results["gpu_memory"] = {
                "before_loading_mb": before_mem,
                "after_loading_mb": after_mem,
                "difference_mb": after_mem - before_mem
            }
            logger.info(f"GPU memory used by models: {after_mem - before_mem:.2f} MB")
        
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {str(e)}")
        results["status"] = "failed"
        results["error"] = str(e)
    
    return results

def main():
    """Run all tests and save results"""
    logger.info("=" * 60)
    logger.info("CUDA and GPU Model Test for Tree ML")
    logger.info("=" * 60)
    
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cuda_test": test_cuda_availability(),
        "cuda_operations": test_basic_cuda_operations(),
        "model_loading": test_model_loading()
    }
    
    # Determine overall status
    cuda_ok = results["cuda_test"]["cuda_available"]
    ops_ok = results["cuda_operations"]["status"] == "success" if cuda_ok else True
    models_ok = all(model.get("on_gpu", False) for model in results["model_loading"].get("models", []))
    
    results["overall"] = {
        "cuda_available": cuda_ok,
        "operations_working": ops_ok,
        "models_on_gpu": models_ok,
        "status": "success" if (cuda_ok and ops_ok and models_ok) else "failed"
    }
    
    # Save results
    output_dir = Path("/home/ss/tree-risk-pro/data/tests/ml_test_results/gpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "model_gpu_test_results.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Test results saved to {output_file}")
    logger.info("=" * 60)
    logger.info("Test Summary:")
    logger.info(f"CUDA Available: {'✅' if cuda_ok else '❌'}")
    logger.info(f"Basic CUDA Operations: {'✅' if ops_ok else '❌'}")
    logger.info(f"Models on GPU: {'✅' if models_ok else '❌'}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()