"""
Basic tests for the model server functionality.
"""

import os
import sys
import unittest
import pytest
from pathlib import Path

# Add the parent directory to the path so we can import the module
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

class TestBasic(unittest.TestCase):
    """Basic tests for the model server functionality."""
    
    def test_fastapi_import(self):
        """Test that FastAPI can be imported."""
        try:
            import fastapi
            from fastapi import FastAPI
            self.assertTrue(True, "FastAPI imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import FastAPI: {e}")
    
    def test_torch_import(self):
        """Test that PyTorch can be imported."""
        try:
            import torch
            self.assertTrue(True, "PyTorch imported successfully")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
        except ImportError as e:
            self.fail(f"Failed to import PyTorch: {e}")
    
    def test_model_server_import(self):
        """Test that the model server module can be imported."""
        try:
            from tree_ml.pipeline import model_server
            self.assertTrue(True, "Model server module imported successfully")
            print("Model server module structure:", dir(model_server))
        except ImportError as e:
            self.fail(f"Failed to import model_server: {e}")
    
    def test_s2sphere_import(self):
        """Test that s2sphere can be imported."""
        try:
            import s2sphere
            self.assertTrue(True, "s2sphere imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import s2sphere: {e}")
    
    def test_ultralytics_import(self):
        """Test that ultralytics can be imported."""
        try:
            import ultralytics
            print(f"Ultralytics version: {ultralytics.__version__}")
            self.assertTrue(True, "Ultralytics imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import ultralytics: {e}")
    
    def test_segment_anything_import(self):
        """Test that segment_anything can be imported."""
        try:
            import segment_anything
            self.assertTrue(True, "Segment Anything imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import segment_anything: {e}")
    
    def test_sample_data_exists(self):
        """Test that sample data exists."""
        sample_job_id = "detection_1748997756"
        sample_job_dir = os.path.join(ROOT_DIR, "data/ml", sample_job_id)
        
        self.assertTrue(os.path.exists(sample_job_dir), 
                      f"Sample job directory exists: {sample_job_dir}")
        
        ml_response_dir = os.path.join(sample_job_dir, "ml_response")
        self.assertTrue(os.path.exists(ml_response_dir),
                      f"ML response directory exists: {ml_response_dir}")
        
        metadata_path = os.path.join(ml_response_dir, "metadata.json")
        self.assertTrue(os.path.exists(metadata_path),
                      f"Metadata file exists: {metadata_path}")
        
        trees_path = os.path.join(ml_response_dir, "trees.json")
        self.assertTrue(os.path.exists(trees_path),
                      f"Trees file exists: {trees_path}")
        
if __name__ == "__main__":
    unittest.main()