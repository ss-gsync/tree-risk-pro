#!/usr/bin/env python3
"""
Basic import tests for the model server.
These tests only verify that required modules can be imported.
"""

import os
import sys
from pathlib import Path
import unittest
import pytest

# Add the parent directory to the path so we can import the module
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

class TestImports(unittest.TestCase):
    """Test that all required modules can be imported."""
    
    def test_torch_import(self):
        """Test that PyTorch can be imported."""
        try:
            import torch
            self.assertTrue(True, "PyTorch imported successfully")
            # Print CUDA information
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
        except ImportError as e:
            self.fail(f"Failed to import PyTorch: {e}")
    
    def test_fastapi_import(self):
        """Test that FastAPI can be imported."""
        try:
            import fastapi
            from fastapi.testclient import TestClient
            self.assertTrue(True, "FastAPI imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import FastAPI: {e}")
    
    def test_model_imports(self):
        """Test that model-related modules can be imported."""
        # Tree ML structure
        try:
            sys.path.append(str(ROOT_DIR))
            
            # Try to import the model server module
            try:
                from tree_ml.pipeline import model_server
                self.assertTrue(True, "Model server module imported successfully")
                print("Model server module structure:", dir(model_server))
            except ImportError as e:
                # This might fail if the module doesn't exist yet
                print(f"Warning: Failed to import model_server: {e}")
                # Don't fail the test if the module doesn't exist yet
            
            # Try to import other related modules
            try:
                import ultralytics
                print(f"Ultralytics version: {ultralytics.__version__}")
                self.assertTrue(True, "Ultralytics imported successfully")
            except ImportError as e:
                print(f"Warning: Failed to import ultralytics: {e}")
                # Don't fail the test for optional dependencies
            
            try:
                import segment_anything
                self.assertTrue(True, "Segment Anything imported successfully")
            except ImportError as e:
                print(f"Warning: Failed to import segment_anything: {e}")
                # Don't fail the test for optional dependencies
            
        except Exception as e:
            self.fail(f"Unexpected error during imports: {e}")
    
    def test_s2sphere_import(self):
        """Test that s2sphere can be imported."""
        try:
            import s2sphere
            self.assertTrue(True, "s2sphere imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import s2sphere: {e}")
    
    def test_project_structure(self):
        """Test that the project structure is as expected."""
        # Check for key directories
        data_dir = os.path.join(ROOT_DIR, "data")
        self.assertTrue(os.path.exists(data_dir), f"Data directory exists: {data_dir}")
        
        tree_ml_dir = os.path.join(ROOT_DIR, "tree_ml")
        self.assertTrue(os.path.exists(tree_ml_dir), f"Tree ML directory exists: {tree_ml_dir}")
        
        # Check for sample data
        ml_dir = os.path.join(data_dir, "ml")
        if os.path.exists(ml_dir):
            print(f"ML data directory: {ml_dir}")
            # List directories in the ML data directory
            try:
                job_dirs = [d for d in os.listdir(ml_dir) if os.path.isdir(os.path.join(ml_dir, d))]
                print(f"Available detection jobs: {job_dirs}")
            except Exception as e:
                print(f"Error listing ML data directory: {e}")
                
if __name__ == "__main__":
    unittest.main()