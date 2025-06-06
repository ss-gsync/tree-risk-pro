#!/usr/bin/env python3
"""
Tests for the tree detection model server.
Tests model initialization, API endpoints, and detection functionality.
"""

import os
import sys
import unittest
import json
from pathlib import Path
import pytest
import torch

# Add the parent directory to the path so we can import the module
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

try:
    # Import the model server module - wrapped in try/except to handle import errors gracefully
    from tree_ml.pipeline.model_server import app, GroundedSAMServer
    from tests.model_server.test_client import create_test_client
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    IMPORTS_AVAILABLE = False

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestModelServer(unittest.TestCase):
    """Test the model server functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test client and check CUDA availability."""
        if IMPORTS_AVAILABLE:
            cls.client = create_test_client(app)
            cls.cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {cls.cuda_available}")
            if cls.cuda_available:
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    def test_imports(self):
        """Test that all required modules can be imported."""
        self.assertTrue(IMPORTS_AVAILABLE, "All required imports are available")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("device", data)
        
        # Check if the device matches CUDA availability
        if self.cuda_available:
            self.assertEqual(data["device"], "cuda")
        else:
            self.assertEqual(data["device"], "cpu")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_model_initialization(self):
        """Test that the model can be initialized."""
        # Skip this test if the model server module couldn't be imported
        if not IMPORTS_AVAILABLE:
            self.skipTest("Model server module not available")
            
        # Initialize the model with a small test to verify it loads
        try:
            model_dir = os.path.join(ROOT_DIR, "tree_ml/pipeline/model")
            output_dir = os.path.join(ROOT_DIR, "data/ml")
            
            # Use CPU for testing if CUDA is not available
            device = "cuda" if self.cuda_available else "cpu"
            
            # Initialize the model with test_mode parameter
            # This parameter should be added to the GroundedSAMServer constructor
            # to skip full model loading for tests
            server = GroundedSAMServer(
                model_dir=model_dir,
                output_dir=output_dir,
                device=device
            )
            
            self.assertIsNotNone(server)
            self.assertEqual(server.device, device)
            self.assertEqual(server.model_dir, model_dir)
            self.assertEqual(server.output_dir, output_dir)
        except Exception as e:
            self.fail(f"Model initialization failed: {str(e)}")
    
    @pytest.mark.skipif(not torch.cuda.is_available() or not IMPORTS_AVAILABLE,
                       reason="CUDA not available or imports missing")
    def test_model_cuda_functionality(self):
        """Test CUDA-specific functionality (skip if CUDA not available)."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")
        
        model_dir = os.path.join(ROOT_DIR, "tree_ml/pipeline/model")
        output_dir = os.path.join(ROOT_DIR, "data/ml")
        
        # Initialize with CUDA
        server = GroundedSAMServer(
            model_dir=model_dir,
            output_dir=output_dir,
            device="cuda"
        )
        
        self.assertEqual(server.device, "cuda")
        # Add more CUDA-specific tests as needed
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_detection_endpoint_validation(self):
        """Test the detection endpoint input validation."""
        # Test with missing required fields
        response = self.client.post("/detect", json={})
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        
        # Test with invalid coordinates
        invalid_data = {
            "image_path": "/path/to/image.jpg",
            "bounds": [[-200, 100], [-190, 110]]  # Invalid longitude
        }
        response = self.client.post("/detect", json=invalid_data)
        self.assertEqual(response.status_code, 422)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_detection_job_status_endpoint(self, sample_job_id):
        """Test the job status endpoint."""
        # Test with non-existent job ID
        non_existent_job_id = "non_existent_job_id"
        response = self.client.get(f"/job/{non_existent_job_id}")
        self.assertEqual(response.status_code, 404)
        
        # Test with existing job ID if available
        if sample_job_id:
            sample_job_path = os.path.join(ROOT_DIR, f"data/ml/{sample_job_id}/ml_response/metadata.json")
            
            if os.path.exists(sample_job_path):
                response = self.client.get(f"/job/{sample_job_id}")
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertEqual(data["job_id"], sample_job_id)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_output_directory_structure(self, output_dir, sample_job_id):
        """Test that the output directory has the expected structure."""
        # Check that the output directory exists
        self.assertTrue(os.path.exists(output_dir), 
                       f"Output directory does not exist: {output_dir}")
        
        # Check for a sample job if available
        if sample_job_id:
            sample_job_dir = os.path.join(output_dir, sample_job_id)
            
            if os.path.exists(sample_job_dir):
                # Check for required files
                ml_response_dir = os.path.join(sample_job_dir, "ml_response")
                self.assertTrue(os.path.exists(ml_response_dir))
                
                metadata_path = os.path.join(ml_response_dir, "metadata.json")
                self.assertTrue(os.path.exists(metadata_path))
                
                # Verify metadata structure
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                required_fields = ['job_id', 'timestamp', 'image_path', 'bounds',
                                  'detection_count', 'coordinate_system']
                for field in required_fields:
                    self.assertIn(field, metadata)
                
                # Verify coordinate system is as expected
                self.assertEqual(metadata['coordinate_system'], 's2')
            

if __name__ == "__main__":
    unittest.main()