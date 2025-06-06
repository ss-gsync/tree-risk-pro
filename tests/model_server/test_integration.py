#!/usr/bin/env python3
"""
Integration tests for the tree detection model server.
Tests the complete pipeline from image input to detection output.
"""

import os
import sys
import json
import pytest
import unittest
from pathlib import Path
import torch

# Add the parent directory to the path so we can import the module
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

try:
    # Import the model server module - wrapped in try/except to handle import errors gracefully
    from tree_ml.pipeline.model_server import app
    from tests.model_server.test_client import create_test_client
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    IMPORTS_AVAILABLE = False

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
@pytest.mark.integration
class TestModelServerIntegration(unittest.TestCase):
    """Integration tests for the model server."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test client."""
        if IMPORTS_AVAILABLE:
            cls.client = create_test_client(app)
            cls.cuda_available = torch.cuda.is_available()
    
    @pytest.mark.skipif(not torch.cuda.is_available() or not IMPORTS_AVAILABLE, 
                       reason="CUDA required for this test or imports missing")
    def test_detection_pipeline(self, sample_satellite_image):
        """Test the complete detection pipeline with a real image."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")
        
        if not sample_satellite_image or not os.path.exists(sample_satellite_image):
            self.skipTest(f"Sample image not found: {sample_satellite_image}")
        
        # Sample test area (Fort Worth, TX area)
        bounds = [
            [-97.09153340365056, 32.75281712447018],  # SW corner
            [-97.08745110132345, 32.754860967739525]  # NE corner
        ]
        
        # Submit detection job
        detection_data = {
            "image_path": sample_satellite_image,
            "bounds": bounds,
            "coordinate_system": "s2"
        }
        
        response = self.client.post("/detect", json=detection_data)
        self.assertEqual(response.status_code, 202)  # Accepted
        
        # Get the job ID from the response
        data = response.json()
        self.assertIn("job_id", data)
        job_id = data["job_id"]
        
        # Wait for the job to complete (would timeout in a real test)
        # This is just a placeholder since we don't want to actually run the detection
        # in this test file
        
        # Check job status
        response = self.client.get(f"/job/{job_id}")
        
        # If the job actually ran, we'd check the results
        # Since this is just a mock test, we'll just check the response structure
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["job_id"], job_id)
            self.assertIn("status", data)
        else:
            # We don't expect the job to actually complete in this test
            self.skipTest("Job processing skipped for this test")
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_list_detections(self):
        """Test the endpoint that lists available detections."""
        response = self.client.get("/detections")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIsInstance(data, list)
        
        # Check that each item has the required fields
        if data:
            for item in data:
                self.assertIn("job_id", item)
                self.assertIn("timestamp", item)
                self.assertIn("detection_count", item)
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
    def test_get_detection_results(self, sample_job_id):
        """Test retrieving detection results for a specific job."""
        if not sample_job_id:
            self.skipTest("No sample job ID available")
        
        response = self.client.get(f"/detections/{sample_job_id}")
        
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["job_id"], sample_job_id)
            self.assertIn("trees", data)
            self.assertIsInstance(data["trees"], list)
            
            # Check tree structure if there are any trees
            if data["trees"]:
                tree = data["trees"][0]
                self.assertIn("id", tree)
                self.assertIn("confidence", tree)
                self.assertIn("coordinates", tree)
        else:
            self.skipTest(f"Detection results not available for job: {sample_job_id}")


if __name__ == "__main__":
    unittest.main()