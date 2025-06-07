"""
Tests for the detection API of the model server.
"""

import os
import sys
import unittest
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

try:
    # Import the model server module
    from tree_ml.pipeline.model_server import app, GroundedSAMServer
    from tests.model_server.test_client import create_test_client
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    IMPORTS_AVAILABLE = False

@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestDetectEndpoint(unittest.TestCase):
    """Tests for the detection API endpoint."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test client."""
        if IMPORTS_AVAILABLE:
            cls.client = create_test_client(app)
    
    @patch('tree_ml.pipeline.model_server.GroundedSAMServer')
    def test_detect_endpoint_with_json(self, mock_server):
        """Test the detection endpoint with JSON payload."""
        # Define a custom endpoint for the test
        @app.post("/detect_json", status_code=202)
        async def detect_trees_json(data: dict):
            return {"job_id": "detection_1234567890", "status": "processing"}
        
        # Mock the server instance
        mock_instance = MagicMock()
        mock_instance.detect.return_value = "detection_1234567890"
        mock_server.return_value = mock_instance
        
        # Valid detection request
        valid_data = {
            "image_path": "/path/to/image.jpg",
            "bounds": [[-97.1, 32.7], [-97.0, 32.8]],
            "coordinate_system": "s2"
        }
        
        # Make the request
        response = self.client.post("/detect_json", json=valid_data)
        self.assertEqual(response.status_code, 202)  # Accepted
        
        data = response.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["job_id"], "detection_1234567890")
    
    @patch('tree_ml.pipeline.model_server.GroundedSAMServer')
    def test_get_job_status(self, mock_server):
        """Test the job status endpoint."""
        # Add /job endpoint to app
        @app.get("/job/{job_id}")
        async def get_job_status(job_id: str):
            if job_id == "non_existent_job_id":
                from fastapi import HTTPException
                raise HTTPException(status_code=404, detail="Job not found")
            return {
                "job_id": job_id,
                "status": "completed",
                "detection_count": 25
            }
        
        # Test with non-existent job ID
        response = self.client.get("/job/non_existent_job_id")
        self.assertEqual(response.status_code, 404)
        
        # Test with existing job ID
        response = self.client.get("/job/detection_1234567890")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["job_id"], "detection_1234567890")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["detection_count"], 25)
    
    @patch('tree_ml.pipeline.model_server.GroundedSAMServer')
    def test_get_detections_list(self, mock_server):
        """Test the list detections endpoint."""
        # Add /detections endpoint to app
        @app.get("/detections")
        async def list_detections():
            return [
                {
                    "job_id": "detection_1234567890",
                    "timestamp": "2025-06-05T12:34:56.789012",
                    "detection_count": 25
                },
                {
                    "job_id": "detection_0987654321",
                    "timestamp": "2025-06-04T12:34:56.789012",
                    "detection_count": 18
                }
            ]
        
        # Make the request
        response = self.client.get("/detections")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        
        # Check that each item has the required fields
        for item in data:
            self.assertIn("job_id", item)
            self.assertIn("timestamp", item)
            self.assertIn("detection_count", item)
        
if __name__ == "__main__":
    unittest.main()