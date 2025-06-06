#!/usr/bin/env python3
"""
Mock tests for the tree detection model server.
These tests use mocks to avoid requiring the actual model.
"""

import os
import sys
import json
import pytest
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

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
class TestModelServerMock(unittest.TestCase):
    """Mock tests for the model server that don't require the actual model."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test client."""
        if IMPORTS_AVAILABLE:
            cls.client = create_test_client(app)
    
    @patch('tree_ml.pipeline.model_server.GroundedSAMServer')
    def test_health_check_mocked(self, mock_server):
        """Test the health check endpoint with a mocked server."""
        # Mock the server instance
        mock_instance = MagicMock()
        mock_instance.device = "cuda"
        mock_server.return_value = mock_instance
        
        # Make the request
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["device"], "cuda")
    
    @patch('tree_ml.pipeline.model_server.GroundedSAMServer')
    def test_detect_endpoint_mocked(self, mock_server):
        """Test the detection endpoint with a mocked server."""
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
        response = self.client.post("/detect", json=valid_data)
        self.assertEqual(response.status_code, 202)  # Accepted
        
        data = response.json()
        self.assertIn("job_id", data)
        self.assertEqual(data["job_id"], "detection_1234567890")
        
        # Verify the detect method was called with correct arguments
        mock_instance.detect.assert_called_once()
        args, kwargs = mock_instance.detect.call_args
        self.assertEqual(kwargs["image_path"], valid_data["image_path"])
        self.assertEqual(kwargs["bounds"], valid_data["bounds"])
        self.assertEqual(kwargs["coordinate_system"], valid_data["coordinate_system"])
    
    @patch('tree_ml.pipeline.model_server.GroundedSAMServer')
    def test_job_status_mocked(self, mock_server):
        """Test the job status endpoint with a mocked server."""
        # Mock the server instance
        mock_instance = MagicMock()
        mock_instance.get_job_status.return_value = {
            "job_id": "detection_1234567890",
            "status": "completed",
            "detection_count": 25,
            "timestamp": "2025-06-05T12:34:56.789012"
        }
        mock_server.return_value = mock_instance
        
        # Make the request
        response = self.client.get("/job/detection_1234567890")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["job_id"], "detection_1234567890")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["detection_count"], 25)
        
        # Verify the get_job_status method was called with correct arguments
        mock_instance.get_job_status.assert_called_once_with("detection_1234567890")
    
    @patch('tree_ml.pipeline.model_server.GroundedSAMServer')
    def test_list_detections_mocked(self, mock_server):
        """Test the list detections endpoint with a mocked server."""
        # Mock the server instance
        mock_instance = MagicMock()
        mock_instance.list_detections.return_value = [
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
        mock_server.return_value = mock_instance
        
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
        
        # Verify the list_detections method was called
        mock_instance.list_detections.assert_called_once()


if __name__ == "__main__":
    unittest.main()