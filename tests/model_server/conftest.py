"""
Pytest configuration for model server tests.
Sets up fixtures and environment variables for testing.
"""

import os
import sys
import pytest
from pathlib import Path
import json

# Add the parent directory to the path so we can import the module
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

# Register custom marks
def pytest_configure(config):
    """Register custom pytest marks."""
    config.addinivalue_line("markers", "integration: mark test as integration test")

@pytest.fixture(scope="session")
def root_dir():
    """Return the root directory of the project."""
    return ROOT_DIR

@pytest.fixture(scope="session")
def model_dir():
    """Return the model directory."""
    return os.path.join(ROOT_DIR, "tree_ml/pipeline/model")

@pytest.fixture(scope="session")
def output_dir():
    """Return the output directory for ML results."""
    return os.path.join(ROOT_DIR, "data/ml")

@pytest.fixture(scope="session")
def sample_job_id():
    """Return a sample job ID for testing."""
    return "detection_1748997756"

@pytest.fixture(scope="session")
def sample_job_metadata(sample_job_id, output_dir):
    """Return the metadata for a sample job."""
    metadata_path = os.path.join(output_dir, f"{sample_job_id}/ml_response/metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

@pytest.fixture(scope="session")
def sample_satellite_image(sample_job_metadata):
    """Return the path to a sample satellite image for testing."""
    if sample_job_metadata and 'image_path' in sample_job_metadata:
        return sample_job_metadata['image_path']
    
    # Fallback to a standard path if metadata is not available
    return os.path.join(ROOT_DIR, "data/images/sample_satellite.jpg")