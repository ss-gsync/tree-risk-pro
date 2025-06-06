"""
Test script for the External Model Service

This script tests the T4 model server client integration.
"""

import os
import sys
import argparse
import logging
import numpy as np
from PIL import Image
import time

# Add necessary paths for imports
sys.path.append('/ttt')
sys.path.append('/ttt/tree_ml/dashboard/backend')
from config import MODEL_SERVER_URL
from tree_ml.dashboard.backend.services.ml.external_model_service import get_external_model_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_external_model_service(image_path, model_server_url=None):
    """
    Test the external model service with an image.
    
    Args:
        image_path: Path to test image
        model_server_url: Optional URL for the model server
    """
    logger.info(f"Testing external model service with image: {image_path}")
    
    # Initialize the model service
    model_service = get_external_model_service(server_url=model_server_url)
    
    # Wait for models to load
    logger.info("Waiting for models to load...")
    if not model_service.wait_for_models(timeout=30):
        logger.error("Models failed to load. Check model server status.")
        return False
    
    # Load test image
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        logger.info(f"Loaded image with shape: {image_array.shape}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return False
    
    # Run detection
    logger.info("Running tree detection...")
    start_time = time.time()
    results = model_service.detect_trees(
        image=image_array,
        confidence_threshold=0.3,
        with_segmentation=True,
        job_id=f"test_{int(time.time())}"
    )
    detection_time = time.time() - start_time
    
    # Check results
    if not results.get('success', False):
        logger.error(f"Detection failed: {results.get('error', 'Unknown error')}")
        return False
    
    # Print results
    detections = results.get('detections', [])
    logger.info(f"Detection successful in {detection_time:.2f}s")
    logger.info(f"Found {len(detections)} objects")
    
    # Print detection details
    for i, detection in enumerate(detections):
        logger.info(f"Detection {i+1}:")
        logger.info(f"  Class: {detection.get('class', 'unknown')}")
        logger.info(f"  Confidence: {detection.get('confidence', 0):.4f}")
        logger.info(f"  Bounding box: {detection.get('bbox', [])}")
        logger.info(f"  Has segmentation: {'segmentation' in detection}")
    
    logger.info("Test completed successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the External Model Service")
    parser.add_argument("--image", required=True, help="Path to the test image")
    parser.add_argument("--server", default=None, help="URL of the model server (defaults to config value)")
    
    args = parser.parse_args()
    
    success = test_external_model_service(args.image, args.server)
    sys.exit(0 if success else 1)