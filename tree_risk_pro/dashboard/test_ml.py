#!/usr/bin/env python3
"""
Test script for ML pipeline detection and segmentation with proper directory organization
"""

import os
import sys
import json
import time
import asyncio
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_pipeline_test')

# Import the detection service
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dashboard.backend.services.detection_service import DetectionService
from dashboard.backend.config import USE_ML_PIPELINE

# Constants for the test
BASE_RESULTS_DIR = '/ttt/data/temp/ml_results_test'
DETECTION_DIR = os.path.join(BASE_RESULTS_DIR, 'detection')
SEGMENTATION_DIR = os.path.join(BASE_RESULTS_DIR, 'segmentation')
DETECTION_RESPONSE_DIR = os.path.join(DETECTION_DIR, 'ml_response')
SEGMENTATION_RESPONSE_DIR = os.path.join(SEGMENTATION_DIR, 'ml_response')

# Test bounds (Central Park area)
TEST_BOUNDS = [
    [-73.9776, 40.7614],  # SW corner
    [-73.9499, 40.7968]   # NE corner
]

# Test coordinates
TEST_CENTER = [-73.96375, 40.7791]
TEST_ZOOM = 16

# Create a test image for consistent results
def create_test_image():
    """Create a test satellite image if needed"""
    from PIL import Image, ImageDraw
    
    # Define the path for our test image
    timestamp = int(time.time())
    image_filename = f"satellite_{TEST_CENTER[1]}_{TEST_CENTER[0]}_{TEST_ZOOM}_{timestamp}.jpg"
    image_path = os.path.join(BASE_RESULTS_DIR, image_filename)
    
    # Check if we need to create the image
    if not os.path.exists(image_path):
        # Create a simple green test image with white shapes for trees
        img = Image.new('RGB', (640, 640), color=(37, 125, 60))
        d = ImageDraw.Draw(img)
        
        # Draw some tree-like shapes
        d.rectangle([(50, 50), (150, 150)], outline=(255, 255, 255), width=2)
        d.rectangle([(200, 100), (300, 200)], outline=(255, 255, 255), width=2)
        d.ellipse([(350, 350), (450, 450)], outline=(255, 255, 255), width=2)
        d.ellipse([(100, 300), (200, 400)], outline=(255, 255, 255), width=2)
        d.ellipse([(250, 250), (350, 350)], outline=(255, 255, 255), width=2)
        
        # Add text for identification
        d.text((10, 10), f"Test Image {timestamp}", fill=(255, 255, 0))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Save the image
        img.save(image_path)
        logger.info(f"Created test image at {image_path}")
    
    return image_path

async def test_detection():
    """Test tree detection with the new directory structure"""
    logger.info("\n--- Testing Detection with New Directory Structure ---")
    
    # Check if ML pipeline is enabled
    if not USE_ML_PIPELINE:
        logger.error("ML pipeline is disabled in config. Set USE_ML_PIPELINE=True in config.py")
        return None
    
    # Create or get the test image
    image_path = create_test_image()
    
    # Clean and prepare directories
    os.makedirs(DETECTION_RESPONSE_DIR, exist_ok=True)
    for item in os.listdir(DETECTION_RESPONSE_DIR):
        item_path = os.path.join(DETECTION_RESPONSE_DIR, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
            
    # Create detection service instance
    detection_service = DetectionService()
    
    # Create test job ID with timestamp
    timestamp = int(time.time())
    test_job_id = f"ml_detection_{timestamp}"
    
    try:
        # Run detection with explicit response directory
        logger.info(f"Running detection with image: {image_path}")
        detection_result = await detection_service.detect_trees_from_satellite(
            image_path=image_path,
            bounds=TEST_BOUNDS,
            job_id=test_job_id,
            ml_response_dir=DETECTION_RESPONSE_DIR
        )
        
        # Check the result
        if not detection_result:
            logger.error("Detection failed to return results")
            return None
        
        # Add timestamp to result for later use
        detection_result['timestamp'] = timestamp
        detection_result['base_dir'] = BASE_RESULTS_DIR
        
        # Log the results
        logger.info(f"Detection completed: {len(detection_result.get('trees', []))} trees found")
        logger.info(f"Job ID: {detection_result.get('job_id')}")
        logger.info(f"ML response dir: {detection_result.get('ml_response_dir')}")
        
        # Check for required files
        files = os.listdir(DETECTION_RESPONSE_DIR)
        logger.info(f"Files in detection response directory ({len(files)}):")
        for file in files:
            logger.info(f"  - {file}")
        
        # Return the result for segmentation test
        return detection_result
        
    except Exception as e:
        logger.error(f"Error in detection test: {str(e)}", exc_info=True)
        return None

async def test_segmentation(detection_result):
    """Test segmentation using results from previous detection"""
    logger.info("\n--- Testing Segmentation with New Directory Structure ---")
    
    if not detection_result:
        logger.error("No detection result provided - cannot run segmentation test")
        return False
    
    # Create detection service instance
    detection_service = DetectionService()
    
    # Clean and prepare directories
    os.makedirs(SEGMENTATION_RESPONSE_DIR, exist_ok=True)
    for item in os.listdir(SEGMENTATION_RESPONSE_DIR):
        item_path = os.path.join(SEGMENTATION_RESPONSE_DIR, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
    
    # Use same timestamp as detection
    timestamp = detection_result.get('timestamp')
    test_job_id = f"ml_segmentation_{timestamp}"
    
    # Get the trees from detection result
    trees = detection_result.get('trees', [])
    if not trees:
        logger.error("No trees found in detection result - cannot run segmentation")
        return False
    
    # Get the image path
    base_dir = detection_result.get('base_dir', BASE_RESULTS_DIR)
    image_path = None
    
    # Find satellite image in the base directory
    for file in os.listdir(base_dir):
        if file.startswith('satellite_') and file.endswith('.jpg'):
            image_path = os.path.join(base_dir, file)
            break
    
    if not image_path or not os.path.exists(image_path):
        # Create a new test image if needed
        image_path = create_test_image()
    
    logger.info(f"Using image: {image_path}")
    logger.info(f"Using {len(trees)} trees from previous detection")
    
    try:
        # Run segmentation using the trees from detection
        segmentation_result = await detection_service.detect_trees_from_satellite(
            image_path=image_path,
            bounds=TEST_BOUNDS,
            job_id=test_job_id,
            ml_response_dir=SEGMENTATION_RESPONSE_DIR,
            existing_trees=trees
        )
        
        if not segmentation_result:
            logger.error("Segmentation failed - no result returned")
            return False
        
        # Add segmentation masks
        logger.info("Adding segmentation masks...")
        await detection_service._add_segmentation_data(
            ml_response_dir=SEGMENTATION_RESPONSE_DIR,
            trees=segmentation_result.get('trees', []),
            image_path=image_path,
            bounds=TEST_BOUNDS
        )
        
        # Set mode to segmentation
        segmentation_result['mode'] = 'segmentation'
        
        # Check for segmentation files
        files = os.listdir(SEGMENTATION_RESPONSE_DIR)
        logger.info(f"Files in segmentation response directory ({len(files)}):")
        for file in files:
            logger.info(f"  - {file}")
        
        # Check for key segmentation files
        required_files = [
            'segmentation_metadata.json',
            'combined_segmentation.png',
            'segmentation_overlay.png'
        ]
        
        missing = [f for f in required_files if f not in files]
        if missing:
            logger.warning(f"Missing required segmentation files: {missing}")
        else:
            logger.info("All required segmentation files present")
        
        # Verify tree mask files
        mask_files = [f for f in files if f.startswith('tree_') and f.endswith('_mask.png')]
        logger.info(f"Found {len(mask_files)} individual tree mask files")
        
        # Log the overall success
        success = len(mask_files) > 0 and len(missing) == 0
        if success:
            logger.info("✅ Segmentation test completed successfully")
        else:
            logger.warning("⚠️ Segmentation test completed but with missing files")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in segmentation test: {str(e)}", exc_info=True)
        return False

async def main():
    """Run all tests in sequence"""
    logger.info("Starting ML pipeline directory structure tests...")
    
    # Create test directories if needed
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    os.makedirs(DETECTION_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_DIR, exist_ok=True)
    os.makedirs(DETECTION_RESPONSE_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_RESPONSE_DIR, exist_ok=True)
    
    # Run detection test
    detection_result = await test_detection()
    
    # Run segmentation test
    if detection_result:
        segmentation_success = await test_segmentation(detection_result)
        
        # Log final status
        if segmentation_success:
            logger.info("\n--- All tests completed successfully ---")
            logger.info(f"Results stored in: {BASE_RESULTS_DIR}")
            logger.info(f"Detection results: {DETECTION_DIR}")
            logger.info(f"Segmentation results: {SEGMENTATION_DIR}")
        else:
            logger.warning("\n--- Tests completed with issues ---")
    else:
        logger.error("Detection failed, skipping segmentation test")
        logger.error("\n--- Tests failed ---")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(main())