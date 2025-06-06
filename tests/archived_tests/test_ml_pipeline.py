#!/usr/bin/env python3
"""
Comprehensive ML Pipeline Test with S2 Geospatial Indexing Integration

This script tests the entire ML pipeline including:
1. YOLO for tree detection
2. SAM for ML overlay generation
3. S2 indexing for spatial organization
"""

import os
import sys
import json
import time
import asyncio
import logging
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_pipeline_test')

# Add necessary paths
sys.path.append('/ttt')

# Import the detection service
from tree_ml.dashboard.backend.services.detection_service import DetectionService, S2IndexManager
from tree_ml.dashboard.backend.config import USE_ML_PIPELINE

# Test constants
BASE_RESULTS_DIR = '/ttt/data/temp/ml_results_test'
DETECTION_DIR = os.path.join(BASE_RESULTS_DIR, 'detection')
SEGMENTATION_DIR = os.path.join(BASE_RESULTS_DIR, 'segmentation')
DETECTION_RESPONSE_DIR = os.path.join(DETECTION_DIR, 'ml_response')
SEGMENTATION_RESPONSE_DIR = os.path.join(SEGMENTATION_DIR, 'ml_response')
MODEL_DIR = '/ttt/tree_risk_pro/pipeline/model'

# Test bounds (Central Park area)
TEST_BOUNDS = [
    [-73.9776, 40.7614],  # SW corner
    [-73.9499, 40.7968]   # NE corner
]

# Test coordinates
TEST_CENTER = [-73.96375, 40.7791]
TEST_ZOOM = 16

def check_ml_models():
    """Check if required ML models are available"""
    # Check for YOLO models
    yolo_models = [
        os.path.join(MODEL_DIR, "yolo11s.pt"),
        os.path.join(MODEL_DIR, "yolo11l.pt"),
        os.path.join(MODEL_DIR, "yolov8n.pt"),
        os.path.join(MODEL_DIR, "yolov8m.pt")
    ]
    
    yolo_found = False
    for model_path in yolo_models:
        if os.path.exists(model_path):
            logger.info(f"Found YOLO model: {model_path}")
            yolo_found = True
            break
    
    if not yolo_found:
        logger.warning("No YOLO models found. Detection may fail.")
    
    # Check for SAM models
    sam_models = [
        os.path.join(MODEL_DIR, "sam_vit_h_4b8939.pth"),
        os.path.join(MODEL_DIR, "sam2.1_hiera_small.pt"),
        os.path.join(MODEL_DIR, "sam2.1_hiera_base_plus.pt")
    ]
    
    sam_found = False
    for model_path in sam_models:
        if os.path.exists(model_path):
            logger.info(f"Found SAM model: {model_path}")
            sam_found = True
            break
    
    if not sam_found:
        logger.warning("No SAM models found. Segmentation may fail.")
    
    return yolo_found and sam_found

def find_or_create_test_image():
    """Find an existing satellite image or create a new test image"""
    # Make sure the results directory exists
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    
    # Look for existing satellite images
    existing_images = []
    for file in os.listdir(BASE_RESULTS_DIR):
        if file.startswith('satellite_') and file.endswith('.jpg'):
            existing_images.append(os.path.join(BASE_RESULTS_DIR, file))
    
    if existing_images:
        logger.info(f"Found {len(existing_images)} existing satellite images")
        return existing_images[0]
    
    # Create a new synthetic satellite image if no real ones exist
    # This is only for testing when real satellite imagery isn't available
    logger.info("No satellite images found, creating a synthetic test image")
    
    # Create a simple satellite-like image with tree shapes
    img = Image.new('RGB', (640, 640), color=(102, 140, 100))  # Green-ish background for satellite
    draw = ImageDraw.Draw(img)
    
    # Draw some tree-like shapes with realistic colors
    for i in range(10):
        # Random positions
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 590)
        size = np.random.randint(30, 70)
        
        # Random green shade for trees
        green = np.random.randint(60, 120)
        tree_color = (30, green, 30)
        
        # Draw tree crown
        draw.ellipse([(x-size, y-size), (x+size, y+size)], fill=tree_color)
        
        # Draw shadow
        shadow_offset = np.random.randint(5, 15)
        shadow_size = size - np.random.randint(5, 10)
        draw.ellipse([(x-shadow_size+shadow_offset, y-shadow_size+shadow_offset), 
                     (x+shadow_size+shadow_offset, y+shadow_size+shadow_offset)], 
                     fill=(max(0, tree_color[0]-20), max(0, tree_color[1]-20), max(0, tree_color[2]-20)))
    
    # Add some building-like shapes
    for i in range(3):
        x = np.random.randint(100, 540)
        y = np.random.randint(100, 540)
        width = np.random.randint(40, 100)
        height = np.random.randint(40, 100)
        
        # Gray for buildings
        gray = np.random.randint(120, 180)
        draw.rectangle([(x, y), (x+width, y+height)], fill=(gray, gray, gray))
        
        # Add shadow
        shadow_offset = np.random.randint(5, 15)
        draw.rectangle([(x+shadow_offset, y+shadow_offset), (x+width+shadow_offset, y+height+shadow_offset)], 
                       fill=(max(0, gray-30), max(0, gray-30), max(0, gray-30)))
    
    # Add some road-like shapes
    road_color = (80, 80, 80)
    road_width = 20
    
    # Horizontal road
    draw.rectangle([(0, 320-road_width//2), (640, 320+road_width//2)], fill=road_color)
    
    # Vertical road
    draw.rectangle([(320-road_width//2, 0), (320+road_width//2, 640)], fill=road_color)
    
    # Save the image
    timestamp = int(time.time())
    image_path = os.path.join(BASE_RESULTS_DIR, f"satellite_{TEST_CENTER[1]}_{TEST_CENTER[0]}_{TEST_ZOOM}_{timestamp}.jpg")
    img.save(image_path)
    logger.info(f"Created synthetic satellite image at {image_path}")
    
    return image_path

async def test_s2_indexing():
    """Test S2 geospatial indexing functionality"""
    logger.info("\n--- Testing S2 Geospatial Indexing ---")
    
    # Create S2 index manager
    s2_manager = S2IndexManager()
    
    # Generate test coordinate points (NYC area)
    test_points = [
        (40.7128, -74.0060),  # NYC
        (40.7614, -73.9776),  # Central Park SW
        (40.7968, -73.9499),  # Central Park NE
    ]
    
    # Test cell generation at different levels
    logger.info("Testing S2 cell generation at different levels:")
    for level in s2_manager.cell_levels.keys():
        logger.info(f"  Level: {level} ({s2_manager.cell_levels[level]})")
        for i, (lat, lng) in enumerate(test_points):
            cell_id = s2_manager.get_cell_id(lat, lng, level)
            logger.info(f"    Point {i+1}: ({lat}, {lng}) -> Cell ID: {cell_id}")
    
    # Test neighbor finding
    logger.info("\nTesting S2 neighbor finding:")
    for level in ['block', 'property']:
        logger.info(f"  Level: {level}")
        lat, lng = test_points[0]  # Use first test point
        neighbors = s2_manager.get_neighbors(lat, lng, level, k=8)
        logger.info(f"    Found {len(neighbors)} neighbors for point ({lat}, {lng})")
        logger.info(f"    First few neighbors: {neighbors[:3]}")
    
    # Test bounds coverage
    logger.info("\nTesting S2 bounds coverage:")
    for level in ['neighborhood', 'block']:
        cells = s2_manager.get_cells_for_bounds(TEST_BOUNDS, level)
        logger.info(f"  Level {level}: Found {len(cells)} cells covering the test bounds")
        logger.info(f"  First few cells: {cells[:3]}")
    
    return True

async def test_detection():
    """Test tree detection with S2 indexing"""
    logger.info("\n--- Testing Tree Detection with S2 Indexing ---")
    
    # Check if ML pipeline is enabled
    if not USE_ML_PIPELINE:
        logger.error("ML pipeline is disabled in config. Set USE_ML_PIPELINE=True in config.py")
        return None
    
    # Find or create the test image
    image_path = find_or_create_test_image()
    
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
        
        # Test S2 cell grouping functionality
        if 'trees' in detection_result and detection_result['trees']:
            logger.info("\nTesting S2 cell grouping for detected trees:")
            grouped_trees = await detection_service.group_trees_by_s2_cell(
                detection_result['trees'], level='block'
            )
            logger.info(f"Grouped {len(detection_result['trees'])} trees into {len(grouped_trees)} S2 cells")
            
            # Calculate stats for grouped trees
            cell_stats = detection_service.calculate_s2_statistics(grouped_trees)
            logger.info(f"Generated statistics for {len(cell_stats)} S2 cells")
            
            # Save the S2 grouping results
            with open(os.path.join(DETECTION_RESPONSE_DIR, 's2_grouping.json'), 'w') as f:
                # Convert to serializable format (cell IDs as keys)
                serializable_grouped = {str(k): v for k, v in grouped_trees.items()}
                json.dump(serializable_grouped, f, indent=2)
            
            with open(os.path.join(DETECTION_RESPONSE_DIR, 's2_statistics.json'), 'w') as f:
                # Convert to serializable format (cell IDs as keys)
                serializable_stats = {str(k): v for k, v in cell_stats.items()}
                json.dump(serializable_stats, f, indent=2)
            
            logger.info("S2 grouping results saved to detection response directory")
        else:
            logger.warning("No trees detected, skipping S2 cell grouping test")
        
        # Return the result for segmentation test
        return detection_result
        
    except Exception as e:
        logger.error(f"Error in detection test: {str(e)}", exc_info=True)
        return None

async def test_segmentation(detection_result):
    """Test tree segmentation using SAM with S2 indexing"""
    logger.info("\n--- Testing SAM Segmentation with S2 Indexing ---")
    
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
        logger.error("No satellite image found for segmentation")
        return False
    
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
        
        # Add segmentation masks using SAM
        logger.info("Adding segmentation masks with SAM...")
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
        
        # Test S2 grouping with segmentation results
        if 'trees' in segmentation_result and segmentation_result['trees']:
            logger.info("\nTesting S2 cell grouping with segmentation results:")
            grouped_trees = await detection_service.group_trees_by_s2_cell(
                segmentation_result['trees'], level='property'
            )
            logger.info(f"Grouped {len(segmentation_result['trees'])} trees into {len(grouped_trees)} S2 cells")
            
            # Save the S2 grouping results for segmentation
            with open(os.path.join(SEGMENTATION_RESPONSE_DIR, 's2_segmentation_grouping.json'), 'w') as f:
                # Convert to serializable format (cell IDs as keys)
                serializable_grouped = {str(k): v for k, v in grouped_trees.items()}
                json.dump(serializable_grouped, f, indent=2)
            
            logger.info("S2 segmentation grouping results saved")
        
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
    logger.info("Starting ML Pipeline Test with S2 Geospatial Indexing")
    
    # Check if ML models are available
    models_available = check_ml_models()
    if not models_available:
        logger.warning("Required ML models not found. Tests may fail.")
    
    # Create test directories if needed
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    os.makedirs(DETECTION_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_DIR, exist_ok=True)
    os.makedirs(DETECTION_RESPONSE_DIR, exist_ok=True)
    os.makedirs(SEGMENTATION_RESPONSE_DIR, exist_ok=True)
    
    # First test S2 indexing
    s2_success = await test_s2_indexing()
    if not s2_success:
        logger.error("S2 indexing test failed")
    
    # Run detection test
    detection_result = await test_detection()
    
    # Run segmentation test if detection succeeded
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