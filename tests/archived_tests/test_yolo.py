#!/usr/bin/env python3
"""
YOLO Tree Detection Test Script

This script tests the YOLO models directly with different configurations to debug
detection issues with satellite imagery.
"""

import os
import sys
import json
import logging
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('yolo_test')

# Add necessary paths
sys.path.append('/ttt')

# Import ultraltyics for YOLO
try:
    import torch
    from ultralytics import YOLO
    logger.info("Successfully imported ultralytics YOLO")
except ImportError as e:
    logger.error(f"Failed to import ultralytics: {e}")
    sys.exit(1)

# Constants
MODEL_DIR = "/ttt/tree_ml/pipeline/model"
TEMP_DIR = "/ttt/data/temp/ml_results_test"

def get_available_models():
    """Find available YOLO models"""
    model_paths = [
        os.path.join(MODEL_DIR, "yolo11s.pt"),
        os.path.join(MODEL_DIR, "yolo11l.pt"),
        os.path.join(MODEL_DIR, "yolov8n.pt"),
        os.path.join(MODEL_DIR, "yolov8m.pt"),
        os.path.join(MODEL_DIR, "yolov8s.pt"),
        os.path.join("/ttt/tree_risk_pro/dashboard/backend", "yolov8m.pt")
    ]
    
    available_models = []
    for path in model_paths:
        if os.path.exists(path):
            available_models.append(path)
            logger.info(f"Found model: {path}")
    
    return available_models

def load_test_image():
    """Find or create a test image"""
    # Look for existing satellite images in the temp directory
    os.makedirs(TEMP_DIR, exist_ok=True)
    existing_images = []
    
    for file in os.listdir(TEMP_DIR):
        if file.startswith('satellite_') and file.endswith('.jpg'):
            existing_images.append(os.path.join(TEMP_DIR, file))
    
    if existing_images:
        logger.info(f"Found {len(existing_images)} existing satellite images")
        return existing_images[0]
    
    # Create a new test image if none exists
    logger.info("No existing images found, creating a test image")
    
    # Create a simple test image with tree-like shapes
    img = Image.new('RGB', (640, 640), color=(37, 125, 60))  # Green background
    draw = ImageDraw.Draw(img)
    
    # Draw some tree-like shapes
    draw.ellipse([(100, 100), (200, 200)], fill=(0, 100, 0))  # Dark green tree
    draw.ellipse([(300, 200), (400, 300)], fill=(0, 100, 0))  # Dark green tree
    draw.ellipse([(150, 350), (250, 450)], fill=(0, 100, 0))  # Dark green tree
    draw.ellipse([(350, 400), (450, 500)], fill=(0, 100, 0))  # Dark green tree
    
    # Add some tree shadows
    draw.ellipse([(120, 120), (180, 180)], fill=(0, 80, 0))  # Darker center
    draw.ellipse([(320, 220), (380, 280)], fill=(0, 80, 0))  # Darker center
    draw.ellipse([(170, 370), (230, 430)], fill=(0, 80, 0))  # Darker center
    draw.ellipse([(370, 420), (430, 480)], fill=(0, 80, 0))  # Darker center
    
    # Add some tree trunks
    draw.rectangle([(145, 200), (155, 220)], fill=(101, 67, 33))  # Brown trunk
    draw.rectangle([(345, 300), (355, 320)], fill=(101, 67, 33))  # Brown trunk
    draw.rectangle([(195, 450), (205, 470)], fill=(101, 67, 33))  # Brown trunk
    draw.rectangle([(395, 500), (405, 520)], fill=(101, 67, 33))  # Brown trunk
    
    # Save the image
    timestamp = int(time.time())
    image_path = os.path.join(TEMP_DIR, f"synthetic_test_{timestamp}.jpg")
    img.save(image_path)
    logger.info(f"Created test image at {image_path}")
    
    return image_path

def test_yolo_model(model_path, image_path, conf_threshold=0.01, save_results=True):
    """Test a YOLO model on an image with the given confidence threshold"""
    logger.info(f"Testing YOLO model {os.path.basename(model_path)} with conf={conf_threshold}")
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Set confidence threshold
        model.conf = conf_threshold
        
        # Load image
        image = Image.open(image_path)
        image_np = np.array(image)
        logger.info(f"Loaded image from {image_path} with shape {image_np.shape}")
        
        # Run detection with maximum verbosity
        logger.info(f"Running detection with YOLO model {os.path.basename(model_path)}")
        results = model(image_np, verbose=True)
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            logger.info(f"Result type: {type(result)}")
            
            # Get detection boxes
            if hasattr(result, 'boxes'):
                logger.info(f"Result has boxes attribute")
                
                # Try to get box data
                if hasattr(result.boxes, 'data'):
                    boxes = result.boxes.data
                    logger.info(f"Found {len(boxes)} detections before filtering")
                    
                    # List all raw detections
                    for i, box in enumerate(boxes):
                        class_id = int(box[5]) if len(box) > 5 else 0
                        confidence = float(box[4]) if len(box) > 4 else 0
                        class_name = model.names[class_id] if hasattr(model, 'names') else f"Class {class_id}"
                        
                        x1, y1, x2, y2 = box[:4]
                        logger.info(f"  Detection {i+1}: Class {class_id} ({class_name}), " 
                                   f"Conf {confidence:.4f}, Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                else:
                    logger.warning(f"Boxes has no data attribute: {type(result.boxes)}")
            else:
                logger.warning("Result has no boxes attribute")
        else:
            logger.warning("No results returned from model")
        
        # Save visualization
        if save_results:
            result_path = os.path.join(TEMP_DIR, f"result_{os.path.basename(model_path)}_{conf_threshold}_"
                                      f"{os.path.basename(image_path)}")
            
            # Save with result boxes
            for r in results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                im.save(result_path)
                logger.info(f"Saved result visualization to {result_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error testing model {model_path}: {e}", exc_info=True)
        return None

def run_all_tests():
    """Run tests on all available models with different configurations"""
    # Get models
    models = get_available_models()
    if not models:
        logger.error("No YOLO models found")
        return
    
    # Get or create test image
    test_image = load_test_image()
    
    # Test with different confidence thresholds
    confidence_thresholds = [0.01, 0.05, 0.1, 0.25]
    
    for model_path in models:
        for conf in confidence_thresholds:
            logger.info(f"\n=== Testing {os.path.basename(model_path)} with conf={conf} ===")
            results = test_yolo_model(model_path, test_image, conf)
            
            # Parse results
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'data'):
                    num_detections = len(result.boxes.data)
                    logger.info(f"Found {num_detections} detections with conf={conf}")
                else:
                    logger.warning(f"No detections found with conf={conf}")
            else:
                logger.warning(f"No results returned with conf={conf}")
            
            logger.info(f"=== Test complete for {os.path.basename(model_path)} with conf={conf} ===\n")

def test_with_real_satellite():
    """Test with real satellite image if available"""
    # Look for satellite images specifically
    satellite_images = []
    
    for file in os.listdir(TEMP_DIR):
        if file.startswith('satellite_') and file.endswith('.jpg'):
            satellite_images.append(os.path.join(TEMP_DIR, file))
    
    if not satellite_images:
        logger.warning("No real satellite images found for testing")
        return
    
    logger.info(f"Found {len(satellite_images)} real satellite images")
    
    # Get the first available model
    models = get_available_models()
    if not models:
        logger.error("No YOLO models found")
        return
    
    # Use the first available model and extremely low confidence threshold
    model_path = models[0]
    conf = 0.001  # Extremely low threshold to catch anything
    
    logger.info(f"\n=== Testing {os.path.basename(model_path)} with real satellite image ===")
    for image_path in satellite_images:
        logger.info(f"Testing with {os.path.basename(image_path)}")
        results = test_yolo_model(model_path, image_path, conf)
        
        # Log the raw predictions
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'data'):
                boxes = result.boxes.data
                
                if len(boxes) > 0:
                    logger.info(f"Found {len(boxes)} raw detections with conf={conf}")
                    
                    # Display all detections, even with very low confidence
                    for i, box in enumerate(boxes):
                        class_id = int(box[5]) if len(box) > 5 else 0
                        confidence = float(box[4]) if len(box) > 4 else 0
                        class_name = result.names[class_id] if hasattr(result, 'names') else f"Class {class_id}"
                        
                        logger.info(f"  Detection {i+1}: Class {class_id} ({class_name}), Conf {confidence:.4f}")
                else:
                    logger.warning("No detections at all, even with extremely low confidence")
            else:
                logger.warning("Results have no boxes or data")
        else:
            logger.warning("No results returned from model")

if __name__ == "__main__":
    logger.info("Starting YOLO test script")
    
    # Run basic tests with all models
    run_all_tests()
    
    # Test with real satellite image
    test_with_real_satellite()
    
    logger.info("YOLO testing complete")