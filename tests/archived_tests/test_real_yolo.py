#!/usr/bin/env python3
"""
Direct YOLO Test on Satellite Image - NO SYNTHETIC DATA

This script tests YOLO detection directly on the satellite image 
without any synthetic data generation or fallbacks.
"""

import os
import sys
import logging
import time
import numpy as np
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('real_yolo_test')

# Add necessary paths
sys.path.append('/ttt')

# Find the satellite image
def find_satellite_image():
    """Find the satellite image in the test directory"""
    test_dir = '/ttt/data/temp/ml_results_test'
    
    # Create directory if it doesn't exist
    os.makedirs(test_dir, exist_ok=True)
    
    # Look for satellite image
    for file in os.listdir(test_dir):
        if file.startswith('satellite_') and file.endswith('.jpg'):
            return os.path.join(test_dir, file)
    
    logger.error("No satellite image found for testing")
    return None

def examine_satellite_image(image_path):
    """Examine the satellite image to understand what we're working with"""
    try:
        # Load the image
        image = Image.open(image_path)
        width, height = image.size
        
        logger.info(f"Satellite image dimensions: {width}x{height}")
        logger.info(f"Image mode: {image.mode}")
        
        # Analyze pixel data
        image_np = np.array(image)
        logger.info(f"Image array shape: {image_np.shape}")
        logger.info(f"Image data type: {image_np.dtype}")
        
        # Calculate basic statistics
        mean_color = np.mean(image_np, axis=(0, 1))
        std_color = np.std(image_np, axis=(0, 1))
        min_color = np.min(image_np, axis=(0, 1))
        max_color = np.max(image_np, axis=(0, 1))
        
        logger.info(f"Mean color values: {mean_color}")
        logger.info(f"Std dev color values: {std_color}")
        logger.info(f"Min color values: {min_color}")
        logger.info(f"Max color values: {max_color}")
        
        # Save a copy for inspection
        copy_path = os.path.join(os.path.dirname(image_path), 'image_analysis.jpg')
        image.save(copy_path)
        logger.info(f"Saved image copy for analysis to {copy_path}")
        
        return image_np
    
    except Exception as e:
        logger.error(f"Error examining image: {e}")
        return None

def test_yolo_model(image_np, model_path, conf_threshold=0.001):
    """Test a specific YOLO model with the given confidence threshold"""
    try:
        # Import YOLO
        from ultralytics import YOLO
        
        logger.info(f"Testing YOLO model: {os.path.basename(model_path)}")
        logger.info(f"Confidence threshold: {conf_threshold}")
        
        # Load the model
        model = YOLO(model_path)
        
        # Set confidence threshold
        model.conf = conf_threshold
        
        # Run detection with verbose output
        logger.info("Running detection...")
        results = model(image_np, verbose=True)
        
        # Process results
        if results:
            result = results[0]
            
            # Check for detections
            if hasattr(result, 'boxes') and hasattr(result.boxes, 'data'):
                boxes = result.boxes.data
                
                if len(boxes) > 0:
                    logger.info(f"Found {len(boxes)} detections")
                    
                    # Display each detection with full details
                    for i, box in enumerate(boxes):
                        class_id = int(box[5]) if len(box) > 5 else 0
                        confidence = float(box[4]) if len(box) > 4 else 0
                        
                        # Get class name if available
                        class_name = "Unknown"
                        if hasattr(model, 'names') and class_id in model.names:
                            class_name = model.names[class_id]
                        
                        x1, y1, x2, y2 = box[:4]
                        
                        logger.info(f"Detection {i+1}:")
                        logger.info(f"  Class: {class_id} ({class_name})")
                        logger.info(f"  Confidence: {confidence:.4f}")
                        logger.info(f"  Bounding box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                        
                    # Save visualization
                    output_dir = '/ttt/data/temp/ml_results_test'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    output_path = os.path.join(
                        output_dir, 
                        f"result_{os.path.basename(model_path)}_{conf_threshold:.4f}.jpg"
                    )
                    
                    im_array = result.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                    im.save(output_path)
                    logger.info(f"Saved detection visualization to {output_path}")
                    
                    return len(boxes), output_path
                else:
                    logger.warning(f"No objects detected with confidence threshold {conf_threshold}")
                    return 0, None
            else:
                logger.warning("Result has no valid boxes data")
                return 0, None
        else:
            logger.warning("No valid results returned from YOLO")
            return 0, None
            
    except Exception as e:
        logger.error(f"Error running YOLO detection: {e}")
        return 0, None

def find_available_models():
    """Find available YOLO models"""
    # Define possible model paths
    model_paths = [
        # Models in the pipeline directory
        "/ttt/tree_ml/pipeline/model/yolo11s.pt",
        "/ttt/tree_ml/pipeline/model/yolo11l.pt",
        "/ttt/tree_ml/pipeline/model/yolov8n.pt",
        "/ttt/tree_ml/pipeline/model/yolov8m.pt",
        "/ttt/tree_ml/pipeline/model/yolov8s.pt",
        
        # Models in the backend directory
        "/ttt/tree_ml/dashboard/backend/yolov8m.pt",
        
        # Standard YOLOv8 models from ultralytics
        "yolov8n.pt",  # Will use pre-trained model from ultralytics
        "yolov8s.pt",
        "yolov8m.pt"
    ]
    
    # Check which models exist
    available_models = []
    for path in model_paths:
        if os.path.exists(path) or not path.startswith("/"):  # Local path or pre-trained
            available_models.append(path)
            logger.info(f"Found model: {path}")
    
    return available_models

def main():
    """Main function to test YOLO on satellite image"""
    logger.info("Starting REAL YOLO test on satellite image - NO SYNTHETIC DATA")
    
    # Find the satellite image
    image_path = find_satellite_image()
    if not image_path:
        logger.error("Exiting: No satellite image found")
        return
    
    logger.info(f"Using satellite image: {image_path}")
    
    # Examine the satellite image
    image_np = examine_satellite_image(image_path)
    if image_np is None:
        logger.error("Exiting: Failed to analyze satellite image")
        return
    
    # Find available models
    models = find_available_models()
    if not models:
        logger.error("Exiting: No YOLO models found")
        return
    
    # Test each model with decreasing confidence thresholds
    confidence_thresholds = [0.25, 0.1, 0.05, 0.01, 0.001]
    
    # Results tracking
    best_result = None
    best_count = 0
    
    for model_path in models:
        for conf in confidence_thresholds:
            logger.info(f"\n========== Testing {os.path.basename(model_path)} with conf={conf} ==========")
            count, output_path = test_yolo_model(image_np, model_path, conf)
            
            if count > best_count:
                best_count = count
                best_result = {
                    'model': model_path,
                    'confidence': conf,
                    'count': count,
                    'output_path': output_path
                }
    
    # Summary
    logger.info("\n===== TEST SUMMARY =====")
    if best_count > 0:
        logger.info(f"Best result: {best_count} detections using {os.path.basename(best_result['model'])} with conf={best_result['confidence']}")
        logger.info(f"Visualization saved to: {best_result['output_path']}")
    else:
        logger.warning("NO DETECTIONS found with any model or confidence threshold")
        logger.warning("The models are NOT detecting any trees in the satellite image")
    
    logger.info("\nNext steps:")
    if best_count > 0:
        logger.info(f"1. Use model {best_result['model']} with confidence threshold {best_result['confidence']}")
        logger.info(f"2. Update object_recognition.py to use this configuration")
    else:
        logger.info("1. Consider using a different satellite image or pre-processing techniques")
        logger.info("2. Evaluate if the YOLO models are appropriate for satellite tree detection")
        logger.info("3. Consider training a specialized model for satellite tree detection")

if __name__ == "__main__":
    main()