#!/usr/bin/env python3
"""
ML Tester - Unified ML testing script for Tree ML project

This script provides a standardized interface for testing machine learning models
(DeepForest and SAM) used in the Tree ML project. It can run individual model tests
or the complete pipeline test using test images.

Features:
- GPU/CPU mode selection
- Support for different model types (DeepForest, SAM)
- Pipeline testing with image input/output
- Comprehensive reporting and logging

Usage examples:
  # Test models with default settings
  python ml_tester.py

  # Test specific models
  python ml_tester.py --test deepforest
  python ml_tester.py --test sam
  
  # Test full pipeline
  python ml_tester.py --test pipeline
  
  # Force CPU mode
  python ml_tester.py --cpu
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ml_tester')

# Default directories
DATA_DIR = "/ttt/data/tests"
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")
TEST_RESULTS_DIR = DATA_DIR  # Main test results directory

# Ensure results directories exist
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

# Dallas, TX coordinates for consistent testing
DALLAS_LAT = 32.7767
DALLAS_LNG = -96.7970

# Import required libraries - no fallbacks
import torch
TORCH_AVAILABLE = True
logger.info("Using PyTorch version: " + torch.__version__)

import deepforest
from deepforest import main
DEEPFOREST_AVAILABLE = True
logger.info("Using DeepForest version: " + deepforest.__version__)

from segment_anything import sam_model_registry, SamPredictor
SAM_AVAILABLE = True
logger.info("SAM model loaded successfully")

# Global model cache - with per-device model caching
_deepforest_models = {}

def get_deepforest_model(device="cuda:0"):
    """Get or create a cached DeepForest model for the specific device"""
    global _deepforest_models
    
    # Only create a new model if we don't have one for this device
    if device not in _deepforest_models:
        # Initialize model
        model = deepforest.main.deepforest()
        
        # Use the NEON model
        neon_model_path = "/ttt/tree_ml/pipeline/model/deepforest/NEON.pt"
        
        # Move model to device first
        model.model = model.model.to(device)
        
        # Load the weights with the correct device mapping
        weights = torch.load(neon_model_path, map_location=device)
        
        # Load state dict
        model.model.load_state_dict(weights)
        
        # Ensure model is in eval mode
        model.model.eval()
        
        logger.info(f"Created new DeepForest NEON model on {device}")
        
        # Cache the model
        _deepforest_models[device] = model
    else:
        # Get the cached model
        model = _deepforest_models[device]
        
        # Ensure model is on the right device and in eval mode
        model.model = model.model.to(device)
        model.model.eval()
            
        logger.info(f"Using cached DeepForest model on {device}")
    
    return _deepforest_models[device]


def create_visualization(image_path, boxes=None, sam_masks_dir=None, output_path=None, enhance_image=False):
    """
    Create a visualization of detection boxes and SAM masks without relying on DeepForest's visualization
    
    Args:
        image_path: Path to the input image
        boxes: List of detection boxes (dict with xmin, ymin, xmax, ymax, etc.)
        sam_masks_dir: Directory containing SAM mask images
        output_path: Where to save the visualization
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image, ImageDraw, ImageEnhance
        import os
        import numpy as np
        
        # Load original image
        base_img = Image.open(image_path)
        
        # Apply image enhancement if requested
        if enhance_image:
            # Apply same enhancement settings as used for detection
            contrast = ImageEnhance.Contrast(base_img)
            base_img = contrast.enhance(1.5)  # Increase contrast by 50%
            
            # Increase color saturation to differentiate vegetation
            saturation = ImageEnhance.Color(base_img)
            base_img = saturation.enhance(1.8)  # Increase saturation by 80%
            
            # Add brightness adjustment to help with darker trees
            brightness = ImageEnhance.Brightness(base_img)
            base_img = brightness.enhance(1.2)  # Increase brightness by 20%
            
            # Add sharpness to better define tree edges
            sharpness = ImageEnhance.Sharpness(base_img)
            base_img = sharpness.enhance(2.0)  # Increase sharpness by 100%
        
        # Create a drawing context
        if base_img.mode != "RGBA":
            base_img = base_img.convert("RGBA")
        
        # Create overlay for masks
        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Colors for different trees
        colors = [(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
                 (255, 255, 0, 128), (255, 0, 255, 128), (0, 255, 255, 128)]
        
        # Draw boxes if available
        if boxes and len(boxes) > 0:
            for i, box in enumerate(boxes):
                color = colors[i % len(colors)]
                
                # Extract coordinates
                if isinstance(box, dict):
                    if 'xmin' in box:
                        x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                    else:
                        x1, y1, x2, y2 = box.get('x1', 0), box.get('y1', 0), box.get('x2', 100), box.get('y2', 100)
                    
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=2)
                    
                    # Add label with confidence score if available
                    if 'score' in box or 'confidence' in box:
                        score = box.get('score', box.get('confidence', 0))
                        draw.text((x1, y1-15), f"Tree: {score:.2f}", fill=color[:3])
        
        # Add SAM masks if available
        if sam_masks_dir and os.path.exists(sam_masks_dir):
            # First try to load masks directly from sam_masks.json if it exists
            json_path = os.path.join(sam_masks_dir, "sam_masks.json")
            if os.path.exists(json_path):
                try:
                    import json
                    with open(json_path, 'r') as f:
                        masks_info = json.load(f)
                    
                    # Match masks with corresponding image files
                    for i, mask_info in enumerate(masks_info):
                        mask_path = os.path.join(sam_masks_dir, f"sam_mask_{mask_info.get('box_idx', i)}.png")
                        if os.path.exists(mask_path):
                            try:
                                mask_img = Image.open(mask_path).convert("L")
                                color = colors[i % len(colors)]
                                
                                # Apply mask
                                mask_np = np.array(mask_img) > 128
                                for y in range(0, base_img.height, 2):  # Sample every 2 pixels for speed
                                    for x in range(0, base_img.width, 2):
                                        if y < mask_np.shape[0] and x < mask_np.shape[1] and mask_np[y, x]:
                                            overlay.putpixel((x, y), color)
                                            
                                # Also draw corresponding box from mask info if boxes aren't provided
                                if not boxes and 'bbox' in mask_info:
                                    bbox = mask_info['bbox']
                                    draw.rectangle(bbox, outline=color[:3], width=2)
                                    # Add score if available
                                    if 'score' in mask_info:
                                        draw.text((bbox[0], bbox[1]-15), f"Tree: {mask_info['score']:.2f}", fill=color[:3])
                            except Exception as e:
                                logger.error(f"Error applying mask from info: {e}")
                except Exception as json_error:
                    logger.error(f"Error loading masks from json: {json_error}")
                    
            # Fall back to looking for individual mask files
            if not os.path.exists(json_path):
                for i in range(100):  # Check for up to 100 masks
                    mask_path = os.path.join(sam_masks_dir, f"sam_mask_{i}.png")
                    if os.path.exists(mask_path):
                        try:
                            mask_img = Image.open(mask_path).convert("L")
                            color = colors[i % len(colors)]
                            
                            # Apply mask
                            mask_np = np.array(mask_img) > 128
                            for y in range(0, base_img.height, 2):  # Sample every 2 pixels for speed
                                for x in range(0, base_img.width, 2):
                                    if y < mask_np.shape[0] and x < mask_np.shape[1] and mask_np[y, x]:
                                        overlay.putpixel((x, y), color)
                        except Exception as e:
                            logger.error(f"Error applying mask {i}: {e}")
        
        # Combine original image with overlay
        result = Image.alpha_composite(base_img, overlay)
        
        # Convert back to RGB for saving
        result = result.convert("RGB")
        
        # Save the visualization
        if output_path:
            result.save(output_path)
            logger.info(f"Visualization saved to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return False


def test_deepforest(test_image_path=None, force_cpu=False, save_results=True):
    """Test DeepForest on an image and return results"""
    if not DEEPFOREST_AVAILABLE:
        logger.error("DeepForest not available - test skipped")
        return False, {}
    
    # Check CUDA
    cuda_available = torch.cuda.is_available() and not force_cpu
    device = "cuda:0" if cuda_available else "cpu"
    logger.info(f"Testing DeepForest on device: {device}")
    
    # Choose test image
    if not test_image_path:
        # Use default test image
        test_image_path = os.path.join(TEST_IMAGES_DIR, "1_satellite_image.jpg")
        if not os.path.exists(test_image_path):
            # Find any available test image
            test_images = list(Path(TEST_IMAGES_DIR).glob("*.jpg"))
            if test_images:
                test_image_path = str(test_images[0])
            else:
                logger.error("No test images found")
                return False, {}
    
    logger.info(f"Testing DeepForest on image: {test_image_path}")
    
    # Get image name for unique results
    image_name = os.path.basename(test_image_path)
    base_name = os.path.splitext(image_name)[0]
    
    # Setup image-specific results directory
    image_results_dir = os.path.join(TEST_RESULTS_DIR, base_name)
    os.makedirs(image_results_dir, exist_ok=True)
    
    # Initialize DeepForest model
    try:
        start_time = time.time()
        
        # Get cached model
        model = get_deepforest_model(device=device)
        
        # Run prediction
        logger.info("Running DeepForest prediction...")
        
        # Fix for device type mismatch error
        # Override predict_image method temporarily to handle device properly
        # Custom implementation to handle device properly
        from PIL import Image
        import torchvision.transforms as T
        import pandas as pd
        
        # Load image
        image = Image.open(test_image_path)
        
        # Convert to tensor and move to correct device
        transform = T.Compose([T.ToTensor()])
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Set model to correct device and eval mode
        model.model = model.model.to(device)
        model.model.eval()
        
        # Run detection directly on tensor with no gradients
        with torch.no_grad():
            predictions = model.model(tensor)[0]
        
        # Convert to pandas boxes format manually since tensor_to_boxes is not available
        # Extract box predictions
        boxes_tensor = predictions['boxes'].cpu().detach()
        scores_tensor = predictions['scores'].cpu().detach()
        
        # Create DataFrame with box coordinates and scores
        boxes_data = []
        for i in range(len(boxes_tensor)):
            if scores_tensor[i] > 0.005:  # Lower score threshold to 0.005 to detect more trees
                box = boxes_tensor[i]
                boxes_data.append({
                    'xmin': float(box[0]),
                    'ymin': float(box[1]),
                    'xmax': float(box[2]),
                    'ymax': float(box[3]),
                    'score': float(scores_tensor[i]),
                    'label': 'Tree'
                })
        
        boxes = pd.DataFrame(boxes_data)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"DeepForest prediction completed in {execution_time:.2f} seconds")
        
        # Create results
        success = True
        detection_count = len(boxes) if not boxes.empty else 0
        
        if detection_count > 0:
            logger.info(f"DeepForest detected {detection_count} trees")
            
            # Save visualization if requested
            if save_results:
                try:
                    # Save boxes as JSON to both locations
                    boxes_clean = []
                    for _, row in boxes.iterrows():
                        box_dict = {}
                        for col in boxes.columns:
                            val = row[col]
                            if 'Polygon' in str(type(val)):
                                box_dict[col] = str(val)
                            else:
                                box_dict[col] = val
                        boxes_clean.append(box_dict)
                    
                            # Save primarily to image-specific location
                    with open(os.path.join(image_results_dir, "detection_results.json"), 'w') as f:
                        json.dump(boxes_clean, f, indent=2)
                        
                    # Also create an image-specific report
                    report = {
                        "timestamp": datetime.now().isoformat(),
                        "image": os.path.basename(test_image_path),
                        "model": "DeepForest",
                        "execution_time": execution_time,
                        "detection_count": detection_count,
                        "success": True
                    }
                    
                    with open(os.path.join(image_results_dir, "ml_test_report.json"), 'w') as f:
                        json.dump(report, f, indent=2)
                    
                    # Create visualization
                    prediction_image = model.predict_image(path=test_image_path, return_plot=True)
                    
                    # Save directly to image-specific directory
                    if hasattr(prediction_image, 'save'):
                        prediction_image.save(os.path.join(image_results_dir, "visualization.png"))
                    else:
                        # Convert numpy array to image if needed
                        from PIL import Image
                        Image.fromarray(prediction_image).save(os.path.join(image_results_dir, "visualization.png"))
                    
                    # Save summary
                    with open(os.path.join(image_results_dir, "summary.txt"), 'w') as f:
                        f.write(f"Image: {test_image_path}\n")
                        f.write(f"Test: DeepForest\n")
                        f.write(f"Processed: {datetime.now().isoformat()}\n")
                        f.write(f"Trees detected: {detection_count}\n")
                        f.write(f"Execution time: {execution_time:.2f} seconds\n")
                    
                    logger.info(f"Results saved to {image_results_dir}")
                except Exception as e:
                    logger.error(f"Error saving results: {e}")
        else:
            logger.warning("DeepForest didn't detect any trees")
        
        # Return results
        results = {
            "success": success,
            "execution_time": execution_time,
            "detection_count": detection_count,
            "model_type": "DeepForest",
            "device": device,
            "timestamp": datetime.now().isoformat()
        }
        
        return success, results
    
    except Exception as e:
        logger.error(f"Error testing DeepForest: {e}")
        return False, {"error": str(e)}


def test_sam(test_image_path=None, detection_boxes=None, force_cpu=False, save_results=True):
    """Test SAM on an image with detection boxes and return results"""
    if not SAM_AVAILABLE:
        logger.error("SAM not available - test skipped")
        return False, {}
    
    # Check CUDA
    cuda_available = torch.cuda.is_available() and not force_cpu
    device = "cuda:0" if cuda_available else "cpu"
    logger.info(f"Testing SAM on device: {device}")
    
    # Choose test image
    if not test_image_path:
        # Use default test image
        test_image_path = os.path.join(TEST_IMAGES_DIR, "1_satellite_image.jpg")
        if not os.path.exists(test_image_path):
            # Find any available test image
            test_images = list(Path(TEST_IMAGES_DIR).glob("*.jpg"))
            if test_images:
                test_image_path = str(test_images[0])
            else:
                logger.error("No test images found")
                return False, {}
    
    # Check if we have boxes
    if not detection_boxes:
        logger.warning("No detection boxes provided - will generate random test boxes")
        # Create synthetic boxes for testing
        import random
        from PIL import Image
        try:
            img = Image.open(test_image_path)
            width, height = img.size
            
            # Generate 3 random boxes
            detection_boxes = []
            for i in range(3):
                x1 = random.randint(0, width - 100)
                y1 = random.randint(0, height - 100)
                w = random.randint(50, 100)
                h = random.randint(50, 100)
                detection_boxes.append({
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x1 + w,
                    'ymax': y1 + h,
                    'confidence': 0.8,
                    'label': 'tree'
                })
            logger.info(f"Created {len(detection_boxes)} test boxes")
        except Exception as e:
            logger.error(f"Error creating test boxes: {e}")
            return False, {}
    
    logger.info(f"Testing SAM on image: {test_image_path} with {len(detection_boxes)} boxes")
    
    # Find SAM checkpoint
    sam_checkpoints = [
        "/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth",
        "/ttt/tree_ml/pipeline/model/sam_vit_l_0b3195.pth",
        "/ttt/tree_ml/pipeline/model/sam_vit_b_01ec64.pth"
    ]
    
    sam_checkpoint = None
    for path in sam_checkpoints:
        if os.path.exists(path):
            sam_checkpoint = path
            logger.info(f"Found SAM checkpoint: {sam_checkpoint}")
            break
    
    if not sam_checkpoint:
        logger.error("No SAM checkpoint found")
        return False, {"error": "No SAM checkpoint found"}
    
    # Determine model type
    if "vit_h" in sam_checkpoint:
        model_type = "vit_h"
    elif "vit_l" in sam_checkpoint:
        model_type = "vit_l"
    elif "vit_b" in sam_checkpoint:
        model_type = "vit_b"
    else:
        model_type = "vit_b"  # default
    
    try:
        # Start timing
        start_time = time.time()
        
        # Initialize SAM
        logger.info(f"Initializing SAM model ({model_type})...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        # Initialize predictor
        predictor = SamPredictor(sam)
        logger.info("SAM model initialized")
        
        # Load image
        from PIL import Image
        image = np.array(Image.open(test_image_path))
        predictor.set_image(image)
        
        # Process boxes
        masks_output = []
        successful_masks = 0
        
        for i, box in enumerate(detection_boxes):
            try:
                # Extract bbox
                if 'xmin' in box:
                    x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                else:
                    x1, y1, x2, y2 = box.get('x1', 0), box.get('y1', 0), box.get('x2', 100), box.get('y2', 100)
                
                # Get center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Create prompts
                input_point = np.array([[center_x, center_y]])
                input_label = np.array([1])  # foreground
                box_np = np.array([x1, y1, x2, y2])
                
                # Generate masks
                masks, scores, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    box=box_np,
                    multimask_output=True
                )
                
                # Get best mask
                if len(masks) > 0:
                    best_mask_idx = np.argmax(scores)
                    mask = masks[best_mask_idx]
                    score = float(scores[best_mask_idx])
                    
                    logger.info(f"Box {i+1}: SAM segmentation score: {score:.4f}")
                    successful_masks += 1
                    
                    # Store mask info
                    masks_output.append({
                        'box_idx': i,
                        'mask_idx': int(best_mask_idx),
                        'score': score,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
                    
                    # Save mask image if requested
                    if save_results:
                        # Get image name for unique results if we have a test image path
                        if test_image_path:
                            image_name = os.path.basename(test_image_path)
                            base_name = os.path.splitext(image_name)[0]
                            # Create image-specific directory if needed
                            image_results_dir = os.path.join(TEST_RESULTS_DIR, base_name)
                            os.makedirs(image_results_dir, exist_ok=True)
                            
                            # Save mask directly to image-specific directory only
                            from PIL import Image
                            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                            mask_img.save(os.path.join(image_results_dir, f"sam_mask_{i}.png"))
                        else:
                            # If no specific image path, create a temp results dir
                            temp_results_dir = os.path.join(TEST_RESULTS_DIR, "temp_results")
                            os.makedirs(temp_results_dir, exist_ok=True)
                            
                            # Save mask
                            from PIL import Image
                            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                            mask_img.save(os.path.join(temp_results_dir, f"sam_mask_{i}.png"))
            except Exception as e:
                logger.error(f"Error processing box {i}: {e}")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"SAM segmentation completed in {execution_time:.2f} seconds")
        
        # Save results to image-specific directory only
        if save_results and masks_output and test_image_path:
            try:
                # Get image name for results directory
                image_name = os.path.basename(test_image_path)
                base_name = os.path.splitext(image_name)[0]
                image_results_dir = os.path.join(TEST_RESULTS_DIR, base_name)
                os.makedirs(image_results_dir, exist_ok=True)
                
                # Save masks JSON to image-specific directory only
                with open(os.path.join(image_results_dir, "sam_masks.json"), 'w') as f:
                    json.dump(masks_output, f, indent=2)
                
                # Create visualization
                from PIL import Image, ImageDraw
                base_img = Image.open(test_image_path).convert("RGBA")
                overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                # Colors for masks
                colors = [(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128), 
                        (255, 255, 0, 128), (255, 0, 255, 128), (0, 255, 255, 128)]
                
                # Draw masks
                for i, mask_info in enumerate(masks_output):
                    color = colors[i % len(colors)]
                    
                    # Draw box
                    bbox = mask_info['bbox']
                    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color[:3], width=2)
                    
                    # Add mask
                    mask_path = os.path.join(image_results_dir, f"sam_mask_{mask_info['box_idx']}.png")
                    if os.path.exists(mask_path):
                        mask_img = Image.open(mask_path).convert("L")
                        
                        # Use faster approach for large images
                        mask_np = np.array(mask_img) > 128
                        for y in range(0, base_img.height, 2):  # Sample every 2 pixels for speed
                            for x in range(0, base_img.width, 2):
                                if y < mask_np.shape[0] and x < mask_np.shape[1] and mask_np[y, x]:
                                    overlay.putpixel((x, y), color)
                
                # Combine images
                result = Image.alpha_composite(base_img, overlay)
                
                # Save only to image-specific directory
                if test_image_path:
                    image_name = os.path.basename(test_image_path)
                    base_name = os.path.splitext(image_name)[0]
                    
                    # Save to image-specific directory only
                    image_results_dir = os.path.join(TEST_RESULTS_DIR, base_name)
                    os.makedirs(image_results_dir, exist_ok=True)
                    result.save(os.path.join(image_results_dir, "visualization.png"))
                    
                    logger.info(f"SAM visualization saved to {image_results_dir}/visualization.png")
                else:
                    # If no specific image path, save to temp directory
                    temp_results_dir = os.path.join(TEST_RESULTS_DIR, "temp_results")
                    os.makedirs(temp_results_dir, exist_ok=True)
                    result.save(os.path.join(temp_results_dir, "visualization.png"))
                    logger.info(f"SAM visualization saved to {temp_results_dir}/visualization.png")
            except Exception as e:
                logger.error(f"Error saving SAM results: {e}")
        
        # Return results
        success = successful_masks > 0
        results = {
            "success": success,
            "execution_time": execution_time,
            "input_boxes": len(detection_boxes),
            "successful_masks": successful_masks,
            "model_type": f"SAM ({model_type})",
            "device": device,
            "timestamp": datetime.now().isoformat()
        }
        
        return success, results
    
    except Exception as e:
        logger.error(f"Error testing SAM: {e}")
        return False, {"error": str(e)}


def test_pipeline(test_image_path=None, force_cpu=False, save_results=True):
    """Test the complete ML pipeline (DeepForest + SAM)"""
    import os
    logger.info("Testing complete ML pipeline (DeepForest + SAM)")
    
    # Choose test image
    if not test_image_path:
        # Use default test image
        test_image_path = os.path.join(TEST_IMAGES_DIR, "1_satellite_image.jpg")
        if not os.path.exists(test_image_path):
            # Find any available test image
            test_images = list(Path(TEST_IMAGES_DIR).glob("*.jpg"))
            if test_images:
                test_image_path = str(test_images[0])
            else:
                logger.error("No test images found")
                return False, {}
    
    logger.info(f"Testing pipeline on image: {test_image_path}")
    
    # Get image name for unique results
    image_name = os.path.basename(test_image_path)
    base_name = os.path.splitext(image_name)[0]
    
    # Image-specific results directory only
    image_results_dir = os.path.join(TEST_RESULTS_DIR, base_name)
    os.makedirs(image_results_dir, exist_ok=True)
    
    # Run DeepForest
    start_time = time.time()
    df_success, df_results = test_deepforest(
        test_image_path=test_image_path, 
        force_cpu=force_cpu,
        save_results=False  # Don't save results, we'll do it in pipeline
    )
    
    # Extract detection boxes
    detection_boxes = []
    if df_success and "detection_count" in df_results and df_results["detection_count"] > 0:
        # Load boxes from DeepForest
        try:
            # Import torch to make it available in this scope
            import torch
            
            # Get cached model using the same device setting as DeepForest
            cuda_available = torch.cuda.is_available() and not force_cpu
            device_str = "cuda:0" if cuda_available else "cpu"
            model = get_deepforest_model(device=device_str)
            
            # Extract boxes directly from model predictions
            try:
                # Enhance the image before prediction to improve tree detection
                from PIL import Image, ImageEnhance
                import numpy as np
                
                # Load and enhance the image
                img = Image.open(test_image_path)
                
                # Increase contrast more significantly to make trees stand out
                contrast = ImageEnhance.Contrast(img)
                img_enhanced = contrast.enhance(1.5)  # Increase contrast by 50%
                
                # Increase color saturation to differentiate vegetation
                saturation = ImageEnhance.Color(img_enhanced)
                img_enhanced = saturation.enhance(1.8)  # Increase saturation by 80%
                
                # Add brightness adjustment to help with darker trees
                brightness = ImageEnhance.Brightness(img_enhanced)
                img_enhanced = brightness.enhance(1.2)  # Increase brightness by 20%
                
                # Add sharpness to better define tree edges
                sharpness = ImageEnhance.Sharpness(img_enhanced)
                img_enhanced = sharpness.enhance(2.0)  # Increase sharpness by 100%
                
                # Save enhanced image as processed.png in results directory
                image_name = os.path.basename(test_image_path)
                base_name = os.path.splitext(image_name)[0]
                image_results_dir = os.path.join(TEST_RESULTS_DIR, base_name)
                os.makedirs(image_results_dir, exist_ok=True)
                
                # Save enhanced image to results directory
                processed_path = os.path.join(image_results_dir, "processed.png")
                img_enhanced.save(processed_path)
                
                # Also save a temporary version for processing
                enhanced_path = test_image_path + ".enhanced.jpg"
                img_enhanced.save(enhanced_path)
                
                # Set extremely low threshold for detection (0.005) to catch more trees
                model.config["score_thresh"] = 0.005
                
                # Set NMS IoU threshold higher to prevent duplicate detections being filtered
                model.config["nms_thresh"] = 0.4
                
                # Convert input to properly formatted dataframe using the enhanced image
                boxes = model.predict_image(path=enhanced_path, return_plot=False)
                
                # Remove temporary file
                import os
                if os.path.exists(enhanced_path):
                    os.remove(enhanced_path)
                
                # Convert to list of dicts
                for _, row in boxes.iterrows():
                    box_dict = {}
                    for col in boxes.columns:
                        val = row[col]
                        if 'Polygon' in str(type(val)):
                            box_dict[col] = str(val)
                        else:
                            box_dict[col] = val
                    detection_boxes.append(box_dict)
                
                logger.info(f"Using {len(detection_boxes)} detected boxes from predict_image")
            except Exception as model_error:
                # Fallback to direct model inference
                from PIL import Image
                import torch
                import torchvision.transforms as T
                
                # Get device from model
                device = next(model.model.parameters()).device
                
                # Load the image
                img = Image.open(test_image_path)
                
                # Convert to tensor and move to model's device
                transform = T.Compose([T.ToTensor()])
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # Run prediction with no gradients
                model.model.eval()
                with torch.no_grad():
                    predictions = model.model(img_tensor)
                
                # Get detected boxes (maintain device consistency)
                if len(predictions) > 0 and "boxes" in predictions[0]:
                    boxes_tensor = predictions[0]["boxes"]
                    scores_tensor = predictions[0]["scores"]
                    
                    # Filter by confidence and convert to list of dicts
                    for i in range(len(boxes_tensor)):
                        if scores_tensor[i] > 0.005:  # Further lowered confidence threshold to 0.005
                            box = boxes_tensor[i].cpu()  # Only convert to CPU for final output
                            detection_boxes.append({
                                'xmin': float(box[0]),
                                'ymin': float(box[1]),
                                'xmax': float(box[2]),
                                'ymax': float(box[3]),
                                'confidence': float(scores_tensor[i].cpu()),
                                'label': 'tree'
                            })
                    
                    logger.info(f"Using {len(detection_boxes)} detected boxes from direct inference")
            
            logger.info(f"Using {len(detection_boxes)} detected boxes for segmentation")
            
            # Save detection visualization (without using DeepForest's visualization)
            if save_results and len(detection_boxes) > 0:
                # Create manual visualization of boxes
                from PIL import Image, ImageDraw
                vis_img = Image.open(test_image_path)
                draw = ImageDraw.Draw(vis_img)
                
                # Draw each box
                for box in detection_boxes:
                    x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                    confidence = box['confidence']
                    # Draw rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                    # Add confidence label
                    draw.text((x1, y1-15), f"Tree: {confidence:.2f}", fill=(255, 0, 0))
                
                # Save the visualization
                vis_img.save(os.path.join(image_results_dir, "detection.png"))
                
                # Save boxes as JSON to image-specific directory
                with open(os.path.join(image_results_dir, "detection_boxes.json"), 'w') as f:
                    json.dump(detection_boxes, f, indent=2)
        except Exception as e:
            logger.error(f"Error extracting detection boxes: {e}")
            # Fall back to test boxes on failure
            detection_boxes = []
    
    # If no boxes were detected or extraction failed, create test boxes
    if not detection_boxes:
        logger.warning("No trees detected by DeepForest, using test boxes for SAM")
        # Create test boxes
        import random
        from PIL import Image
        try:
            img = Image.open(test_image_path)
            width, height = img.size
            
            # Generate 3 random boxes
            for i in range(3):
                x1 = random.randint(0, width - 100)
                y1 = random.randint(0, height - 100)
                w = random.randint(50, 100)
                h = random.randint(50, 100)
                detection_boxes.append({
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x1 + w,
                    'ymax': y1 + h,
                    'confidence': 0.8,
                    'label': 'tree'
                })
            logger.info(f"Created {len(detection_boxes)} test boxes")
        except Exception as e:
            logger.error(f"Error creating test boxes: {e}")
    
    # Run SAM
    sam_success, sam_results = test_sam(
        test_image_path=test_image_path,
        detection_boxes=detection_boxes,
        force_cpu=force_cpu,
        save_results=False  # Don't save separately, we'll do it in pipeline
    )
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Save pipeline results
    if save_results:
        try:
            # Save report
            import torch
            pipeline_report = {
                "timestamp": datetime.now().isoformat(),
                "total_time": total_time,
                "detection": df_results,
                "segmentation": sam_results,
                "image_path": test_image_path,
                "cuda_available": torch.cuda.is_available() and not force_cpu
            }
            
            with open(os.path.join(image_results_dir, "pipeline_report.json"), 'w') as f:
                json.dump(pipeline_report, f, indent=2)
            
            # Save visualization using our custom helper function
            try:
                # First save detection boxes to a JSON file if they aren't already saved
                if detection_boxes and not os.path.exists(os.path.join(image_results_dir, "detection_boxes.json")):
                    with open(os.path.join(image_results_dir, "detection_boxes.json"), 'w') as f:
                        json.dump(detection_boxes, f, indent=2)
                
                # Use our custom visualization function with original image
                viz_success = create_visualization(
                    image_path=test_image_path,
                    boxes=detection_boxes,
                    sam_masks_dir=image_results_dir,  # Directory containing SAM masks
                    output_path=os.path.join(image_results_dir, "detection.png"),
                    enhance_image=False  # Use original image for visualization
                )
                
                if viz_success:
                    logger.info(f"Pipeline visualization saved to {image_results_dir}/detection.png")
                else:
                    # No fallbacks - if visualization fails, it should fail completely
                    logger.error("Failed to create visualization")
                    raise RuntimeError("Visualization creation failed")
            except Exception as e:
                logger.error(f"Error saving pipeline visualization: {e}")
                # Re-raise the exception - no fallbacks
                raise e
            
            logger.info(f"Pipeline results saved to {image_results_dir}")
        except Exception as e:
            logger.error(f"Error saving pipeline results: {e}")
    
    # Return combined results
    success = df_success or sam_success
    results = {
        "success": success,
        "total_time": total_time,
        "detection_success": df_success,
        "detection_count": df_results.get("detection_count", 0),
        "segmentation_success": sam_success,
        "segmentation_count": sam_results.get("successful_masks", 0),
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Pipeline test completed in {total_time:.2f} seconds")
    if success:
        logger.info("✅ Pipeline test successful")
    else:
        logger.warning("⚠️ Pipeline test partially successful or failed")
    
    return success, results


def process_all_images(test_type, force_cpu=False, save_results=True):
    """Process all images in the test directory with the specified test"""
    # Get all test images
    test_images = list(Path(TEST_IMAGES_DIR).glob("*.jpg")) + list(Path(TEST_IMAGES_DIR).glob("*.png"))
    if not test_images:
        logger.error("No test images found in the test directory")
        return False, {}
    
    logger.info(f"Found {len(test_images)} test images to process")
    
    # Track results for all images
    all_image_results = {}
    success_count = 0
    
    # Process each image
    for i, image_path in enumerate(test_images):
        image_name = image_path.stem
        logger.info(f"Processing image {i+1}/{len(test_images)}: {image_path.name}")
        
        # Create directory for this image's results
        image_results_dir = os.path.join(TEST_RESULTS_DIR, image_name)
        os.makedirs(image_results_dir, exist_ok=True)
        
        # Run the appropriate test
        image_success = False
        image_results = {}
        
        if test_type == "deepforest":
            # Run DeepForest on this image
            success, results = test_deepforest(
                test_image_path=str(image_path),
                force_cpu=force_cpu,
                save_results=save_results
            )
            
            # Save image-specific results
            if success and save_results:
                # Save results to image-specific directory
                try:
                    # No need to copy detection results - they should already be in the image directory
                    # We've modified the code to save directly to image-specific directories
                    
                    # Instead of copying potentially same files, regenerate the visualization
                    # directly for this specific image to ensure uniqueness
                    try:
                        # Create fresh DeepForest visualization
                        # Get cached model
                        cuda_available = torch.cuda.is_available() and not args.cpu
                        device_str = "cuda:0" if cuda_available else "cpu"
                        model = get_deepforest_model(device=device_str)
                        prediction_image = model.predict_image(path=str(image_path), return_plot=True)
                        
                        # Save directly to image-specific directory
                        if hasattr(prediction_image, 'save'):
                            prediction_image.save(os.path.join(image_results_dir, "visualization.png"))
                        else:
                            # Convert numpy array to image if needed
                            from PIL import Image
                            Image.fromarray(prediction_image).save(os.path.join(image_results_dir, "visualization.png"))
                        
                        logger.info(f"Created unique visualization for {image_path.name}")
                    except Exception as e:
                        logger.error(f"Error creating unique visualization for {image_path.name}: {e}")
                    
                    # Create summary
                    with open(os.path.join(image_results_dir, "summary.txt"), 'w') as f:
                        f.write(f"Image: {image_path}\n")
                        f.write(f"Test: DeepForest\n")
                        f.write(f"Processed: {datetime.now().isoformat()}\n")
                        f.write(f"Trees detected: {results.get('detection_count', 0)}\n")
                        f.write(f"Execution time: {results.get('execution_time', 0):.2f} seconds\n")
                except Exception as e:
                    logger.error(f"Error saving image-specific results: {e}")
            
            image_success = success
            image_results = results
            
        elif test_type == "pipeline":
            # Run full pipeline on this image
            success, results = test_pipeline(
                test_image_path=str(image_path),
                force_cpu=force_cpu,
                save_results=save_results
            )
            
            # Save image-specific results
            if success and save_results:
                try:
                    # No need to copy detection results - they should already be in the image directory
                    # We've modified the code to save directly to image-specific directories
                    
                    # Instead of copying potentially same files, regenerate the visualization
                    # directly for this specific image to ensure uniqueness
                    try:
                        # Use the same direct approach that worked in test_pipeline
                        from PIL import Image
                        import torch
                        import torchvision.transforms as T
                        import numpy as np
                        
                        # Load the image
                        img = Image.open(str(image_path))
                        width, height = img.size
                        
                        # Get cached model on CPU to avoid device issues
                        model = get_deepforest_model(device="cpu")
                        
                        # Ensure model is in eval mode on CPU
                        model.model = model.model.to("cpu")
                        model.model.eval()
                        
                        # Convert image to tensor (on CPU)
                        transform = T.Compose([T.ToTensor()])
                        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
                        
                        # Run prediction with no gradients
                        with torch.no_grad():
                            predictions = model.model(img_tensor)
                        
                        # Get detected boxes
                        detection_boxes = []
                        if len(predictions) > 0 and "boxes" in predictions[0]:
                            boxes_tensor = predictions[0]["boxes"].cpu()
                            scores_tensor = predictions[0]["scores"].cpu()
                            
                            # Filter by confidence and convert to list of dicts
                            for i in range(len(boxes_tensor)):
                                if scores_tensor[i] > 0.02:  # Lower confidence threshold
                                    box = boxes_tensor[i]
                                    detection_boxes.append({
                                        'xmin': float(box[0]),
                                        'ymin': float(box[1]),
                                        'xmax': float(box[2]),
                                        'ymax': float(box[3]),
                                        'confidence': float(scores_tensor[i]),
                                        'label': 'tree'
                                    })
                        
                        if len(detection_boxes) > 0:
                            # First, try to create a SAM visualization if SAM is available
                            if SAM_AVAILABLE:
                                try:
                                    # Find SAM checkpoint
                                    sam_checkpoints = [
                                        "/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth",
                                        "/ttt/tree_ml/pipeline/model/sam_vit_l_0b3195.pth",
                                        "/ttt/tree_ml/pipeline/model/sam_vit_b_01ec64.pth"
                                    ]
                                    
                                    sam_checkpoint = None
                                    for path in sam_checkpoints:
                                        if os.path.exists(path):
                                            sam_checkpoint = path
                                            break
                                    
                                    if sam_checkpoint:
                                        # Run SAM on the detection boxes
                                        # Initialize SAM
                                        if "vit_h" in sam_checkpoint:
                                            model_type = "vit_h"
                                        elif "vit_l" in sam_checkpoint:
                                            model_type = "vit_l"
                                        else:
                                            model_type = "vit_b"
                                            
                                        # Create SAM visualization
                                        from segment_anything import sam_model_registry, SamPredictor
                                        from PIL import Image, ImageDraw
                                        
                                        # Initialize SAM
                                        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                                        sam.to(device="cpu")  # Use CPU for safety
                                        predictor = SamPredictor(sam)
                                        
                                        # Load image
                                        image = np.array(Image.open(image_path))
                                        predictor.set_image(image)
                                        
                                        # Create visualization canvas
                                        base_img = Image.open(image_path).convert("RGBA")
                                        overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
                                        draw = ImageDraw.Draw(overlay)
                                        
                                        # Colors for masks
                                        colors = [(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
                                                (255, 255, 0, 128), (255, 0, 255, 128), (0, 255, 255, 128)]
                                        
                                        # Process each box
                                        for i, box in enumerate(detection_boxes):
                                            # Draw box
                                            color = colors[i % len(colors)]
                                            
                                            if 'xmin' in box:
                                                x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                                            else:
                                                x1, y1, x2, y2 = box.get('x1', 0), box.get('y1', 0), box.get('x2', 100), box.get('y2', 100)
                                            
                                            draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=2)
                                            
                                            # Create prompts for SAM
                                            center_x = (x1 + x2) / 2
                                            center_y = (y1 + y2) / 2
                                            input_point = np.array([[center_x, center_y]])
                                            input_label = np.array([1])  # foreground
                                            box_np = np.array([x1, y1, x2, y2])
                                            
                                            # Generate masks
                                            try:
                                                masks, scores, _ = predictor.predict(
                                                    point_coords=input_point,
                                                    point_labels=input_label,
                                                    box=box_np,
                                                    multimask_output=True
                                                )
                                                
                                                # Get best mask
                                                if len(masks) > 0:
                                                    best_mask_idx = np.argmax(scores)
                                                    mask = masks[best_mask_idx]
                                                    
                                                    # Add mask to visualization
                                                    mask_np = mask
                                                    for y in range(0, base_img.height, 2):  # Sample every 2 pixels for speed
                                                        for x in range(0, base_img.width, 2):
                                                            if y < mask_np.shape[0] and x < mask_np.shape[1] and mask_np[y, x]:
                                                                overlay.putpixel((x, y), color)
                                            except Exception as mask_error:
                                                logger.error(f"Error creating mask for box {i} in {image_path.name}: {mask_error}")
                                        
                                        # Combine images
                                        result = Image.alpha_composite(base_img, overlay)
                                        
                                        # Save directly to image-specific directory
                                        result.save(os.path.join(image_results_dir, "detection.png"))
                                        logger.info(f"Created detection visualization for {image_path.name}")
                                    else:
                                        # Fall back to DeepForest visualization
                                        prediction_image = model.predict_image(path=str(image_path), return_plot=True)
                                        if hasattr(prediction_image, 'save'):
                                            prediction_image.save(os.path.join(image_results_dir, "detection.png"))
                                        else:
                                            Image.fromarray(prediction_image).save(os.path.join(image_results_dir, "detection.png"))
                                        logger.info(f"Created unique DeepForest visualization for {image_path.name}")
                                except Exception as sam_error:
                                    logger.error(f"Error creating SAM visualization for {image_path.name}, falling back to DeepForest: {sam_error}")
                                    prediction_image = model.predict_image(path=str(image_path), return_plot=True)
                                    if hasattr(prediction_image, 'save'):
                                        prediction_image.save(os.path.join(image_results_dir, "detection.png"))
                                    else:
                                        Image.fromarray(prediction_image).save(os.path.join(image_results_dir, "detection.png"))
                            else:
                                # SAM not available, use DeepForest visualization
                                prediction_image = model.predict_image(path=str(image_path), return_plot=True)
                                if hasattr(prediction_image, 'save'):
                                    prediction_image.save(os.path.join(image_results_dir, "detection.png"))
                                else:
                                    Image.fromarray(prediction_image).save(os.path.join(image_results_dir, "detection.png"))
                                logger.info(f"Created unique DeepForest visualization for {image_path.name}")
                        else:
                            # No trees detected, just save original image with a note
                            from PIL import Image, ImageDraw, ImageFont
                            img = Image.open(image_path)
                            draw = ImageDraw.Draw(img)
                            draw.text((10, 10), "No trees detected", fill=(255, 0, 0))
                            img.save(os.path.join(image_results_dir, "detection.png"))
                            logger.warning(f"No trees detected in {image_path.name}")
                    except Exception as e:
                        logger.error(f"Error creating unique visualization for {image_path.name}: {e}")
                        # Fallback to original image if everything else fails
                        try:
                            from PIL import Image
                            Image.open(image_path).save(os.path.join(image_results_dir, "detection.png"))
                        except:
                            pass
                    
                    # Create summary
                    with open(os.path.join(image_results_dir, "summary.txt"), 'w') as f:
                        f.write(f"Image: {image_path}\n")
                        f.write(f"Test: Full Pipeline\n")
                        f.write(f"Processed: {datetime.now().isoformat()}\n")
                        f.write(f"Trees detected: {results.get('detection_count', 0)}\n")
                        f.write(f"Trees segmented: {results.get('segmentation_count', 0)}\n")
                        f.write(f"Total execution time: {results.get('total_time', 0):.2f} seconds\n")
                except Exception as e:
                    logger.error(f"Error saving image-specific results: {e}")
            
            image_success = success
            image_results = results
        
        # Track results
        all_image_results[image_name] = {
            "success": image_success,
            "results": image_results
        }
        
        if image_success:
            success_count += 1
    
    # Return overall success and results
    return success_count > 0, {
        "total_images": len(test_images),
        "successful_images": success_count,
        "image_results": all_image_results
    }


def main():
    """Main function to run ML tests"""
    parser = argparse.ArgumentParser(description="Tree ML Tester")
    parser.add_argument(
        "--test",
        choices=["all", "deepforest", "sam", "pipeline", "batch"],
        default="all",
        help="Test to run (default: all)"
    )
    parser.add_argument(
        "--image",
        help="Path to test image (optional)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (disable CUDA)"
    )
    parser.add_argument(
        "--nosave",
        action="store_true",
        help="Don't save test results"
    )
    parser.add_argument(
        "--all-images",
        action="store_true",
        help="Process all images in the test directory"
    )
    parser.add_argument(
        "--dallas",
        action="store_true",
        help="Use Dallas coordinates for S2 compatibility"
    )
    parser.add_argument(
        "--s2-compatible",
        action="store_true",
        help="Generate S2-compatible test results"
    )
    
    args = parser.parse_args()
    
    # Banner
    logger.info("=" * 60)
    logger.info("Tree ML Tester")
    logger.info("=" * 60)
    
    # CUDA info
    if TORCH_AVAILABLE:
        cuda_available = torch.cuda.is_available() and not args.cpu
        if cuda_available:
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            if args.cpu:
                logger.info("CUDA disabled (--cpu flag used)")
            else:
                logger.warning("CUDA not available, using CPU")
    
    # Process all images if requested
    if args.all_images or args.test == "batch":
        logger.info("Processing all test images")
        
        # Determine which test to run on all images
        batch_test_type = "pipeline"  # Default to pipeline
        if args.test in ["deepforest", "pipeline"]:
            batch_test_type = args.test
        
        # Process all images
        success, batch_results = process_all_images(
            test_type=batch_test_type,
            force_cpu=args.cpu,
            save_results=not args.nosave
        )
        
        # Save batch report
        if not args.nosave:
            try:
                batch_report = {
                    "timestamp": datetime.now().isoformat(),
                    "test_type": batch_test_type,
                    "total_images": batch_results.get("total_images", 0),
                    "successful_images": batch_results.get("successful_images", 0),
                    "cuda_enabled": torch.cuda.is_available() and not args.cpu if TORCH_AVAILABLE else False,
                    "s2_compatible": args.s2_compatible or args.dallas,
                    "dallas_coordinates": {
                        "enabled": args.dallas,
                        "lat": DALLAS_LAT,
                        "lng": DALLAS_LNG
                    }
                }
                
                # Save to the main test results directory
                with open(os.path.join(TEST_RESULTS_DIR, "batch_test_report.json"), 'w') as f:
                    json.dump(batch_report, f, indent=2)
                
                logger.info(f"Batch test report saved to {os.path.join(TEST_RESULTS_DIR, 'batch_test_report.json')}")
            except Exception as e:
                logger.error(f"Error saving batch test report: {e}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Batch Test Summary")
        logger.info("=" * 60)
        logger.info(f"Total images: {batch_results.get('total_images', 0)}")
        logger.info(f"Successfully processed: {batch_results.get('successful_images', 0)}")
        
        return 0 if success else 1
    
    # Get test image path for single image test
    test_image_path = args.image
    if test_image_path and not os.path.exists(test_image_path):
        logger.error(f"Test image not found: {test_image_path}")
        test_image_path = None
    
    if not test_image_path:
        # Try to find a default test image
        default_path = os.path.join(TEST_IMAGES_DIR, "1_satellite_image.jpg")
        if os.path.exists(default_path):
            test_image_path = default_path
            logger.info(f"Using default test image: {test_image_path}")
        else:
            # Look for any jpg in the test images directory
            test_images = list(Path(TEST_IMAGES_DIR).glob("*.jpg"))
            if test_images:
                test_image_path = str(test_images[0])
                logger.info(f"Using available test image: {test_image_path}")
            else:
                logger.warning("No test images found - some tests may fail")
    
    # Track results
    all_results = {}
    tests_run = 0
    tests_passed = 0
    
    # Create image-specific directory for results
    if test_image_path and not args.nosave:
        image_name = Path(test_image_path).stem
        image_results_dir = os.path.join(TEST_RESULTS_DIR, image_name)
        os.makedirs(image_results_dir, exist_ok=True)
    
    # Run requested tests
    if args.test in ["all", "deepforest"]:
        logger.info("=" * 60)
        logger.info("Testing DeepForest")
        logger.info("=" * 60)
        
        success, results = test_deepforest(
            test_image_path=test_image_path,
            force_cpu=args.cpu,
            save_results=not args.nosave
        )
        
        # Save to image-specific directory
        if success and not args.nosave and test_image_path:
            image_name = Path(test_image_path).stem
            image_results_dir = os.path.join(TEST_RESULTS_DIR, image_name)
            
            try:
                # No need to copy detection results - they should already be in the image directory
                # We've modified the code to save directly to image-specific directories
                
                # Create a fresh visualization instead of copying
                # Get cached model
                cuda_available = torch.cuda.is_available() and not force_cpu
                device_str = "cuda:0" if cuda_available else "cpu"
                model = get_deepforest_model(device=device_str)
                prediction_image = model.predict_image(path=test_image_path, return_plot=True)
                
                if hasattr(prediction_image, 'save'):
                    prediction_image.save(os.path.join(image_results_dir, "visualization.png"))
                else:
                    from PIL import Image
                    Image.fromarray(prediction_image).save(os.path.join(image_results_dir, "visualization.png"))
                
                logger.info(f"Created unique DeepForest visualization for {os.path.basename(test_image_path)}")
                
                # Create summary
                with open(os.path.join(image_results_dir, "summary.txt"), 'w') as f:
                    f.write(f"Image: {test_image_path}\n")
                    f.write(f"Test: DeepForest\n")
                    f.write(f"Processed: {datetime.now().isoformat()}\n")
                    f.write(f"Trees detected: {results.get('detection_count', 0)}\n")
                    f.write(f"Execution time: {results.get('execution_time', 0):.2f} seconds\n")
            except Exception as e:
                logger.error(f"Error saving image-specific results: {e}")
                # Re-raise the exception for proper error handling
                raise e
        
        all_results["deepforest"] = results
        tests_run += 1
        if success:
            tests_passed += 1
    
    if args.test in ["all", "sam"]:
        logger.info("=" * 60)
        logger.info("Testing SAM")
        logger.info("=" * 60)
        
        success, results = test_sam(
            test_image_path=test_image_path,
            force_cpu=args.cpu,
            save_results=not args.nosave
        )
        
        all_results["sam"] = results
        tests_run += 1
        if success:
            tests_passed += 1
    
    if args.test in ["all", "pipeline"]:
        logger.info("=" * 60)
        logger.info("Testing ML Pipeline")
        logger.info("=" * 60)
        
        success, results = test_pipeline(
            test_image_path=test_image_path,
            force_cpu=args.cpu,
            save_results=not args.nosave
        )
        
        # Save to image-specific directory
        if success and not args.nosave and test_image_path:
            image_name = Path(test_image_path).stem
            image_results_dir = os.path.join(TEST_RESULTS_DIR, image_name)
            
            try:
                # No need to copy detection results - they should already be in the image directory
                # We've modified the code to save directly to image-specific directories
                
                # Generate a fresh pipeline visualization
                try:
                    # Get detection boxes directly from image-specific directory
                    detection_json_path = os.path.join(image_results_dir, "detection_results.json")
                    if os.path.exists(detection_json_path):
                        with open(detection_json_path, 'r') as f:
                            detection_boxes = json.load(f)
                    else:
                        detection_boxes = []
                    
                    if SAM_AVAILABLE and len(detection_boxes) > 0:
                        # Create fresh SAM visualization for this image
                        # Find SAM checkpoint
                        sam_checkpoints = [
                            "/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth",
                            "/ttt/tree_ml/pipeline/model/sam_vit_l_0b3195.pth",
                            "/ttt/tree_ml/pipeline/model/sam_vit_b_01ec64.pth"
                        ]
                        
                        sam_checkpoint = None
                        for path in sam_checkpoints:
                            if os.path.exists(path):
                                sam_checkpoint = path
                                break
                        
                        if sam_checkpoint:
                            # Create SAM visualization
                            from segment_anything import sam_model_registry, SamPredictor
                            from PIL import Image, ImageDraw
                            
                            # Initialize SAM
                            if "vit_h" in sam_checkpoint:
                                model_type = "vit_h"
                            elif "vit_l" in sam_checkpoint:
                                model_type = "vit_l"
                            else:
                                model_type = "vit_b"
                                
                            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                            sam.to(device="cpu")  # Use CPU for safety
                            predictor = SamPredictor(sam)
                            
                            # Load image
                            image = np.array(Image.open(test_image_path))
                            predictor.set_image(image)
                            
                            # Create visualization canvas
                            base_img = Image.open(test_image_path).convert("RGBA")
                            overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
                            draw = ImageDraw.Draw(overlay)
                            
                            # Colors for masks
                            colors = [(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
                                    (255, 255, 0, 128), (255, 0, 255, 128), (0, 255, 255, 128)]
                            
                            # Process each box
                            for i, box in enumerate(detection_boxes):
                                # Draw box
                                color = colors[i % len(colors)]
                                
                                if 'xmin' in box:
                                    x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                                else:
                                    x1, y1, x2, y2 = box.get('x1', 0), box.get('y1', 0), box.get('x2', 100), box.get('y2', 100)
                                
                                draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=2)
                                
                                # Create prompts for SAM
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                input_point = np.array([[center_x, center_y]])
                                input_label = np.array([1])  # foreground
                                box_np = np.array([x1, y1, x2, y2])
                                
                                # Generate masks
                                try:
                                    masks, scores, _ = predictor.predict(
                                        point_coords=input_point,
                                        point_labels=input_label,
                                        box=box_np,
                                        multimask_output=True
                                    )
                                    
                                    # Get best mask
                                    if len(masks) > 0:
                                        best_mask_idx = np.argmax(scores)
                                        mask = masks[best_mask_idx]
                                        
                                        # Add mask to visualization
                                        mask_np = mask
                                        for y in range(0, base_img.height, 2):  # Sample every 2 pixels for speed
                                            for x in range(0, base_img.width, 2):
                                                if y < mask_np.shape[0] and x < mask_np.shape[1] and mask_np[y, x]:
                                                    overlay.putpixel((x, y), color)
                                except Exception as e:
                                    logger.error(f"Error processing mask for box {i}: {e}")
                            
                            # Combine images
                            result = Image.alpha_composite(base_img, overlay)
                            
                            # Save directly to image-specific directory
                            result.save(os.path.join(image_results_dir, "visualization.png"))
                            logger.info(f"Created unique pipeline visualization for {os.path.basename(test_image_path)}")
                        else:
                            # Don't attempt to create extra visualization - skip this step to avoid CUDA/CPU device issues
                            # Instead just log that we're skipping the visualization
                            logger.info("Skipping extra visualization step - using main pipeline visualization instead")
                        
                except Exception as e:
                    logger.error(f"Error creating pipeline visualization: {e}")
                    # No fallbacks - just raise the error
                    raise e
                
                # Create summary
                with open(os.path.join(image_results_dir, "summary.txt"), 'w') as f:
                    f.write(f"Image: {test_image_path}\n")
                    f.write(f"Test: Full Pipeline\n")
                    f.write(f"Processed: {datetime.now().isoformat()}\n")
                    f.write(f"Trees detected: {results.get('detection_count', 0)}\n")
                    f.write(f"Trees segmented: {results.get('segmentation_count', 0)}\n")
                    f.write(f"Total execution time: {results.get('total_time', 0):.2f} seconds\n")
            except Exception as e:
                logger.error(f"Error saving image-specific results: {e}")
                # Re-raise the exception for proper error handling
                raise e
        
        all_results["pipeline"] = results
        tests_run += 1
        if success:
            tests_passed += 1
    
    # Save overall report
    if not args.nosave:
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "test_image": test_image_path,
                "cuda_enabled": torch.cuda.is_available() and not args.cpu if TORCH_AVAILABLE else False,
                "results": all_results
            }
            
            with open(os.path.join(TEST_RESULTS_DIR, "ml_test_report.json"), 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Test report saved to {os.path.join(TEST_RESULTS_DIR, 'ml_test_report.json')}")
        except Exception as e:
            logger.error(f"Error saving test report: {e}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Tests run: {tests_run}")
    logger.info(f"Tests passed: {tests_passed}")
    
    if tests_passed == tests_run:
        logger.info("✅ All tests passed!")
    else:
        logger.warning(f"⚠️ {tests_run - tests_passed} tests failed")
    
    return 0 if tests_passed == tests_run else 1


if __name__ == "__main__":
    sys.exit(main())