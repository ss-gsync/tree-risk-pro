#!/usr/bin/env python3
"""
Grounded SAM Tester - ML testing script for Tree ML project using Grounded SAM

This script provides a standardized interface for testing Grounded SAM, 
a combination of Grounding-DINO (for zero-shot detection) and SAM (for segmentation)
used in the Tree ML project for improved tree and infrastructure detection.

Features:
- GPU/CPU mode selection
- Zero-shot detection of trees and infrastructure
- High-quality segmentation with SAM
- Comprehensive reporting and logging

Usage examples:
  # Test models with default settings
  python grounded_sam_tester.py
  
  # Test with specific prompt
  python grounded_sam_tester.py --prompt "tree"
  
  # Test multiple objects
  python grounded_sam_tester.py --prompt "tree, power line, roof, solar panel"
  
  # Force CPU mode
  python grounded_sam_tester.py --cpu
  
  # Process all images
  python grounded_sam_tester.py --all-images
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
logger = logging.getLogger('grounded_sam_tester')

# Default directories
DATA_DIR = "/ttt/data/tests"
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "ml_test_images")
TEST_RESULTS_DIR = os.path.join(DATA_DIR, "ml_test_results")

# Ensure results directory exists
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# Import required libraries
import torch
TORCH_AVAILABLE = True
logger.info("Using PyTorch version: " + torch.__version__)

# Check for Grounding-DINO and SAM
try:
    # Add paths to ensure imports work
    sys.path.append("/ttt/tree_ml/pipeline/grounded-sam")
    sys.path.append("/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO")
    
    # Try to import transformers for GroundingDINO
    import transformers
    from transformers import pipeline
    
    GROUNDINGDINO_AVAILABLE = True
    logger.info("Grounding-DINO appears to be available")
except Exception as e:
    GROUNDINGDINO_AVAILABLE = False
    logger.warning(f"Grounding-DINO import failed: {e}")

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    logger.info("SAM appears to be available")
except Exception as e:
    SAM_AVAILABLE = False
    logger.warning(f"SAM import failed: {e}")

# Global model cache
_groundingdino_model = None
_sam_predictor = None

def initialize_groundingdino(device="cuda:0"):
    """Initialize and cache the Grounding-DINO model"""
    global _groundingdino_model
    
    if _groundingdino_model is not None:
        logger.info("Using cached Grounding-DINO model")
        return _groundingdino_model
    
    try:
        # Use the transformers pipeline for zero-shot object detection
        from transformers import pipeline
        
        # Try with the tiny model first which is faster
        model_id = "IDEA-Research/grounding-dino-tiny"
        
        # Alternative models if needed:
        alternatives = [
            "IDEA-Research/grounding-dino-base",
            "IDEA-Research/grounding-dino-b"
        ]
        
        # Initialize the detection pipeline
        detector = pipeline(
            "zero-shot-object-detection",
            model=model_id,
            device=device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        
        _groundingdino_model = detector
        logger.info(f"Grounding-DINO model initialized using {model_id} on {device}")
        
        return detector
    except Exception as e:
        logger.error(f"Error initializing Grounding-DINO: {e}")
        raise

def initialize_sam(device="cuda:0"):
    """Initialize and cache the SAM predictor"""
    global _sam_predictor
    
    if _sam_predictor is not None:
        logger.info("Using cached SAM predictor")
        return _sam_predictor
    
    try:
        # Find SAM checkpoint
        sam_checkpoint = "/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth"
        
        # Check if checkpoint exists
        if not os.path.exists(sam_checkpoint):
            logger.warning(f"SAM checkpoint not found at {sam_checkpoint}")
            logger.info("Checking alternative paths...")
            
            # Try alternative paths
            alternatives = [
                "/ttt/tree_ml/pipeline/grounded-sam/weights/sam_vit_h_4b8939.pth",
                "/ttt/tree_ml/pipeline/grounded-sam/segment_anything/weights/sam_vit_h_4b8939.pth",
                "/ttt/tree_ml/pipeline/model/sam_vit_l_0b3195.pth",
                "/ttt/tree_ml/pipeline/model/sam_vit_b_01ec64.pth"
            ]
            for path in alternatives:
                if os.path.exists(path):
                    sam_checkpoint = path
                    break
        
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found")
        
        # Determine model type
        if "vit_h" in sam_checkpoint:
            model_type = "vit_h"
        elif "vit_l" in sam_checkpoint:
            model_type = "vit_l"
        elif "vit_b" in sam_checkpoint:
            model_type = "vit_b"
        else:
            model_type = "vit_h"  # Default
        
        # Initialize SAM
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        # Initialize predictor
        predictor = SamPredictor(sam)
        
        _sam_predictor = predictor
        logger.info(f"SAM model ({model_type}) initialized on {device}")
        
        return predictor
    except Exception as e:
        logger.error(f"Error initializing SAM: {e}")
        raise

def get_groundingdino_detections(image_path, text_prompt, box_threshold=0.35, text_threshold=0.25, device="cuda:0"):
    """
    Get object detections using Grounding-DINO
    
    Args:
        image_path: Path to the input image
        text_prompt: Text prompt for the objects to detect (comma-separated for multiple objects)
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text prompts
        device: Device to run the model on
        
    Returns:
        boxes: List of bounding boxes (normalized coordinates [x0, y0, x1, y1])
        phrases: List of detected phrases
        scores: List of confidence scores
    """
    try:
        # Initialize model
        detector = initialize_groundingdino(device=device)
        
        # Load image
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        
        # Parse the comma-separated text prompts
        labels = text_prompt.split(',')
        labels = [label.strip() for label in labels if label.strip()]
        
        # Make sure each label ends with a period as required by the model
        labels = [label if label.endswith(".") else label + "." for label in labels]
        
        # Run detection with the given threshold
        results = detector(
            image, 
            candidate_labels=labels,
            threshold=box_threshold
        )
        
        # Extract results
        boxes = []
        phrases = []
        scores = []
        
        # Process detection results
        if results:
            # Get image dimensions for normalization
            width, height = image.size
            
            for result in results:
                # Extract box (normalized coordinates)
                box = result['box']
                x0, y0, x1, y1 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                
                # Normalize coordinates to 0-1 range
                x0, y0, x1, y1 = x0/width, y0/height, x1/width, y1/height
                
                # Store results
                boxes.append(np.array([x0, y0, x1, y1]))
                phrases.append(result['label'])
                scores.append(result['score'])
        
        # Convert to numpy arrays for consistency
        boxes = np.array(boxes) if boxes else np.array([])
        scores = np.array(scores) if scores else np.array([])
        
        # Log results
        logger.info(f"Grounding-DINO detected {len(boxes)} objects")
        for i, (box, phrase, score) in enumerate(zip(boxes, phrases, scores)):
            logger.info(f"  {i+1}. {phrase}: {score:.4f} at {box}")
        
        return boxes, phrases, scores
    except Exception as e:
        logger.error(f"Error running Grounding-DINO detection: {e}")
        return [], [], []

def get_sam_masks(predictor, image_path, boxes):
    """
    Get segmentation masks using SAM
    
    Args:
        predictor: SAM predictor instance
        image_path: Path to the input image
        boxes: Bounding boxes from Grounding-DINO (normalized coordinates)
        
    Returns:
        masks: List of segmentation masks
        scores: List of confidence scores
    """
    try:
        # Load image
        from PIL import Image
        image = np.array(Image.open(image_path))
        
        # Set image in predictor
        predictor.set_image(image)
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Process boxes
        masks_output = []
        scores_output = []
        
        for i, box in enumerate(boxes):
            # Convert normalized coordinates to pixel coordinates
            x0, y0, x1, y1 = box
            x0 = int(x0 * width)
            y0 = int(y0 * height)
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            
            # Transform box for SAM
            sam_box = np.array([x0, y0, x1, y1])
            
            # Get center point
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            
            # Generate masks
            masks, scores, _ = predictor.predict(
                box=sam_box,
                point_coords=np.array([[center_x, center_y]]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            
            # Get best mask
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            best_score = scores[best_mask_idx]
            
            masks_output.append(best_mask)
            scores_output.append(best_score)
            
            logger.info(f"  Mask {i+1}: SAM score {best_score:.4f}")
        
        return masks_output, scores_output
    except Exception as e:
        logger.error(f"Error generating SAM masks: {e}")
        return [], []

def create_visualization(image_path, boxes, phrases, scores, masks=None, mask_scores=None, output_path=None):
    """
    Create a visualization of the detection and segmentation results
    
    Args:
        image_path: Path to the input image
        boxes: Bounding boxes (normalized coordinates)
        phrases: Detected phrases
        scores: Confidence scores for boxes
        masks: Segmentation masks (optional)
        mask_scores: Confidence scores for masks (optional)
        output_path: Path to save the visualization
        
    Returns:
        Success flag
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Load image
        image = Image.open(image_path)
        
        # Get image dimensions
        width, height = image.size
        
        # Create transparent overlay for masks
        if masks is not None:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw_overlay = ImageDraw.Draw(overlay)
        
        # Create drawing context
        draw = ImageDraw.Draw(image)
        
        # Colors for different tree risk categories
        class_colors = {
            "healthy tree": (22, 163, 74),      # Green
            "hazardous tree": (139, 92, 246),   # Purple
            "dead tree": (107, 114, 128),       # Gray
            "low canopy tree": (14, 165, 233),  # Light Blue
            "pest disease tree": (132, 204, 22), # Lime Green
            "flood prone tree": (8, 145, 178),  # Teal
            "utility conflict tree": (59, 130, 246), # Blue
            "structural hazard tree": (13, 148, 136), # Teal Green
            "fire risk tree": (79, 70, 229),    # Indigo
        }
        
        # Default color
        default_color = (200, 200, 200)
        
        # Process each detection
        for i, (box, phrase, score) in enumerate(zip(boxes, phrases, scores)):
            # Convert normalized coordinates to pixel coordinates
            x0, y0, x1, y1 = box
            x0 = int(x0 * width)
            y0 = int(y0 * height)
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            
            # Determine color based on class
            color = default_color
            for class_name, class_color in class_colors.items():
                if class_name in phrase.lower():
                    color = class_color
                    break
            
            # Draw box
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            
            # Draw label
            label = f"{phrase}: {score:.2f}"
            draw.text((x0, y0-15), label, fill=color)
            
            # Draw mask if available
            if masks is not None and i < len(masks):
                mask = masks[i]
                mask_alpha = 128  # Semi-transparent
                
                # Create colored mask
                mask_color = (*color, mask_alpha)
                
                # Draw mask
                for y in range(0, height, 2):  # Sample every 2 pixels for speed
                    for x in range(0, width, 2):
                        if y < mask.shape[0] and x < mask.shape[1] and mask[y, x]:
                            overlay.putpixel((x, y), mask_color)
        
        # Combine image with overlay if masks are available
        if masks is not None:
            result = Image.alpha_composite(image.convert("RGBA"), overlay)
            result = result.convert("RGB")
        else:
            result = image
        
        # Save the visualization
        if output_path:
            result.save(output_path)
            logger.info(f"Visualization saved to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        return False

def test_grounded_sam(
    test_image_path=None, 
    text_prompt="tree, power line, roof, solar panel, railroad, building", 
    box_threshold=0.35, 
    text_threshold=0.25,
    force_cpu=False, 
    save_results=True
):
    """
    Test the Grounded SAM pipeline on an image
    
    Args:
        test_image_path: Path to the test image
        text_prompt: Text prompt for object detection
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text prompts
        force_cpu: Force CPU usage even if CUDA is available
        save_results: Save results to disk
        
    Returns:
        success: Success flag
        results: Dictionary with results
    """
    if not GROUNDINGDINO_AVAILABLE or not SAM_AVAILABLE:
        logger.error("Grounding-DINO or SAM not available - test skipped")
        return False, {"error": "Required models not available"}
    
    # Check CUDA
    cuda_available = torch.cuda.is_available() and not force_cpu
    device = "cuda:0" if cuda_available else "cpu"
    logger.info(f"Testing Grounded SAM on device: {device}")
    
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
                return False, {"error": "No test images found"}
    
    logger.info(f"Testing Grounded SAM on image: {test_image_path}")
    logger.info(f"Text prompt: {text_prompt}")
    
    # Get image name for unique results
    image_name = os.path.basename(test_image_path)
    base_name = os.path.splitext(image_name)[0]
    
    # Setup image-specific results directory
    image_results_dir = os.path.join(TEST_RESULTS_DIR, base_name)
    os.makedirs(image_results_dir, exist_ok=True)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Step 1: Get detections using Grounding-DINO
        logger.info("Running Grounding-DINO detection...")
        dino_start_time = time.time()
        boxes, phrases, scores = get_groundingdino_detections(
            image_path=test_image_path,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
        )
        dino_time = time.time() - dino_start_time
        
        detection_count = len(boxes)
        logger.info(f"Detected {detection_count} objects in {dino_time:.2f} seconds")
        
        # Exit early if no detections
        if detection_count == 0:
            logger.warning("No objects detected by Grounding-DINO")
            if save_results:
                report = {
                    "timestamp": datetime.now().isoformat(),
                    "image": os.path.basename(test_image_path),
                    "prompt": text_prompt,
                    "execution_time": time.time() - start_time,
                    "detection_count": 0,
                    "segmentation_count": 0,
                    "device": device,
                    "success": False
                }
                with open(os.path.join(image_results_dir, "grounded_sam_report.json"), 'w') as f:
                    json.dump(report, f, indent=2)
            
            return False, {
                "success": False,
                "execution_time": time.time() - start_time,
                "detection_count": 0,
                "segmentation_count": 0,
                "device": device
            }
        
        # Step 2: Get masks using SAM
        logger.info("Running SAM segmentation...")
        sam_start_time = time.time()
        predictor = initialize_sam(device=device)
        masks, mask_scores = get_sam_masks(predictor, test_image_path, boxes)
        sam_time = time.time() - sam_start_time
        
        segmentation_count = len(masks)
        logger.info(f"Generated {segmentation_count} masks in {sam_time:.2f} seconds")
        
        # Step 3: Create visualization
        if save_results:
            # Save detection results
            detection_results = []
            for i, (box, phrase, score) in enumerate(zip(boxes, phrases, scores)):
                mask_score = mask_scores[i] if i < len(mask_scores) else None
                detection_results.append({
                    "box": box.tolist(),
                    "phrase": phrase,
                    "score": float(score),
                    "mask_score": float(mask_score) if mask_score is not None else None
                })
            
            with open(os.path.join(image_results_dir, "grounded_sam_detections.json"), 'w') as f:
                json.dump(detection_results, f, indent=2)
            
            # Save visualization
            output_path = os.path.join(image_results_dir, "grounded_sam_visualization.png")
            create_visualization(
                image_path=test_image_path,
                boxes=boxes,
                phrases=phrases,
                scores=scores,
                masks=masks,
                mask_scores=mask_scores,
                output_path=output_path
            )
            
            # Save masks
            for i, mask in enumerate(masks):
                mask_path = os.path.join(image_results_dir, f"grounded_sam_mask_{i}.png")
                from PIL import Image
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img.save(mask_path)
            
            # Save report
            total_time = time.time() - start_time
            report = {
                "timestamp": datetime.now().isoformat(),
                "image": os.path.basename(test_image_path),
                "prompt": text_prompt,
                "execution_time": total_time,
                "detection_time": dino_time,
                "segmentation_time": sam_time,
                "detection_count": detection_count,
                "segmentation_count": segmentation_count,
                "device": device,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "success": True
            }
            
            with open(os.path.join(image_results_dir, "grounded_sam_report.json"), 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save summary
            with open(os.path.join(image_results_dir, "grounded_sam_summary.txt"), 'w') as f:
                f.write(f"Image: {test_image_path}\n")
                f.write(f"Prompt: {text_prompt}\n")
                f.write(f"Processed: {datetime.now().isoformat()}\n")
                f.write(f"Objects detected: {detection_count}\n")
                f.write(f"Detection time: {dino_time:.2f} seconds\n")
                f.write(f"Segmentation time: {sam_time:.2f} seconds\n")
                f.write(f"Total execution time: {total_time:.2f} seconds\n")
                f.write("\nDetections:\n")
                for i, (phrase, score) in enumerate(zip(phrases, scores)):
                    f.write(f"  {i+1}. {phrase}: {score:.4f}\n")
            
            logger.info(f"Results saved to {image_results_dir}")
        
        # Return results
        total_time = time.time() - start_time
        success = detection_count > 0 and segmentation_count > 0
        results = {
            "success": success,
            "execution_time": total_time,
            "detection_time": dino_time,
            "segmentation_time": sam_time,
            "detection_count": detection_count,
            "segmentation_count": segmentation_count,
            "device": device,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Test completed in {total_time:.2f} seconds")
        return success, results
    
    except Exception as e:
        logger.error(f"Error testing Grounded SAM: {e}")
        return False, {"error": str(e)}

def process_all_images(text_prompt, box_threshold=0.35, text_threshold=0.25, force_cpu=False, save_results=True):
    """
    Process all images in the test directory
    
    Args:
        text_prompt: Text prompt for object detection
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text prompts
        force_cpu: Force CPU usage even if CUDA is available
        save_results: Save results to disk
        
    Returns:
        success: Success flag
        results: Dictionary with overall results
    """
    # Get all test images
    test_images = list(Path(TEST_IMAGES_DIR).glob("*.jpg")) + list(Path(TEST_IMAGES_DIR).glob("*.png"))
    if not test_images:
        logger.error("No test images found in the test directory")
        return False, {}
    
    logger.info(f"Found {len(test_images)} test images to process")
    
    # Track results for all images
    all_image_results = {}
    success_count = 0
    total_objects = 0
    
    # Process each image
    for i, image_path in enumerate(test_images):
        image_name = image_path.stem
        logger.info(f"Processing image {i+1}/{len(test_images)}: {image_path.name}")
        
        # Run test on this image
        success, results = test_grounded_sam(
            test_image_path=str(image_path),
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            force_cpu=force_cpu,
            save_results=save_results
        )
        
        # Track results
        all_image_results[image_name] = {
            "success": success,
            "results": results
        }
        
        if success:
            success_count += 1
            total_objects += results.get("detection_count", 0)
    
    # Save batch report
    if save_results:
        batch_report = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(test_images),
            "successful_images": success_count,
            "total_objects": total_objects,
            "prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "cuda_enabled": torch.cuda.is_available() and not force_cpu,
            "image_results": {name: result["results"] for name, result in all_image_results.items()}
        }
        
        with open(os.path.join(TEST_RESULTS_DIR, "grounded_sam_batch_report.json"), 'w') as f:
            json.dump(batch_report, f, indent=2)
        
        logger.info(f"Batch report saved to {os.path.join(TEST_RESULTS_DIR, 'grounded_sam_batch_report.json')}")
    
    # Return overall success and results
    return success_count > 0, {
        "total_images": len(test_images),
        "successful_images": success_count,
        "total_objects": total_objects,
        "image_results": all_image_results
    }

def main():
    """Main function to run Grounded SAM tests"""
    parser = argparse.ArgumentParser(description="Grounded SAM Tester")
    parser.add_argument(
        "--prompt",
        default="healthy tree, hazardous tree, dead tree, low canopy tree, pest disease tree, flood prone tree, utility conflict tree, structural hazard tree, fire risk tree",
        help="Text prompt for tree risk assessment detection"
    )
    parser.add_argument(
        "--image",
        help="Path to test image (optional)"
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for bounding boxes (default: 0.35)"
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for text prompts (default: 0.25)"
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
    
    args = parser.parse_args()
    
    # Banner
    logger.info("=" * 60)
    logger.info("Grounded SAM Tester")
    logger.info("=" * 60)
    
    # CUDA info
    cuda_available = torch.cuda.is_available() and not args.cpu
    if cuda_available:
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        if args.cpu:
            logger.info("CUDA disabled (--cpu flag used)")
        else:
            logger.warning("CUDA not available, using CPU")
    
    # Process all images if requested
    if args.all_images:
        logger.info("Processing all test images")
        
        success, batch_results = process_all_images(
            text_prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            force_cpu=args.cpu,
            save_results=not args.nosave
        )
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Batch Test Summary")
        logger.info("=" * 60)
        logger.info(f"Total images: {batch_results.get('total_images', 0)}")
        logger.info(f"Successfully processed: {batch_results.get('successful_images', 0)}")
        logger.info(f"Total objects detected: {batch_results.get('total_objects', 0)}")
        
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
                logger.warning("No test images found - tests may fail")
    
    # Run test on single image
    success, results = test_grounded_sam(
        test_image_path=test_image_path,
        text_prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        force_cpu=args.cpu,
        save_results=not args.nosave
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    logger.info(f"Image: {test_image_path}")
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Objects detected: {results.get('detection_count', 0)}")
    logger.info(f"Masks generated: {results.get('segmentation_count', 0)}")
    logger.info(f"Execution time: {results.get('execution_time', 0):.2f} seconds")
    
    if success:
        logger.info("✅ Test successful!")
    else:
        logger.warning("⚠️ Test failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())