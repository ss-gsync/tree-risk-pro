#!/usr/bin/env python3
"""
Test script to verify DeepForest and SAM on a specific image
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from datetime import datetime

# Try importing DeepForest
try:
    import deepforest
    from deepforest import main
    DEEPFOREST_AVAILABLE = True
    print("DeepForest is available")
except ImportError:
    DEEPFOREST_AVAILABLE = False
    print("DeepForest is not available")

# Try importing SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("SAM is available")
except ImportError:
    SAM_AVAILABLE = False
    print("SAM is not available")
    
# Check CUDA availability with more robust error handling
try:
    import torch
    import os
    
    # Try to force CPU mode first for troubleshooting
    FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"
    
    if FORCE_CPU:
        print("Forcing CPU mode based on environment variable")
        CUDA_AVAILABLE = False
    else:
        # Try to initialize CUDA with better error handling
        try:
            # Check CUDA initialization
            CUDA_AVAILABLE = torch.cuda.is_available()
            if CUDA_AVAILABLE:
                CUDA_DEVICE = torch.cuda.get_device_name(0)
                CUDA_VERSION = torch.version.cuda
                print(f"CUDA is available: {CUDA_DEVICE} (CUDA {CUDA_VERSION})")
                
                # Test CUDA with a small tensor operation
                try:
                    # Create a small tensor on CPU first
                    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
                    print("CPU tensor created successfully")
                    
                    # Try to move it to CUDA
                    cuda_tensor = cpu_tensor.to("cuda:0")
                    print("Tensor moved to CUDA successfully")
                    
                    # Try a simple operation
                    result = cuda_tensor * 2.0
                    print("CUDA tensor operation successful")
                    
                    # Make sure we can access the result
                    result_cpu = result.to("cpu").numpy()
                    print(f"CUDA→CPU transfer successful: {result_cpu}")
                except Exception as e:
                    print(f"WARNING: CUDA tensor operations failed: {e}")
                    print("Falling back to CPU mode")
                    CUDA_AVAILABLE = False
            else:
                print("CUDA is not available (torch.cuda.is_available() returned False)")
                print(f"CUDA version: {torch.version.cuda}")
        except Exception as e:
            CUDA_AVAILABLE = False
            print(f"Error during CUDA initialization: {e}")
            print("Falling back to CPU mode")
except Exception as e:
    CUDA_AVAILABLE = False
    print(f"Error importing PyTorch or checking CUDA: {e}")
    print("Falling back to CPU mode")

# Image path
IMAGE_PATH = "/ttt/data/temp/ml_results_test/satellite_40.7791_-73.96375_16_1746715815.jpg"
OUTPUT_DIR = "/ttt/data/temp/test_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_deepforest():
    """Test DeepForest on the image"""
    if not DEEPFOREST_AVAILABLE:
        print("Skipping DeepForest test - not available")
        return []
    
    print(f"Testing DeepForest on {IMAGE_PATH}")
    
    # Initialize DeepForest with CUDA support if available
    model = deepforest.main.deepforest()
    model.use_release()
    
    # The issue is that the model is on GPU but input is on CPU
    # Let's use a simpler approach - use CPU for prediction
    print("Running DeepForest on CPU to avoid device mismatch issues")
    model.model.to("cpu")
    
    try:
        # Run prediction on CPU
        boxes = model.predict_image(path=IMAGE_PATH, return_plot=False)
        
        # Save boxes to JSON
        if not boxes.empty:
            # Convert to dict and handle polygon objects
            boxes_clean = []
            for _, row in boxes.iterrows():
                box_dict = {}
                for col in boxes.columns:
                    val = row[col]
                    # Handle non-serializable types
                    if 'Polygon' in str(type(val)):
                        box_dict[col] = str(val)
                    else:
                        box_dict[col] = val
                boxes_clean.append(box_dict)
                
            with open(os.path.join(OUTPUT_DIR, "deepforest_boxes.json"), 'w') as f:
                json.dump(boxes_clean, f, indent=2)
            
            print(f"DeepForest detected {len(boxes_clean)} trees")
            
            # Generate visualization
            # Save prediction image
            try:
                prediction_image = model.predict_image(path=IMAGE_PATH, return_plot=True)
                # Check if it's a numpy array and convert to PIL Image
                if isinstance(prediction_image, np.ndarray):
                    Image.fromarray(prediction_image).save(os.path.join(OUTPUT_DIR, "deepforest_prediction.png"))
                else:
                    prediction_image.save(os.path.join(OUTPUT_DIR, "deepforest_prediction.png"))
                print(f"Saved prediction visualization to {os.path.join(OUTPUT_DIR, 'deepforest_prediction.png')}")
            except Exception as e:
                print(f"Error saving prediction image: {e}")
            
            # If CUDA is available, verify it works with a simple test
            if CUDA_AVAILABLE:
                print("Testing CUDA capabilities...")
                try:
                    with torch.cuda.device(0):
                        # Create a dummy tensor and move it to GPU 
                        dummy = torch.tensor([1.0]).cuda()
                        print(f"Test tensor on device: {dummy.device}")
                    print("CUDA test successful")
                except Exception as e:
                    print(f"CUDA test failed: {e}")
                    
            return boxes_clean
        else:
            print("DeepForest didn't detect any trees")
            return []
    except Exception as e:
        print(f"Error during prediction: {e}")
        return []

def test_sam(boxes):
    """Test SAM on the image using DeepForest boxes"""
    if not SAM_AVAILABLE:
        print("Skipping SAM test - not available")
        return
    
    if not boxes:
        print("No boxes provided for SAM test")
        return
    
    print(f"Testing SAM on {IMAGE_PATH} with {len(boxes)} boxes")
    
    # Look for SAM checkpoint
    sam_checkpoint_paths = [
        "/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth",
        "/ttt/tree_ml/pipeline/model/sam_vit_l_0b3195.pth",
        "/ttt/tree_ml/pipeline/model/sam_vit_b_01ec64.pth"
    ]
    
    # Find first available SAM model
    sam_checkpoint = None
    for path in sam_checkpoint_paths:
        if os.path.exists(path):
            sam_checkpoint = path
            print(f"Found SAM model at {sam_checkpoint}")
            break
    
    if not sam_checkpoint:
        print("No SAM checkpoint found, using 'default' model")
        sam_type = "default"
    else:
        # Determine model type from path
        if "vit_h" in sam_checkpoint:
            sam_type = "vit_h"
        elif "vit_l" in sam_checkpoint:
            sam_type = "vit_l"
        elif "vit_b" in sam_checkpoint:
            sam_type = "vit_b"
        else:
            sam_type = "default"
    
    # Load SAM model
    print(f"Loading SAM model ({sam_type})")
    sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)
    
    # Use consistent device approach for SAM
    # For safety, use CPU by default just like with DeepForest
    device = "cpu"
    print(f"Using device for SAM: {device}")
    sam.to(device=device)
    print(f"SAM model moved to {device} successfully")
    
    # If we have CUDA, we'll do a simple verification test
    if CUDA_AVAILABLE:
        print("Testing CUDA capability for SAM...")
        try:
            with torch.cuda.device(0):
                # Simple tensor test
                test_tensor = torch.tensor([1.0, 2.0]).cuda()
                print(f"SAM CUDA test tensor on device: {test_tensor.device}")
            print("SAM CUDA test successful")
        except Exception as e:
            print(f"SAM CUDA test failed: {e}")
    
    # Initialize predictor
    predictor = SamPredictor(sam)
    
    # Load image
    image = np.array(Image.open(IMAGE_PATH))
    predictor.set_image(image)
    
    # Process each detection
    masks_output = []
    
    for i, box in enumerate(boxes):
        # Extract bbox - different key names depending on DeepForest version
        if 'xmin' in box:
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        else:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        
        # Get center point for point prompt
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Create point prompts
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])  # 1 for foreground
        
        # Create box prompt
        box_np = np.array([x1, y1, x2, y2])
        
        # Generate masks
        try:
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=box_np,
                multimask_output=True
            )
            
            # Get the best mask
            if len(masks) > 0:
                best_mask_idx = np.argmax(scores)
                mask = masks[best_mask_idx]
                score = float(scores[best_mask_idx])
                
                print(f"Tree {i+1}: SAM segmentation score: {score:.4f}")
                
                # Store mask info
                masks_output.append({
                    'tree_idx': i,
                    'mask_idx': int(best_mask_idx),
                    'score': score,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
                
                # Visualize this mask (convert boolean mask to image)
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img.save(os.path.join(OUTPUT_DIR, f"sam_mask_{i}.png"))
        except Exception as e:
            print(f"Error generating mask for tree {i}: {str(e)}")
    
    # Save masks to JSON
    with open(os.path.join(OUTPUT_DIR, "sam_masks.json"), 'w') as f:
        json.dump(masks_output, f, indent=2)
    
    # Generate a combined visualization with original image + masks
    if masks_output:
        try:
            # Start with original image
            base_img = Image.open(IMAGE_PATH).convert("RGBA")
            overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Draw boxes and load masks
            colors = [(255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128), 
                      (255, 255, 0, 128), (255, 0, 255, 128), (0, 255, 255, 128)]
            
            for i, mask_info in enumerate(masks_output):
                # Get color (cycle through colors)
                color = colors[i % len(colors)]
                
                # Draw box
                bbox = mask_info['bbox']
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color[:3], width=2)
                
                # Load and add mask
                try:
                    mask_path = os.path.join(OUTPUT_DIR, f"sam_mask_{mask_info['tree_idx']}.png")
                    if os.path.exists(mask_path):
                        mask_img = Image.open(mask_path).convert("L")
                        # For each pixel in the mask, draw with color if mask pixel is white
                        for y in range(base_img.height):
                            for x in range(base_img.width):
                                if x < mask_img.width and y < mask_img.height and mask_img.getpixel((x, y)) > 128:
                                    overlay.putpixel((x, y), color)
                except Exception as e:
                    print(f"Error adding mask {i}: {str(e)}")
            
            # Composite the overlay onto the base image
            result = Image.alpha_composite(base_img, overlay)
            result.save(os.path.join(OUTPUT_DIR, "combined_visualization.png"))
            print(f"Combined visualization saved to {os.path.join(OUTPUT_DIR, 'combined_visualization.png')}")
            
        except Exception as e:
            print(f"Error creating combined visualization: {str(e)}")

def main():
    print(f"Testing DeepForest and SAM on {IMAGE_PATH}")
    print(f"Output will be saved to {OUTPUT_DIR}")
    
    # Run DeepForest
    boxes = test_deepforest()
    
    # Run SAM with boxes from DeepForest
    test_sam(boxes)
    
    print("Testing complete!")

if __name__ == "__main__":
    main()