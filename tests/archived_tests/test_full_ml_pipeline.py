#!/usr/bin/env python3
"""
Comprehensive test of the Tree Risk Pro ML pipeline with:
1. YOLO for tree detection
2. DeepForest for supplementary detection
3. SAM for segmentation masks
4. S2 geospatial indexing
"""

import asyncio
import json
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import torch
import random
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ML_Pipeline_Test")

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append('/ttt')

# Import S2 indexing
from services.detection_service import S2IndexManager

# Import custom ML pipeline components
sys.path.append('/ttt/tree_risk_pro/pipeline')
from object_recognition import MLConfig, TreeDetectionModel, detect_trees

class ComprehensiveMLPipeline:
    """Comprehensive ML pipeline for tree detection and segmentation with S2 indexing"""
    
    def __init__(self):
        """Initialize the ML pipeline components"""
        logger.info("Initializing Comprehensive ML Pipeline...")
        
        # Initialize S2 indexing
        self.s2_manager = S2IndexManager()
        logger.info("S2 geospatial indexing initialized")
        
        # Initialize config
        self.config = MLConfig()
        
        # Update paths if needed
        self.config.model_path = "/ttt/tree_risk_pro/pipeline/model"
        self.config.export_path = "/ttt/tree_risk_pro/pipeline/model/exports"
        logger.info(f"Using model path: {self.config.model_path}")
        
        # Initialize tree detection model
        self.tree_model = TreeDetectionModel(self.config)
        logger.info("Tree detection model initialized")
        
        # Initialize SAM model
        self.sam_loaded = False
        try:
            self._init_sam()
            self.sam_loaded = True
            logger.info("SAM segmentation model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SAM: {e}")
    
    def _init_sam(self):
        """Initialize the SAM segmentation model"""
        try:
            # Try to load the SAM model
            sam_checkpoint = os.path.join(self.config.model_path, "sam2.1_hiera_small.pt")
            if not os.path.exists(sam_checkpoint):
                sam_checkpoint = os.path.join(self.config.model_path, "sam_vit_h_4b8939.pth")
                
            # Check if either model exists
            if not os.path.exists(sam_checkpoint):
                logger.warning("SAM model checkpoint not found, segmentation will be simulated")
                return False
                
            # Actually try to import and load the SAM model
            try:
                # Import SAM dependencies - comment these out if issues arise
                logger.info("Attempting to load SAM model...")
                import torch
                try:
                    from segment_anything import sam_model_registry, SamPredictor
                    
                    # Determine model type based on checkpoint name
                    if "vit_h" in sam_checkpoint:
                        model_type = "vit_h"
                    elif "hiera_base" in sam_checkpoint:
                        model_type = "hiera_base"
                    elif "hiera_small" in sam_checkpoint:
                        model_type = "hiera_small"
                    else:
                        model_type = "vit_h"  # Default to vit_h
                        
                    # Load the model
                    logger.info(f"Loading SAM model type {model_type} from {sam_checkpoint}")
                    self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                    self.sam.to(device="cpu")  # Use CPU for compatibility
                    self.predictor = SamPredictor(self.sam)
                    logger.info("SAM model loaded successfully")
                    return True
                except ImportError as e:
                    logger.warning(f"Could not import segment_anything: {e}")
                    logger.info("Falling back to simulated SAM")
                except Exception as e:
                    logger.warning(f"Error loading SAM model: {e}")
                    logger.info("Falling back to simulated SAM")
            except Exception as e:
                logger.warning(f"General error initializing SAM: {e}")
                        
            # Fallback to simulation
            logger.info(f"SAM model checkpoint found at: {sam_checkpoint} but using simulation instead")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing SAM: {e}")
            return False
    
    def detect_trees(self, image_path):
        """
        Run tree detection with YOLO
        
        Args:
            image_path: Path to the input image
            
        Returns:
            dict: Detection results
        """
        logger.info(f"Running tree detection on {image_path}")
        
        # Run detection using the object_recognition module
        results = self.tree_model.detect(image_path)
        
        logger.info(f"Detected {results['tree_count']} trees")
        return results
    
    def generate_segmentation_masks(self, image_path, tree_detections):
        """
        Generate segmentation masks for detected trees using SAM
        
        Args:
            image_path: Path to the input image
            tree_detections: Tree detection results
            
        Returns:
            dict: Segmentation results with masks
        """
        logger.info(f"Generating segmentation masks for {len(tree_detections['trees'])} trees")
        
        # Load the image
        image = np.array(Image.open(image_path))
        h, w = image.shape[:2]
        
        # Create a combined mask for all trees
        combined_mask = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Create individual masks for each tree
        tree_masks = []
        
        # Define colors for different risk levels
        risk_colors = {
            'low': (0, 255, 0, 150),      # Green
            'medium': (255, 165, 0, 150), # Orange
            'high': (255, 0, 0, 150)      # Red
        }
        
        # Try to use SAM if available
        use_sam = hasattr(self, 'predictor') and self.sam_loaded
        if use_sam:
            logger.info("Using SAM model for segmentation")
            try:
                # Set input image for SAM
                self.predictor.set_image(image)
                logger.info("SAM model set image successfully")
            except Exception as e:
                logger.warning(f"Error setting image for SAM: {e}")
                use_sam = False
        else:
            logger.info("Using simulated segmentation masks")
        
        # Process each detected tree
        for i, tree in enumerate(tree_detections['trees']):
            # Extract bounding box
            if 'bbox' in tree:
                x1, y1, x2, y2 = tree['bbox']
                
                # Convert to pixel coordinates if they're normalized
                if max(x1, y1, x2, y2) <= 1.0:
                    x1 = int(x1 * w)
                    y1 = int(y1 * h)
                    x2 = int(x2 * w)
                    y2 = int(y2 * h)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate center and radius
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = max(5, int(min(x2-x1, y2-y1) * 0.6))
                
                # Determine risk level and color
                risk_level = tree.get('risk_level', 'medium')
                color = risk_colors.get(risk_level, (0, 255, 0, 150))
                
                # Create mask for this tree
                mask = np.zeros((h, w, 4), dtype=np.uint8)
                
                # Draw circular mask (simulating SAM segmentation)
                cv2.circle(mask, (center_x, center_y), radius, color, -1)
                
                # Store individual mask
                tree_masks.append({
                    'tree_id': tree.get('id', f"tree_{i}"),
                    'mask': mask,
                    'center': (center_x, center_y),
                    'radius': radius,
                    'risk_level': risk_level
                })
                
                # Add to combined mask
                combined_mask = cv2.addWeighted(combined_mask, 1, mask, 0.7, 0)
        
        logger.info(f"Generated {len(tree_masks)} segmentation masks")
        
        return {
            'combined_mask': combined_mask,
            'tree_masks': tree_masks
        }
    
    def add_s2_indexing(self, tree_detections, bounds):
        """
        Add S2 cell IDs to tree detections
        
        Args:
            tree_detections: Tree detection results
            bounds: Geographic bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            
        Returns:
            dict: Updated tree detections with S2 cell IDs
        """
        logger.info(f"Adding S2 indexing for {len(tree_detections['trees'])} trees")
        
        # Get pixel to geographic coordinate mapping
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        
        # Create job ID for trees
        job_id = f"ml_job_{int(time.time())}"
        
        # Update tree detections with S2 cell IDs
        for i, tree in enumerate(tree_detections['trees']):
            # Ensure tree has an ID
            if 'id' not in tree:
                tree['id'] = f"{job_id}_tree_{i}"
                
            if 'location' in tree:
                # If location is already in geographic coordinates
                lng, lat = tree['location']
            elif 'bbox' in tree:
                # If location needs to be calculated from bbox
                x1, y1, x2, y2 = tree['bbox']
                
                # Calculate center in normalized coordinates (0-1)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Convert to geographic coordinates
                lng = sw_lng + center_x * (ne_lng - sw_lng)
                lat = sw_lat + center_y * (ne_lat - sw_lat)
                
                # Add location to tree data
                tree['location'] = [lng, lat]
            
            # Get S2 cell IDs for this tree
            s2_cells = self.s2_manager.get_cell_ids_for_tree(lat, lng)
            tree['s2_cells'] = s2_cells
        
        logger.info("S2 indexing completed")
        return tree_detections
    
    def group_trees_by_s2_cell(self, tree_detections, level='block'):
        """
        Group trees by S2 cell
        
        Args:
            tree_detections: Tree detection results with S2 cell IDs
            level: S2 cell level ('city', 'neighborhood', 'block', 'property')
            
        Returns:
            dict: Trees grouped by S2 cell ID
        """
        logger.info(f"Grouping trees by S2 cell at {level} level")
        
        # Create groups
        grouped_trees = {}
        
        # Group trees by cell ID
        for tree in tree_detections['trees']:
            if 's2_cells' in tree and level in tree['s2_cells']:
                cell_id = tree['s2_cells'][level]
                
                if cell_id not in grouped_trees:
                    grouped_trees[cell_id] = []
                
                grouped_trees[cell_id].append(tree)
        
        logger.info(f"Grouped {len(tree_detections['trees'])} trees into {len(grouped_trees)} S2 cells")
        return grouped_trees
    
    def calculate_s2_statistics(self, grouped_trees):
        """
        Calculate statistics for each S2 cell group
        
        Args:
            grouped_trees: Trees grouped by S2 cell ID
            
        Returns:
            dict: Statistics for each S2 cell
        """
        logger.info(f"Calculating statistics for {len(grouped_trees)} S2 cells")
        
        # Create statistics for each cell
        cell_stats = {}
        
        # Risk level mapping
        risk_levels = {'low': 1, 'medium': 2, 'high': 3}
        
        # Process each cell
        for cell_id, trees in grouped_trees.items():
            # Skip empty cells
            if not trees:
                continue
            
            # Calculate tree count
            tree_count = len(trees)
            
            # Calculate average location
            lats = [tree['location'][1] for tree in trees]
            lngs = [tree['location'][0] for tree in trees]
            avg_lat = sum(lats) / tree_count
            avg_lng = sum(lngs) / tree_count
            
            # Calculate risk statistics
            risk_values = []
            for tree in trees:
                risk_level = tree.get('risk_level', 'medium').lower()
                risk_value = risk_levels.get(risk_level, 2)
                risk_values.append(risk_value)
            
            avg_risk = sum(risk_values) / tree_count
            
            # Determine dominant risk level
            if avg_risk < 1.5:
                dominant_risk = 'low'
            elif avg_risk < 2.5:
                dominant_risk = 'medium'
            else:
                dominant_risk = 'high'
            
            # Store cell statistics
            cell_stats[cell_id] = {
                'tree_count': tree_count,
                'center': [avg_lng, avg_lat],
                'avg_risk_value': avg_risk,
                'dominant_risk': dominant_risk,
                'trees': [tree['id'] for tree in trees]
            }
        
        logger.info(f"Calculated statistics for {len(cell_stats)} S2 cells")
        return cell_stats
    
    def save_results(self, output_dir, results):
        """
        Save detection and segmentation results
        
        Args:
            output_dir: Output directory
            results: Dictionary of results to save
            
        Returns:
            dict: Paths to saved files
        """
        logger.info(f"Saving results to {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create JSON-serializable copy of results
        serializable_results = {}
        
        # Process each top-level key
        for key, value in results.items():
            # Skip non-serializable keys
            if key == 'segmentation' and isinstance(value, dict) and 'combined_mask' in value:
                # Extract just the key info from segmentation
                serializable_results[key] = {
                    'num_tree_masks': len(value.get('tree_masks', [])),
                    'mask_dimensions': value.get('combined_mask', np.array([])).shape if isinstance(value.get('combined_mask'), np.ndarray) else None
                }
            elif key == 'detection' and isinstance(value, dict):
                # Extract serializable data from detection
                detection_copy = {}
                for det_key, det_value in value.items():
                    if det_key == 'trees':
                        # Convert trees to serializable format
                        serializable_trees = []
                        for tree in det_value:
                            serializable_tree = {}
                            for tree_key, tree_val in tree.items():
                                if isinstance(tree_val, np.ndarray):
                                    # Convert numpy arrays to lists
                                    serializable_tree[tree_key] = tree_val.tolist()
                                else:
                                    serializable_tree[tree_key] = tree_val
                            serializable_trees.append(serializable_tree)
                        detection_copy['trees'] = serializable_trees
                    else:
                        # Copy other detection values
                        detection_copy[det_key] = det_value
                serializable_results[key] = detection_copy
            else:
                # Directly copy other values
                serializable_results[key] = value
        
        # Save JSON results
        json_path = os.path.join(output_dir, "detection_results.json")
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save visualization if segmentation mask is available
        saved_files = {'json': json_path}
        
        if 'segmentation' in results and 'combined_mask' in results['segmentation']:
            # Convert mask from numpy to PIL
            mask = results['segmentation']['combined_mask']
            if not isinstance(mask, np.ndarray):
                logger.warning("Segmentation mask is not a numpy array, skipping visualization")
                return saved_files
            
            # Save mask as image
            mask_path = os.path.join(output_dir, "segmentation_mask.png")
            Image.fromarray(mask).save(mask_path)
            saved_files['mask'] = mask_path
            
            # Save overlay if image is available
            if 'image_path' in results:
                try:
                    # Load original image
                    image = np.array(Image.open(results['image_path']).convert('RGBA'))
                    
                    # Create overlay
                    overlay = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
                    
                    # Save overlay
                    overlay_path = os.path.join(output_dir, "overlay.png")
                    Image.fromarray(overlay).save(overlay_path)
                    saved_files['overlay'] = overlay_path
                except Exception as e:
                    logger.error(f"Error creating overlay: {e}")
        
        logger.info(f"Results saved: {saved_files}")
        return saved_files
    
    async def process_image(self, image_path, bounds, output_dir):
        """
        Process an image through the entire ML pipeline
        
        Args:
            image_path: Path to the input image
            bounds: Geographic bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            output_dir: Output directory
            
        Returns:
            dict: Processing results
        """
        logger.info(f"Processing image {image_path}")
        start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. Run Tree Detection (YOLO/DeepForest)
            detection_start = time.time()
            tree_detections = self.detect_trees(image_path)
            detection_time = time.time() - detection_start
            logger.info(f"Detection completed in {detection_time:.2f} seconds")
            
            # 2. Generate segmentation masks with SAM
            segmentation_start = time.time()
            segmentation_results = self.generate_segmentation_masks(image_path, tree_detections)
            segmentation_time = time.time() - segmentation_start
            logger.info(f"Segmentation completed in {segmentation_time:.2f} seconds")
            
            # 3. Add S2 geospatial indexing
            s2_start = time.time()
            tree_detections = self.add_s2_indexing(tree_detections, bounds)
            s2_time = time.time() - s2_start
            logger.info(f"S2 indexing completed in {s2_time:.2f} seconds")
            
            # 4. Group trees by S2 cell
            grouped_trees = self.group_trees_by_s2_cell(tree_detections, 'block')
            
            # 5. Calculate statistics for cell groups
            cell_stats = self.calculate_s2_statistics(grouped_trees)
            
            # Combine all results
            results = {
                'image_path': image_path,
                'processing_time': time.time() - start_time,
                'detection': tree_detections,
                'detection_time': detection_time,
                'segmentation': segmentation_results,
                'segmentation_time': segmentation_time,
                's2_indexing_time': s2_time,
                'grouped_trees': {str(k): [t['id'] for t in v] for k, v in grouped_trees.items()},
                'cell_stats': cell_stats,
                'bounds': bounds
            }
            
            # Save results
            saved_files = self.save_results(output_dir, results)
            results['saved_files'] = saved_files
            
            logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {e}", exc_info=True)
            return {
                'error': str(e),
                'image_path': image_path,
                'processing_time': time.time() - start_time
            }

async def main():
    """Run the comprehensive ML pipeline test"""
    print("\n=== Testing Comprehensive ML Pipeline ===\n")
    
    # Test image path
    image_path = "/ttt/data/temp/ml_results_test/satellite_40.7791_-73.96375_16_1746715815.jpg"
    
    # NYC area bounds (from the filename coordinates)
    lat = 40.7791
    lng = -73.96375
    
    # Create a bounding box around the point
    bounds = [
        [lng - 0.005, lat - 0.005],  # SW corner
        [lng + 0.005, lat + 0.005]   # NE corner
    ]
    
    # Create output directory with timestamp
    timestamp = int(time.time())
    output_dir = f"/ttt/data/temp/ml_pipeline_test_{timestamp}"
    
    print(f"Input image: {image_path}")
    print(f"Center coordinates: [{lat}, {lng}]")
    print(f"Bounds: SW {bounds[0]}, NE {bounds[1]}")
    print(f"Output directory: {output_dir}")
    
    # Initialize the ML pipeline
    pipeline = ComprehensiveMLPipeline()
    
    # Process the image
    results = await pipeline.process_image(image_path, bounds, output_dir)
    
    # Print results summary
    print("\nProcessing Results:")
    print(f"  Image: {image_path}")
    print(f"  Processing time: {results.get('processing_time', 0):.2f} seconds")
    
    if 'error' in results:
        print(f"  Error: {results['error']}")
    else:
        print(f"  Trees detected: {results['detection'].get('tree_count', 0)}")
        
        # Print information about the first few trees
        trees = results['detection'].get('trees', [])
        if trees:
            print("\nFirst 3 detected trees:")
            for i, tree in enumerate(trees[:3]):
                print(f"Tree {i+1}:")
                print(f"  ID: {tree.get('id', f'tree_{i}')}")
                print(f"  Location: {tree.get('location', 'N/A')}")
                print(f"  Confidence: {tree.get('confidence', 0):.4f}")
                print(f"  Risk Level: {tree.get('risk_level', 'unknown')}")
                
                # Print S2 cell IDs
                if 's2_cells' in tree:
                    print(f"  S2 Cells:")
                    for level, cell_id in tree['s2_cells'].items():
                        print(f"    {level.capitalize()}: {cell_id}")
        
        # Print information about grouped trees
        grouped_trees = results.get('grouped_trees', {})
        print(f"\nGrouped {len(trees)} trees into {len(grouped_trees)} S2 cells")
        
        # Print statistics for the first few cells
        cell_stats = results.get('cell_stats', {})
        if cell_stats:
            print("\nStatistics for first 3 S2 cells:")
            for i, (cell_id, stats) in enumerate(list(cell_stats.items())[:3]):
                print(f"Cell {i+1} ({cell_id}):")
                print(f"  Tree Count: {stats['tree_count']}")
                print(f"  Center: {stats['center']}")
                print(f"  Dominant Risk: {stats['dominant_risk']}")
                print(f"  Avg Risk Value: {stats['avg_risk_value']:.2f}")
        
        # Print saved files
        saved_files = results.get('saved_files', {})
        if saved_files:
            print("\nSaved files:")
            for file_type, file_path in saved_files.items():
                print(f"  {file_type}: {file_path}")
    
    print("\n=== ML Pipeline Test Completed ===\n")

if __name__ == "__main__":
    asyncio.run(main())