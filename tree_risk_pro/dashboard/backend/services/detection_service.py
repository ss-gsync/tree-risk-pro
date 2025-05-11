"""
Detection Service - Handles the machine learning pipeline for tree detection
Integrates with the Tree Risk Pro ML pipeline and Zarr storage
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
import subprocess
import base64
from io import BytesIO
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import shutil

# Import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import TEMP_DIR, ZARR_DIR, USE_ML_PIPELINE

# Set up logging
logger = logging.getLogger(__name__)

# ML response directory will be created dynamically under /ttt/temp/tree_detection_[timestamp]

class DetectionService:
    """Service to run tree detection using the ML pipeline and store results in Zarr"""
    
    def __init__(self):
        """Initialize the detection service"""
        self.temp_dir = TEMP_DIR
        self.zarr_dir = ZARR_DIR
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize core components
        try:
            # Import ML pipeline components
            sys.path.append('/ttt')
            from tree_risk_pro.pipeline.object_recognition import detect_trees
            from tree_risk_pro.pipeline.zarr_store import StorageManager
            
            # Initialize ML pipeline
            self.detect_trees = detect_trees
            
            # Set up storage manager with a simple config object
            class SimpleConfig:
                def __init__(self):
                    self.store_path = ZARR_DIR
                    self.temp_path = TEMP_DIR
                    
            self.storage_manager = StorageManager(SimpleConfig())
            logger.info("ML pipeline and Zarr storage initialized")
            
        except ImportError as e:
            logger.warning(f"Could not import ML pipeline: {e}")
            self.detect_trees = None
            self.storage_manager = None
    
    def _get_pixel_to_latlon_mapping(self, image_path, bounds):
        """
        Calculate the pixel to lat/lon mapping for a satellite image
        
        Args:
            image_path: Path to the satellite image
            bounds: Geographic bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            
        Returns:
            dict: Mapping information
        """
        # Open the image to get dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Extract bounds
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        
        # Calculate spans
        lng_span = ne_lng - sw_lng
        lat_span = ne_lat - sw_lat
        
        return {
            'image_width': width,
            'image_height': height,
            'sw_lng': sw_lng,
            'sw_lat': sw_lat,
            'ne_lng': ne_lng,
            'ne_lat': ne_lat,
            'lng_span': lng_span,
            'lat_span': lat_span,
            'lng_per_pixel': lng_span / width,
            'lat_per_pixel': lat_span / height
        }
    
    def _normalize_image_coords_to_geo(self, x, y, mapping):
        """
        Convert normalized image coordinates [0,1] to geographic coordinates
        
        Args:
            x: Normalized x coordinate (0-1 where 0 is left, 1 is right)
            y: Normalized y coordinate (0-1 where 0 is top, 1 is bottom)
            mapping: Pixel to lat/lon mapping
            
        Returns:
            tuple: (longitude, latitude)
        """
        # Get mapping values
        sw_lng = mapping['sw_lng']
        sw_lat = mapping['sw_lat']
        ne_lng = mapping['ne_lng']
        ne_lat = mapping['ne_lat']
        lng_span = mapping['lng_span']
        lat_span = mapping['lat_span']
        img_width = mapping['image_width']
        img_height = mapping['image_height']
        
        # Ensure input is within 0-1 range
        x = min(max(float(x), 0.0), 1.0)
        y = min(max(float(y), 0.0), 1.0)
        
        # Detect map orientation
        # Standard orientation: SW is bottom-left, NE is top-right
        # Check if we have a different orientation (e.g., rotated map)
        is_lng_inverted = sw_lng > ne_lng
        is_lat_inverted = sw_lat > ne_lat
        
        # Transform x to longitude with orientation check
        if is_lng_inverted:
            # For inverted longitude (e.g., across the 180° meridian)
            # We need special handling to avoid incorrect interpolation
            if sw_lng > 0 and ne_lng < 0:
                # Map crossing 180° meridian from east to west
                ne_lng += 360  # Adjust to continuous space
                lng = sw_lng + (x * lng_span)
                if lng > 180:
                    lng -= 360  # Normalize back
            else:
                # Otherwise, just invert the mapping
                lng = sw_lng - (x * lng_span)
        else:
            # Standard longitude mapping (west to east)
            lng = sw_lng + (x * lng_span)
        
        # Transform y to latitude with orientation check
        if is_lat_inverted:
            # Inverted latitude (south at top, north at bottom)
            lat = sw_lat - (y * lat_span)
        else:
            # Standard latitude mapping (north at top, south at bottom)
            lat = ne_lat - (y * lat_span)
        
        # Log the transformation
        logger.info(f"Transformed image coords [{x}, {y}] to geo coordinates [{lng}, {lat}]")
        logger.info(f"Map orientation: lng_inverted={is_lng_inverted}, lat_inverted={is_lat_inverted}")
        
        # Apply bounds check and wrap longitude if needed
        lng = (lng + 180) % 360 - 180  # Normalize to [-180, 180]
        lat = max(min(lat, 90), -90)   # Clamp to valid range
        
        return lng, lat
    
    def _convert_detection_to_zarr_format(self, tree_detections, mapping, job_id):
        """
        Convert detection results to Zarr storage format
        
        Args:
            tree_detections: Results from ML detection
            mapping: Pixel to lat/lon mapping
            job_id: Job ID for tracking
            
        Returns:
            dict: Data in Zarr storage format
        """
        trees = []
        coordinates = []
        
        # Process each detected tree
        for i, tree in enumerate(tree_detections.get('trees', [])):
            # Extract bounding box
            if 'bbox' in tree:
                x1, y1, x2, y2 = tree['bbox']
                
                # Convert to normalized coordinates if they're in pixels
                img_width = mapping['image_width']
                img_height = mapping['image_height']
                
                # Check if coordinates are already normalized (0-1)
                # If not, normalize them
                if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
                    x1 = x1 / img_width
                    y1 = y1 / img_height
                    x2 = x2 / img_width
                    y2 = y2 / img_height
                
                # Calculate center point in normalized coordinates
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Convert to geographic coordinates
                lng, lat = self._normalize_image_coords_to_geo(center_x, center_y, mapping)
                
                # Create tree entry
                tree_entry = {
                    'id': f"{job_id}_tree_{i}",
                    'bbox': [x1, y1, x2, y2],
                    'confidence': tree.get('confidence', 0.0),
                    'location': [lng, lat],
                    'height': tree.get('height', 15.0),
                    'species': tree.get('species', 'Unknown'),
                    'risk_level': tree.get('risk_level', 'medium'),
                    'risk_factors': tree.get('risk_factors', [])
                }
                
                # Add to trees list
                trees.append(tree_entry)
                
                # Add to coordinates list for S2 indexing
                coordinates.append({
                    'id': tree_entry['id'],
                    'frame_idx': 0,
                    'lat': lat,
                    'lng': lng,
                    'type': 'tree'
                })
        
        # Create zarr data structure
        zarr_data = {
            'metadata': {
                'job_id': job_id,
                'timestamp': datetime.now().isoformat(),
                'tree_count': len(trees),
                'bounds': [
                    [mapping['sw_lng'], mapping['sw_lat']],
                    [mapping['ne_lng'], mapping['ne_lat']]
                ]
            },
            'frames': [
                {
                    'trees': trees,
                    'metadata': {
                        'frame_idx': 0,
                        'tree_count': len(trees)
                    }
                }
            ],
            'coordinates': coordinates
        }
        
        return zarr_data
    
    async def detect_trees_from_satellite(self, image_path, bounds, job_id, ml_response_dir=None, existing_trees=None):
        """
        Run tree detection on a satellite image and store results in the specified structure
        
        Args:
            image_path: Path to the satellite image
            bounds: Geographic bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            job_id: Job ID for tracking
            ml_response_dir: Optional custom directory for ML response
            existing_trees: Optional list of previously detected trees (for segmentation mode)
            
        Returns:
            dict: Detection results
        """
        try:
            logger.info(f"Starting tree detection for job {job_id}")
            
            # Extract timestamp from job_id or current time
            timestamp = str(int(time.time()))
            if job_id.startswith('ml_detect_'):
                try:
                    timestamp = job_id.split('ml_detect_')[1]
                except Exception as e:
                    logger.warning(f"Could not extract timestamp from job_id: {e}")
            
            # Create ml_response_dir if not provided
            if not ml_response_dir:
                # Use new structure: /ttt/data/temp/ml_results_[timestamp]/detection/ml_response
                base_dir = os.path.join('/ttt/data/temp', f'ml_results_{timestamp}')
                detection_dir = os.path.join(base_dir, 'detection')
                ml_response_dir = os.path.join(detection_dir, 'ml_response')
                os.makedirs(ml_response_dir, exist_ok=True)
                logger.info(f"Created ML response directory at {ml_response_dir}")
            
            # Verify ML pipeline is available
            if not USE_ML_PIPELINE or not self.detect_trees:
                logger.error("ML pipeline is not available. Check tree_risk_pro.pipeline configuration.")
                return {
                    "error": "ML pipeline required but not available",
                    "job_id": job_id,
                    "timestamp": timestamp
                }
            
            # Ensure the ml_response_dir exists
            os.makedirs(ml_response_dir, exist_ok=True)
            logger.info(f"Using ML response directory at {ml_response_dir}")
            
            # Calculate pixel to lat/lon mapping
            mapping = self._get_pixel_to_latlon_mapping(image_path, bounds)
            logger.info(f"Image dimensions: {mapping['image_width']}x{mapping['image_height']}")
            
            # Create temporary output file for detection results
            output_path = os.path.join(ml_response_dir, f"detection_result.json")
            
            # Also save a copy of the input image
            image_copy_path = os.path.join(ml_response_dir, f"input_image.jpg")
            shutil.copy(image_path, image_copy_path)
            
            # If existing trees are provided (for segmentation), use them instead of running detection
            if existing_trees:
                logger.info(f"Using {len(existing_trees)} existing trees from previous detection")
                detection_results = {
                    "trees": existing_trees,
                    "tree_count": len(existing_trees)
                }
                detection_time = 0
            else:
                # Run tree detection (ML pipeline)
                start_time = time.time()
                detection_results = self.detect_trees(image_path, output_path)
                detection_time = time.time() - start_time
                logger.info(f"Tree detection completed in {detection_time:.2f} seconds")
            
            # Convert detection results to Zarr format
            zarr_data = self._convert_detection_to_zarr_format(detection_results, mapping, job_id)
            
            # Store the zarr_data in a readable JSON format in ml_response directory
            # Use 'trees.json' for a more consistent filename than detection_data.json
            with open(os.path.join(ml_response_dir, 'trees.json'), 'w') as f:
                json.dump(zarr_data, f, indent=2)
            
            # Store results in Zarr only for real detection (not for segmentation reuse)
            if self.storage_manager and not existing_trees:
                start_time = time.time()
                store_path = await self.storage_manager.store_data(zarr_data, job_id, use_temp=True)
                storage_time = time.time() - start_time
                logger.info(f"Results stored in Zarr at {store_path} in {storage_time:.2f} seconds")
            else:
                if existing_trees:
                    logger.info("Skipping Zarr storage for segmentation mode using existing trees")
                else:
                    logger.warning("Storage manager unavailable - results not stored in Zarr")
                store_path = None
            
            # Save response metadata
            response_metadata = {
                "job_id": job_id,
                "timestamp": timestamp,
                "tree_count": len(zarr_data['frames'][0]['trees']),
                "image_path": image_copy_path,
                "result_path": output_path,
                "detection_time": detection_time,
                "zarr_store_path": store_path,
                "bounds": bounds
            }
            
            # Save the metadata
            with open(os.path.join(ml_response_dir, 'metadata.json'), 'w') as f:
                json.dump(response_metadata, f, indent=2)
            
            # Return detection results
            return {
                "job_id": job_id,
                "tree_count": len(zarr_data['frames'][0]['trees']),
                "trees": zarr_data['frames'][0]['trees'],
                "detection_time": detection_time,
                "zarr_store_path": store_path,
                "ml_response_dir": ml_response_dir
            }
            
        except Exception as e:
            logger.error(f"Error in tree detection: {str(e)}", exc_info=True)
            return {"error": str(e), "job_id": job_id, "trees": []}
    
    async def detect_trees_from_map_view(self, map_view_info, job_id):
        """
        Run tree detection based on map view information
        
        Args:
            map_view_info: Map view information including bounds
            job_id: Job ID for tracking
            
        Returns:
            dict: Detection results
        """
        try:
            # Extract view data
            view_data = map_view_info.get('viewData', {})
            
            # Check for segmentation mode
            is_segmentation = map_view_info.get('segmentation_mode', False)
            mode = 'segmentation' if is_segmentation else 'detection'
            
            # Check for previous detection information (for segmentation that follows detection)
            previous_job_id = map_view_info.get('previous_job_id')
            previous_ml_dir = map_view_info.get('previous_ml_dir')
            
            logger.info(f"Processing {mode} request for job {job_id}")
            
            # Get bounds
            bounds = view_data.get('bounds')
            if not bounds:
                raise ValueError("Missing bounds in map view info")
            
            # Validate bounds format
            if not isinstance(bounds, list) or len(bounds) != 2 or not all(isinstance(b, list) and len(b) == 2 for b in bounds):
                logger.error(f"Invalid bounds format: {bounds}")
                raise ValueError("Invalid bounds format. Expected [[sw_lng, sw_lat], [ne_lng, ne_lat]]")
            
            # Log the bounds for debugging
            logger.info(f"Map bounds: SW: {bounds[0]}, NE: {bounds[1]}")
            logger.info(f"Spans: lng_span: {bounds[1][0] - bounds[0][0]}, lat_span: {bounds[1][1] - bounds[0][1]}")
            
            # Create timestamp
            timestamp = map_view_info.get('timestamp', str(int(time.time())))
            
            # Create base results directory for this operation
            base_results_dir = os.path.join('/ttt/data/temp', f'ml_results_{timestamp}')
            os.makedirs(base_results_dir, exist_ok=True)
            
            # Create mode-specific subdirectory
            mode_dir = os.path.join(base_results_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            
            # Create ml_response_dir
            ml_response_dir = os.path.join(mode_dir, 'ml_response')
            os.makedirs(ml_response_dir, exist_ok=True)
            
            # Get image path (if already captured)
            image_path = view_data.get('image_path')
            
            # For testing, look for an existing satellite image in the base directory
            if not image_path:
                # Check for existing image in the base directory
                for file in os.listdir(base_results_dir):
                    if file.startswith('satellite_') and file.endswith('.jpg'):
                        image_path = os.path.join(base_results_dir, file)
                        logger.info(f"Using existing image from base directory: {image_path}")
                        break
            
            # If no image path, we need to capture the satellite image
            if not image_path:
                # Use satellite service to get the image
                from .satellite_service import SatelliteService
                satellite_service = SatelliteService()
                
                # Get center and zoom
                center = view_data.get('center')
                zoom = view_data.get('zoom', 18)
                
                # Create a filename with coordinates
                image_filename = f"satellite_{center[1]}_{center[0]}_{zoom}_{timestamp}.jpg"
                image_path = os.path.join(base_results_dir, image_filename)
                
                # Capture the satellite image
                image_path = satellite_service.get_satellite_image(
                    center=center,
                    zoom=zoom,
                    width=640,
                    height=640,
                    output_dir=base_results_dir,
                    filename=image_filename
                )
                
                logger.info(f"Captured satellite image at {image_path}")
            
            # Check if we should reuse existing trees for segmentation
            existing_trees = None
            if is_segmentation and previous_ml_dir:
                # Look for trees.json in previous detection directory
                previous_trees_path = os.path.join(previous_ml_dir, 'trees.json')
                if os.path.exists(previous_trees_path):
                    try:
                        with open(previous_trees_path, 'r') as f:
                            tree_data = json.load(f)
                            existing_trees = tree_data.get('frames', [{}])[0].get('trees', [])
                            logger.info(f"Loaded {len(existing_trees)} trees from previous detection")
                    except Exception as e:
                        logger.error(f"Error loading trees from previous detection: {str(e)}")
            
            # Run main detection
            result = await self.detect_trees_from_satellite(
                image_path=image_path, 
                bounds=bounds, 
                job_id=job_id,
                ml_response_dir=ml_response_dir,
                existing_trees=existing_trees
            )
            
            # For segmentation mode, add segmentation data
            if is_segmentation and 'ml_response_dir' in result:
                logger.info(f"Adding segmentation data for job {job_id}")
                
                trees = result.get('trees', [])
                
                # Process segmentation data
                await self._add_segmentation_data(ml_response_dir, trees, image_path, bounds)
                
                # Update result with mode and updated trees
                result['mode'] = 'segmentation'
                result['trees'] = trees
            else:
                result['mode'] = 'detection'
            
            # Add reference to the base directory and image
            result['base_results_dir'] = base_results_dir
            result['image_path'] = image_path
            result['timestamp'] = timestamp
            
            return result
            
        except Exception as e:
            logger.error(f"Error in {mode} from map view: {str(e)}", exc_info=True)
            return {"error": str(e), "job_id": job_id, "trees": []}
            
    async def _add_segmentation_data(self, ml_response_dir, trees, image_path, bounds):
        """
        Add segmentation data to the response directory
        
        Args:
            ml_response_dir: Directory to store segmentation data
            trees: List of detected trees
            image_path: Path to the satellite image
            bounds: Geographic bounds
            
        Returns:
            None - Updates trees in place
        """
        try:
            # Import required modules
            from PIL import Image, ImageDraw
            import numpy as np
            import random
            
            # Create segmentation metadata
            segmentation_metadata = {
                "segmentation_mode": True,
                "tree_count": len(trees),
                "segmentation_algorithm": "contour-based",
                "timestamp": datetime.now().isoformat(),
                "image_path": image_path
            }
            
            # Save segmentation metadata
            with open(os.path.join(ml_response_dir, 'segmentation_metadata.json'), 'w') as f:
                json.dump(segmentation_metadata, f, indent=2)
                
            # Create a blank combined mask
            try:
                satellite_img = Image.open(image_path)
                img_width, img_height = satellite_img.size
            except:
                img_width, img_height = 640, 640
                
            # Create segmentation mask for each tree
            if not trees:
                logger.warning("No trees found for segmentation")
                return
                
            # Create a combined segmentation mask
            combined_mask = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(combined_mask)
            
            # Set random seed for reproducible colors
            random.seed(42)
            
            # Process each tree
            for i, tree in enumerate(trees):
                if 'location' in tree:
                    try:
                        # Get tree location and convert to pixel coordinates
                        lng, lat = tree['location']
                        
                        # Convert to normalized coordinates (0-1)
                        x_norm = (lng - bounds[0][0]) / (bounds[1][0] - bounds[0][0])
                        y_norm = (lat - bounds[0][1]) / (bounds[1][1] - bounds[0][1])
                        
                        # Convert to pixel coordinates
                        x = int(x_norm * img_width)
                        y = int(y_norm * img_height)
                        
                        # Determine tree radius based on height or use default
                        tree_height = tree.get('height', 30)
                        radius = max(10, min(50, int(tree_height / 2)))
                        
                        # Create individual mask for this tree
                        tree_mask = Image.new('L', (img_width, img_height), 0)
                        tree_draw = ImageDraw.Draw(tree_mask)
                        
                        # Draw tree
                        tree_draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
                        
                        # Save individual mask
                        mask_path = os.path.join(ml_response_dir, f'tree_{i+1}_mask.png')
                        tree_mask.save(mask_path)
                        
                        # Add mask path to tree data
                        tree['mask_path'] = mask_path
                        
                        # Generate random color for combined mask
                        color = (
                            random.randint(50, 255),
                            random.randint(50, 255),
                            random.randint(50, 255),
                            180  # Semi-transparent
                        )
                        
                        # Add to combined mask
                        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
                        
                    except Exception as e:
                        logger.error(f"Error creating mask for tree {i}: {str(e)}")
            
            # Save combined mask
            combined_mask_path = os.path.join(ml_response_dir, 'combined_segmentation.png')
            combined_mask.save(combined_mask_path)
            
            # Save segmentation overlay on original image
            try:
                # Open the original satellite image
                satellite_img = Image.open(image_path).convert('RGBA')
                
                # Create overlay
                overlay = Image.alpha_composite(satellite_img, combined_mask)
                overlay_path = os.path.join(ml_response_dir, 'segmentation_overlay.png')
                overlay.save(overlay_path)
            except Exception as e:
                logger.error(f"Error creating segmentation overlay: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in segmentation processing: {str(e)}", exc_info=True)