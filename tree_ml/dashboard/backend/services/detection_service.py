"""
Detection Service - ML Pipeline for Satellite Imagery Tree Detection
===================================================================

The Detection Service provides comprehensive machine learning capabilities for 
detecting trees and other objects in satellite imagery. It handles the complete
pipeline from image acquisition to model inference and result persistence.

Key Components and Flow:
-----------------------

1. Coordinate Management
   - Uses standardized geographic coordinates from the frontend
   - Supports S2 geospatial indexing for efficient spatial queries
   - Handles conversion between pixel and geographic coordinates

2. Image Acquisition
   - Retrieves satellite imagery that matches the user's current map view
   - Supports both Google Maps Static API and uploaded imagery
   - Ensures consistent resolution and projection

3. ML Model Inference
   - Primary detection using either DeepForest or Grounded-SAM models
   - DeepForest: Specialized forest detection with high precision
   - Grounded-SAM: Zero-shot object detection with segmentation masks
   - Runs inference on GPU when available for faster processing

4. Result Processing
   - Converts detection results to standardized format
   - Adds geospatial metadata including S2 cell IDs
   - Generates visualizations of detected objects

5. Data Persistence
   - Stores results in Zarr format for efficient retrieval
   - Includes metadata about detection parameters and coordinates
   - Maintains job-specific organization of results

Data Flow Diagram:
----------------
User Request → Coordinate Standardization → Satellite Image Acquisition →
ML Model Inference → Geospatial Processing → Result Visualization →
Zarr Storage → Response to Frontend

The implementation ensures accurate ML detection based on the user's current view
by maintaining consistent coordinate references throughout the pipeline.
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
import s2sphere  # For S2 geospatial indexing

# Import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import ML_DIR, ZARR_DIR, USE_ML_PIPELINE

# Import services
from services.geolocation_service import GeolocationService

# Set up logging
logger = logging.getLogger(__name__)

# ML response directory will be created dynamically under /ttt/data/ml/detection_[timestamp]

class S2IndexManager:
    """Handles S2 geospatial indexing for tree detection results"""
    
    def __init__(self):
        """Initialize the S2 geospatial index manager"""
        # S2 cell levels for different zoom levels
        # Level 10: ~0.3km² (city scale)
        # Level 13: ~0.02km² (neighborhood)
        # Level 15: ~1400m² (block)
        # Level 18: ~40m² (property)
        self.cell_levels = {
            'city': 10,      # For city-wide view (low zoom)
            'neighborhood': 13,  # For neighborhood view (medium zoom)
            'block': 15,     # For block view (medium-high zoom)
            'property': 18   # For property level (high zoom)
        }
        
    def get_cell_id(self, lat, lng, level='property'):
        """
        Get the S2 cell ID for a given lat/lng at the specified level
        
        Args:
            lat: Latitude
            lng: Longitude
            level: Cell level name ('city', 'neighborhood', 'block', 'property')
            
        Returns:
            str: S2 cell ID as a string
        """
        # Get numerical level
        cell_level = self.cell_levels.get(level, 18)  # Default to property level
        
        # Create S2 Cell
        latlng = s2sphere.LatLng.from_degrees(lat, lng)
        cell = s2sphere.CellId.from_lat_lng(latlng).parent(cell_level)
        
        # Return cell ID as a string
        return str(cell.id())
    
    def get_cell_ids_for_tree(self, lat, lng):
        """
        Get all S2 cell IDs for a tree location at different levels
        
        Args:
            lat: Latitude
            lng: Longitude
            
        Returns:
            dict: Dictionary of S2 cell IDs at different levels
        """
        cell_ids = {}
        for level_name, level in self.cell_levels.items():
            cell_ids[level_name] = self.get_cell_id(lat, lng, level_name)
        
        return cell_ids
    
    def get_cells_for_bounds(self, bounds, level='property'):
        """
        Get S2 cells covering the given bounds
        
        Args:
            bounds: Geographic bounds as [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            level: Cell level name ('city', 'neighborhood', 'block', 'property')
            
        Returns:
            list: List of S2 cell IDs covering the bounds
        """
        # Extract bounds
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        
        # Get numerical level
        cell_level = self.cell_levels.get(level, 18)  # Default to property level
        
        # Create region covering the bounds
        region = s2sphere.LatLngRect(
            s2sphere.LatLng.from_degrees(sw_lat, sw_lng),
            s2sphere.LatLng.from_degrees(ne_lat, ne_lng)
        )
        
        # Get cells that cover the region
        coverer = s2sphere.RegionCoverer()
        coverer.min_level = cell_level
        coverer.max_level = cell_level
        coverer.max_cells = 100  # Limit the number of cells
        
        cells = coverer.get_covering(region)
        
        # Convert to string IDs
        return [str(cell.id()) for cell in cells]
    
    def get_neighbors(self, lat, lng, level='property', k=8):
        """
        Get neighboring S2 cells for a given location
        
        Args:
            lat: Latitude
            lng: Longitude
            level: Cell level name ('city', 'neighborhood', 'block', 'property')
            k: Number of neighbors to return (8 for all adjacent cells)
            
        Returns:
            list: List of neighboring S2 cell IDs
        """
        # Get numerical level
        cell_level = self.cell_levels.get(level, 18)  # Default to property level
        
        # Create center cell
        latlng = s2sphere.LatLng.from_degrees(lat, lng)
        center_cell = s2sphere.CellId.from_lat_lng(latlng).parent(cell_level)
        
        # Get all neighbors
        neighbors = []
        for i in range(4):  # Four face-adjacent neighbors
            neighbor = center_cell.get_edge_neighbors()[i]
            neighbors.append(str(neighbor.id()))
            
        if k > 4:  # If we want more than just face-adjacent neighbors
            # Add vertex-adjacent neighbors (diagonals)
            for i in range(4):
                vertex_neighbor = center_cell.get_vertex_neighbors(cell_level)[i]
                if str(vertex_neighbor.id()) not in neighbors:
                    neighbors.append(str(vertex_neighbor.id()))
                if len(neighbors) >= k:
                    break
                    
        return neighbors[:k]  # Return up to k neighbors

class DetectionService:
    """Service to run tree detection using the ML pipeline and store results in Zarr"""
    
    def __init__(self):
        """Initialize the detection service"""
        self.ml_dir = ML_DIR
        self.zarr_dir = ZARR_DIR
        
        # Create all required directories
        os.makedirs(self.ml_dir, exist_ok=True)
        os.makedirs(self.zarr_dir, exist_ok=True)
        validation_dir = os.path.join(self.zarr_dir, 'validation')
        os.makedirs(validation_dir, exist_ok=True)
        
        # Create empty validation queue file if it doesn't exist
        validation_queue_path = os.path.join(self.zarr_dir, 'validation_queue.json')
        if not os.path.exists(validation_queue_path):
            with open(validation_queue_path, 'w') as f:
                f.write('[]')
            logger.info(f"Created empty validation queue at {validation_queue_path}")
        
        # Initialize S2 geospatial indexing with enhanced logging
        self.s2_manager = S2IndexManager()
        logger.info("S2 geospatial indexing initialized for precise coordinate mapping")
        
        # Verify S2 functionality with a test coordinate in Dallas, Texas
        try:
            test_lat, test_lng = 32.7767, -96.7970  # Dallas, Texas coordinates
            test_cell_id = self.s2_manager.get_cell_id(test_lat, test_lng, 'block')
            test_cells = self.s2_manager.get_cell_ids_for_tree(test_lat, test_lng)
            logger.info(f"S2 geospatial indexing test successful: {test_cell_id} (block level)")
            logger.info(f"S2 cell hierarchy for Dallas: city={test_cells.get('city', 'N/A')}, neighborhood={test_cells.get('neighborhood', 'N/A')}, block={test_cells.get('block', 'N/A')}")
        except Exception as e:
            logger.warning(f"S2 geospatial indexing test failed: {e}. Coordinate mapping may be less precise.")
        
        # Initialize GeolocationService
        self.geolocation_service = GeolocationService()
        logger.info("Geolocation service initialized")
        
        # Initialize ML components
        self.ml_service = None
        self.detect_trees = None
        self.storage_manager = None
        self._initialize_ml_components()
        
    def _get_satellite_image_from_gmaps(self, view_data, output_path=None):
        """
        Retrieve a satellite image from Google Maps Static API using the GeolocationService
        
        Args:
            view_data: Map view information including center, zoom, and dimensions
            output_path: Optional path to save the image
            
        Returns:
            tuple: (success, image_data) where image_data is either the PIL Image or 
                  a dict with imageUrl, mapWidth, mapHeight for the view_data
        """
        logger.info(f"Delegating satellite image retrieval to GeolocationService")
        
        try:
            # Use the GeolocationService to handle all coordinate processing and API calls
            success, result = self.geolocation_service.get_satellite_image_from_gmaps(view_data, output_path)
            
            if not success:
                logger.error(f"GeolocationService failed to get satellite image: {result.get('error', 'Unknown error')}")
                return False, None
            
            # Extract all the necessary data from the result
            image = result.get("image")
            bounds = result.get("bounds")
            center = result.get("center")
            zoom = result.get("zoom")
            map_width = result.get("width")
            map_height = result.get("height")
            raw_params = result.get("raw_params", {})
            
            logger.info(f"GeolocationService returned success with bounds: {bounds}")
            logger.info(f"Center: {center}, Zoom: {zoom}, Dimensions: {map_width}x{map_height}")
            
            # Return success with the image data and parameters in the expected format for downstream processing
            return True, {
                'bounds': bounds,
                'center': center,
                'zoom': zoom,
                'mapWidth': map_width,
                'mapHeight': map_height,
                'url': raw_params.get('url', '')
            }
            
        except Exception as e:
            logger.error(f"Error in _get_satellite_image_from_gmaps: {e}", exc_info=True)
            return False, None
    
    def _initialize_ml_components(self):
        """Initialize ML pipeline components and models"""
        # Initialize core components
        try:
            # Import ML pipeline components with proper error handling
            logger.info("Initializing ML pipeline components...")
            sys.path.append('/ttt')
            
            # Import the ML service based on configuration - this uses the updated __init__.py
            # which will only import the appropriate model service
            try:
                from .ml import get_model_service
                
                # Get the appropriate model service (external or local)
                self.ml_service = get_model_service()
                service_type = "External T4" if hasattr(self.ml_service, 'server_url') else "Local"
                logger.info(f"{service_type} ML service initialized - waiting for models to load...")
                
                # Wait for models to load with a 30 second timeout - CRITICAL STEP
                timeout = 30
                start_time = time.time()
                while not self.ml_service.models_loaded and (time.time() - start_time) < timeout:
                    time.sleep(1)
                    
                if self.ml_service.models_loaded:
                    logger.info(f"{service_type} ML models successfully loaded")
                    self.detect_trees = self.ml_service.detect_trees
                else:
                    logger.warning(f"ML models did not load within {timeout} seconds")
                    self.detect_trees = None
                    
            except (ImportError, AttributeError) as e:
                logger.warning(f"ML service not available: {e}")
                self.ml_service = None
                self.detect_trees = None
                
            # Import Zarr storage utilities
            try:
                from tree_ml.pipeline.zarr_store import ZarrStorageManager
                self.storage_manager = ZarrStorageManager(self.zarr_dir)
                logger.info("Zarr storage manager initialized")
            except ImportError as e:
                logger.warning(f"Zarr storage manager not available: {e}")
                self.storage_manager = None
                
        except Exception as e:
            logger.error(f"Error initializing ML components: {e}", exc_info=True)
            self.ml_service = None
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
        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
            
        # Extract bounds
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        
        # Calculate spans
        lng_span = ne_lng - sw_lng
        lat_span = ne_lat - sw_lat
        
        # Calculate degrees per pixel
        lng_per_pixel = lng_span / width
        lat_per_pixel = lat_span / height
        
        # Create mapping
        mapping = {
            'width': width,
            'height': height,
            'bounds': bounds,
            'sw_lng': sw_lng,
            'sw_lat': sw_lat,
            'ne_lng': ne_lng,
            'ne_lat': ne_lat,
            'lng_per_pixel': lng_per_pixel,
            'lat_per_pixel': lat_per_pixel
        }
        
        return mapping
        
    def _get_s2_cells_for_bounds(self, bounds, mapping):
        """
        Get S2 cells covering the image bounds for precise coordinate mapping
        
        Args:
            bounds: Geographic bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            mapping: The existing pixel-to-latlon mapping
            
        Returns:
            dict: S2 cell mapping information
        """
        try:
            # Get S2 cells covering the bounds
            cells = self.s2_manager.get_cells_for_bounds(bounds, 'block')
            
            if not cells:
                logging.warning("No S2 cells found for bounds")
                return None
                
            # Get cell details for the center cell
            center_lng = (bounds[0][0] + bounds[1][0]) / 2
            center_lat = (bounds[0][1] + bounds[1][1]) / 2
            center_cell_id = self.s2_manager.get_cell_id(center_lat, center_lng, 'block')
            
            # Create S2 mapping with all necessary parameters
            s2_mapping = {
                'center_cell': center_cell_id,
                'cells': cells,
                'cell_level': 'block',
                'width': mapping['width'],
                'height': mapping['height'],
                'sw_lng': mapping['sw_lng'],
                'sw_lat': mapping['sw_lat'],
                'ne_lng': mapping['ne_lng'],
                'ne_lat': mapping['ne_lat'],
                'lng_per_pixel': mapping['lng_per_pixel'],
                'lat_per_pixel': mapping['lat_per_pixel'],
                'mapping_type': 's2'
            }
            
            # Calculate additional S2-specific parameters for coordinate precision
            # These will be used by the calculateCoordinates function on the frontend
            s2_mapping['cell_width'] = s2_mapping['ne_lng'] - s2_mapping['sw_lng']
            s2_mapping['cell_height'] = s2_mapping['ne_lat'] - s2_mapping['sw_lat']
            
            return s2_mapping
            
        except Exception as e:
            logging.error(f"Error generating S2 cells for bounds: {e}")
            return None
    
    def _normalize_image_coords_to_geo(self, x, y, mapping):
        """
        Convert normalized image coordinates [0,1] to geographic coordinates
        with support for S2 cell-based coordinate mapping
        
        Args:
            x: Normalized x coordinate (0-1 where 0 is left, 1 is right)
            y: Normalized y coordinate (0-1 where 0 is top, 1 is bottom)
            mapping: Pixel to lat/lon mapping with optional S2 cell information
            
        Returns:
            tuple: (longitude, latitude)
        """
        # Check if we have S2 cell mapping for more precise coordinate transformation
        if 's2_cells' in mapping:
            try:
                # Get S2 mapping data
                s2_mapping = mapping['s2_cells']
                
                # Convert to pixel coordinates
                pixel_x = x * s2_mapping['width']
                pixel_y = y * s2_mapping['height']
                
                # Convert to geographic coordinates with S2 cell precision
                # Note: y is inverted because pixel coordinates increase downward
                lng = s2_mapping['sw_lng'] + pixel_x * s2_mapping['lng_per_pixel']
                lat = s2_mapping['ne_lat'] - pixel_y * s2_mapping['lat_per_pixel']
                
                # Log that we're using S2-based mapping
                logger.debug(f"Using S2 cell-based coordinate mapping: [{lng:.6f}, {lat:.6f}]")
                return lng, lat
                
            except Exception as e:
                # Log error and fall back to standard mapping
                logger.warning(f"Error using S2 cell mapping: {e}. Falling back to standard mapping.")
        
        # Standard mapping approach (fallback if S2 mapping fails or isn't available)
        # Convert to pixel coordinates
        pixel_x = x * mapping['width']
        pixel_y = y * mapping['height']
        
        # Convert to geographic coordinates
        # Note: y is inverted because pixel coordinates increase downward
        lng = mapping['sw_lng'] + pixel_x * mapping['lng_per_pixel']
        lat = mapping['ne_lat'] - pixel_y * mapping['lat_per_pixel']
        
        return lng, lat
    
    def _convert_detection_to_zarr_format(self, detections, mapping, job_id):
        """
        Convert detection results to Zarr storage format
        
        Args:
            detections: List of detection objects from ML models or 'detections' field from results
            mapping: Pixel to lat/lon mapping
            job_id: Job ID for tracking
            
        Returns:
            dict: Data in Zarr storage format
        """
        try:
            # Create the base data structure
            data = {
                'job_id': job_id,
                'timestamp': datetime.now().isoformat(),
                'trees': [],
                'metadata': {
                    'source': 'satellite',
                    'resolution': {
                        'width': mapping['width'],
                        'height': mapping['height']
                    },
                    'bounds': mapping['bounds']
                }
            }
            
            # Check if detections is a dict with 'detections' key (from object_recognition.py)
            if isinstance(detections, dict) and 'detections' in detections:
                logger.info(f"Processing detection result with {len(detections['detections'])} detections from ML pipeline")
                # Use the detections field in the result dict
                detection_list = detections['detections']
            else:
                # Use detections directly (backward compatibility)
                logger.info(f"Processing {len(detections)} direct detections")
                detection_list = detections
                
            # Apply additional filtering to make sure no whole-image detections slip through
            filtered_list = []
            for detection in detection_list:
                # Extract bounding box
                bbox = detection.get('bbox', [0, 0, 0, 0])
                if len(bbox) == 4:
                    # Calculate area
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    
                    # Skip if covering too much of the image (20%)
                    if area > 0.2:
                        logger.warning(f"Zarr filter: Skipping detection with large area {area:.2f}")
                        continue
                        
                    # Skip if at the edges
                    if (bbox[0] < 0.03 and bbox[1] < 0.03 and 
                        bbox[2] > 0.97 and bbox[3] > 0.97):
                        logger.warning(f"Zarr filter: Skipping edge-spanning detection")
                        continue
                        
                    # Accept this detection
                    filtered_list.append(detection)
                    
            if len(detection_list) > 0 and len(filtered_list) == 0:
                logger.warning("All detections were filtered out in Zarr conversion!")
            
            logger.info(f"After Zarr filtering: {len(filtered_list)} of {len(detection_list)} detections kept")
            
            # Convert each detection
            for i, detection in enumerate(filtered_list):
                # Extract bounding box in normalized coordinates [0-1]
                bbox = detection.get('bbox', [0, 0, 0, 0])
                
                # Calculate center of bounding box in normalized coordinates
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Convert to geographic coordinates
                lng, lat = self._normalize_image_coords_to_geo(center_x, center_y, mapping)
                
                # Get S2 cell IDs for this location
                s2_cells = self.s2_manager.get_cell_ids_for_tree(lat, lng)
                
                # Get class information (now can include trees, buildings, etc.)
                class_name = detection.get('class', 'tree')
                
                # Create a unique ID based on class
                object_id = f"{class_name}_{job_id}_{i}"
                
                # Create data structure (we'll still call it 'trees' for backward compatibility)
                object_data = {
                    'id': object_id,
                    'coordinates': {
                        'lat': lat,
                        'lng': lng
                    },
                    's2_cells': s2_cells,
                    'detection': {
                        'confidence': detection.get('confidence', 0.0),
                        'bbox': bbox,
                        'class': class_name,
                        'source': 'satellite',
                        'coordinate_system': 's2' if s2_cells else 'standard',
                        'mapping_precision': 'high' if s2_cells else 'standard'
                    },
                    'attributes': {
                        'health': None,
                        'species': None,
                        'height': None,
                        'canopy_width': None,
                        'risk_score': None
                    },
                    'validation': {
                        'status': 'pending',
                        'timestamp': None,
                        'validator': None,
                        'notes': None
                    },
                    'geospatial': {
                        'cell_id': s2_cells.get('block') if s2_cells else None,
                        'precision': 'high' if s2_cells else 'standard',
                        'coordinate_source': 's2_mapping' if s2_cells else 'standard_mapping'
                    }
                }
                
                # Add segmentation mask if available
                if 'segmentation' in detection:
                    object_data['detection']['segmentation'] = detection['segmentation']
                
                # Add mask score if available
                if 'mask_score' in detection:
                    object_data['detection']['mask_score'] = detection['mask_score']
                    
                # Add to trees list
                data['trees'].append(object_data)
                
            return data
            
        except Exception as e:
            logger.error(f"Error converting detection to Zarr format: {e}", exc_info=True)
            return None
    
    async def detect_trees_from_canvas_capture(self, image_path, bounds, job_id, ml_response_dir=None, existing_trees=None):
        """
        Run tree detection on a canvas capture image and store results in the specified structure
        
        Args:
            image_path: Path to the canvas capture image
            bounds: Geographic bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            job_id: Job ID for tracking
            ml_response_dir: Optional custom directory for ML response
            existing_trees: Optional list of previously detected trees (for segmentation mode)
            
        Returns:
            dict: Detection results
        """
        try:
            logger.info(f"Starting tree detection from canvas capture: {image_path}")
            
            # Create ML response directory with the correct naming pattern
            if not ml_response_dir:
                # Create directory structure for ML results
                # Job ID already includes the correct ml_ prefix
                ml_dir = os.path.join(self.ml_dir, job_id)
                ml_response_dir = os.path.join(ml_dir, "ml_response")
            os.makedirs(ml_response_dir, exist_ok=True)
            logger.info(f"Using ML response directory: {ml_response_dir}")
            
            # Create pixel to lat/lon mapping
            mapping = self._get_pixel_to_latlon_mapping(image_path, bounds)
            
            # Add S2 cell information to mapping for more accurate coordinates
            s2_cells_mapping = self._get_s2_cells_for_bounds(bounds, mapping)
            if s2_cells_mapping:
                mapping['s2_cells'] = s2_cells_mapping
                logger.info(f"Added S2 cell mapping information for precise coordinate projection")
            
            # Check if we have a working ML detection function
            if self.detect_trees is None:
                logger.error("No ML detection function available")
                return {
                    'success': False,
                    'message': "ML detection not initialized",
                    'job_id': job_id,
                    'trees': []
                }
                
            # Run ML detection
            try:
                # Load the image as a numpy array
                with Image.open(image_path) as img:
                    # Convert to RGB to ensure compatibility
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Convert to numpy array
                    image_array = np.array(img)
                
                # Process the image for detection with the correct parameters
                results = self.detect_trees(
                    image=image_array,
                    confidence_threshold=0.3,
                    with_segmentation=True,
                    job_id=job_id
                )
                
                # Save all required files to the output directory
                
                # 1. Save detection results as trees.json
                trees_path = os.path.join(ml_response_dir, "trees.json")
                with open(trees_path, 'w') as f:
                    json.dump(results, f)
                logger.info(f"Saved detection results to {trees_path}")
                
                # Get detection count from the new format
                detections = results.get('detections', [])
                detection_count = len(detections)
                
                # This method is called from detect_trees_from_map_view where user_lat, user_lng, etc. are defined
                # We need to ensure those parameters are passed correctly
                # We'll fix this in the detect_trees_from_map_view method
                
                metadata = {
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat(),
                    "image_path": image_path,
                    "bounds": bounds,
                    "mapping": mapping,
                    "detection_count": detection_count,
                    "source": "satellite",
                    "model_type": "grounded_sam" if self.ml_service and getattr(self.ml_service, 'use_grounded_sam', False) else "deepforest",
                    "coordinate_system": "s2" if "s2_cells" in mapping else "standard"
                    # Note: exact_coordinates will be added by the calling method
                }
                
                metadata_path = os.path.join(ml_response_dir, "metadata.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved metadata to {metadata_path}")
                
                # 3. Generate visualization image if possible
                try:
                    # Create a visualization of the detection results
                    visualization_path = os.path.join(ml_response_dir, "combined_visualization.jpg")
                    
                    # Use the model service to generate the visualization
                    if self.ml_service:
                        # Load the original image
                        with Image.open(image_path) as img:
                            image_array = np.array(img.convert('RGB'))
                        
                        # Generate visualization
                        vis_img_array = self.ml_service.generate_combined_visualization(image_array, results)
                        
                        # Convert back to PIL and save
                        vis_img = Image.fromarray(vis_img_array)
                        vis_img.save(visualization_path)
                        logger.info(f"Saved visualization to {visualization_path}")
                    else:
                        # Fallback to simple visualization if model service is not available
                        # Load the original image
                        with Image.open(image_path) as img:
                            vis_img = img.copy().convert('RGB')
                        
                        # Create a draw object
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(vis_img)
                        
                        # Draw bounding boxes for each detection
                        for i, detection in enumerate(detections):
                            bbox = detection.get('bbox', [0, 0, 0, 0])
                            confidence = detection.get('confidence', 0)
                            class_name = detection.get('class', 'unknown')
                            
                            # Convert normalized coordinates to pixel coordinates
                            width, height = vis_img.size
                            x1, y1, x2, y2 = [
                                int(bbox[0] * width),
                                int(bbox[1] * height),
                                int(bbox[2] * width),
                                int(bbox[3] * height)
                            ]
                            
                            # Draw rectangle with color based on confidence
                            # Higher confidence = more green, lower = more red
                            green = min(255, int(confidence * 255))
                            red = min(255, int((1 - confidence) * 255))
                            color = (red, green, 0)
                            
                            # Draw the bounding box
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            
                            # Add label with confidence
                            label = f"{class_name}: {confidence:.2f}"
                            draw.text((x1, y1-10), label, fill=color)
                        
                        # Save the visualization
                        vis_img.save(visualization_path)
                        logger.info(f"Saved visualization to {visualization_path}")
                    
                except Exception as e:
                    logger.error(f"Error creating visualization: {e}")
                    # Continue processing even if visualization fails
                
                # Extract detection results
                logger.info(f"Detected {detection_count} objects in {image_path}")
                
                # Convert results to Zarr format
                zarr_data = self._convert_detection_to_zarr_format(detections, mapping, job_id)
                
                if zarr_data is not None and self.storage_manager:
                    # Store in Zarr
                    self.storage_manager.store_detection(zarr_data)
                    logger.info(f"Stored detection results in Zarr: {job_id}")
                    
                # Return results
                return {
                    'success': True,
                    'message': f"Detected {detection_count} objects",
                    'job_id': job_id,
                    'ml_response_dir': ml_response_dir,
                    'trees': zarr_data['trees'] if zarr_data else [],
                    'model_type': metadata['model_type']
                }
                
            except Exception as e:
                logger.error(f"Error running ML detection: {e}", exc_info=True)
                return {
                    'success': False,
                    'message': f"ML detection error: {str(e)}",
                    'job_id': job_id,
                    'trees': []
                }
                
        except Exception as e:
            logger.error(f"Error in detect_trees_from_canvas_capture: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"Detection error: {str(e)}",
                'job_id': job_id,
                'trees': []
            }
    
    async def detect_trees_from_map_view(self, map_view_info, job_id):
        """
        Run tree detection based on map view information
        
        This function gets a satellite image using the current map center coordinates,
        then runs ML detection on it.
        
        Args:
            map_view_info: Map view information including center coordinates from single source of truth
            job_id: Job ID for tracking
            
        Returns:
            dict: Detection results
        """
        try:
            # Extract view data - We trust this data is accurate from the frontend
            view_data = map_view_info.get('viewData', {})
            
            logger.info(f"Running tree detection from map view. Job ID: {job_id}")
            logger.info(f"Map view info keys: {list(map_view_info.keys())}")
            logger.info(f"View data keys: {list(view_data.keys())}")
            
            # Log all available coordinates for debugging
            if 'center' in view_data:
                logger.info(f"CENTER coordinates: {view_data['center']}")
            if 'userCoordinates' in view_data:
                logger.info(f"USER coordinates: {view_data['userCoordinates']}")
            
            # STEP 1: CRITICAL - Use exact coordinates from global single source of truth
            # We DO NOT use defaults or fallbacks anywhere in this flow
            center_coords = view_data.get('center')
            if not center_coords or len(center_coords) != 2:
                logger.error("CRITICAL: Missing center coordinates from global single source of truth")
                return {
                    'success': False,
                    'message': "Missing valid map center coordinates. Please try again.",
                    'job_id': job_id,
                    'trees': []
                }
            
            # Extract exact coordinates and zoom directly from the view data
            user_lng, user_lat = center_coords
            user_zoom = view_data.get('zoom')
            if not user_zoom:
                logger.error("Missing zoom level in view_data")
                return {
                    'success': False, 
                    'message': "Missing zoom level. Please try again.",
                    'job_id': job_id,
                    'trees': []
                }
            
            # Removed default Dallas coordinates check as it was rejecting valid coordinates
            # Only log for debug purposes but don't block the detection
            is_default = abs(user_lng - (-96.78)) < 0.05 and abs(user_lat - 32.86) < 0.05
            if is_default:
                logger.info(f"Note: Coordinates are in Dallas area: lng={user_lng}, lat={user_lat}")
                # Continue with detection instead of returning an error
            
            logger.info(f"Using EXACT center coordinates: lng={user_lng}, lat={user_lat}, zoom={user_zoom}")
            logger.info(f"Coordinates source: global single source of truth")
            
            # STEP 2: Create directories - use job_id directly
            ml_dir = os.path.join(self.ml_dir, job_id)
            ml_response_dir = os.path.join(ml_dir, "ml_response")
            os.makedirs(ml_dir, exist_ok=True)
            os.makedirs(ml_response_dir, exist_ok=True)
            
            # STEP 3: Extract timestamp from job_id for satellite filename - SINGLE SOURCE OF TRUTH
            timestamp = job_id.split('_')[1]  # Use exact timestamp from job_id
            logger.info(f"Using timestamp {timestamp} from job_id {job_id} for satellite image")
            satellite_path = os.path.join(ml_dir, f"satellite_{timestamp}.jpg")
            
            # STEP 4: Get satellite image using GeolocationService
            # Pass view_data UNMODIFIED to maintain coordinate integrity
            logger.info("Getting satellite image with EXACT coordinates from frontend")
            success, image_data = self._get_satellite_image_from_gmaps(view_data, output_path=satellite_path)
            
            if not success or not image_data:
                logger.error(f"Failed to get satellite image for coordinates: {center_coords}")
                return {
                    'success': False,
                    'message': "Failed to get satellite image. Please check coordinates.",
                    'job_id': job_id,
                    'trees': []
                }
            
            # STEP 5: Get bounds from image data for ML detection
            bounds = image_data.get('bounds')
            if not bounds:
                logger.error("Missing bounds in image data")
                return {
                    'success': False,
                    'message': "Missing geographic bounds information. Cannot perform detection.",
                    'job_id': job_id,
                    'trees': []
                }
            
            logger.info(f"Using bounds: {bounds}")
            
            # STEP 6: Save coordinates metadata
            metadata = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "coordinates": {
                    "lng": user_lng,
                    "lat": user_lat,
                    "zoom": user_zoom
                },
                "source": "global_single_source_of_truth",
                "is_default_location": is_default
            }
            
            metadata_path = os.path.join(ml_response_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            # STEP 7: Run ML detection
            detection_results = await self.detect_trees_from_canvas_capture(
                satellite_path, bounds, job_id, ml_response_dir
            )
            
            # Get mapping from the detection results
            result_metadata = detection_results.get('metadata', {})
            has_s2_cells = 's2_cells' in result_metadata.get('mapping', {})
            
            # STEP 8: Add coordinates, S2 cell information, and paths to results
            detection_results.update({
                'coordinates': {
                    'center': [user_lng, user_lat],
                    'bounds': bounds,
                    'zoom': user_zoom
                },
                'ml_dir': ml_dir,
                'ml_response_dir': ml_response_dir,
                'imageUrl': image_data.get('url', ''),
                'coordinate_system': 's2' if has_s2_cells else 'standard',
                's2_cells': result_metadata.get('mapping', {}).get('s2_cells', None)
            })
            
            return detection_results
            
        except Exception as e:
            logger.error(f"Error in detect_trees_from_map_view: {e}", exc_info=True)
            return {
                'success': False,
                'message': f"Detection error: {str(e)}",
                'job_id': job_id,
                'trees': []
            }