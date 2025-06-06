"""
Geolocation Service - Handles all geographic coordinate processing and transformations
Provides utilities for working with map coordinates, bounds, and Google Maps Static API

This service is responsible for:
1. Coordinate formatting and validation
2. Converting between different coordinate formats
3. Calculating geographic bounds from center and zoom
4. Geographic transformations and projections
5. Handling Google Maps Static API parameters
"""

import os
import sys
import json
import logging
import math
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Union, Optional, Any

# Import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import GOOGLE_MAPS_API_KEY

# Set up logging
logger = logging.getLogger(__name__)

class GeolocationService:
    """Service to handle geographic coordinate processing and transformations"""
    
    def __init__(self):
        """Initialize the geolocation service"""
        self.api_key = GOOGLE_MAPS_API_KEY
        
        # Constants for Earth radius and calculations
        self.EARTH_RADIUS_METERS = 6378137.0  # Earth radius in meters at the equator
        self.EARTH_CIRCUMFERENCE = 2 * math.pi * self.EARTH_RADIUS_METERS
        self.DEGREES_TO_RADIANS = math.pi / 180.0
        self.RADIANS_TO_DEGREES = 180.0 / math.pi
        
        # Google Maps API limits
        self.MAX_MAP_WIDTH = 640
        self.MAX_MAP_HEIGHT = 640
        self.MAX_SCALE = 2  # Maximum scale factor for high-resolution images

    def format_coordinates_for_google_maps(self, coordinates: List[float]) -> str:
        """
        Format coordinates for Google Maps API (latitude,longitude format)
        
        Args:
            coordinates: Coordinates in [longitude, latitude] format
            
        Returns:
            str: Coordinates formatted as "latitude,longitude"
        """
        if not coordinates or len(coordinates) < 2:
            logger.warning("Invalid coordinates provided to format_coordinates_for_google_maps")
            raise ValueError("Invalid coordinates: both longitude and latitude must be provided")
            
        # Convert from [longitude, latitude] to "latitude,longitude"
        try:
            lng, lat = coordinates[0], coordinates[1]
            # Format to 6 decimal places to match what's shown in the UI
            formatted_lat = float(format(lat, '.6f'))
            formatted_lng = float(format(lng, '.6f'))
            return f"{formatted_lat},{formatted_lng}"
        except Exception as e:
            logger.error(f"Error formatting coordinates: {e}")
            raise ValueError(f"Failed to format coordinates: {e}")
    
    def parse_coordinates_from_string(self, coordinate_string: str) -> List[float]:
        """
        Parse coordinates from string format to [longitude, latitude] list
        
        Args:
            coordinate_string: Coordinates in "latitude,longitude" format
            
        Returns:
            List[float]: Coordinates in [longitude, latitude] format
        """
        if not coordinate_string or ',' not in coordinate_string:
            logger.warning(f"Invalid coordinate string: {coordinate_string}")
            return [0, 0]
            
        try:
            lat, lng = map(float, coordinate_string.split(','))
            return [lng, lat]
        except Exception as e:
            logger.error(f"Error parsing coordinate string '{coordinate_string}': {e}")
            return [0, 0]
    
    def validate_coordinates(self, coordinates: Union[List[float], str]) -> bool:
        """
        Validate that coordinates are well-formed and within valid ranges
        
        Args:
            coordinates: Coordinates as either [longitude, latitude] or "latitude,longitude"
            
        Returns:
            bool: True if coordinates are valid
        """
        try:
            # Convert to [lng, lat] format if string
            if isinstance(coordinates, str):
                lng, lat = self.parse_coordinates_from_string(coordinates)
            else:
                lng, lat = coordinates[0], coordinates[1]
                
            # Check ranges
            if not (-180 <= lng <= 180 and -90 <= lat <= 90):
                logger.warning(f"Coordinates out of range: lng={lng}, lat={lat}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating coordinates: {e}")
            return False
    
    def adjust_dimensions_for_api_limits(self, width: int, height: int) -> Tuple[int, int]:
        """
        Adjust dimensions to fit within API limits while preserving aspect ratio
        
        Args:
            width: Original width in pixels
            height: Original height in pixels
            
        Returns:
            Tuple[int, int]: Adjusted (width, height) in pixels
        """
        max_width = self.MAX_MAP_WIDTH
        max_height = self.MAX_MAP_HEIGHT
        
        # If dimensions are already within limits, return as is
        if width <= max_width and height <= max_height:
            return width, height
            
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Adjust based on which dimension exceeds limits more
        if width / max_width > height / max_height:
            # Width is the limiting factor
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            
        logger.info(f"Adjusted dimensions from {width}x{height} to {new_width}x{new_height}")
        return new_width, new_height
    
    def calculate_bounds_from_center_and_zoom(
        self, 
        center: List[float], 
        zoom: int, 
        width_px: int, 
        height_px: int
    ) -> List[List[float]]:
        """
        Calculate geographic bounds based on center coordinates, zoom level, and image dimensions
        
        Args:
            center: Center coordinates [longitude, latitude]
            zoom: Zoom level (0-21)
            width_px: Image width in pixels
            height_px: Image height in pixels
            
        Returns:
            List[List[float]]: Bounds as [[sw_lng, sw_lat], [ne_lng, ne_lat]]
        """
        # Validate inputs
        if not self.validate_coordinates(center):
            logger.warning("Invalid center coordinates, using defaults")
            center = [0, 0]
            
        zoom = max(0, min(21, zoom))  # Clamp zoom between 0-21
        
        # Extract center coordinates
        center_lng, center_lat = center
        
        # Convert latitude to radians
        lat_rad = center_lat * self.DEGREES_TO_RADIANS
        
        # Calculate meters per pixel at this zoom level
        meters_per_pixel = self.EARTH_CIRCUMFERENCE / (256 * 2**zoom)
        
        # Calculate width and height in meters
        width_meters = width_px * meters_per_pixel
        height_meters = height_px * meters_per_pixel
        
        # Calculate lat/lng degrees per meter (latitude-adjusted)
        lat_degrees_per_meter = self.RADIANS_TO_DEGREES / self.EARTH_RADIUS_METERS
        lng_degrees_per_meter = lat_degrees_per_meter / math.cos(lat_rad)
        
        # Calculate offsets in degrees
        lat_offset = height_meters * lat_degrees_per_meter / 2
        lng_offset = width_meters * lng_degrees_per_meter / 2
        
        # Calculate bounds
        sw_lat = center_lat - lat_offset
        sw_lng = center_lng - lng_offset
        ne_lat = center_lat + lat_offset
        ne_lng = center_lng + lng_offset
        
        # Clamp latitude to valid range (-90 to 90)
        sw_lat = max(-90, min(90, sw_lat))
        ne_lat = max(-90, min(90, ne_lat))
        
        # Normalize longitude to valid range (-180 to 180)
        sw_lng = ((sw_lng + 180) % 360) - 180
        ne_lng = ((ne_lng + 180) % 360) - 180
        
        return [[sw_lng, sw_lat], [ne_lng, ne_lat]]
    
    def estimate_zoom_level_from_bounds(self, bounds: List[List[float]]) -> int:
        """
        Estimate appropriate zoom level for given geographic bounds
        
        Args:
            bounds: Geographic bounds as [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            
        Returns:
            int: Estimated zoom level (0-21)
        """
        # Extract bounds
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        
        # Calculate center
        center_lat = (sw_lat + ne_lat) / 2
        
        # Calculate span in degrees
        lat_span = abs(ne_lat - sw_lat)
        lng_span = abs(ne_lng - sw_lng)
        
        # Convert latitude to radians for the calculation
        lat_rad = center_lat * self.DEGREES_TO_RADIANS
        
        # Calculate meters per degree at this latitude
        meters_per_lat_degree = self.EARTH_CIRCUMFERENCE / 360
        meters_per_lng_degree = meters_per_lat_degree * math.cos(lat_rad)
        
        # Calculate span in meters
        height_meters = lat_span * meters_per_lat_degree
        width_meters = lng_span * meters_per_lng_degree
        
        # Calculate required zoom level
        # Each zoom level halves the scale, so we need log2
        max_dimension_meters = max(width_meters, height_meters)
        zoom_level = math.log2(self.EARTH_CIRCUMFERENCE / (max_dimension_meters * 256))
        
        # Clamp to valid zoom levels
        return max(0, min(21, round(zoom_level)))
    
    def convert_pixel_to_geo_coordinates(
        self, 
        x: int, 
        y: int, 
        bounds: List[List[float]], 
        width_px: int, 
        height_px: int
    ) -> List[float]:
        """
        Convert pixel coordinates to geographic coordinates
        
        Args:
            x: X pixel coordinate (0 is left)
            y: Y pixel coordinate (0 is top)
            bounds: Geographic bounds as [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            width_px: Image width in pixels
            height_px: Image height in pixels
            
        Returns:
            List[float]: Geographic coordinates as [longitude, latitude]
        """
        # Extract bounds
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        
        # Calculate normalized coordinates (0-1)
        x_norm = x / width_px
        y_norm = y / height_px
        
        # Calculate interpolated coordinates
        # Y is inverted because pixel Y increases downward
        lng = sw_lng + x_norm * (ne_lng - sw_lng)
        lat = ne_lat - y_norm * (ne_lat - sw_lat)
        
        return [lng, lat]
    
    def convert_geo_to_pixel_coordinates(
        self, 
        lat: float, 
        lng: float, 
        bounds: List[List[float]], 
        width_px: int, 
        height_px: int
    ) -> Tuple[int, int]:
        """
        Convert geographic coordinates to pixel coordinates
        
        Args:
            lat: Latitude
            lng: Longitude
            bounds: Geographic bounds as [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            width_px: Image width in pixels
            height_px: Image height in pixels
            
        Returns:
            Tuple[int, int]: Pixel coordinates as (x, y)
        """
        # Extract bounds
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        
        # Calculate normalized coordinates (0-1)
        # Handle edge cases where bounds have zero span
        lng_span = ne_lng - sw_lng
        lat_span = ne_lat - sw_lat
        
        if abs(lng_span) < 1e-10:
            x_norm = 0.5
        else:
            x_norm = (lng - sw_lng) / lng_span
            
        if abs(lat_span) < 1e-10:
            y_norm = 0.5
        else:
            # Y is inverted because pixel Y increases downward
            y_norm = (ne_lat - lat) / lat_span
            
        # Convert to pixel coordinates
        x = int(x_norm * width_px)
        y = int(y_norm * height_px)
        
        return (x, y)
    
    def get_satellite_image_from_gmaps(
        self, 
        view_data: Dict[str, Any], 
        output_path: Optional[str] = None
    ) -> Tuple[bool, Any]:
        """
        Retrieve a satellite image from Google Maps Static API
        
        Args:
            view_data: Map view information including center, zoom, and dimensions
            output_path: Optional path to save the image
            
        Returns:
            Tuple[bool, Any]: (success, image_data) where image_data is either the PIL Image or 
                             a dict with imageUrl, mapWidth, mapHeight for the view_data
        """
        try:
            # Log all available information from view_data for debugging
            logger.info(f"View data keys: {list(view_data.keys())}")
            
            # STEP 1: Extract exact center coordinates - CRITICAL
            # We expect these to be from the global single source of truth
            center_coords = view_data.get('center')
            if not center_coords:
                error_msg = "CRITICAL ERROR: No center coordinates provided in view_data. Cannot proceed with detection."
                logger.error(error_msg)
                return False, {"error": error_msg}
            
            # Removed default Dallas coordinates check as it was rejecting valid coordinates
            # Only log for debug purposes but don't block the geolocation
            is_default = abs(center_coords[0] - (-96.78)) < 0.05 and abs(center_coords[1] - 32.86) < 0.05
            if is_default:
                logger.info(f"Note: Coordinates are in Dallas area: {center_coords}")
                # Continue with geolocation instead of returning an error
                
            # Format coordinates for Google Maps API (convert from [lng, lat] to "lat,lng")
            # We avoid any rounding here to maintain maximum precision
            center = f"{center_coords[1]},{center_coords[0]}"
            logger.info(f"Using EXACT center coordinates from global single source of truth: {center_coords}")
            
            # STEP 2: Extract zoom level - no defaults allowed
            zoom = view_data.get('zoom')
            if not zoom:
                error_msg = "ERROR: No zoom level provided in view_data."
                logger.error(error_msg)
                return False, {"error": error_msg}
                
            # STEP 3: Get container dimensions
            # Use default dimensions only if absolutely necessary for API
            width = view_data.get('containerWidth', 600)
            height = view_data.get('containerHeight', 400)
            logger.info(f"Using container dimensions: {width}x{height}")
            
            # Adjust dimensions to fit within API limits
            map_width, map_height = self.adjust_dimensions_for_api_limits(width, height)
            
            # Log what we're using
            logger.info(f"Using EXACT center: {center}, zoom: {zoom}, dimensions: {map_width}x{map_height}")
            logger.info(f"This matches what's shown in LocationInfo.jsx")
            
            # Set up parameters for the Google Maps Static API request
            params = {
                'center': center,
                'zoom': zoom,
                'size': f"{map_width}x{map_height}",
                'maptype': 'satellite',
                'key': self.api_key,
                'scale': 2  # Use high resolution
            }
            
            # Calculate bounds for the requested view
            bounds = self.calculate_bounds_from_center_and_zoom(
                center_coords, zoom, map_width, map_height
            )
            logger.info(f"Calculated bounds: {bounds}")
            
            # Make the request to Google Maps Static API
            logger.info(f"Requesting Google Maps Static API with params: {params}")
            response = requests.get('https://maps.googleapis.com/maps/api/staticmap', params=params)
            
            # Check if the request was successful
            if response.status_code != 200:
                logger.error(f"Google Maps API request failed: {response.status_code} - {response.text}")
                return False, {"error": f"API request failed: {response.status_code}"}
                
            # Process the image data
            from PIL import Image
            import io
            
            image_data = Image.open(io.BytesIO(response.content))
            
            # Save the image if output path is provided
            if output_path:
                # Convert the image to RGB mode if it's in palette mode (P)
                if image_data.mode == 'P':
                    image_data = image_data.convert('RGB')
                    logger.info("Converted image from palette mode (P) to RGB for JPEG compatibility")
                
                image_data.save(output_path)
                logger.info(f"Saved satellite image to {output_path}")
                
                # Save parameters to gsync_params.json in the same directory
                params_path = os.path.join(os.path.dirname(output_path), "gsync_params.json")
                params_data = {
                    "center": center_coords,
                    "zoom": zoom,
                    "size": f"{map_width}x{map_height}",
                    "maptype": "satellite",
                    "bounds": bounds,
                    "url": response.url,
                    "timestamp": datetime.now().isoformat()
                }
                
                try:
                    import json
                    with open(params_path, 'w') as f:
                        json.dump(params_data, f, indent=2)
                    logger.info(f"Saved satellite image parameters to {params_path}")
                except Exception as e:
                    logger.error(f"Error saving params file: {e}")
                
            # Ensure the image is in RGB mode for downstream processing
            if image_data.mode != 'RGB':
                image_data = image_data.convert('RGB')
                logger.info("Converted image to RGB mode for downstream processing")
                
            # Return success and the image data
            return True, {
                "image": image_data,
                "bounds": bounds,
                "center": center_coords,
                "zoom": zoom,
                "width": map_width,
                "height": map_height,
                "raw_params": params
            }
                
        except Exception as e:
            logger.error(f"Error retrieving satellite image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, {"error": str(e)}