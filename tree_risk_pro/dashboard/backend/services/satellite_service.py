"""
Satellite imagery service - retrieves satellite images for given coordinates
Uses Google Maps Static API to get satellite imagery directly
"""

import os
import requests
import logging
import time
import base64
from io import BytesIO
from PIL import Image
from config import TEMP_DIR

logger = logging.getLogger(__name__)

class SatelliteService:
    """Service for retrieving satellite imagery directly from Google Maps API"""
    
    def __init__(self):
        """Initialize the satellite imagery service"""
        # Get API key from config
        from config import GOOGLE_MAPS_API_KEY
        self.api_key = GOOGLE_MAPS_API_KEY
        
        # Cache to avoid duplicate requests
        self.image_cache = {}
        
    def get_satellite_image(self, center, zoom, width=640, height=640, heading=0, pitch=0, format='jpg', output_dir=None):
        """
        Retrieve a satellite image from Google Maps Static API
        
        Args:
            center (list): [longitude, latitude] coordinates
            zoom (int): Zoom level (0-21)
            width (int): Image width in pixels
            height (int): Image height in pixels
            heading (float): Camera heading in degrees
            pitch (float): Camera pitch in degrees (0 = straight down)
            format (str): Image format ('jpg' or 'png')
            output_dir (str): Optional output directory to save the image to (overrides TEMP_DIR)
            
        Returns:
            str: Path to the downloaded image
        """
        if not self.api_key:
            logger.error("Google Maps API key not configured")
            raise ValueError("Google Maps API key not configured")
        
        # Format center coordinates
        center_str = f"{center[1]},{center[0]}"  # Google Maps uses lat,lng order
        
        # Create a cache key to avoid duplicate requests
        cache_key = f"{center_str}_{zoom}_{width}_{height}_{heading}_{pitch}_{format}"
        
        # Check cache first
        if cache_key in self.image_cache:
            logger.info(f"Using cached satellite image for {center_str}, zoom {zoom}")
            return self.image_cache[cache_key]
        
        # Prepare API URL
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        
        # Build parameters
        params = {
            'center': center_str,
            'zoom': zoom,
            'size': f"{width}x{height}",
            'maptype': 'satellite',
            'key': self.api_key,
            'format': format,
        }
        
        # Add heading and pitch if specified (for 45° imagery where available)
        if heading != 0:
            params['heading'] = heading
        if pitch != 0:
            # Google uses positive values for pitch
            params['pitch'] = abs(pitch)
            
        # Make request
        try:
            logger.info(f"Retrieving satellite image for {center_str}, zoom {zoom}")
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Use the specified output directory or fall back to TEMP_DIR
            save_dir = output_dir if output_dir else TEMP_DIR
            
            # Create a unique filename
            timestamp = int(os.path.getmtime(save_dir)) if os.path.exists(save_dir) else int(time.time())
            filename = f"satellite_{center[1]}_{center[0]}_{zoom}_{timestamp}.{format}"
            filepath = os.path.join(save_dir, filename)
            
            # Ensure the directory exists
            os.makedirs(save_dir, exist_ok=True)
            
            # Save image to file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Cache the result
            self.image_cache[cache_key] = filepath
            
            logger.info(f"Satellite image saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error retrieving satellite image: {str(e)}")
            raise
            
    def get_satellite_image_base64(self, center, zoom, width=640, height=640, heading=0, pitch=0, format='jpg'):
        """
        Retrieve a satellite image and return as base64 string
        
        Args: Same as get_satellite_image
            
        Returns:
            str: Base64 encoded image data
        """
        try:
            # Get the image file
            image_path = self.get_satellite_image(center, zoom, width, height, heading, pitch, format)
            
            # Read the file and convert to base64
            with open(image_path, 'rb') as f:
                image_data = f.read()
                
            # Convert to base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/{format};base64,{base64_data}"
            
        except Exception as e:
            logger.error(f"Error getting base64 satellite image: {str(e)}")
            raise
            
    def convert_cesium_params(self, view_info):
        """
        Convert Cesium camera parameters to Google Maps parameters
        
        Args:
            view_info (dict): Cesium view parameters
                {
                    center: [longitude, latitude],
                    cameraHeight: float,
                    heading: float,
                    pitch: float,
                    zoom: int
                }
                
        Returns:
            dict: Parameters for Google Maps Static API
        """
        # Extract parameters
        view_data = view_info.get('viewData') or {}
        center = view_info.get('center') or view_data.get('center')
        if not center:
            raise ValueError("Missing center coordinates in view info")
            
        # Extract zoom level - default to 18 for high detail if not provided
        zoom = view_info.get('zoom') or view_data.get('zoom') or 18
        
        # Extract heading/pitch, defaulting to 0
        heading = view_info.get('heading') or view_data.get('heading') or 0
        pitch = view_info.get('tilt') or view_data.get('tilt') or 0
        
        # For Cesium's birdseye view, adjust zoom slightly
        if pitch < -45:  # If looking straight down (Cesium uses negative pitch)
            # Reduce zoom slightly to get more context
            zoom = max(1, zoom - 1)
            # Force pitch to 0 (straight down) since that's most similar to satellite view
            pitch = 0
            
        return {
            'center': center,
            'zoom': zoom,
            'heading': heading,
            'pitch': pitch
        }