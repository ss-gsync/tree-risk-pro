"""
External ML Model Service Client for T4 GPU Server Integration

This service provides a client interface to the Grounded-SAM model running on a T4 GPU instance.
It maintains API compatibility with the in-memory MLModelService while offloading the actual
computation to a dedicated server.
"""

import os
import sys
import json
import time
import logging
import traceback
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Dict, List, Any, Optional, Union
import threading

# Add backend directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import MODEL_SERVER_URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExternalModelService:
    """Client service for the external T4 model server."""
    
    def __init__(self, server_url: str = None, use_gpu: bool = True):
        """
        Initialize the External Model Service client.
        
        Args:
            server_url: URL of the T4 model server (overrides config)
            use_gpu: Not used but kept for API compatibility
        """
        self.server_url = server_url or MODEL_SERVER_URL
        self.models_loaded = False
        self.loading_error = None
        self.health_check_interval = 30  # seconds
        self.last_health_check = 0
        self.health_status = {}
        
        # Performance tracking
        self.last_inference_time = 0
        self.average_inference_time = 0
        self.inference_count = 0
        
        # Flags for API compatibility
        self.use_grounded_sam = True  # T4 server always uses Grounded-SAM
        
        # Check server health in background
        threading.Thread(target=self._check_server_health, daemon=True).start()
    
    def _check_server_health(self):
        """Check server health and update status."""
        try:
            logger.info(f"Checking health of T4 model server at {self.server_url}")
            
            # First try the /health endpoint (new API)
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                
                if response.status_code == 200:
                    self.health_status = response.json()
                    self.models_loaded = self.health_status.get("models_loaded", False) or self.health_status.get("model_initialized", False)
                    logger.info(f"T4 model server health check (new API): {self.health_status}")
                    
                    if self.models_loaded:
                        logger.info("T4 model server is healthy and models are loaded")
                    else:
                        logger.warning("T4 model server is healthy but models are not loaded")
                    return  # Success, no need to try alternative endpoint
            except Exception as health_error:
                logger.warning(f"New health endpoint not available, trying status endpoint: {str(health_error)}")
            
            # Fall back to /status endpoint (old API)
            response = requests.get(f"{self.server_url}/status", timeout=5)
            
            if response.status_code == 200:
                self.health_status = response.json()
                self.models_loaded = self.health_status.get("model_initialized", False)
                logger.info(f"T4 model server status check (fallback): {self.health_status}")
                
                if self.models_loaded:
                    logger.info("T4 model server is healthy and models are loaded (via status endpoint)")
                else:
                    logger.warning("T4 model server is available but models are not loaded (via status endpoint)")
            else:
                logger.error(f"T4 model server health/status check failed: {response.status_code}")
                self.models_loaded = False
                self.loading_error = f"Health check failed: {response.status_code}"
        except Exception as e:
            logger.error(f"Error checking T4 model server health: {str(e)}")
            self.models_loaded = False
            self.loading_error = str(e)
    
    def wait_for_models(self, timeout: int = 60) -> bool:
        """
        Wait for models to load with a timeout.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        start_time = time.time()
        
        # Force health check immediately
        self._check_server_health()
        
        while not self.models_loaded and (time.time() - start_time) < timeout:
            logger.info(f"Waiting for T4 model server models to load ({int(time.time() - start_time)}s)")
            time.sleep(5)
            self._check_server_health()
            
        return self.models_loaded
    
    def detect_trees(self, 
                    image: np.ndarray, 
                    confidence_threshold: float = 0.5,
                    with_segmentation: bool = True,
                    job_id: str = None) -> Dict[str, Any]:
        """
        Detect trees in an image using the external T4 model server.
        
        Args:
            image: Image as numpy array (RGB)
            confidence_threshold: Minimum confidence for detections
            with_segmentation: Whether to add segmentation masks
            job_id: Optional job ID for tracking
            
        Returns:
            Dict containing detection results
        """
        if not self.models_loaded:
            # Check health if needed
            current_time = time.time()
            if current_time - self.last_health_check > self.health_check_interval:
                self._check_server_health()
                self.last_health_check = current_time
                
            if not self.models_loaded:
                logger.error("T4 model server is not available")
                return {
                    'success': False,
                    'error': "T4 model server is not available",
                    'detections': []
                }
        
        start_time = time.time()
        
        try:
            # Convert image to JPEG bytes
            img = Image.fromarray(image)
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            # Create multipart form data
            files = {
                'image': ('image.jpg', img_byte_arr, 'image/jpeg')
            }
            
            # Create form data
            data = {
                'job_id': job_id or f"detection_{int(time.time() * 1000)}",
                'save_results': 'true'
            }
            
            # Send request to server
            logger.info(f"Sending detection request to T4 model server for job {data['job_id']}")
            response = requests.post(
                f"{self.server_url}/detect",
                files=files,
                data=data,
                timeout=60  # Longer timeout for inference
            )
            
            # Process response
            if response.status_code == 200:
                result = response.json()
                
                # Update performance metrics
                self.last_inference_time = time.time() - start_time
                self.inference_count += 1
                self.average_inference_time = (
                    (self.average_inference_time * (self.inference_count - 1) + self.last_inference_time) 
                    / self.inference_count
                )
                
                # Log result summary
                detection_count = len(result.get('detections', []))
                logger.info(f"T4 model server detected {detection_count} objects in {self.last_inference_time:.2f}s")
                
                # Add success flag if not present
                if 'success' not in result:
                    result['success'] = True
                    
                # Add execution time
                result['execution_time'] = self.last_inference_time
                
                return result
            else:
                error_msg = f"T4 model server error: {response.status_code}"
                try:
                    error_detail = response.json().get('error', 'Unknown error')
                    error_msg = f"{error_msg} - {error_detail}"
                except:
                    pass
                
                logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'detections': [],
                    'execution_time': time.time() - start_time
                }
                
        except Exception as e:
            error_msg = f"Error sending request to T4 model server: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': error_msg,
                'detections': [],
                'execution_time': time.time() - start_time
            }
    
    def generate_combined_visualization(self, 
                                      image: np.ndarray, 
                                      detection_results: Dict[str, Any]) -> np.ndarray:
        """
        Generate visualization of detection results - uses local rendering to avoid
        sending the image back to the server.
        
        Args:
            image: Original image as numpy array (RGB)
            detection_results: Results from detect_trees method
            
        Returns:
            Visualization image as numpy array (RGB)
        """
        try:
            # Import required modules
            import cv2
            from PIL import Image, ImageDraw
            
            # Convert image to PIL for easier manipulation
            pil_image = Image.fromarray(image).convert("RGBA")
            
            # Create a transparent overlay for masks
            overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Class colors
            class_colors = {
                "tree": (22, 163, 74, 128),        # Green
                "healthy tree": (22, 163, 74, 128), # Green
                "hazardous tree": (139, 92, 246, 128),  # Purple
                "dead tree": (107, 114, 128, 128),      # Gray
                "low canopy tree": (14, 165, 233, 128), # Light Blue
                "pest disease tree": (132, 204, 22, 128), # Lime Green
                "flood prone tree": (8, 145, 178, 128), # Teal
                "utility conflict tree": (59, 130, 246, 128), # Blue
                "structural hazard tree": (13, 148, 136, 128), # Teal Green
                "fire risk tree": (79, 70, 229, 128),   # Indigo
                "default": (200, 200, 200, 128)         # Gray
            }
            
            height, width = image.shape[:2]
            
            # Process each detection
            for i, detection in enumerate(detection_results.get('detections', [])):
                # Get normalized box and convert to pixel coordinates
                bbox = detection.get('bbox', [0, 0, 0, 0])
                x_min, y_min, x_max, y_max = [
                    int(bbox[0] * width),
                    int(bbox[1] * height),
                    int(bbox[2] * width),
                    int(bbox[3] * height)
                ]
                
                # Get class and confidence
                class_name = detection.get('class', 'tree')
                confidence = detection.get('confidence', 0.0)
                
                # Get color
                color = class_colors.get(class_name, class_colors['default'])
                
                # Draw box
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color[:3] + (200,), width=3)
                
                # Draw label
                label = f"{class_name}: {confidence:.2f}"
                draw.text((x_min, y_min-15), label, fill=color[:3] + (255,))
                
                # Draw mask if available
                if 'segmentation' in detection:
                    mask_array = np.array(detection['segmentation'])
                    mask_color = color
                    
                    # Ensure mask is right size
                    if mask_array.shape[0] != height or mask_array.shape[1] != width:
                        # Resize mask
                        from PIL import Image
                        mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))
                        mask_img = mask_img.resize((width, height))
                        mask_array = np.array(mask_img) > 0
                    
                    # Draw mask on overlay
                    for y in range(0, height, 2):  # Sample every 2 pixels for speed
                        for x in range(0, width, 2):
                            if y < mask_array.shape[0] and x < mask_array.shape[1] and mask_array[y, x]:
                                overlay.putpixel((x, y), mask_color)
            
            # Combine image with overlay
            result = Image.alpha_composite(pil_image, overlay)
            result = result.convert("RGB")
            
            # Convert back to numpy array
            return np.array(result)
        
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            logger.error(traceback.format_exc())
            return image  # Return original image if visualization fails
    
    def get_model_status(self) -> Dict[str, Any]:
        """Return the current status of the T4 model server."""
        # Update health status if needed
        current_time = time.time()
        if current_time - self.last_health_check > self.health_check_interval:
            self._check_server_health()
            self.last_health_check = current_time
        
        status = {
            "models_loaded": self.models_loaded,
            "loading_error": self.loading_error,
            "server_url": self.server_url,
            "average_inference_time": self.average_inference_time,
            "inference_count": self.inference_count,
            "server_health": self.health_status,
            "using_grounded_sam": True,
            "cuda_available": self.health_status.get("cuda_available", False),
            "device": self.health_status.get("device", "unknown"),
            "model_type": "External Grounded-SAM (T4)"
        }
        
        return status


# Create a global instance for singleton-like access
_external_model_service_instance = None

def get_external_model_service(server_url: str = None) -> ExternalModelService:
    """
    Get or create the global external model service instance.
    
    Args:
        server_url: URL of the T4 model server (overrides config)
        
    Returns:
        ExternalModelService instance
    """
    global _external_model_service_instance
    
    if _external_model_service_instance is None:
        logger.info(f"Creating new ExternalModelService instance for T4 server")
        _external_model_service_instance = ExternalModelService(server_url=server_url)
    else:
        logger.info("Reusing existing ExternalModelService instance")
    
    return _external_model_service_instance