"""ML service package for model inference using any available GPU."""

import os
import sys
import json
import time
import logging
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import traceback
from typing import Dict, List, Any, Optional

# Add backend directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import MODEL_SERVER_URL

# Configure logging
logger = logging.getLogger(__name__)

class ModelService:
    """
    GPU-accelerated model service for tree detection.
    
    This service can use any available GPU for model inference,
    supporting both local and remote GPU resources without requiring
    specific hardware.
    """
    
    def __init__(self, server_url: str = None):
        """
        Initialize the model service.
        
        Args:
            server_url: URL of the model server (defaults to config value)
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
        
        # Flags
        self.use_grounded_sam = True
        
        # Check server health in background
        logger.info(f"Initializing GPU-accelerated ModelService with URL: {self.server_url}")
        self._check_server_health()
    
    def _check_server_health(self):
        """Check server health and update status."""
        try:
            logger.info(f"Checking health of GPU-accelerated model server at {self.server_url}")
            
            # Try the /health endpoint (primary API)
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                
                if response.status_code == 200:
                    self.health_status = response.json()
                    # Use the 'ready' flag to indicate if the server is ready
                    self.models_loaded = self.health_status.get("ready", False)
                    logger.info(f"Model server health check: {self.health_status}")
                    
                    if self.models_loaded:
                        logger.info("Model server is healthy and ready for requests")
                    else:
                        logger.warning("Model server is healthy but not ready for requests")
                    return  # Success, no need to try alternative endpoint
            except Exception as health_error:
                logger.warning(f"Health endpoint not available, trying status endpoint: {str(health_error)}")
            
            # Fall back to /status endpoint (alternative endpoint)
            response = requests.get(f"{self.server_url}/status", timeout=5)
            
            if response.status_code == 200:
                self.health_status = response.json()
                # Status endpoint also uses the 'ready' flag
                self.models_loaded = self.health_status.get("ready", False)
                logger.info(f"Model server status check: {self.health_status}")
                
                if self.models_loaded:
                    logger.info("Model server is ready for requests (via status endpoint)")
                else:
                    logger.warning("Model server is available but not ready for requests (via status endpoint)")
            else:
                logger.error(f"Model server health/status check failed: {response.status_code}")
                self.models_loaded = False
                self.loading_error = f"Health check failed: {response.status_code}"
        except Exception as e:
            logger.error(f"Error checking model server health: {str(e)}")
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
            logger.info(f"Waiting for model server models to load ({int(time.time() - start_time)}s)")
            time.sleep(5)
            self._check_server_health()
            
        return self.models_loaded
    
    def detect_trees(self, 
                    image: np.ndarray, 
                    confidence_threshold: float = 0.5,
                    with_segmentation: bool = True,
                    job_id: str = None) -> Dict[str, Any]:
        """
        Detect trees in an image using the model server.
        
        Args:
            image: Image as numpy array (RGB)
            confidence_threshold: Minimum confidence for detections
            with_segmentation: Whether to include segmentation masks
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
                logger.error("Model server is not available")
                return {
                    'success': False,
                    'error': "Model server is not available",
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
                'confidence_threshold': str(confidence_threshold),
                'with_segmentation': 'true' if with_segmentation else 'false'
            }
            
            # Send request to server
            logger.info(f"Sending detection request to model server for job {data['job_id']}")
            response = requests.post(
                f"{self.server_url}/detect",
                files=files,
                data=data,
                timeout=60  # Longer timeout for inference
            )
            
            # Process response
            if response.status_code == 200:
                response_data = response.json()
                
                # Update performance metrics
                self.last_inference_time = time.time() - start_time
                self.inference_count += 1
                self.average_inference_time = (
                    (self.average_inference_time * (self.inference_count - 1) + self.last_inference_time) 
                    / self.inference_count
                )
                
                # Check if we need to read the detection results from the output files
                if 'output_dir' in response_data and 'files_created' in response_data:
                    try:
                        output_dir = response_data.get('output_dir')
                        trees_json_path = os.path.join(output_dir, "trees.json")
                        
                        if os.path.exists(trees_json_path):
                            # Read the complete detection results from the file
                            with open(trees_json_path, 'r') as f:
                                file_data = json.load(f)
                                
                                # Include full detection data and metadata
                                if 'detection_result' in file_data:
                                    result = file_data['detection_result']
                                else:
                                    # Sometimes trees.json contains the result directly
                                    result = file_data
                                
                                # Add execution time
                                result['execution_time'] = self.last_inference_time
                                result['job_id'] = data['job_id']
                                
                                # Log success
                                detection_count = len(result.get('detections', []))
                                logger.info(f"Read {detection_count} detections from {trees_json_path}")
                                return result
                    except Exception as read_error:
                        logger.error(f"Error reading detection results from file: {read_error}")
                        # Continue with response data
                
                # Fall back to response data if we couldn't read from file
                result = response_data
                
                # Add success flag if not present
                if 'success' not in result:
                    result['success'] = True
                    
                # Add execution time
                result['execution_time'] = self.last_inference_time
                
                # Ensure we have an empty detections array if none provided
                if 'detections' not in result:
                    result['detections'] = []
                
                # Log result summary
                detection_count = len(result.get('detections', []))
                logger.info(f"Model server detected {detection_count} objects in {self.last_inference_time:.2f}s")
                
                return result
            else:
                error_msg = f"Model server error: {response.status_code}"
                try:
                    error_data = response.json()
                    # Get detailed error information
                    if 'detail' in error_data:
                        error_detail = error_data['detail']
                        error_msg = f"{error_msg} - {error_detail}"
                    else:
                        error_detail = error_data.get('error', 'Unknown error')
                        error_msg = f"{error_msg} - {error_detail}"
                    
                    # Log the full error response for debugging
                    logger.error(f"Full error response: {json.dumps(error_data)}")
                except Exception as parse_error:
                    logger.error(f"Failed to parse error response: {str(parse_error)}")
                    try:
                        # Try to get the text response if JSON parsing failed
                        error_msg = f"{error_msg} - {response.text[:200]}"
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
            error_msg = f"Error sending request to model server: {str(e)}"
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
        Generate visualization of detection results.
        
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
                "building": (0, 0, 255, 128),  # Blue
                "power line": (255, 0, 0, 128), # Red
                "default": (200, 200, 200, 128)         # Gray
            }
            
            height, width = image.shape[:2]
            
            # Process each detection
            for i, detection in enumerate(detection_results.get('detections', [])):
                # Get normalized box and convert to pixel coordinates
                # bbox format is [x, y, width, height] in normalized coordinates
                bbox = detection.get('bbox', [0, 0, 0, 0])
                x_min = int(bbox[0] * width)
                y_min = int(bbox[1] * height)
                x_max = int((bbox[0] + bbox[2]) * width)  # x + width
                y_max = int((bbox[1] + bbox[3]) * height) # y + height
                logger.info(f"Drawing bbox with coordinates: {x_min},{y_min},{x_max},{y_max} from {bbox}")
                
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
        """Return the current status of the model server."""
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
            "model_type": "GPU-Accelerated Grounded-SAM"
        }
        
        return status


# Global instance for singleton-like access
_model_service_instance = None

def get_model_service(use_gpu=True, use_grounded_sam=True):
    """
    Get or create the global model service instance.
    
    Args:
        use_gpu: Whether to use GPU for inference if available (kept for API compatibility)
        use_grounded_sam: Whether to use Grounded SAM (kept for API compatibility)
        
    Returns:
        ModelService instance
    """
    global _model_service_instance
    
    if _model_service_instance is None:
        logger.info(f"Creating new ModelService instance")
        _model_service_instance = ModelService()
    else:
        logger.info("Reusing existing ModelService instance")
    
    return _model_service_instance

# Export the service
__all__ = ['ModelService', 'get_model_service']