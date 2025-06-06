"""
ML Model Service for in-memory ML models - supports both DeepForest/SAM and Grounded SAM.

This service loads ML models once during initialization and keeps them in memory
for fast inference without having to reload models for each request.
"""

import os
import time
import logging
import traceback
import numpy as np
import threading
import torch
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to track model status
_models_loaded = False
_loading_lock = threading.Lock()

class MLModelService:
    """Service that maintains ML models in memory for fast inference."""
    
    def __init__(self, use_gpu: bool = True, use_grounded_sam: bool = True):
        """
        Initialize the ML Model Service.
        
        Args:
            use_gpu: Whether to use GPU for inference if available
            use_grounded_sam: Whether to use Grounded SAM instead of DeepForest
        """
        self.use_gpu = use_gpu
        self.use_grounded_sam = use_grounded_sam
        
        # DeepForest + SAM models
        self.deepforest_model = None
        self.sam_model = None
        self.sam_predictor = None
        self.detection_model = None
        self.model_manager = None
        self.inference_engine = None
        
        # Grounded SAM models
        self.groundingdino_model = None
        
        # Track model status
        self.models_loaded = False
        self.loading_error = None
        
        # Device settings
        self.device = None
        self.cuda_available = False
        
        # Performance tracking
        self.last_inference_time = 0
        self.average_inference_time = 0
        self.inference_count = 0
        
        # Initialize models in a separate thread to avoid blocking
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models in a background thread."""
        threading.Thread(target=self._load_models, daemon=True).start()
    
    def _load_models(self):
        """Load ML models with proper device management and progressive loading."""
        global _models_loaded
        
        # Use lock to prevent multiple threads from loading models simultaneously
        with _loading_lock:
            if _models_loaded:
                logger.info("Models already loaded by another instance")
                self.models_loaded = True
                return
                
            try:
                # Log start of model loading
                logger.info("Starting model loading in background thread - this may take a minute")
                
                # Import ML libraries
                import torch
                import sys
                
                # Add root directory to Python path to fix imports
                sys.path.append('/ttt')
                
                # Check CUDA availability
                self.cuda_available = torch.cuda.is_available()
                self.device = torch.device("cuda" if self.cuda_available and self.use_gpu else "cpu")
                logger.info(f"Using device: {self.device}")
                
                # Set initial loading progress state
                self.loading_progress = {
                    "status": "initializing",
                    "progress": 0,
                    "message": "Starting model initialization"
                }
                
                # Create a progressive loading sequence
                loading_steps = []
                
                if self.use_grounded_sam:
                    # Split Grounded SAM loading into multiple steps
                    loading_steps = [
                        {
                            "function": self._load_grounding_dino_model,
                            "message": "Loading Grounding DINO model",
                            "progress": 40
                        },
                        {
                            "function": self._load_sam_model,
                            "message": "Loading Segment Anything (SAM) model",
                            "progress": 80
                        }
                    ]
                else:
                    # Split DeepForest + SAM loading into multiple steps
                    loading_steps = [
                        {
                            "function": self._load_deepforest_model,
                            "message": "Loading DeepForest model",
                            "progress": 40
                        },
                        {
                            "function": self._load_sam_for_deepforest,
                            "message": "Loading SAM model for DeepForest",
                            "progress": 80
                        }
                    ]
                
                # Execute each loading step sequentially
                for step in loading_steps:
                    self.loading_progress = {
                        "status": "loading",
                        "progress": step["progress"],
                        "message": step["message"]
                    }
                    logger.info(f"Model loading: {step['message']} ({step['progress']}%)")
                    
                    try:
                        # Execute the loading function
                        step["function"]()
                    except Exception as step_error:
                        logger.error(f"Error during {step['message']}: {step_error}")
                        logger.error(traceback.format_exc())
                        # Continue to next step - models might be able to partially function
                
                # Mark models as loaded - even if some steps failed, 
                # the system can operate with partially loaded models
                self.models_loaded = True
                global _models_initialized
                _models_loaded = True
                _models_initialized = True
                
                self.loading_progress = {
                    "status": "complete",
                    "progress": 100,
                    "message": "ML models initialized and ready"
                }
                
                logger.info("ML models initialized and marked as loaded globally")
                
            except Exception as e:
                error_msg = f"Error loading ML models: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                self.loading_error = error_msg
                self.models_loaded = False
                
                self.loading_progress = {
                    "status": "error",
                    "progress": 0,
                    "message": f"Error loading models: {str(e)}"
                }
    
    def _load_grounding_dino_model(self):
        """Load just the Grounding DINO model component."""
        try:
            # Import and initialize transformers components
            from transformers import pipeline
            
            # Load Grounding DINO model
            logger.info("Loading Grounding DINO model...")
            
            # Use tiny model for faster inference
            model_id = "IDEA-Research/grounding-dino-tiny"
            device_str = str(self.device)
            
            self.groundingdino_model = pipeline(
                "zero-shot-object-detection",
                model=model_id,
                device=device_str if self.cuda_available else "cpu"
            )
            logger.info(f"Grounding DINO model loaded successfully using {model_id}")
        except Exception as e:
            logger.error(f"Error loading Grounding DINO model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_sam_model(self):
        """Load just the SAM model component."""
        try:
            # Import SAM components
            from segment_anything import sam_model_registry, SamPredictor
            
            # Find SAM checkpoint
            sam_checkpoint_paths = [
                "/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth",
                "/ttt/tree_ml/pipeline/grounded-sam/weights/sam_vit_h_4b8939.pth",
                "/ttt/tree_ml/pipeline/grounded-sam/segment_anything/weights/sam_vit_h_4b8939.pth",
                "/ttt/tree_ml/pipeline/model/sam_vit_l_0b3195.pth",
                "/ttt/tree_ml/pipeline/model/sam_vit_b_01ec64.pth"
            ]
            
            sam_checkpoint = None
            for path in sam_checkpoint_paths:
                if os.path.exists(path):
                    sam_checkpoint = path
                    break
            
            if not sam_checkpoint:
                raise FileNotFoundError("SAM checkpoint not found")
            
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
            sam.to(device=self.device)
            
            # Initialize predictor
            self.sam_predictor = SamPredictor(sam)
            self.sam_model = sam
            
            logger.info(f"SAM model ({model_type}) loaded successfully")
        except Exception as e:
            logger.error(f"Error loading SAM model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_deepforest_model(self):
        """Load just the DeepForest model component."""
        try:
            from tree_ml.pipeline.object_recognition import (
                TreeDetectionModel, 
                ModelManager,
                InferenceEngine,
                MLConfig
            )
            
            # Create config object for TreeDetectionModel with GPU settings
            config = MLConfig()
            config.gpu_device_id = 0  # Use first GPU
            
            # Log CUDA availability and device selection
            logger.info(f"CUDA available: {self.cuda_available}, using GPU device: {config.gpu_device_id}")
            
            # Initialize TreeDetectionModel with config
            self.detection_model = TreeDetectionModel(config=config)
            
            # CRITICAL: Ensure models are in eval mode, which fixes GPU compatibility issues
            if hasattr(self.detection_model, 'model') and hasattr(self.detection_model.model, 'model'):
                logger.info("Setting DeepForest model to eval mode")
                self.detection_model.model.model.eval()
                
                # Verify model is in eval mode and on the correct device
                model_device = next(self.detection_model.model.model.parameters()).device
                logger.info(f"DeepForest model is on device: {model_device}")
                
                # Check if model is using the expected device
                using_correct_device = 'cuda' in str(model_device) if self.cuda_available else 'cpu' in str(model_device)
                if not using_correct_device:
                    logger.warning(f"DeepForest model is on {model_device} but should be on {'CUDA' if self.cuda_available else 'CPU'}")
                    if self.cuda_available:
                        try:
                            # Try to move model to CUDA again
                            logger.info("Attempting to move DeepForest model to CUDA...")
                            self.detection_model.model.model.to("cuda:0")
                            new_device = next(self.detection_model.model.model.parameters()).device
                            logger.info(f"DeepForest model is now on device: {new_device}")
                        except Exception as e:
                            logger.error(f"Failed to move DeepForest model to CUDA: {e}")
            
            # Initialize model manager with proper parameters
            self.model_manager = ModelManager(config=config)
            self.model_manager.model = self.detection_model
            
            # Create inference engine
            self.inference_engine = InferenceEngine(config=config)
            self.inference_engine.model_manager = self.model_manager
            
            # Get direct reference to the underlying model
            self.deepforest_model = self.detection_model.model
            
            logger.info("DeepForest model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading DeepForest model: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_sam_for_deepforest(self):
        """Load SAM model for DeepForest integration."""
        try:
            # Get SAM model from the detection model if it's already loaded
            if hasattr(self.detection_model, 'sam_model'):
                self.sam_predictor = self.detection_model.sam_model
                if self.sam_predictor:
                    self.sam_model = self.sam_predictor.model
                    
                    # Check SAM model device
                    if self.sam_model:
                        sam_device = next(self.sam_model.model.parameters()).device
                        logger.info(f"SAM model is on device: {sam_device}")
                        
                        # It's OK if SAM is on CPU while DeepForest is on GPU
                        if 'cpu' in str(sam_device) and self.cuda_available:
                            logger.info("Using mixed precision: DeepForest on GPU, SAM on CPU for compatibility")
                            logger.info("This won't affect tree detection performance, only segmentation")
                
                logger.info("SAM model loaded from DeepForest integration")
            else:
                logger.warning("SAM model not available in DeepForest integration")
        except Exception as e:
            logger.error(f"Error loading SAM model for DeepForest: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_grounded_sam_models(self):
        """Load Grounded SAM models."""
        try:
            start_time = time.time()
            
            # Import and initialize transformers components
            from transformers import pipeline
            
            # Load Grounding DINO model
            logger.info("Loading Grounding DINO model...")
            try:
                # Use tiny model for faster inference
                model_id = "IDEA-Research/grounding-dino-tiny"
                device_str = str(self.device)
                
                self.groundingdino_model = pipeline(
                    "zero-shot-object-detection",
                    model=model_id,
                    device=device_str if self.cuda_available else "cpu"
                )
                logger.info(f"Grounding DINO model loaded successfully using {model_id}")
            except Exception as e:
                logger.error(f"Error loading Grounding DINO model: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Load SAM model
            logger.info("Loading SAM model...")
            try:
                # Import SAM components
                from segment_anything import sam_model_registry, SamPredictor
                
                # Find SAM checkpoint
                sam_checkpoint_paths = [
                    "/ttt/tree_ml/pipeline/model/sam_vit_h_4b8939.pth",
                    "/ttt/tree_ml/pipeline/grounded-sam/weights/sam_vit_h_4b8939.pth",
                    "/ttt/tree_ml/pipeline/grounded-sam/segment_anything/weights/sam_vit_h_4b8939.pth",
                    "/ttt/tree_ml/pipeline/model/sam_vit_l_0b3195.pth",
                    "/ttt/tree_ml/pipeline/model/sam_vit_b_01ec64.pth"
                ]
                
                sam_checkpoint = None
                for path in sam_checkpoint_paths:
                    if os.path.exists(path):
                        sam_checkpoint = path
                        break
                
                if not sam_checkpoint:
                    raise FileNotFoundError("SAM checkpoint not found")
                
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
                sam.to(device=self.device)
                
                # Initialize predictor
                self.sam_predictor = SamPredictor(sam)
                self.sam_model = sam
                
                logger.info(f"SAM model ({model_type}) loaded successfully")
            except Exception as e:
                logger.error(f"Error loading SAM model: {e}")
                logger.error(traceback.format_exc())
                raise
            
            load_time = time.time() - start_time
            logger.info(f"Grounded SAM models loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading Grounded SAM models: {e}")
            raise
    
    def _load_deepforest_sam_models(self):
        """Load DeepForest and SAM models."""
        try:
            from tree_ml.pipeline.object_recognition import (
                TreeDetectionModel, 
                ModelManager,
                InferenceEngine
            )
            
            # Initialize the detection model (includes DeepForest and SAM)
            start_time = time.time()
            
            # Initialize TreeDetectionModel which loads both DeepForest and SAM
            logger.info("Loading ML models (DeepForest and SAM)...")
            
            # Import required modules
            from tree_ml.pipeline.object_recognition import MLConfig
            
            # Create config object for TreeDetectionModel with GPU settings
            config = MLConfig()
            config.gpu_device_id = 0  # Use first GPU
            
            # Log CUDA availability and device selection
            logger.info(f"CUDA available: {self.cuda_available}, using GPU device: {config.gpu_device_id}")
            
            # Initialize TreeDetectionModel with config
            self.detection_model = TreeDetectionModel(config=config)
            
            # CRITICAL: Ensure models are in eval mode, which fixes GPU compatibility issues
            if hasattr(self.detection_model, 'model') and hasattr(self.detection_model.model, 'model'):
                logger.info("Setting DeepForest model to eval mode")
                self.detection_model.model.model.eval()
                
                # Verify model is in eval mode and on the correct device
                model_device = next(self.detection_model.model.model.parameters()).device
                logger.info(f"DeepForest model is on device: {model_device}")
                
                # Check if model is using the expected device
                using_correct_device = 'cuda' in str(model_device) if self.cuda_available else 'cpu' in str(model_device)
                if not using_correct_device:
                    logger.warning(f"DeepForest model is on {model_device} but should be on {'CUDA' if self.cuda_available else 'CPU'}")
                    if self.cuda_available:
                        try:
                            # Try to move model to CUDA again
                            logger.info("Attempting to move DeepForest model to CUDA...")
                            self.detection_model.model.model.to("cuda:0")
                            new_device = next(self.detection_model.model.model.parameters()).device
                            logger.info(f"DeepForest model is now on device: {new_device}")
                        except Exception as e:
                            logger.error(f"Failed to move DeepForest model to CUDA: {e}")
                
                # Also check SAM model separately - it might be on CPU even if DeepForest is on GPU
                if hasattr(self.detection_model, 'sam_model') and self.detection_model.sam_model is not None:
                    try:
                        sam_device = next(self.detection_model.sam_model.model.parameters()).device
                        logger.info(f"SAM model is on device: {sam_device}")
                        
                        # It's OK if SAM is on CPU while DeepForest is on GPU
                        if 'cpu' in str(sam_device) and 'cuda' in str(model_device):
                            logger.info("Using mixed precision: DeepForest on GPU, SAM on CPU for compatibility")
                            logger.info("This won't affect tree detection performance, only segmentation")
                    except Exception as e:
                        logger.warning(f"Could not verify SAM device or SAM is not properly loaded: {e}")
                        logger.info("Will continue with DeepForest detection regardless of SAM status")
                
            # Verify model device
            if hasattr(self.detection_model, 'model') and hasattr(self.detection_model.model, 'model'):
                model_device = next(self.detection_model.model.model.parameters()).device
                logger.info(f"DeepForest model is on device: {model_device}")
            
            # Initialize model manager and inference engine with proper parameters
            self.model_manager = ModelManager(config=config)
            self.model_manager.model = self.detection_model
            
            # Create inference engine
            self.inference_engine = InferenceEngine(config=config)
            self.inference_engine.model_manager = self.model_manager
            
            logger.info("ML model components initialized with GPU support")
            
            # Get direct references to the underlying models
            self.deepforest_model = self.detection_model.model
            self.sam_predictor = self.detection_model.sam_model
            self.sam_model = self.sam_predictor.model if self.sam_predictor else None
            
            load_time = time.time() - start_time
            logger.info(f"DeepForest and SAM models loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading DeepForest and SAM models: {e}")
            raise
    
    def wait_for_models(self, timeout: int = 60) -> bool:
        """
        Wait for models to load with a timeout.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        start_time = time.time()
        while not self.models_loaded and (time.time() - start_time) < timeout:
            time.sleep(0.5)
            
        return self.models_loaded
    
    def detect_trees(self, 
                    image: np.ndarray, 
                    confidence_threshold: float = 0.5,  # Increased from 0.3 to 0.5 for higher precision
                    with_segmentation: bool = True,
                    job_id: str = None) -> Dict[str, Any]:
        """
        Detect trees and other objects in an image with optional segmentation.
        
        Args:
            image: Image as numpy array (RGB)
            confidence_threshold: Minimum confidence for detections
            with_segmentation: Whether to add segmentation masks
            job_id: Optional job ID for tracking
            
        Returns:
            Dict containing detection results with detections, success flag and execution time
        """
        if not self.models_loaded:
            if not self.wait_for_models(timeout=30):
                raise RuntimeError("ML models not loaded. Check logs for errors.")
        
        start_time = time.time()
        
        try:
            # Run inference using the appropriate model (Grounded SAM or DeepForest)
            if self.use_grounded_sam:
                return self._detect_with_grounded_sam(image, confidence_threshold, with_segmentation, job_id)
            else:
                return self._detect_with_deepforest(image, confidence_threshold, with_segmentation, job_id)
        
        except Exception as e:
            logger.error(f"Error during tree detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'execution_time': time.time() - start_time
            }
    
    def _detect_with_grounded_sam(self, 
                                image: np.ndarray, 
                                confidence_threshold: float = 0.5,  # Increased from 0.3 to 0.5 for higher precision
                                with_segmentation: bool = True,
                                job_id: str = None) -> Dict[str, Any]:
        """
        Detect objects using Grounded SAM.
        
        Args:
            image: Image as numpy array (RGB)
            confidence_threshold: Minimum confidence for detections
            with_segmentation: Whether to add segmentation masks
            job_id: Optional job ID for tracking
            
        Returns:
            Dict containing detection results
        """
        start_time = time.time()
        
        try:
            # Prepare the image
            pil_image = Image.fromarray(image)
            
            # Define tree categories for risk assessment
            classes = [
                "healthy tree", 
                "hazardous tree", 
                "dead tree", 
                "low canopy tree", 
                "pest disease tree", 
                "flood prone tree", 
                "utility conflict tree", 
                "structural hazard tree", 
                "fire risk tree"
            ]
            labels = [class_name + "." for class_name in classes]
            
            # Run Grounding DINO detection
            logger.info(f"Running Grounding DINO detection with {len(labels)} classes")
            detection_results = self.groundingdino_model(
                pil_image,
                candidate_labels=labels,
                threshold=confidence_threshold
            )
            
            # Process detection results
            detections = []
            boxes = []
            
            # Get image dimensions for normalization
            height, width = image.shape[:2]
            
            for result in detection_results:
                # Extract box (normalized coordinates)
                box = result['box']
                x_min, y_min, x_max, y_max = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                
                # Normalize coordinates to 0-1 range
                normalized_box = [
                    x_min / width,
                    y_min / height,
                    x_max / width,
                    y_max / height
                ]
                
                # Create detection object
                detection = {
                    'bbox': normalized_box,
                    'confidence': result['score'],
                    'class': result['label'].rstrip('.'),  # Remove trailing period
                }
                
                detections.append(detection)
                boxes.append([x_min, y_min, x_max, y_max])  # Keep pixel coordinates for SAM
            
            logger.info(f"Detected {len(detections)} objects")
            
            # Run segmentation if requested and if we have detections
            if with_segmentation and detections and self.sam_predictor:
                logger.info("Running SAM segmentation")
                
                # Set image for segmentation
                self.sam_predictor.set_image(image)
                
                # Process each detection to add segmentation mask
                for i, (detection, box) in enumerate(zip(detections, boxes)):
                    try:
                        # Generate mask
                        masks, scores, _ = self.sam_predictor.predict(
                            box=np.array(box),
                            multimask_output=False
                        )
                        
                        # Get best mask
                        mask = masks[0]
                        
                        # Convert to boolean array and store
                        detection['segmentation'] = mask.tolist()
                        detection['mask_score'] = float(scores[0])
                        
                        logger.debug(f"Added segmentation to detection {i+1} with score {scores[0]:.4f}")
                    except Exception as e:
                        logger.error(f"Error in segmentation for detection {i+1}: {e}")
            
            # Create result object
            result = {
                'success': True,
                'detections': detections,
                'execution_time': time.time() - start_time
            }
            
            # Update performance metrics
            self.last_inference_time = time.time() - start_time
            self.inference_count += 1
            self.average_inference_time = (
                (self.average_inference_time * (self.inference_count - 1) + self.last_inference_time) 
                / self.inference_count
            )
            
            # Log performance
            logger.info(f"Inference completed in {self.last_inference_time:.2f}s (avg: {self.average_inference_time:.2f}s)")
            
            return result
        
        except Exception as e:
            logger.error(f"Error during Grounded SAM detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'execution_time': time.time() - start_time
            }
    
    def _detect_with_deepforest(self, 
                              image: np.ndarray, 
                              confidence_threshold: float = 0.5,  # Increased from 0.3 to 0.5 for higher precision
                              with_segmentation: bool = True,
                              job_id: str = None) -> Dict[str, Any]:
        """
        Detect trees using DeepForest with optional SAM segmentation.
        
        Args:
            image: Image as numpy array (RGB)
            confidence_threshold: Minimum confidence for detections
            with_segmentation: Whether to add segmentation masks
            job_id: Optional job ID for tracking
            
        Returns:
            Dict containing detection results
        """
        start_time = time.time()
        
        try:
            # Make sure the model is in eval mode before detection
            if hasattr(self.detection_model, 'model') and hasattr(self.detection_model.model, 'model'):
                self.detection_model.model.model.eval()
                model_device = next(self.detection_model.model.model.parameters()).device
                logger.info(f"Running detection with model on device: {model_device}")
            
            # Make sure the DeepForest model config has the right device settings
            # This is critical for GPU compatibility
            if hasattr(self.detection_model, 'model') and hasattr(self.detection_model.model, 'config'):
                if 'cuda' in str(model_device):
                    gpu_id = int(str(model_device).split(':')[1]) if ':' in str(model_device) else 0
                    self.detection_model.model.config["devices"] = [gpu_id]
                    logger.info(f"Updated DeepForest config to use device: {gpu_id}")
            
            # Call the detect method from the model using torch.no_grad() for efficiency
            import torch
            import PIL.Image
            
            # Make sure required imports are accessible for object_recognition.py
            # These are needed within the TreeDetectionModel.detect method
            from PIL import Image
            import numpy as np
            
            with torch.no_grad():
                logger.info("Running ML inference with torch.no_grad() for efficiency")
                # Pass job_id to model for consistent file naming
                if job_id and hasattr(self.detection_model, 'config'):
                    # Set job_id on model config
                    self.detection_model.config.job_id = job_id
                    self.detection_model.job_id = job_id
                    logger.info(f"Using job_id {job_id} for consistent file naming")
                
                detect_results = self.detection_model.detect(image)
            
            # Format results for the newer API format
            detections = []
            
            for tree in detect_results.get('trees', []):
                if 'bbox' in tree and 'confidence' in tree:
                    detection = {
                        'bbox': tree['bbox'],
                        'confidence': tree['confidence'],
                        'class': 'tree'  # DeepForest only detects trees
                    }
                    
                    # Add segmentation if available
                    if 'segmentation' in tree:
                        detection['segmentation'] = tree['segmentation']
                        detection['mask_score'] = tree.get('mask_score', 0.9)  # Default if not provided
                    
                    detections.append(detection)
            
            result = {
                'success': True,
                'detections': detections,
                'execution_time': time.time() - start_time
            }
            
            logger.info(f"Tree detection completed with {len(detections)} trees")
            
            # Update performance metrics
            self.last_inference_time = time.time() - start_time
            self.inference_count += 1
            self.average_inference_time = (
                (self.average_inference_time * (self.inference_count - 1) + self.last_inference_time) 
                / self.inference_count
            )
            
            # Log performance
            logger.info(f"Inference completed in {self.last_inference_time:.2f}s (avg: {self.average_inference_time:.2f}s)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during DeepForest detection: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'execution_time': time.time() - start_time
            }
    
    def segment_trees(self, 
                     image: np.ndarray, 
                     boxes: List[List[float]]) -> Dict[str, Any]:
        """
        Segment trees using SAM model with provided bounding boxes.
        
        Args:
            image: Image as numpy array (RGB)
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            Dict containing segmentation masks
        """
        if not self.models_loaded:
            if not self.wait_for_models(timeout=30):
                raise RuntimeError("ML models not loaded. Check logs for errors.")
        
        if self.sam_predictor is None:
            return {"masks": [], "scores": []}
        
        start_time = time.time()
        
        try:
            # Prepare image
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)
            
            # Set image for segmentation
            self.sam_predictor.set_image(image)
            
            # Process boxes
            masks = []
            scores = []
            
            for box in boxes:
                try:
                    # Generate masks
                    mask_result, score_result, _ = self.sam_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=False
                    )
                    
                    # Add results
                    masks.append(mask_result[0])
                    scores.append(score_result[0])
                except Exception as e:
                    logger.error(f"Error in segmentation: {e}")
            
            logger.info(f"Segmentation completed with {len(masks)} masks")
            
            # Format results
            result = {
                "masks": masks,
                "scores": scores,
                "processing_time": time.time() - start_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during tree segmentation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def generate_combined_visualization(self, 
                                       image: np.ndarray, 
                                       detection_results: Dict[str, Any]) -> np.ndarray:
        """
        Generate combined visualization with detection boxes and segmentation masks.
        
        Args:
            image: Original image as numpy array (RGB)
            detection_results: Results from detect_trees method
            
        Returns:
            Visualization image as numpy array (RGB)
        """
        if not self.models_loaded:
            if not self.wait_for_models(timeout=30):
                raise RuntimeError("ML models not loaded. Check logs for errors.")
        
        try:
            # Check if we have the new format (with 'detections') or old format (with 'boxes')
            if 'detections' in detection_results:
                return self._generate_visualization_new_format(image, detection_results)
            else:
                return self._generate_visualization_old_format(image, detection_results)
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            logger.error(traceback.format_exc())
            # Return the original image if visualization fails
            return image
    
    def _generate_visualization_new_format(self, 
                                          image: np.ndarray, 
                                          detection_results: Dict[str, Any]) -> np.ndarray:
        """
        Generate visualization with the new format (used by Grounded SAM).
        
        Args:
            image: Original image as numpy array (RGB)
            detection_results: Results from detect_trees method with 'detections' key
            
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
                "healthy tree": (22, 163, 74, 128),     # Green
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
                class_name = detection.get('class', 'unknown')
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
            logger.error(f"Error generating visualization (new format): {str(e)}")
            logger.error(traceback.format_exc())
            return image  # Return original image if visualization fails
    
    def _generate_visualization_old_format(self, 
                                          image: np.ndarray, 
                                          detection_results: Dict[str, Any]) -> np.ndarray:
        """
        Generate visualization with the old format (used by DeepForest).
        
        Args:
            image: Original image as numpy array (RGB)
            detection_results: Results from detect_trees method with 'boxes' and 'scores' keys
            
        Returns:
            Visualization image as numpy array (RGB)
        """
        try:
            # Import visualization utilities
            import cv2
            
            # Create a copy of the image for visualization
            vis_image = image.copy()
            
            # Get boxes and scores
            boxes = detection_results.get("boxes", [])
            scores = detection_results.get("scores", [])
            
            # Draw bounding boxes
            for i, (box, score) in enumerate(zip(boxes, scores)):
                # Get pixel coordinates
                height, width = image.shape[:2]
                if box[0] < 1 and box[1] < 1 and box[2] < 1 and box[3] < 1:
                    # If coordinates are normalized (0-1), convert to pixel coordinates
                    x1 = int(box[0] * width)
                    y1 = int(box[1] * height)
                    x2 = int(box[2] * width)
                    y2 = int(box[3] * height)
                else:
                    # Otherwise use as-is (assuming they're already pixels)
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                
                confidence = float(score)
                
                # Draw box with confidence-based color (green for high confidence, red for low)
                color = (0, int(255 * confidence), int(255 * (1 - confidence)))
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Add confidence label
                label = f"{confidence:.2f}"
                cv2.putText(vis_image, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw segmentation masks if available
            if "masks" in detection_results and detection_results["masks"] is not None:
                masks = detection_results["masks"]
                
                # Create a semi-transparent overlay for masks
                mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
                
                # Process each mask
                for i, mask in enumerate(masks):
                    # Use different colors for each mask
                    color = np.array([
                        (i * 50) % 255,  # B
                        (i * 100) % 255, # G
                        (i * 150) % 255  # R
                    ])
                    
                    # Convert binary mask to image
                    binary_mask = mask.astype(np.uint8) * 255
                    colored_mask = np.zeros_like(vis_image, dtype=np.uint8)
                    
                    # Apply color to mask
                    for c in range(3):  # RGB channels
                        colored_mask[:, :, c] = binary_mask * color[c]
                    
                    # Add to overlay
                    mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.5, 0)
                
                # Combine original image with mask overlay
                vis_image = cv2.addWeighted(vis_image, 1.0, mask_overlay, 0.5, 0)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Error generating visualization (old format): {str(e)}")
            logger.error(traceback.format_exc())
            return image  # Return original image if visualization fails
    
    def get_model_status(self) -> Dict[str, Any]:
        """Return the current status of ML models."""
        status = {
            "models_loaded": self.models_loaded,
            "loading_error": self.loading_error,
            "cuda_available": self.cuda_available,
            "device": str(self.device) if self.device else None,
            "average_inference_time": self.average_inference_time,
            "inference_count": self.inference_count,
            "sam_model_loaded": self.sam_model is not None,
            "using_grounded_sam": self.use_grounded_sam
        }
        
        # Add model-specific status
        if self.use_grounded_sam:
            status.update({
                "groundingdino_model_loaded": self.groundingdino_model is not None,
                "model_type": "Grounded SAM"
            })
        else:
            status.update({
                "deepforest_model_loaded": self.deepforest_model is not None,
                "model_type": "DeepForest + SAM"
            })
            
        return status

# Create a global instance for singleton-like access
_model_service_instance = None
_models_initialized = False

def get_model_service(use_gpu: bool = True, use_grounded_sam: bool = True) -> MLModelService:
    """
    Get or create the global ML model service instance.
    
    Args:
        use_gpu: Whether to use GPU for inference if available
        use_grounded_sam: Whether to use Grounded SAM instead of DeepForest
        
    Returns:
        MLModelService instance
    """
    global _model_service_instance, _models_initialized
    
    if _model_service_instance is None:
        logger.info(f"Creating new MLModelService instance using {'Grounded SAM' if use_grounded_sam else 'DeepForest'}")
        _model_service_instance = MLModelService(use_gpu=use_gpu, use_grounded_sam=use_grounded_sam)
    else:
        logger.info("Reusing existing MLModelService instance")
        
        # Check if we're trying to use a different model type than what's already loaded
        if _model_service_instance.use_grounded_sam != use_grounded_sam:
            logger.warning(
                f"Requested model type ({'Grounded SAM' if use_grounded_sam else 'DeepForest'}) "
                f"differs from loaded model ({'Grounded SAM' if _model_service_instance.use_grounded_sam else 'DeepForest'}). "
                f"Will continue using the already loaded model."
            )
        
        if _models_initialized:
            # If models were already loaded, mark them as loaded in this instance
            # to avoid reloading them again
            logger.info("Models were already loaded in a previous instance, marking as initialized")
            _model_service_instance.models_loaded = True
    
    return _model_service_instance