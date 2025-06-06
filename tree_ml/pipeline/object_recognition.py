"""
ML Pipeline for Tree Detection with DeepForest, SAM, and Gemini Integration
"""

import logging
import os
import sys
import asyncio
import json
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from types import SimpleNamespace
import io
from PIL import Image

# =====================================
# Configuration
# =====================================
@dataclass
class MLConfig:
    """Configuration for the tree detection pipeline"""
    
    # Paths
    model_path: str = "/ttt/tree_ml/pipeline/model"
    export_path: str = "/ttt/tree_ml/pipeline/model/exports"
    store_path: str = "/ttt/data/zarr"
    ml_path: str = "/ttt/data/ml"
    
    # Hardware configuration
    gpu_device_id: int = 0
    num_cpu_cores: int = 24
    
    # Model parameters
    batch_size: int = 8
    input_shape: Tuple[int, int, int] = (3, 640, 640)
    
    # DeepForest parameters
    deepforest_score_threshold: float = 0.2  # Default confidence threshold
    
    # SAM parameters
    sam_model_type: str = "vit_h"  # Model type for SAM
    
    # Gemini parameters
    use_gemini: bool = True  # Whether to use Gemini API
    
    # Output parameters
    include_bounding_boxes: bool = True
    output_path: Optional[str] = None
    store_masks_separately: bool = True  # Store masks in separate files to reduce trees.json size
    
    # Job tracking
    job_id: Optional[str] = None  # For consistent file naming across the pipeline
    
    def __post_init__(self):
        # Create necessary directories
        for path_attr in ['model_path', 'export_path', 'store_path', 'ml_path']:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)
            
# Global singleton instances for models to avoid loading multiple times
_deepforest_model = None
_sam_model = None
_model_loaded = False

# =====================================
# Tree Detection Model
# =====================================
        
class TreeDetectionModel:
    """Tree detection model using DeepForest"""
    
    def __init__(self, config):
        """Initialize the tree detection model with DeepForest"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Store job_id from config for consistent file naming
        self.job_id = getattr(config, 'job_id', None)
        
        # Set device for computation with better error handling
        try:
            # Check CUDA device and driver compatibility
            if torch.cuda.is_available():
                self.device = f"cuda:{config.gpu_device_id}"
                # Verify CUDA device is working properly
                test_tensor = torch.tensor([1.0], device=self.device)
                self.logger.info(f"CUDA initialization successful. Using device: {self.device} - {torch.cuda.get_device_name(config.gpu_device_id)}")
            else:
                self.device = "cpu"
                self.logger.warning("CUDA not available, using CPU for computation")
        except Exception as e:
            self.logger.error(f"Error initializing CUDA: {e}. Falling back to CPU.")
            self.device = "cpu"
        
        # Initialize DeepForest model
        self.model = self._init_deepforest_model()
        
        # Initialize SAM model
        self.sam_model = self._init_sam_model()
        
        # Skip Gemini initialization to avoid event loop issues
        self.gemini_client = None
        
    def _init_deepforest_model(self):
        """Initialize DeepForest model for tree crown detection"""
        global _deepforest_model, _model_loaded
        
        try:
            # Use global model if already loaded
            if _deepforest_model is not None and _model_loaded:
                self.logger.info("Using existing global DeepForest model")
                model = _deepforest_model
                # Just update the score threshold
                model.model.score_thresh = self.config.deepforest_score_threshold
                return model
                
            # Load a new model if not already loaded
            from deepforest import main
            model = main.deepforest()
            
            # Check for custom weights
            custom_weights_path = os.path.join(self.config.model_path, "deepforest_weights.pt")
            if os.path.exists(custom_weights_path):
                self.logger.info(f"Loading custom DeepForest weights from {custom_weights_path}")
                model.use_release(custom_weights_path)
            else:
                self.logger.info("Loading pre-trained DeepForest model")
                model.use_release()
            
            # Set score threshold
            model.model.score_thresh = self.config.deepforest_score_threshold
            
            # Move to appropriate device with error handling
            try:
                if 'cuda' in self.device:
                    self.logger.info(f"Moving DeepForest model to {self.device}")
                    # Important: Set model to eval mode before moving to GPU
                    model.model.eval()
                    model.model.to(self.device)  # Access the internal PyTorch model
                    
                    # Force all model components to the same device
                    for module in model.model.modules():
                        if hasattr(module, 'to'):
                            module.to(self.device)
                    
                    # Update DeepForest config to use GPU devices
                    gpu_id = int(self.device.split(':')[1]) if ':' in self.device else 0
                    model.config["devices"] = [gpu_id]
                    
                    # Recreate the trainer to use GPU properly
                    model.create_trainer()
                    
                    # Verify model is on the correct device
                    self.logger.info(f"Model device check after moving to {self.device}: {next(model.model.parameters()).device}")
                    
                    # Test with torch.no_grad to avoid training mode issues
                    with torch.no_grad():
                        # Just verify the parameters are on the right device
                        for param in model.model.parameters():
                            _ = param.device
                            break
                            
                    self.logger.info(f"DeepForest CUDA test successful")
                else:
                    self.logger.info("Using CPU for DeepForest model")
            except Exception as e:
                self.logger.error(f"Error moving DeepForest model to GPU: {e}. Using CPU instead.")
                self.device = "cpu"
                # Try to move the model back to CPU if it failed on GPU
                try:
                    model.model.eval()  # Set to eval mode
                    model.model.to("cpu")
                except Exception as cpu_err:
                    self.logger.error(f"Error moving model to CPU: {cpu_err}")
            
            # Store the model in the global variable for reuse
            _deepforest_model = model
            _model_loaded = True
            self.logger.info("Stored DeepForest model in global cache for reuse")
                
            self.logger.info(f"Initialized DeepForest model with score threshold {model.model.score_thresh} on {self.device}")
            return model
            
        except ImportError as e:
            self.logger.error(f"Failed to import DeepForest: {e}")
            raise RuntimeError(f"DeepForest is required but not installed: {e}")
        except Exception as e:
            self.logger.error(f"Error initializing DeepForest model: {e}")
            raise RuntimeError(f"Failed to initialize DeepForest model: {e}")
            
    def _init_sam_model(self):
        """Initialize Segment Anything (SAM) model for tree segmentation"""
        global _sam_model, _model_loaded
        
        try:
            # Use global model if already loaded
            if _sam_model is not None and _model_loaded:
                self.logger.info("Using existing global SAM model")
                return _sam_model
            
            # Check if SAM model file exists
            sam_checkpoint_paths = [
                os.path.join(self.config.model_path, "sam_vit_h_4b8939.pth"),
                os.path.join(self.config.model_path, "sam_vit_l_0b3195.pth"),
                os.path.join(self.config.model_path, "sam_vit_b_01ec64.pth"),
                os.path.join("/ttt/tree_ml/dashboard/backend", "sam_vit_h_4b8939.pth")
            ]
            
            # Find first available SAM model
            sam_checkpoint = None
            for path in sam_checkpoint_paths:
                if os.path.exists(path):
                    sam_checkpoint = path
                    self.logger.info(f"Found SAM model at {sam_checkpoint}")
                    break
            
            if sam_checkpoint:
                from segment_anything import sam_model_registry, SamPredictor
                
                # Load SAM model
                self.logger.info(f"Loading SAM model ({self.config.sam_model_type}) from {sam_checkpoint}")
                sam = sam_model_registry[self.config.sam_model_type](checkpoint=sam_checkpoint)
                
                # Move to device with error handling
                try:
                    if 'cuda' in self.device:
                        self.logger.info(f"Moving SAM model to {self.device}")
                        # First set model to eval mode 
                        sam.eval()
                        # Then move to GPU
                        sam.to(device=self.device)
                        
                        # Verify model device
                        sam_device = next(sam.parameters()).device
                        self.logger.info(f"SAM model is on device: {sam_device}")
                        
                        # Test with a correctly sized tensor for SAM's image encoder
                        # SAM's image encoder expects 1024x1024 inputs
                        self.logger.info("Creating appropriately sized test tensor for SAM (1024x1024)")
                        test_tensor = torch.ones(1, 3, 1024, 1024, device=self.device)
                        
                        with torch.no_grad():
                            # Run a small inference test
                            self.logger.info("Testing SAM image encoder with properly sized tensor")
                            test_output = sam.image_encoder(test_tensor)
                            
                        self.logger.info(f"SAM CUDA test successful, device: {test_output.device}, output shape: {test_output.shape}")
                    else:
                        self.logger.info("Using CPU for SAM model")
                        sam.eval()  # Set to eval mode first
                        sam.to(device="cpu")
                except Exception as e:
                    self.logger.error(f"Error with SAM model on GPU: {e}. Using CPU for SAM only.")
                    self.logger.warning("SAM will use CPU but DeepForest will still use GPU if available")
                    
                    # Only change the device for SAM, not for DeepForest
                    sam_device = "cpu"
                    
                    # Reset the model and try again with CPU
                    try:
                        sam = sam_model_registry[self.config.sam_model_type](checkpoint=sam_checkpoint)
                        sam.eval()  # Set to eval mode
                        sam.to(device=sam_device)
                        self.logger.info(f"SAM model successfully loaded on CPU")
                    except Exception as e2:
                        self.logger.error(f"Failed to load SAM even on CPU: {e2}")
                
                # Initialize predictor
                predictor = SamPredictor(sam)
                
                # Store in global variable for reuse
                _sam_model = predictor
                _model_loaded = True
                self.logger.info("Stored SAM model in global cache for reuse")
                
                self.logger.info("SAM model loaded successfully")
                return predictor
            else:
                self.logger.warning("No SAM checkpoint found, segmentation will not be available")
                return None
                
        except ImportError as e:
            self.logger.warning(f"Failed to import SAM: {e}. Segmentation will not be available.")
            return None
        except Exception as e:
            self.logger.warning(f"Error initializing SAM model: {e}. Segmentation will not be available.")
            return None
            
    def _init_gemini_client(self):
        """Initialize Gemini API client for enhanced tree detection"""
        try:
            # Import Gemini service from backend
            sys.path.append('/ttt/tree_ml/dashboard/backend')
            from services.gemini_service import GeminiService
            
            # Initialize Gemini service
            gemini_service = GeminiService()
            asyncio.create_task(gemini_service.initialize())
            
            self.logger.info("Gemini API client initialized")
            return gemini_service
        except ImportError as e:
            self.logger.warning(f"Failed to import Gemini service: {e}. Gemini detection will not be available.")
            return None
        except Exception as e:
            self.logger.warning(f"Error initializing Gemini service: {e}. Gemini detection will not be available.")
            return None
        
    def detect(self, image):
        """Run tree detection on an image"""
        self.logger.info("Starting tree detection with DeepForest")
        
        # Ensure PIL.Image is imported locally within this method
        # to avoid relying on external scope
        import sys
        try:
            from PIL import Image as PILImage
        except ImportError:
            self.logger.error("Failed to import PIL.Image - trying to install it")
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
                from PIL import Image as PILImage
            except Exception as e:
                self.logger.error(f"Failed to install/import PIL: {e}")
                return {'tree_count': 0, 'trees': []}
        
        # Ensure image is in the right format for DeepForest
        if isinstance(image, str) and os.path.isfile(image):
            # If image is a file path
            image_path = image
            # Load image for potential SAM use
            try:
                image_data = np.array(PILImage.open(image_path))
            except Exception as e:
                self.logger.error(f"Error loading image: {e}")
                image_data = None
        elif isinstance(image, np.ndarray):
            # If image is already a numpy array
            image_data = image
            
            # Check if we already have a job ID in the path, otherwise use timestamp
            timestamp_str = datetime.now().timestamp()
            # Instead of creating a separate temp file, save it in a consistent location
            if hasattr(self, 'job_id') and self.job_id:
                # Use consistent directory structure: detection_[job_id]
                ml_dir_name = f"detection_{job_id}"
                
                # First, check if the image already exists in the ml_response directory
                ml_response_dir = os.path.join(self.config.ml_path, ml_dir_name, "ml_response")
                existing_image_path = os.path.join(ml_response_dir, "input_image.jpg")
                
                if os.path.exists(existing_image_path):
                    # Use the existing image instead of creating a new one
                    self.logger.info(f"Using existing image at {existing_image_path}")
                    image_path = existing_image_path
                else:
                    # Only save satellite image once in the ml_response directory
                    # This prevents duplicate images across multiple directories
                    ml_response_dir = os.path.join(self.config.ml_path, ml_dir_name, "ml_response")
                    os.makedirs(ml_response_dir, exist_ok=True)
                    image_path = os.path.join(ml_response_dir, "input_image.jpg")
                    
                    # Save image as high-quality JPEG
                    PILImage.fromarray(image_data).convert('RGB').save(
                        image_path, format='JPEG', quality=95, optimize=True
                    )
                    self.logger.info(f"Saved image to unified location: {image_path}")
            else:
                # No saving of duplicates - return the path that should be used
                # without actually saving a duplicate file
                image_path = os.path.join(self.config.ml_path, f"input_image_{timestamp_str}.jpg")
        else:
            self.logger.error(f"Unsupported image format: {type(image)}")
            return {'tree_count': 0, 'trees': []}
        
        try:
            # DeepForest detection
            self.logger.info(f"Running DeepForest detection on {image_path}")
            # Log model device for monitoring
            if hasattr(self.model, 'model'):
                device = next(self.model.model.parameters()).device
                self.logger.info(f"DeepForest model is on device: {device}")
                
            # Verify we're in eval mode (not training)
            if hasattr(self.model, 'model'):
                self.model.model.eval()
                self.logger.info("Confirmed model is in eval mode for inference")
            
            # Handle image directly to ensure device consistency
            try:
                # Log the device we're using for prediction
                self.logger.info(f"Using direct tensor handling for prediction on device: {self.device}")
                
                # Load image and process directly - ensure local import
                try:
                    from PIL import Image as PILImage
                except ImportError:
                    self.logger.error("Failed to import PIL.Image in tensor handling")
                    raise ImportError("PIL.Image not available")
                
                import torch
                from deepforest import preprocess
                
                # Load image
                self.logger.info(f"Loading image from {image_path}")
                image_array = np.array(PILImage.open(image_path))
                
                # Create a direct custom inference approach that bypasses DeepForest's predict_image
                # to ensure device consistency
                self.logger.info("Using custom inference with explicit device handling")
                
                # Preprocess the image manually to match DeepForest's preprocessing
                from deepforest import preprocess
                from deepforest.utilities import annotations_to_shapefile
                import pandas as pd
                
                # Preprocess image
                self.logger.info("Preprocessing image")
                image_tensor = preprocess.preprocess_image(image_array)
                
                # Convert to tensor and ensure it's on the right device
                self.logger.info(f"Converting to tensor and explicitly moving to {device}")
                image_tensor = torch.tensor(image_tensor, dtype=torch.float32, device=device)
                
                # Log tensor details
                self.logger.info(f"Input tensor shape: {image_tensor.shape}, device: {image_tensor.device}")
                
                # Check if model device matches tensor device
                model_device = next(self.model.model.parameters()).device
                if str(image_tensor.device) != str(model_device):
                    self.logger.warning(f"Device mismatch! Tensor on {image_tensor.device}, model on {model_device}")
                    self.logger.info("Moving tensor to match model device")
                    image_tensor = image_tensor.to(model_device)
                
                # Run inference with torch.no_grad
                with torch.no_grad():
                    self.logger.info(f"Running forward pass on device {image_tensor.device}")
                    # Keep batch dimension as DeepForest expects
                    if len(image_tensor.shape) == 3:
                        image_tensor = image_tensor.unsqueeze(0)
                    predictions = self.model.model(image_tensor)[0]
                    
                    # Process predictions manually to DataFrame format
                    if len(predictions['boxes']) > 0:
                        self.logger.info("Processing prediction results")
                        # Convert tensors to numpy
                        boxes_np = predictions['boxes'].cpu().numpy()
                        scores_np = predictions['scores'].cpu().numpy()
                        labels_np = predictions['labels'].cpu().numpy()
                        
                        # Create DataFrame manually instead of using annotations_to_shapefile 
                        # since the API seems different from what we expected
                        self.logger.info("Creating DataFrame manually from detection results")
                        import pandas as pd
                        
                        # Initialize empty DataFrame
                        boxes = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
                        
                        # Add detections
                        for i, (box, score, label_idx) in enumerate(zip(boxes_np, scores_np, labels_np)):
                            # Convert label index to string
                            label_name = self.model.numeric_to_label_dict.get(label_idx, f"class_{label_idx}")
                            
                            # Add row to DataFrame
                            boxes.loc[i] = [
                                float(box[0]), float(box[1]), 
                                float(box[2]), float(box[3]), 
                                float(score), label_name
                            ]
                    else:
                        self.logger.info("No detections found")
                        boxes = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
                
                self.logger.info(f"Direct tensor inference complete, found {len(boxes)} boxes")
            except Exception as e:
                self.logger.error(f"Error in predict_image with device handling: {e}")
                # Fall back to standard prediction if our wrapper fails
                boxes = self.model.predict_image(path=image_path, return_plot=False)
            
            # Process results
            detections = []
            if not boxes.empty:
                for _, box in boxes.iterrows():
                    x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                    score = box['score']
                    
                    # Assign a tree risk category - in a real system this would use ML
                    # but for demo purposes we'll randomly assign categories based on score
                    risk_categories = [
                        'healthy_tree',
                        'hazardous_tree',
                        'dead_tree',
                        'low_canopy_tree',
                        'pest_disease_tree',
                        'flood_prone_tree',
                        'utility_conflict_tree',
                        'structural_hazard_tree',
                        'fire_risk_tree'
                    ]
                    
                    # Use the confidence score to seed the category selection
                    # This gives a deterministic but varied selection of categories
                    category_index = int((score * 1000) % len(risk_categories))
                    tree_class = risk_categories[category_index]
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(score),
                        'class': tree_class,  # Use specific tree risk categories
                        'class_id': category_index,
                        'detection_type': 'deepforest'
                    })
                    
            # Log the detection count
            self.logger.info(f"DeepForest detected {len(detections)} trees")
            
            # Add segmentation with SAM if available
            if self.sam_model is not None and image_data is not None and len(detections) > 0:
                self._add_segmentation(detections, image_data)
            
            # Add Gemini detection if available and no trees were found with DeepForest
            if self.gemini_client is not None and len(detections) == 0:
                gemini_detections = self._detect_with_gemini(image_path)
                if gemini_detections:
                    detections.extend(gemini_detections)
                    
            # Add one final filtering step to absolutely ensure no whole-image detections are returned
            filtered_detections = []
            for detection in detections:
                if 'bbox' in detection:
                    bbox = detection['bbox']
                    # Calculate area
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    
                    # Very aggressive filtering - reject anything covering more than 20% of image
                    if area > 0.2:
                        self.logger.warning(f"Final filter: Rejecting large detection with area {area:.2f}")
                        continue
                    
                    # Also reject if it's too close to the image edges
                    if (bbox[0] < 0.03 and bbox[1] < 0.03 and 
                        bbox[2] > 0.97 and bbox[3] > 0.97):
                        self.logger.warning(f"Final filter: Rejecting edge-spanning detection")
                        continue
                        
                    # Accept this detection
                    filtered_detections.append(detection)
                    
            if len(detections) > 0 and len(filtered_detections) == 0:
                # If we filtered out all detections, use DeepForest as fallback
                self.logger.warning("All detections were filtered out, trying DeepForest fallback")
                try:
                    # Import fallback detector if needed
                    from deepforest import main
                    model = main.deepforest()
                    model.use_release()
                    
                    # Use temporary path for consistency
                    if isinstance(image, str) and os.path.isfile(image):
                        image_path = image
                    else:
                        # If we don't have a valid path, create a temporary file
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                            temp_path = tmp.name
                            Image.fromarray(image).save(temp_path)
                            image_path = temp_path
                    
                    # Run DeepForest detection directly - it's usually more conservative
                    boxes = model.predict_image(path=image_path, return_plot=False)
                    
                    # Process results
                    if not boxes.empty:
                        for _, box in boxes.iterrows():
                            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                            
                            # Only add if it's not too large
                            width = x2 - x1
                            height = y2 - y1
                            if (width * height) < 0.2:  # 20% area threshold
                                filtered_detections.append({
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': float(box['score']),
                                    'class': 'tree',
                                    'class_id': 0,
                                    'detection_type': 'deepforest_fallback'
                                })
                except Exception as e:
                    self.logger.error(f"DeepForest fallback failed: {e}")
            
            # Log detection counts
            self.logger.info(f"Detection complete: {len(filtered_detections)} valid trees (filtered from {len(detections)} initial detections)")
            
            # Return the filtered detections
            return {
                'success': True, 
                'tree_count': len(filtered_detections),
                'trees': filtered_detections,
                'detections': filtered_detections  # Duplicate for backward compatibility
            }
            
        except Exception as e:
            self.logger.error(f"Error in tree detection: {e}")
            return {'tree_count': 0, 'trees': []}
            
    def _add_segmentation(self, detections, image_data):
        """Add segmentation masks to detections using SAM and GroundingDINO if available"""
        self.logger.info("Adding segmentation masks with SAM")
        
        # Create directory for mask storage if needed
        masks_dir = None
        if hasattr(self, 'job_id') and self.job_id:
            masks_dir = os.path.join(self.config.ml_path, f"detection_{self.job_id}", "masks")
            os.makedirs(masks_dir, exist_ok=True)
            self.logger.info(f"Storing masks in {masks_dir}")
            
            # Create directory for visualizations
            vis_dir = os.path.join(self.config.ml_path, f"detection_{self.job_id}", "ml_response")
            os.makedirs(vis_dir, exist_ok=True)
        
        try:
            # Set the image for SAM
            # Ensure the image is on the same device as the SAM model
            self.logger.info(f"Setting image for SAM on device: {self.device}")
            
            # Try to load GroundingDINO if available
            grounding_dino_model = None
            try:
                # Check if GroundingDINO is installed
                from groundingdino.util.inference import Model
                
                # Check for model paths
                grounding_dino_config_paths = [
                    os.path.join(self.config.model_path, "grounded-sam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                    os.path.join("/ttt/tree_ml/pipeline/grounded-sam/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
                ]
                
                grounding_dino_checkpoint_paths = [
                    os.path.join(self.config.model_path, "grounded-sam/groundingdino_swint_ogc.pth"),
                    os.path.join("/ttt/tree_ml/pipeline/grounded-sam/groundingdino_swint_ogc.pth")
                ]
                
                # Find first available config and checkpoint
                config_path = None
                checkpoint_path = None
                
                for path in grounding_dino_config_paths:
                    if os.path.exists(path):
                        config_path = path
                        self.logger.info(f"Found GroundingDINO config at {config_path}")
                        break
                        
                for path in grounding_dino_checkpoint_paths:
                    if os.path.exists(path):
                        checkpoint_path = path
                        self.logger.info(f"Found GroundingDINO checkpoint at {checkpoint_path}")
                        break
                
                if config_path and checkpoint_path:
                    self.logger.info("Initializing GroundingDINO model")
                    grounding_dino_model = Model(
                        model_config_path=config_path,
                        model_checkpoint_path=checkpoint_path
                    )
                    self.logger.info("GroundingDINO model loaded successfully")
            except ImportError as e:
                self.logger.warning(f"GroundingDINO not available: {e}. Will use basic SAM segmentation.")
            except Exception as e:
                self.logger.warning(f"Error loading GroundingDINO: {e}. Will use basic SAM segmentation.")
                
            # Use SAM to segment the image based on prompts
            if grounding_dino_model is not None:
                # Convert image from numpy array to the format expected by GroundingDINO
                # GroundingDINO expects BGR format (OpenCV format)
                import cv2
                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                    # If RGB, convert to BGR
                    if image_data.dtype != np.uint8:
                        image_data_uint8 = (image_data * 255).astype(np.uint8)
                    else:
                        image_data_uint8 = image_data
                    
                    # Set image for SAM (which expects RGB)
                    self.sam_model.set_image(image_data_uint8)
                    
                    # Make a copy for GroundingDINO (which might expect BGR)
                    grounding_image = cv2.cvtColor(image_data_uint8, cv2.COLOR_RGB2BGR)
                    
                    # Enhanced prompt classes for more precise tree detection in aerial/satellite imagery
                    CLASSES = [
                        # Primary tree detection classes - focused and specific
                        "single tree",
                        "individual tree",
                        "isolated tree",
                        "tree crown",
                        "tree canopy top view",
                        "individual tree in yard",
                        "distinct tree",
                        "standalone tree",
                        "mature tree top-down view",
                        "tree from above",
                        "lone tree aerial view",
                        "solitary tree satellite view",
                        
                        # Secondary tree detection classes
                        "small tree",
                        "large tree",
                        "tree next to house",
                        "tree in residential area",
                        "tree in suburban setting",
                        
                        # Original non-tree classes for contrast
                        "Residential Roof",
                        "Building Roof",
                        "Parking Lot"
                    ]
                    
                    # Use extremely aggressive thresholds to explicitly prevent whole-image detections
                    BOX_THRESHOLD = 0.6    # Much higher threshold to require very strong detections
                    TEXT_THRESHOLD = 0.6   # Much higher threshold for very precise text matching
                    NMS_THRESHOLD = 0.1    # Much lower to allow many overlapping trees
                    
                    # Filter out whole-image detections with very aggressive area threshold
                    WHOLE_IMAGE_AREA_THRESHOLD = 0.2   # Reject any detection covering more than 20% of the image
                    
                    self.logger.info(f"Running GroundingDINO with prompts: {CLASSES}")
                    
                    # Detect objects using GroundingDINO
                    try:
                        detections_grounding = grounding_dino_model.predict_with_classes(
                            image=grounding_image,
                            classes=CLASSES,
                            box_threshold=BOX_THRESHOLD,
                            text_threshold=TEXT_THRESHOLD
                        )
                        
                        self.logger.info(f"GroundingDINO detected {len(detections_grounding.xyxy)} potential trees")
                        
                        # Apply NMS to filter overlapping detections
                        import torchvision
                        nms_idx = torchvision.ops.nms(
                            torch.from_numpy(detections_grounding.xyxy), 
                            torch.from_numpy(detections_grounding.confidence), 
                            NMS_THRESHOLD
                        ).numpy().tolist()
                        
                        filtered_xyxy = detections_grounding.xyxy[nms_idx]
                        filtered_confidence = detections_grounding.confidence[nms_idx]
                        filtered_class_id = detections_grounding.class_id[nms_idx]
                        
                        self.logger.info(f"After NMS: {len(filtered_xyxy)} tree detections")
                        
                        # Get segmentation masks for each detection
                        self.logger.info("Generating masks with SAM")
                        all_masks = []
                        
                        for box in filtered_xyxy:
                            masks, scores, _ = self.sam_model.predict(
                                box=box,
                                multimask_output=True
                            )
                            
                            # Select best mask
                            if len(masks) > 0:
                                best_mask_idx = np.argmax(scores)
                                all_masks.append(masks[best_mask_idx])
                        
                        # Process each mask and add to detections
                        grounding_detections = []
                        
                        # Try running with multiple different prompts to improve tree detection
                        additional_detections = []
                        additional_class_ids = []
                        additional_scores = []
                        
                        # Optimized prompts for urban tree detection in satellite imagery
                        urban_tree_prompts = [
                            "individual tree in neighborhood top view",
                            "single tree in yard satellite view",
                            "isolated tree in residential area",
                            "standalone tree aerial view",
                            "solitary tree next to house",
                            "distinct tree crown from above",
                            "mature tree in urban setting from satellite",
                            "tree canopy top-down view"
                        ]
                        
                        # Try each prompt separately for better tree detection
                        for prompt in urban_tree_prompts:
                            try:
                                self.logger.info(f"Trying additional prompt: '{prompt}'")
                                prompt_detections = grounding_dino_model.predict_with_classes(
                                    image=grounding_image,
                                    classes=[prompt],
                                    box_threshold=BOX_THRESHOLD + 0.1,  # Use higher threshold for these specific prompts
                                    text_threshold=TEXT_THRESHOLD + 0.1
                                )
                                
                                # Only use detections that aren't whole-image detections
                                for j, box in enumerate(prompt_detections.xyxy):
                                    # Skip boxes that are almost the entire image
                                    if (box[0] < 0.05 and box[1] < 0.05 and 
                                        box[2] > 0.95 and box[3] > 0.95):
                                        continue
                                        
                                    # Skip tiny boxes
                                    width = box[2] - box[0]
                                    height = box[3] - box[1]
                                    if width * height < 0.001:  # Skip if less than 0.1% of image
                                        continue
                                        
                                    additional_detections.append(box)
                                    additional_scores.append(prompt_detections.confidence[j])
                                    additional_class_ids.append(prompt_detections.class_id[j])
                            except Exception as e:
                                self.logger.error(f"Error with prompt '{prompt}': {e}")
                        
                        # Add additional detections to the filtered results if we found any
                        if len(additional_detections) > 0:
                            self.logger.info(f"Found {len(additional_detections)} additional trees from specific prompts")
                            
                            # Combine with original detections for NMS
                            combined_boxes = np.vstack([filtered_xyxy, np.array(additional_detections)])
                            combined_scores = np.hstack([filtered_confidence, np.array(additional_scores)])
                            combined_class_ids = np.hstack([filtered_class_id, np.array(additional_class_ids)])
                            
                            # Apply NMS to remove duplicates
                            import torchvision
                            nms_idx = torchvision.ops.nms(
                                torch.from_numpy(combined_boxes), 
                                torch.from_numpy(combined_scores), 
                                NMS_THRESHOLD
                            ).numpy().tolist()
                            
                            # Update filtered results
                            filtered_xyxy = combined_boxes[nms_idx]
                            filtered_confidence = combined_scores[nms_idx]
                            filtered_class_id = combined_class_ids[nms_idx]
                            
                            # Get new masks for the combined set
                            all_masks = []
                            for box in filtered_xyxy:
                                # SAM expects pixel coordinates
                                h, w = image_data_uint8.shape[:2]
                                pixel_box = box * np.array([w, h, w, h])
                                
                                masks, scores, _ = self.sam_model.predict(
                                    box=pixel_box,
                                    multimask_output=True
                                )
                                
                                # Select best mask
                                if len(masks) > 0:
                                    best_mask_idx = np.argmax(scores)
                                    all_masks.append(masks[best_mask_idx])
                            
                            self.logger.info(f"After combining and NMS: {len(filtered_xyxy)} tree detections with {len(all_masks)} masks")
                        
                        # Filter out whole-image detections before processing
                        valid_indices = []
                        for i, box in enumerate(filtered_xyxy):
                            # Skip boxes that cover too much of the image
                            width = box[2] - box[0]
                            height = box[3] - box[1]
                            box_area = width * height
                            
                            if box_area > WHOLE_IMAGE_AREA_THRESHOLD:
                                self.logger.warning(f"Skipping box {i} which covers too much of the image (area: {box_area:.2f})")
                                continue
                                
                            # Skip boxes that are almost the entire image based on position
                            if (box[0] < 0.05 and box[1] < 0.05 and 
                                box[2] > 0.95 and box[3] > 0.95):
                                self.logger.warning(f"Skipping box {i} which spans the entire image boundaries")
                                continue
                                
                            # Skip tiny boxes
                            if box_area < 0.001:
                                self.logger.warning(f"Skipping box {i} which is too small")
                                continue
                                
                            valid_indices.append(i)
                        
                        # Process only valid detections
                        for i in valid_indices:
                            if i >= len(all_masks):
                                self.logger.warning(f"No mask available for detection {i}")
                                continue
                                
                            mask = all_masks[i]
                            box = filtered_xyxy[i]
                            score = filtered_confidence[i]
                            class_id = filtered_class_id[i]
                            
                            # Create detection object
                            x1, y1, x2, y2 = box
                            detection = {
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(score),
                                'class': 'tree',  # Basic class, will be refined later
                                'class_id': 0,  # Tree class
                                'detection_type': 'groundingsam'
                            }
                            
                            # Calculate mask coverage percentage
                            true_pixels = np.sum(mask)
                            total_pixels = mask.shape[0] * mask.shape[1]
                            coverage_percent = float(true_pixels) / total_pixels if total_pixels > 0 else 0
                            detection['mask_coverage'] = coverage_percent
                            detection['mask_score'] = float(score)
                            
                            # Add a unique mask ID for reference
                            mask_id = f"mask_grounding_{i}"
                            detection['mask_id'] = mask_id
                            
                            # Save mask to separate file if directory is configured
                            if masks_dir:
                                mask_path = os.path.join(masks_dir, f"{mask_id}.npz")
                                np.savez_compressed(mask_path, mask=mask, score=score)
                                detection['mask_path'] = os.path.relpath(mask_path, self.config.ml_path)
                                self.logger.info(f"Saved mask {mask_id} to {mask_path}")
                            
                            # Add to detections list
                            grounding_detections.append(detection)
                            
                        # If no valid detections, try segmenting the image directly
                        if len(grounding_detections) == 0:
                            self.logger.warning("No valid detections found, trying direct image segmentation")
                            try:
                                # Use SAM to generate masks directly
                                from segment_anything import SamAutomaticMaskGenerator
                                
                                # Create mask generator with parameters optimized for trees
                                mask_generator = SamAutomaticMaskGenerator(
                                    self.sam_model.model,
                                    points_per_side=32,        # Higher density of points
                                    pred_iou_thresh=0.86,      # Higher threshold for higher quality masks
                                    stability_score_thresh=0.92, # Higher threshold for stability
                                    crop_n_layers=1,           # Use cropping for large images
                                    crop_n_points_downscale_factor=2,
                                    min_mask_region_area=100,  # Minimum size in pixels
                                )
                                
                                # Generate masks
                                masks = mask_generator.generate(image_data_uint8)
                                self.logger.info(f"Generated {len(masks)} masks directly with SAM")
                                
                                # Sort masks by area (descending)
                                masks = sorted(masks, key=lambda x: x['area'], reverse=True)
                                
                                # Take only the largest masks (likely to be trees)
                                h, w = image_data_uint8.shape[:2]
                                for i, mask_data in enumerate(masks[:20]):  # Limit to top 20 masks
                                    # Skip tiny masks
                                    if mask_data['area'] < 400:  # Skip if too small
                                        continue
                                        
                                    # Get bounding box from mask
                                    binary_mask = mask_data['segmentation']
                                    y_indices, x_indices = np.where(binary_mask)
                                    if len(y_indices) == 0 or len(x_indices) == 0:
                                        continue
                                        
                                    x1, x2 = np.min(x_indices), np.max(x_indices)
                                    y1, y2 = np.min(y_indices), np.max(y_indices)
                                    
                                    # Convert to normalized coordinates
                                    x1_norm, y1_norm = float(x1) / w, float(y1) / h
                                    x2_norm, y2_norm = float(x2) / w, float(y2) / h
                                    
                                    # Calculate area to filter out whole-image detections
                                    box_area = (x2_norm - x1_norm) * (y2_norm - y1_norm)
                                    
                                    # Skip if box covers too much of the image (likely a false detection)
                                    if box_area > WHOLE_IMAGE_AREA_THRESHOLD:
                                        self.logger.warning(f"Skipping large SAM automatic detection with area {box_area:.2f}")
                                        continue
                                        
                                    # Also skip if the box is at the image edges
                                    if (x1_norm < 0.03 and y1_norm < 0.03 and 
                                        x2_norm > 0.97 and y2_norm > 0.97):
                                        self.logger.warning(f"Skipping edge-spanning SAM automatic detection")
                                        continue
                                    
                                    # Create detection
                                    detection = {
                                        'bbox': [x1_norm, y1_norm, x2_norm, y2_norm],
                                        'confidence': float(mask_data['stability_score']),
                                        'class': 'tree',
                                        'class_id': 0,
                                        'detection_type': 'sam_automatic'
                                    }
                                    
                                    # Add mask info
                                    mask_id = f"mask_auto_{i}"
                                    detection['mask_id'] = mask_id
                                    detection['mask_score'] = float(mask_data['stability_score'])
                                    
                                    # Calculate coverage
                                    true_pixels = np.sum(binary_mask)
                                    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
                                    coverage_percent = float(true_pixels) / total_pixels
                                    detection['mask_coverage'] = coverage_percent
                                    
                                    # Save mask
                                    if masks_dir:
                                        mask_path = os.path.join(masks_dir, f"{mask_id}.npz")
                                        np.savez_compressed(mask_path, mask=binary_mask, score=mask_data['stability_score'])
                                        detection['mask_path'] = os.path.relpath(mask_path, self.config.ml_path)
                                    
                                    # Add to list
                                    grounding_detections.append(detection)
                                    all_masks.append(binary_mask)
                            except Exception as e:
                                self.logger.error(f"Error with direct segmentation: {e}")
                                
                        self.logger.info(f"Final detection count: {len(grounding_detections)}")
                            
                        # Create visualization to verify results
                        if vis_dir:
                            try:
                                # First check if we have any valid detections (not just the entire image)
                                valid_xyxy = []
                                valid_confidence = []
                                valid_class_id = []
                                valid_masks = []
                                
                                for i, box in enumerate(filtered_xyxy):
                                    # Skip boxes that are almost the entire image
                                    if box[0] < 0.05 and box[1] < 0.05 and box[2] > 0.95 and box[3] > 0.95:
                                        continue
                                    valid_xyxy.append(box)
                                    valid_confidence.append(filtered_confidence[i])
                                    valid_class_id.append(filtered_class_id[i])
                                    if i < len(all_masks):
                                        valid_masks.append(all_masks[i])
                                
                                # Convert normalized coordinates to pixel coordinates for visualization
                                h, w = image_data_uint8.shape[:2]
                                pixel_xyxy = []
                                for box in valid_xyxy:
                                    x1, y1, x2, y2 = box
                                    pixel_xyxy.append([
                                        x1 * w, y1 * h, x2 * w, y2 * h
                                    ])
                                
                                # Create visualization with all masks
                                import supervision as sv
                                
                                # Create detections object for visualization
                                from supervision.detection.core import Detections
                                
                                if len(valid_xyxy) > 0 and len(valid_masks) == len(valid_xyxy):
                                    # Use the valid detections
                                    sv_detections = Detections(
                                        xyxy=np.array(pixel_xyxy),
                                        confidence=np.array(valid_confidence),
                                        class_id=np.array(valid_class_id),
                                        mask=np.array(valid_masks)
                                    )
                                    
                                    # Create annotators
                                    box_annotator = sv.BoxAnnotator()
                                    mask_annotator = sv.MaskAnnotator()
                                    
                                    # Create labels
                                    labels = [
                                        f"Tree {confidence:0.2f}" 
                                        for confidence
                                        in valid_confidence
                                    ]
                                    
                                    # Create visualization
                                    annotated_image = mask_annotator.annotate(scene=image_data_uint8.copy(), detections=sv_detections)
                                    annotated_image = box_annotator.annotate(scene=annotated_image, detections=sv_detections, labels=labels)
                                    
                                    # Save visualization
                                    vis_path = os.path.join(vis_dir, "combined_visualization.jpg")
                                    cv2.imwrite(vis_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                                    self.logger.info(f"Saved visualization with {len(valid_xyxy)} valid detections to {vis_path}")
                                elif len(detections) > 0:
                                    # Use existing detections if available and no valid grounding detections
                                    self.logger.info(f"Creating visualization from {len(detections)} existing detections")
                                    
                                    # Create a simple visualization with bounding boxes
                                    image_copy = image_data_uint8.copy()
                                    
                                    for i, detection in enumerate(detections):
                                        bbox = detection['bbox']
                                        x1, y1, x2, y2 = [int(bbox[0] * w), int(bbox[1] * h), 
                                                          int(bbox[2] * w), int(bbox[3] * h)]
                                        confidence = detection['confidence']
                                        
                                        # Draw box
                                        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        
                                        # Add label
                                        label = f"Tree {confidence:.2f}"
                                        cv2.putText(image_copy, label, (x1, y1-10), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    # Save visualization
                                    vis_path = os.path.join(vis_dir, "combined_visualization.jpg")
                                    cv2.imwrite(vis_path, cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
                                    self.logger.info(f"Saved basic visualization with {len(detections)} detections to {vis_path}")
                                else:
                                    # No valid detections at all - create a placeholder visualization
                                    self.logger.warning("No valid detections for visualization")
                                    vis_path = os.path.join(vis_dir, "combined_visualization.jpg")
                                    cv2.imwrite(vis_path, cv2.cvtColor(image_data_uint8, cv2.COLOR_RGB2BGR))
                                    self.logger.info(f"Saved original image without detections to {vis_path}")
                            except Exception as e:
                                self.logger.error(f"Error creating visualization: {e}")
                                
                        # Update detections with GroundingDINO results
                        # Check if we got any valid detections (not just one big box)
                        valid_detections = [d for d in grounding_detections 
                                           if not (d['bbox'][0] < 0.05 and d['bbox'][1] < 0.05 and 
                                                  d['bbox'][2] > 0.95 and d['bbox'][3] > 0.95)]
                        
                        if len(valid_detections) > 0:
                            self.logger.info(f"Adding {len(valid_detections)} valid GroundingDINO detections")
                            
                            # If we have both existing detections and valid new ones, use whichever set is larger
                            if len(valid_detections) > len(detections):
                                # Replace detections with the valid grounding detections
                                detections.clear()
                                detections.extend(valid_detections)
                            else:
                                # Enhance existing detections with better masks
                                self._enhance_existing_masks(detections, all_masks, filtered_xyxy)
                        else:
                            self.logger.warning("GroundingDINO only found the entire image - falling back to DeepForest detections")
                            
                            # If no valid GroundingDINO detections, try with DeepForest as fallback
                            if len(detections) == 0:
                                self.logger.info("No existing detections - trying DeepForest as fallback")
                                from deepforest import main
                                try:
                                    model = main.deepforest()
                                    model.use_release()
                                    # Get boxes using DeepForest's predict_image
                                    # Save a temporary copy of the image for DeepForest
                                    temp_path = os.path.join(self.config.ml_path, "temp_deepforest.jpg")
                                    cv2.imwrite(temp_path, cv2.cvtColor(image_data_uint8, cv2.COLOR_RGB2BGR))
                                    
                                    # Run DeepForest prediction
                                    boxes = model.predict_image(path=temp_path, return_plot=False)
                                    
                                    # Convert DeepForest boxes to our format
                                    for _, box in boxes.iterrows():
                                        # Convert coordinates to normalized format
                                        h, w = image_data_uint8.shape[:2]
                                        x1 = box['xmin'] / w
                                        y1 = box['ymin'] / h
                                        x2 = box['xmax'] / w
                                        y2 = box['ymax'] / h
                                        
                                        detection = {
                                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                            'confidence': float(box['score']),
                                            'class': 'tree',
                                            'detection_type': 'deepforest'
                                        }
                                        detections.append(detection)
                                    
                                    # Now get segmentation masks for these detections
                                    self._fallback_segmentation(detections, image_data_uint8, masks_dir)
                                    
                                    # Clean up temp file
                                    if os.path.exists(temp_path):
                                        os.remove(temp_path)
                                except Exception as df_err:
                                    self.logger.error(f"DeepForest fallback failed: {df_err}")
                    
                    except Exception as e:
                        self.logger.error(f"Error in GroundingDINO detection: {e}")
                        # Fall back to basic SAM segmentation
                        self._fallback_segmentation(detections, image_data, masks_dir)
                else:
                    self.logger.warning(f"Unexpected image format for GroundingDINO: {image_data.shape}, using basic SAM")
                    # Fall back to basic SAM segmentation
                    self._fallback_segmentation(detections, image_data, masks_dir)
            else:
                # Fall back to basic SAM segmentation
                self._fallback_segmentation(detections, image_data, masks_dir)
                
            # After we have the segmentation masks, assign tree categories for analytics
            # This happens as a separate step from segmentation
            for detection in detections:
                if 'mask_coverage' in detection:
                    self._assign_tree_category(detection, mask_coverage=detection['mask_coverage'])
                else:
                    self._assign_tree_category(detection)
                
        except Exception as e:
            self.logger.error(f"Error adding segmentation masks: {e}")
            
    def _enhance_existing_masks(self, detections, all_masks, filtered_xyxy):
        """Enhance existing detections with better masks from GroundingDINO + SAM"""
        self.logger.info("Enhancing existing masks with GroundingDINO + SAM results")
        
        # Create a mapping from bbox to mask
        better_masks = {}
        for i, box in enumerate(filtered_xyxy):
            x1, y1, x2, y2 = box
            box_key = f"{x1:.1f}_{y1:.1f}_{x2:.1f}_{y2:.1f}"
            better_masks[box_key] = all_masks[i] if i < len(all_masks) else None
            
        # Update existing detections if there's a matching bbox
        for detection in detections:
            bbox = detection['bbox']
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                box_key = f"{x1:.1f}_{y1:.1f}_{x2:.1f}_{y2:.1f}"
                
                if box_key in better_masks and better_masks[box_key] is not None:
                    mask = better_masks[box_key]
                    
                    # Calculate mask coverage percentage
                    true_pixels = np.sum(mask)
                    total_pixels = mask.shape[0] * mask.shape[1]
                    coverage_percent = float(true_pixels) / total_pixels if total_pixels > 0 else 0
                    detection['mask_coverage'] = coverage_percent
                    
                    # Mark as enhanced
                    detection['mask_enhanced'] = True
                    
                    self.logger.info(f"Enhanced mask for detection with confidence {detection['confidence']}")
            
    def _fallback_segmentation(self, detections, image_data, masks_dir):
        """Fall back to basic SAM segmentation when GroundingDINO is not available"""
        self.logger.info("Using basic SAM segmentation")
        
        # Set the image for SAM
        self.sam_model.set_image(image_data)
        
        # Process each detection
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            # Convert bbox format for SAM
            x1, y1, x2, y2 = bbox
            
            # Get center point for point prompt
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Create point prompts
            input_point = np.array([[center_x, center_y]])
            input_label = np.array([1])  # 1 for foreground
            
            # Generate masks
            masks, scores, _ = self.sam_model.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=np.array([x1, y1, x2, y2]),
                multimask_output=True
            )
            
            # Get the best mask
            if len(masks) > 0:
                best_mask_idx = np.argmax(scores)
                mask = masks[best_mask_idx]
                
                # Store only the mask score, not the full mask data
                detection['mask_score'] = float(scores[best_mask_idx])
                
                # Calculate mask coverage percentage for a lightweight representation
                true_pixels = np.sum(mask)
                total_pixels = mask.shape[0] * mask.shape[1]
                coverage_percent = float(true_pixels) / total_pixels if total_pixels > 0 else 0
                detection['mask_coverage'] = coverage_percent
                
                # Add a unique mask ID for reference
                mask_id = f"mask_{i}"
                detection['mask_id'] = mask_id
                
                # Save mask to separate file if directory is configured
                if masks_dir:
                    mask_path = os.path.join(masks_dir, f"{mask_id}.npz")
                    np.savez_compressed(mask_path, mask=mask, score=scores[best_mask_idx])
                    detection['mask_path'] = os.path.relpath(mask_path, self.config.ml_path)
                    self.logger.info(f"Saved mask {mask_id} to {mask_path}")
                
                self.logger.info(f"Added mask score for detection with confidence {detection['confidence']}")
            
    def _assign_tree_category(self, detection, mask_coverage=None):
        """
        Assign tree risk categories based on detection characteristics.
        This is a separate analytics step after segmentation.
        """
        # Calculate bounding box area (proxy for tree size)
        bbox = detection['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        confidence = detection['confidence']
        
        # If this is a GroundingDINO detection, set a higher base confidence
        # as these are more reliable for tree segmentation
        if detection.get('detection_type') == 'groundingsam':
            confidence = max(confidence, 0.7)
        
        # Preserve the original class if it exists (for UI compatibility)
        original_class = detection.get('class', 'tree')
        
        # Base categories that match frontend expectations
        risk_categories = [
            'healthy_tree',
            'hazardous_tree',
            'dead_tree',
            'low_canopy_tree',
            'pest_disease_tree',
            'flood_prone_tree',
            'utility_conflict_tree',
            'structural_hazard_tree',
            'fire_risk_tree'
        ]
        
        # Size-based classification - this is analytics after the fact
        # We don't use these categories for the initial segmentation
        if area > 0.05:  # Large trees
            if confidence < 0.3:
                category = 'hazardous_tree'
                risk = 7
            elif confidence < 0.5:
                category = 'utility_conflict_tree'
                risk = 6
            else:
                category = 'healthy_tree'
                risk = 0
        elif area > 0.02:  # Medium trees
            if confidence < 0.4:
                category = 'pest_disease_tree'
                risk = 4
            elif mask_coverage and mask_coverage < 0.3:
                category = 'dead_tree'
                risk = 2
            else:
                category = 'healthy_tree'
                risk = 0
        else:  # Small trees
            if confidence < 0.35:
                category = 'low_canopy_tree'
                risk = 3
            elif confidence < 0.6:
                category = 'flood_prone_tree'
                risk = 5
            else:
                category = 'fire_risk_tree'
                risk = 8
        
        # Update detection with category and risk data for analytics
        detection['category'] = category
        detection['analytics_class'] = category.replace('_', ' ')
        detection['risk_score'] = risk
        
        # Keep the original class (usually just 'tree') as the main class
        # This ensures that segmentation is focused on just finding trees
        # and not trying to segment specific types of trees
        detection['class'] = original_class
            
    async def _detect_with_gemini(self, image_path):
        """Use Gemini API for tree detection as fallback"""
        self.logger.info("Attempting tree detection with Gemini API")
        
        if not self.gemini_client:
            return []
            
        try:
            # Encode image for Gemini API
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                
            # Create prompt for Gemini
            prompt = """
            Analyze this satellite or aerial image and identify all trees present.
            For each tree you detect, provide:
            1. The bounding box coordinates [x1, y1, x2, y2] in normalized format (0-1 range)
            2. A confidence score between 0 and 1
            3. Any distinguishing features of the tree (species if identifiable, size, health)
            
            Format your response as JSON with the following structure:
            {
              "trees": [
                {
                  "bbox": [x1, y1, x2, y2],
                  "confidence": 0.95,
                  "features": "large healthy oak tree"
                },
                ...
              ]
            }
            """
            
            # Call Gemini API
            response = await self.gemini_client.query_image(prompt, image_b64)
            
            if response and response.get('success'):
                # Parse JSON from response
                try:
                    # Extract JSON from text response
                    response_text = response.get('response', '')
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        result = json.loads(json_str)
                        
                        # Format detections
                        gemini_detections = []
                        for i, tree in enumerate(result.get('trees', [])):
                            gemini_detections.append({
                                'bbox': tree.get('bbox', [0, 0, 1, 1]),
                                'confidence': tree.get('confidence', 0.5),
                                'class_id': 0,  # Always tree
                                'detection_type': 'gemini',
                                'features': tree.get('features', '')
                            })
                            
                        self.logger.info(f"Gemini detected {len(gemini_detections)} trees")
                        return gemini_detections
                except Exception as e:
                    self.logger.error(f"Error parsing Gemini response: {e}")
                    
            return []
                
        except Exception as e:
            self.logger.error(f"Error in Gemini tree detection: {e}")
            return []

# =====================================
# Model Management
# =====================================
class ModelManager:
    """Handles model loading and management"""
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = TreeDetectionModel(config)
        
    async def initialize(self):
        """Initialize model and prepare for inference"""
        self.logger.info("Initializing model for inference")
        return True
    
    def detect(self, image_path):
        """Run detection on an image"""
        return self.model.detect(image_path)
        
    async def export_onnx(self) -> str:
        """Export model to ONNX format for inference engine"""
        try:
            self.logger.info("Exporting model to ONNX format")
            
            # Create paths for export
            export_path = Path(self.config.export_path) / "tree_detection_model.onnx"
            os.makedirs(os.path.dirname(str(export_path)), exist_ok=True)
            
            # Create a simple ONNX model with expected outputs
            import onnx
            from onnx import helper
            from onnx import TensorProto
            
            # Input (Batch x Channels x Height x Width)
            X = helper.make_tensor_value_info('input', TensorProto.FLOAT, 
                                             [None, self.config.input_shape[0], 
                                              self.config.input_shape[1], self.config.input_shape[2]])
            
            # Outputs - compatible with the inference engine 
            # (yolo_detections, deepforest_detections, segmentation_masks)
            Y1 = helper.make_tensor_value_info('yolo_detections', TensorProto.FLOAT, [None, 100, 6])
            Y2 = helper.make_tensor_value_info('deepforest_detections', TensorProto.FLOAT, [None, 100, 5])
            Y3 = helper.make_tensor_value_info('segmentation_masks', TensorProto.FLOAT, [None, 10, 640, 640])
            
            # Create nodes for the graph - identity nodes for now
            identity_node1 = helper.make_node(
                'Identity',
                ['input'],
                ['temp1'],
                name='identity_node1'
            )
            
            # Create constant nodes for outputs
            zero_tensor_y1 = helper.make_tensor('zero_y1', TensorProto.FLOAT, [1, 0, 6], [])
            const_node_y1 = helper.make_node(
                'Constant',
                [],
                ['yolo_detections'],
                name='const_node_y1',
                value=zero_tensor_y1
            )
            
            zero_tensor_y2 = helper.make_tensor('zero_y2', TensorProto.FLOAT, [1, 0, 5], [])
            const_node_y2 = helper.make_node(
                'Constant',
                [],
                ['deepforest_detections'],
                name='const_node_y2',
                value=zero_tensor_y2
            )
            
            zero_tensor_y3 = helper.make_tensor('zero_y3', TensorProto.FLOAT, [1, 0, 10, 10], [])
            const_node_y3 = helper.make_node(
                'Constant',
                [],
                ['segmentation_masks'],
                name='const_node_y3',
                value=zero_tensor_y3
            )
            
            # Create the graph
            graph_def = helper.make_graph(
                [identity_node1, const_node_y1, const_node_y2, const_node_y3],
                'tree_detection_model',
                [X],
                [Y1, Y2, Y3]
            )
            
            # Create the model
            model_def = helper.make_model(graph_def, producer_name='tree_detection_onnx')
            model_def.opset_import[0].version = 11
            
            # Save the model
            onnx.save(model_def, str(export_path))
            
            self.logger.info(f"Model exported to {export_path}")
            return str(export_path)
            
        except Exception as e:
            self.logger.error(f"Failed to export model to ONNX: {e}")
            # Create a fallback ONNX model if the export fails
            fallback_path = Path(self.config.export_path) / "fallback_model.onnx"
            try:
                import onnx
                from onnx import helper
                from onnx import TensorProto
                
                # Input (Batch x Channels x Height x Width)
                X = helper.make_tensor_value_info('input', TensorProto.FLOAT, 
                                                [None, self.config.input_shape[0], 
                                                self.config.input_shape[1], self.config.input_shape[2]])
                
                # Outputs
                Y1 = helper.make_tensor_value_info('yolo_detections', TensorProto.FLOAT, [None, 0, 6])
                Y2 = helper.make_tensor_value_info('deepforest_detections', TensorProto.FLOAT, [None, 0, 5])
                Y3 = helper.make_tensor_value_info('segmentation_masks', TensorProto.FLOAT, [None, 0, 10, 10])
                
                # Create an identity node
                identity_node = helper.make_node(
                    'Identity',
                    ['input'],
                    ['temp_output'],
                    name='identity_node'
                )
                
                # Create constant nodes for outputs
                zero_tensor_y1 = helper.make_tensor('y1_value', TensorProto.FLOAT, [1, 0, 6], [])
                const_node_y1 = helper.make_node(
                    'Constant',
                    [],
                    ['yolo_detections'],
                    name='const_node_y1',
                    value=zero_tensor_y1
                )
                
                zero_tensor_y2 = helper.make_tensor('y2_value', TensorProto.FLOAT, [1, 0, 5], [])
                const_node_y2 = helper.make_node(
                    'Constant',
                    [],
                    ['deepforest_detections'],
                    name='const_node_y2',
                    value=zero_tensor_y2
                )
                
                zero_tensor_y3 = helper.make_tensor('y3_value', TensorProto.FLOAT, [1, 0, 10, 10], [])
                const_node_y3 = helper.make_node(
                    'Constant',
                    [],
                    ['segmentation_masks'],
                    name='const_node_y3',
                    value=zero_tensor_y3
                )
                
                # Create the graph
                graph_def = helper.make_graph(
                    [identity_node, const_node_y1, const_node_y2, const_node_y3],
                    'fallback_model',
                    [X],
                    [Y1, Y2, Y3]
                )
                
                # Create the model
                model_def = helper.make_model(graph_def, producer_name='tree_detection_fallback')
                model_def.opset_import[0].version = 11
                
                # Save the model
                onnx.save(model_def, str(fallback_path))
                
                self.logger.info(f"Created fallback ONNX model at {fallback_path}")
                return str(fallback_path)
            except Exception as inner_e:
                self.logger.error(f"Failed to create fallback ONNX model: {inner_e}")
                raise RuntimeError(f"Failed to export model to ONNX: {e}, and fallback failed: {inner_e}")

# =====================================
# Inference Engine
# =====================================
class InferenceEngine:
    """Handles model inference with ONNX Runtime"""
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = None
        self.model_manager = ModelManager(config)
        
    async def initialize(self, model_path: str):
        """Initialize ONNX Runtime session or direct model inference"""
        self.logger.info(f"Initializing inference engine with model {model_path}")
        
        # We won't use the ONNX model directly but will use direct model inference
        # Still need to initialize the model manager
        await self.model_manager.initialize()
        
        # Store the model path for reference
        self.model_path = model_path
        self.logger.info("Inference engine initialized for direct model inference")
        return True
        
    async def infer(self, input_data):
        """Run inference on input data using the model manager"""
        try:
            # Log the input data
            if isinstance(input_data, np.ndarray):
                self.logger.info(f"Running inference on input data with shape {input_data.shape}")
            
            # Start timing
            start_time = datetime.now()
            
            # Use the model manager's detect method directly
            results = self.model_manager.detect(input_data)
            
            # Calculate inference time
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Log results
            yolo_count = len(results.get('trees', []))
            self.logger.info(f"Inference completed in {inference_time:.4f} seconds with {yolo_count} detections")
            
            # Convert to expected format for the rest of the pipeline
            tree_boxes = []
            for tree in results.get('trees', []):
                # Convert to format expected by tree_features extraction
                bbox = tree.get('bbox', [0, 0, 0, 0])
                confidence = tree.get('confidence', 0.0)
                class_id = tree.get('class_id', 0)
                
                # Format as [x1, y1, x2, y2, confidence, class_id]
                box = np.array([bbox[0], bbox[1], bbox[2], bbox[3], confidence, class_id])
                tree_boxes.append(box)
            
            # Format results in the expected structure
            output = {
                'yolo_detections': np.array(tree_boxes) if tree_boxes else np.array([]),
                'deepforest_detections': {'boxes': np.array([]), 'scores': np.array([])},
                'segmentation_masks': np.array([]),
                'inference_time': inference_time
            }
            
            return output
            
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            
            # Return minimal inference results
            return {
                'yolo_detections': np.array([]),
                'deepforest_detections': {'boxes': np.array([]), 'scores': np.array([])},
                'segmentation_masks': np.array([]),
                'inference_time': 0.0
            }

# =====================================
# Feature Extraction
# =====================================
class TreeFeatureExtractor:
    """Basic feature extraction from detections"""
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def extract(self, detections, image_data, lidar_data=None):
        """Extract features from detections"""
        # Just return the detections directly
        return {
            'tree_features': detections.get('trees', [])
        }

# =====================================
# Risk Assessment
# =====================================
class TreeRiskAssessment:
    """Minimal tree risk assessment"""
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def assess_risk(self, features):
        """Simple risk assessment"""
        trees = features.get('tree_features', [])
        
        # Just assign a random risk level
        risk_factors = []
        for i, tree in enumerate(trees):
            risk_factors.append({
                'tree_id': i,
                'bbox': tree.get('bbox', [0, 0, 0, 0]),
                'risk_score': 3.0,  # Default score
                'risk_level': "Medium",  # Default level
                'health_score': 0.7,  # Default health
                'confidence': tree.get('confidence', 0.0)
            })
            
        return {
            'risk_factors': risk_factors,
            'overall_risk': {
                'risk_level': "Medium",
                'risk_score': 3.0
            }
        }
        
    async def generate_json_results(self, features, area_id, geo_bounds=None):
        """Generate JSON results for frontend"""
        trees = features.get('tree_features', [])
        
        # Check if masks directory exists
        masks_dir = os.path.join(self.config.ml_path, f"detection_{area_id}", "masks")
        masks_dir_exists = os.path.isdir(masks_dir)
        
        # Group trees by category for frontend processing
        categorized_trees = {}
        
        # Remove segmentation data from trees to keep the JSON file size manageable
        cleaned_trees = []
        for tree in trees:
            # Create a filtered copy of the tree data without segmentation
            cleaned_tree = {}
            for key, value in tree.items():
                # Skip segmentation data and large binary mask arrays
                if key not in ['segmentation', 'segmentation_mask', 'mask_data']:
                    cleaned_tree[key] = value
            
            # Add the tree with calculated location
            if 'bbox' in cleaned_tree:
                # If geo_bounds are available, calculate real-world coordinates
                if geo_bounds and len(geo_bounds) == 2:
                    bbox = cleaned_tree['bbox']
                    # Calculate center point of the bbox
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Convert normalized coordinates to geographic coordinates
                    # Interpolate between SW and NE corners
                    sw_lng, sw_lat = geo_bounds[0]
                    ne_lng, ne_lat = geo_bounds[1]
                    
                    # Calculate longitude and latitude
                    lng = center_x * ne_lng + (1 - center_x) * sw_lng
                    lat = center_y * ne_lat + (1 - center_y) * sw_lat
                    
                    # Add location to the tree data
                    cleaned_tree['location'] = [lng, lat]
                    
            # Add to categorized trees
            category = cleaned_tree.get('category', 'healthy_tree')  # Default to healthy_tree
            if category not in categorized_trees:
                categorized_trees[category] = []
            categorized_trees[category].append(cleaned_tree)
            
            # Also keep in the main trees list for backward compatibility
            cleaned_trees.append(cleaned_tree)
        
        # Create a basic results structure
        json_output = {
            "job_id": area_id,
            "status": "complete",
            "timestamp": datetime.now().isoformat(),
            "tree_count": len(cleaned_trees),
            "trees": cleaned_trees,  # Keep main trees array for backward compatibility
            "metadata": {
                "include_bounding_boxes": self.config.include_bounding_boxes,
                "masks_directory": f"detection_{area_id}/masks" if masks_dir_exists else None
            }
        }
        
        # Add categorized trees to output
        for category, trees_list in categorized_trees.items():
            json_output[category] = trees_list
        
        # Add geo bounds if available
        if geo_bounds:
            json_output["metadata"]["bounds"] = geo_bounds
        
        # Add specific counts for frontend
        category_counts = {category: len(trees_list) for category, trees_list in categorized_trees.items()}
        json_output["category_counts"] = category_counts
        
        # Check if trees were found
        if not trees:
            json_output["status"] = "complete_no_detections"
            json_output["message"] = "No trees were detected in this area. Try a different location."
            
        return json_output

# =====================================
# Entry Point
# =====================================
async def detect_trees_async(image_path, output_path=None, geo_bounds=None, area_id=None):
    """
    Detect trees in an image using DeepForest, SAM with GroundingDINO, and Gemini
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save results
        geo_bounds: Optional geographic bounds for the image [sw_corner, ne_corner]
        area_id: Optional area/job ID for consistent file naming
        
    Returns:
        Dictionary with detection results
    """
    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize config
    config = MLConfig()
    if output_path:
        config.output_path = output_path
    
    # Set job_id from area_id for consistent file naming
    if area_id:
        config.job_id = area_id
        logger.info(f"Using area_id {area_id} for consistent file naming")
    
    # Create tree detection model
    model = TreeDetectionModel(config)
    
    # Run detection
    results = model.detect(image_path)
    
    # Try to get Gemini detections if enabled and no trees were found
    if config.use_gemini and results['tree_count'] == 0 and model.gemini_client:
        try:
            gemini_detections = await model._detect_with_gemini(image_path)
            if gemini_detections:
                results = {
                    'tree_count': len(gemini_detections),
                    'trees': gemini_detections
                }
                logger.info(f"Using {len(gemini_detections)} trees detected by Gemini")
        except Exception as e:
            logger.error(f"Error in Gemini detection: {e}")
    
    # If geo_bounds are provided, add locations to tree detections
    if geo_bounds and len(geo_bounds) == 2:
        logger.info(f"Adding geographic coordinates based on bounds: {geo_bounds}")
        sw_lng, sw_lat = geo_bounds[0]
        ne_lng, ne_lat = geo_bounds[1]
        
        for tree in results.get('trees', []):
            if 'bbox' in tree:
                bbox = tree['bbox']
                # Calculate center point of the bbox
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Convert normalized coordinates to geographic coordinates
                lng = sw_lng + center_x * (ne_lng - sw_lng)
                lat = sw_lat + center_y * (ne_lat - sw_lat)
                
                # Add location to tree
                tree['location'] = [float(lng), float(lat)]
                
        # Add bounds to results metadata
        if 'metadata' not in results:
            results['metadata'] = {}
        results['metadata']['bounds'] = geo_bounds
    
    # Add S2 cell information if needed for geospatial indexing
    try:
        if geo_bounds and len(geo_bounds) == 2:
            import s2sphere
            # Calculate center point of the image
            center_lng = (geo_bounds[0][0] + geo_bounds[1][0]) / 2
            center_lat = (geo_bounds[0][1] + geo_bounds[1][1]) / 2
            
            # Create S2 cell at appropriate level
            s2_level = 15  # Good balance for tree-sized objects
            s2_latlng = s2sphere.LatLng.from_degrees(center_lat, center_lng)
            s2_cellid = s2sphere.CellId.from_lat_lng(s2_latlng).parent(s2_level)
            
            # Add S2 cell information to results
            if 'metadata' not in results:
                results['metadata'] = {}
            results['metadata']['s2_cell'] = str(s2_cellid)
            results['metadata']['s2_token'] = s2_cellid.to_token()
            results['metadata']['s2_level'] = s2_level
            
            logger.info(f"Added S2 cell information: {s2_cellid.to_token()} (level {s2_level})")
    except ImportError:
        logger.warning("s2sphere library not available, skipping S2 cell indexing")
    except Exception as e:
        logger.error(f"Error adding S2 cell information: {e}")
    
    # Save results if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return results

def detect_trees(image_path, output_path=None, job_id=None, geo_bounds=None):
    """
    Synchronous wrapper for tree detection
    
    Args:
        image_path: Path to the input image
        output_path: Optional path to save results
        job_id: Optional job ID for consistent file naming
        geo_bounds: Optional geographic bounds for the image [sw_corner, ne_corner]
        
    Returns:
        Dictionary with detection results
    """
    # Fixed version that uses global models if available
    try:
        # Create config
        config = MLConfig()
        
        # Store job_id for consistent temp file naming
        if job_id:
            config.job_id = job_id
        
        global _deepforest_model, _model_loaded
        
        # Use global model if already loaded
        if _deepforest_model is not None and _model_loaded:
            logging.info("Using existing global DeepForest model")
            model = _deepforest_model
        else:
            # Import needed components
            from deepforest import main
            import pandas as pd
            import torch
            
            # Initialize DeepForest model
            logging.info("Initializing DeepForest for tree detection with GPU")
            model = main.deepforest()
            model.use_release()
            
            # Set model to eval mode BEFORE moving to GPU
            model.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                logging.info(f"Moving DeepForest model to GPU: {torch.cuda.get_device_name(0)}")
                model.model.to("cuda:0")
                # Update config for GPU
                model.config["devices"] = [0]
                # Create trainer with updated config
                model.create_trainer()
                
                # Verify model device
                device = next(model.model.parameters()).device
                logging.info(f"DeepForest model is on device: {device}")
            
            # Store model in global variable
            _deepforest_model = model
            _model_loaded = True
            logging.info("Stored DeepForest model in global cache for reuse")
        
        # Run detection with proper device handling
        try:
            # Direct tensor-based inference with device handling
            logging.info("Using direct tensor-based inference with proper device handling")
            
            # Get the model device
            device = next(model.model.parameters()).device
            logging.info(f"Model is on device: {device}")
            
            # Import required modules - ensure local import
            try:
                from PIL import Image as PILImage
            except ImportError:
                logging.error("Failed to import PIL.Image in detect_trees")
                raise ImportError("PIL.Image not available")
                
            from deepforest import preprocess
            
            # Load and preprocess image
            image_array = np.array(PILImage.open(image_path))
            processed_image = preprocess.preprocess_image(image_array)
            
            # Create tensor with explicit device placement
            logging.info(f"Creating tensor and explicitly placing on {device}")
            tensor_input = torch.tensor(processed_image, dtype=torch.float32, device=device)
            
            # Add batch dimension if needed
            if len(tensor_input.shape) == 3:
                tensor_input = tensor_input.unsqueeze(0)
                
            logging.info(f"Input tensor on device: {tensor_input.device}, shape: {tensor_input.shape}")
            
            # Verify device match
            model_device = next(model.model.parameters()).device
            if str(tensor_input.device) != str(model_device):
                logging.warning(f"Device mismatch! Tensor on {tensor_input.device}, model on {model_device}")
                tensor_input = tensor_input.to(model_device)
                logging.info(f"Moved tensor to {tensor_input.device}")
            
            # Run inference with explicit device handling
            from deepforest.utilities import annotations_to_shapefile
            import pandas as pd
            
            with torch.no_grad():
                logging.info(f"Running forward pass on {tensor_input.device}")
                predictions = model.model(tensor_input)[0]  # Get first item from batch
                
                # Process predictions
                if len(predictions['boxes']) > 0:
                    # Convert tensors to numpy
                    boxes_np = predictions['boxes'].cpu().numpy() 
                    scores_np = predictions['scores'].cpu().numpy()
                    labels_np = predictions['labels'].cpu().numpy()
                    
                    # Create DataFrame manually instead of using annotations_to_shapefile
                    logging.info("Creating DataFrame manually from detection results")
                    
                    # Initialize empty DataFrame
                    boxes = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
                    
                    # Add detections
                    for i, (box, score, label_idx) in enumerate(zip(boxes_np, scores_np, labels_np)):
                        # Convert label index to string
                        label_name = model.numeric_to_label_dict.get(label_idx, f"class_{label_idx}")
                        
                        # Add row to DataFrame
                        boxes.loc[i] = [
                            float(box[0]), float(box[1]), 
                            float(box[2]), float(box[3]), 
                            float(score), label_name
                        ]
                else:
                    # No detections
                    boxes = pd.DataFrame(columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
                
            logging.info(f"Successfully ran inference with tensor on {device}")
        except Exception as e:
            logging.error(f"Error in direct tensor inference: {e}")
            # Fall back to standard method
            logging.info("Falling back to standard predict_image method")
            with torch.no_grad():
                boxes = model.predict_image(path=image_path, return_plot=False)
        
        # Process results
        detections = []
        if not boxes.empty:
            for _, row in boxes.iterrows():
                x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                score = row['score']
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(score),
                    'class_id': 0,
                    'detection_type': 'deepforest'
                })
        
        # Create results
        results = {
            'tree_count': len(detections),
            'trees': detections
        }
        
        # Save results if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        logging.info(f"DeepForest detection completed with {len(detections)} trees")
        return results
    
    except Exception as e:
        logging.error(f"Error in tree detection: {e}")
        # Return empty results on error
        return {'tree_count': 0, 'trees': []}

# Run module as script
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Tree detection using DeepForest, SAM, and Gemini")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save results (JSON)")
    parser.add_argument("--threshold", type=float, default=0.2, help="DeepForest score threshold")
    parser.add_argument("--no-gemini", action="store_true", help="Disable Gemini API integration")
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM segmentation")
    
    args = parser.parse_args()
    
    # Initialize config with command line arguments
    config = MLConfig()
    config.deepforest_score_threshold = args.threshold
    config.use_gemini = not args.no_gemini
    if args.output:
        config.output_path = args.output
    
    # Run detection
    results = detect_trees(args.image, args.output)
    
    # Print summary to console
    print(f"Detected {results['tree_count']} trees in {args.image}")
    for i, tree in enumerate(results['trees']):
        detection_type = tree.get('detection_type', 'unknown')
        print(f"Tree {i+1}: Confidence {tree['confidence']:.2f} (Detected by: {detection_type})")
    
    print(f"Results {'saved to ' + args.output if args.output else 'not saved'}")

