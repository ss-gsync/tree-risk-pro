"""ML services package for model inference, supporting both in-memory and external models."""

import os
import sys
import logging

# Add backend directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import USE_EXTERNAL_MODEL_SERVER

# Configure logging
logger = logging.getLogger(__name__)

# Import model service implementations
from .model_service import MLModelService, get_model_service as get_local_model_service
from .external_model_service import ExternalModelService, get_external_model_service

def get_model_service(use_gpu=True, use_grounded_sam=True):
    """
    Get the appropriate model service based on configuration.
    
    Args:
        use_gpu: Whether to use GPU for inference if available
        use_grounded_sam: Whether to use Grounded SAM instead of DeepForest
        
    Returns:
        Either ExternalModelService or MLModelService instance
    """
    if USE_EXTERNAL_MODEL_SERVER:
        logger.info("Using external T4 model server")
        return get_external_model_service()
    else:
        logger.info("Using local model service")
        return get_local_model_service(use_gpu=use_gpu, use_grounded_sam=use_grounded_sam)

__all__ = ['MLModelService', 'ExternalModelService', 'get_model_service']