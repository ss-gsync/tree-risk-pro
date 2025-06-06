"""Gemini Service.

Integrates with Google's Gemini API for advanced analytics on tree risk assessment data.
This service handles ONLY Gemini API calls.

This service is exclusively for Gemini API interactions and should not reference or
interact with the ML pipeline in any way. Any tree detection and analysis is performed
solely through Gemini API, not local ML models.
"""

import os
import io
import json
import math
import base64
import logging
import aiohttp
import asyncio
from typing import Dict, List
from datetime import datetime

# Import config - support both module and direct imports
try:
    # When imported as module (from parent directory)
    from config import ML_DIR, GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL
except ImportError:
    # When running from tree_ml root
    from dashboard.backend.config import ML_DIR, GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL

# Try to import PIL for image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL (Pillow) not available. Image processing capabilities will be limited.")

# Configure logging
logger = logging.getLogger(__name__)

class GeminiService:
    """
    Service for Google Gemini Pro API integration and tree risk analytics.
    
    This class provides a comprehensive interface to integrate with Google's 
    Gemini large language model for tree risk assessment analysis, including:
    
    1. Structured tree risk assessments and recommendations
    2. Property-level report generation with risk summaries
    3. Multimodal imagery analysis for tree health evaluation
    4. Tree detection from satellite imagery and map data
    
    The service handles all aspects of API communication including authentication,
    request formatting, error handling, and response parsing to provide a clean
    interface for the rest of the application.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Gemini service with API configuration.
        
        This constructor sets up the Gemini service with proper configuration
        parameters for API access, including credentials and endpoints.
        The actual API connection is established later via the initialize() method.
        
        Args:
            config_path (str, optional): Path to external JSON configuration file
                                        containing Gemini API settings.
                                        If None, loads config from environment/defaults.
                                        Defaults to None.
        """
        # Use the API key directly from config.py - don't reload or transform it
        self.api_key = GEMINI_API_KEY
        # For consistency, load other settings through the config method
        self.config = self._load_config(config_path)
        # Store the base API URL - actual URL with version will be constructed when needed
        self.api_url = self.config['gemini']['api_url']
        self.model = self.config['gemini']['model'] 
        # Determine the API version based on model name
        # Gemini 2.0 requires v1beta endpoint
        self.api_version = "v1beta" if "2.0" in self.model else "v1"
        self.session = None
        self.is_initialized = False
        
        # Log important configuration details (without sensitive information)
        logger.info(f"Initializing GeminiService with model: {self.model}")
        logger.info(f"Using API version: {self.api_version} for Gemini API")
        
        # Set the base API URL with the correct version
        self.api_base_url = f"https://generativelanguage.googleapis.com/{self.api_version}"
    
    def _load_config(self, config_path: str = None) -> Dict:
        """
        Load configuration settings from file or environment variables.
        
        Args:
            config_path (str, optional): Path to JSON configuration file.
                                      If None, uses environment variables.
                                      Defaults to None.
        
        Returns:
            Dict: Configuration settings for the Gemini service
        """
        # Default configuration using environment variables
        config = {
            'gemini': {
                'api_url': GEMINI_API_URL,
                'model': GEMINI_MODEL
            }
        }
        
        # If config file provided, load and merge settings
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    
                # Merge file config with defaults (file takes precedence)
                if 'gemini' in file_config:
                    for key, value in file_config['gemini'].items():
                        if key != 'api_key':  # Don't override API key from config.py
                            config['gemini'][key] = value
                            
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                
        return config
    
    def _get_gemini_url(self, endpoint: str) -> str:
        """
        Get a properly formatted Gemini API URL for the given endpoint.
        
        Args:
            endpoint (str): The endpoint to access (e.g., 'models/gemini-2.0-flash:generateContent')
            
        Returns:
            str: The complete URL with API version, endpoint, and API key
        """
        # Always use self.api_key which is already loaded from config during initialization
        return f"https://generativelanguage.googleapis.com/{self.api_version}/{endpoint}?key={self.api_key}"

    async def initialize(self) -> bool:
        """
        Initialize the service and prepare it for API requests.
        
        This asynchronous method validates the API key and initializes
        the service for Gemini API access. It intentionally doesn't test
        the actual connection to avoid startup delays or failures that
        could prevent the application from starting.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Check if API key is available
            if not self.api_key:
                logger.error("No Gemini API key found - service will not be available")
                return False
                
            # Log initialization details for debugging
            logger.info(f"Initializing Gemini service with API key: {'*' * 10}")
            logger.info(f"Gemini API URL: {self.api_base_url}")
            logger.info(f"Gemini model: {self.model}")
            
            # Initialize as ready without testing connections
            # This avoids issues during startup
            self.is_initialized = True
            logger.info("Gemini service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Gemini service: {str(e)}")
            # Log more details about the exception
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            return False

    async def query(self, prompt: str, context: Dict = None) -> Dict:
        """
        Send a text query to the Gemini API
        
        Args:
            prompt: The text prompt to send to Gemini
            context: Optional dictionary of context information
            
        Returns:
            Dict: Response from Gemini API
        """
        if not self.is_initialized and not prompt.startswith("Test connection"):
            return {
                "success": False,
                "message": "Gemini service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        # Create a new session for each query to avoid timeout issues
        session = None
        
        try:
            # Create the request payload with properly configured safety settings
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 2048
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            # Add context if provided
            if context:
                payload["contents"][0]["parts"].append({
                    "text": f"\nContext: {json.dumps(context)}"
                })
            
            # Use the helper method for consistent URL formatting
            url = self._get_gemini_url(f"models/{self.model}:generateContent")
            # Log with masked API key
            key_param = f"key={self.api_key}"
            masked_param = f"key=****"
            masked_url = url.replace(key_param, masked_param)
            logger.info(f"Using Gemini API URL: {masked_url}")
            
            # Create a new session for this request
            session = aiohttp.ClientSession()
            
            # Make the request with proper timeout handling
            async with session.post(url, json=payload, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Gemini API error: Status {response.status}, Response: {error_text}")
                    
                    # Log detailed error for easier debugging
                    try:
                        error_json = json.loads(error_text)
                        if "error" in error_json:
                            logger.error(f"Gemini API error details: {json.dumps(error_json['error'], indent=2)}")
                    except (json.JSONDecodeError, KeyError):
                        pass
                        
                    return {
                        "success": False,
                        "message": f"API error: {response.status}",
                        "error_details": error_text,
                        "timestamp": datetime.now().isoformat()
                    }
                
                result = await response.json()
                
                # Extract the response text
                try:
                    response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    return {
                        "success": True,
                        "response": response_text,
                        "full_result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                except (KeyError, IndexError) as e:
                    logger.error(f"Failed to parse Gemini response: {str(e)}")
                    return {
                        "success": False,
                        "message": f"Failed to parse Gemini response",
                        "error_details": str(e),
                        "full_result": result,
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Error in Gemini API query: {str(e)}")
            return {
                "success": False,
                "message": f"Gemini API error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            if session:
                await session.close()
            
    async def query_image(self, prompt: str, image_base64: str) -> Dict:
        """
        Send a multimodal query with image to the Gemini API
        
        Args:
            prompt: The text prompt for analysis
            image_base64: Base64-encoded image data
            
        Returns:
            Dict: Response from Gemini API
        """
        if not self.is_initialized:
            return {
                "success": False,
                "message": "Gemini service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        # Create a new session for each query to avoid timeout issues
        session = None
        
        try:
            # Create the request payload with image
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inlineData": {
                                    "mimeType": "image/jpeg",
                                    "data": image_base64
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,  # Lower temperature for more precise analysis
                    "topP": 0.95,
                    "topK": 40,
                    "maxOutputTokens": 2048
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                    }
                ]
            }
            
            # Use the helper method for consistent URL formatting
            # Ensure we're using a model that supports multimodal (like gemini-pro-vision)
            vision_model = "gemini-pro-vision"
            if "2.0" in self.model:
                vision_model = "gemini-2.0-pro-vision"
                
            url = self._get_gemini_url(f"models/{vision_model}:generateContent")
            
            # Log with masked URL
            key_param = f"key={self.api_key}"
            masked_param = f"key=****"
            masked_url = url.replace(key_param, masked_param)
            logger.info(f"Using Gemini API URL: {masked_url}")
            
            # Create a new session for this request
            session = aiohttp.ClientSession()
            
            # Make the request with proper timeout handling
            async with session.post(url, json=payload, timeout=60) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Gemini API error: Status {response.status}, Response: {error_text}")
                    
                    # Log detailed error for easier debugging
                    try:
                        error_json = json.loads(error_text)
                        if "error" in error_json:
                            logger.error(f"Gemini API error details: {json.dumps(error_json['error'], indent=2)}")
                    except (json.JSONDecodeError, KeyError):
                        pass
                        
                    return {
                        "success": False,
                        "message": f"API error: {response.status}",
                        "error_details": error_text,
                        "timestamp": datetime.now().isoformat()
                    }
                
                result = await response.json()
                
                # Extract the response text
                try:
                    response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    return {
                        "success": True,
                        "response": response_text,
                        "full_result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                except (KeyError, IndexError) as e:
                    logger.error(f"Failed to parse Gemini response: {str(e)}")
                    return {
                        "success": False,
                        "message": f"Failed to parse Gemini response",
                        "error_details": str(e),
                        "full_result": result,
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            logger.error(f"Error in Gemini image query: {str(e)}")
            return {
                "success": False,
                "message": f"Gemini API error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            if session:
                await session.close()

    async def analyze_tree_from_satellite(self, image_path: str, location_data: Dict = None) -> Dict:
        """
        Analyze a tree from satellite imagery using Gemini Vision
        
        Args:
            image_path: Path to the satellite image
            location_data: Optional dictionary with location information
            
        Returns:
            Dict: Analysis results
        """
        try:
            # Read image and encode as base64
            with open(image_path, 'rb') as img_file:
                image_bytes = img_file.read()
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                
            # Create prompt for tree analysis
            prompt = """
            Analyze this satellite or aerial image of trees and provide:
            1. Tree count - how many distinct trees can you identify?
            2. For each tree, provide bounding box coordinates [x1, y1, x2, y2] in normalized format (0-1 range)
            3. A risk assessment based on visible factors like proximity to structures, visible damage, or environmental factors
            
            Format your response as JSON with the following structure:
            {
              "tree_count": 5,
              "trees": [
                {
                  "id": "tree_1",
                  "bbox": [0.2, 0.3, 0.4, 0.5],
                  "confidence": 0.95,
                  "risk_level": "medium",
                  "notes": "Healthy tree away from structures"
                },
                ...
              ],
              "summary": "Brief overall assessment of the trees and potential risks"
            }
            """
            
            # Add location context if provided
            if location_data:
                location_str = json.dumps(location_data)
                prompt += f"\n\nLocation context: {location_str}"
                
            # Call Gemini Vision API
            return await self.query_image(prompt, image_b64)
            
        except Exception as e:
            logger.error(f"Error in tree satellite analysis: {str(e)}")
            return {
                "success": False,
                "message": f"Analysis error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }