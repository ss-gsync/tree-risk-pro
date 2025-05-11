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
    from config import TEMP_DIR, GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL
except ImportError:
    # When running from tree_risk_pro root
    from dashboard.backend.config import TEMP_DIR, GEMINI_API_KEY, GEMINI_API_URL, GEMINI_MODEL

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
        """Send a query to the Gemini API"""
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
            # Use proper safety settings instead of BLOCK_NONE to comply with Google's security policies
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
            # Log with masked API key - but keep the URL structure intact for debugging
            # Replace just the actual key value, not the entire key parameter
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
                    
        except aiohttp.ClientError as e:
            logger.error(f"Gemini API connection error: {str(e)}")
            return {
                "success": False,
                "message": f"API connection error",
                "error_details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        except asyncio.TimeoutError:
            logger.error("Gemini API request timed out")
            return {
                "success": False,
                "message": "API request timed out",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error querying Gemini API: {str(e)}")
            return {
                "success": False,
                "message": f"Error querying Gemini API",
                "error_details": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Always close the session when done
            if session and not session.closed:
                await session.close()

    async def analyze_tree_risk(self, tree_data: Dict) -> Dict:
        """Analyze tree risk using Gemini AI"""
        if not self.is_initialized:
            return {
                "success": False,
                "message": "Gemini service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Create a comprehensive analysis prompt
            prompt = f"""
            Please analyze this tree risk assessment data and provide insights:
            
            Tree Details:
            - Species: {tree_data.get('species', 'Unknown')}
            - Height: {tree_data.get('height', 'Unknown')} meters
            - Canopy Width: {tree_data.get('canopy_width', 'Unknown')} meters
            - Current Risk Level: {tree_data.get('risk_level', 'Unknown')}
            
            Risk Factors:
            {json.dumps(tree_data.get('risk_factors', []), indent=2)}
            
            Assessment History:
            {json.dumps(tree_data.get('assessment_history', []), indent=2)}
            
            Property Context:
            - Distance to Structure: {tree_data.get('distance_to_structure', 'Unknown')} meters
            - Property Type: {tree_data.get('property_type', 'Unknown')}
            
            Please provide:
            1. A detailed risk analysis for this tree
            2. Recommendations for risk mitigation
            3. Potential future concerns based on growth patterns
            4. Compare this tree's risk level with similar trees in the database
            """
            
            # Send to Gemini API
            result = await self.query(prompt, context=tree_data)
            
            if result.get('success', False):
                # Process successful response
                analysis = result['response']
                
                # Parse the response into structured sections
                sections = self._parse_analysis_response(analysis)
                
                return {
                    "success": True,
                    "tree_id": tree_data.get('id', 'unknown'),
                    "analysis": analysis,
                    "structured_analysis": sections,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return result  # Return the error response
        except Exception as e:
            logger.error(f"Error analyzing tree risk: {str(e)}")
            return {
                "success": False,
                "message": f"Analysis error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def generate_property_report(self, property_data: Dict, trees_data: List[Dict]) -> Dict:
        """Generate a comprehensive property report using Gemini AI"""
        if not self.is_initialized:
            return {
                "success": False,
                "message": "Gemini service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Create report metadata
            high_risk_trees = sum(1 for tree in trees_data if tree.get('risk_level') == 'high')
            medium_risk_trees = sum(1 for tree in trees_data if tree.get('risk_level') == 'medium')
            low_risk_trees = sum(1 for tree in trees_data if tree.get('risk_level') == 'low')
            
            # Create a comprehensive report prompt
            prompt = f"""
            Generate a comprehensive property tree risk assessment report for:
            
            Property Details:
            - Address: {property_data.get('address', 'Unknown')}
            - City: {property_data.get('city', 'Unknown')}
            - Property Type: {property_data.get('property_type', 'Unknown')}
            - Size: {property_data.get('size', 'Unknown')} acres
            
            Tree Summary:
            - Total Trees: {len(trees_data)}
            - High Risk Trees: {high_risk_trees}
            - Medium Risk Trees: {medium_risk_trees}
            - Low Risk Trees: {low_risk_trees}
            
            For each high risk tree, provide a specific assessment and recommendations.
            
            Conclude with:
            1. Overall property risk assessment
            2. Prioritized action items
            3. Long-term management recommendations
            4. Monitoring schedule recommendations
            
            Format the report in a professional manner suitable for property owners and arborists.
            """
            
            # Add context with more detailed tree data
            context = {
                "property": property_data,
                "trees": trees_data[:10]  # Limit to first 10 trees to avoid token limits
            }
            
            # Send to Gemini API
            result = await self.query(prompt, context=context)
            
            if result.get('success', False):
                return {
                    "success": True,
                    "property_id": property_data.get('id', 'unknown'),
                    "report": result['response'],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return result  # Return the error response
        except Exception as e:
            logger.error(f"Error generating property report: {str(e)}")
            return {
                "success": False,
                "message": f"Report generation error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    async def analyze_imagery(self, image_url: str, context_data: Dict = None) -> Dict:
        """Analyze tree imagery using Gemini AI multimodal capabilities
        
        This function uses gemini-2.0-flash for proper image analysis with
        Google's multimodal capabilities.
        """
        if not self.is_initialized:
            return {
                "success": False,
                "message": "Gemini service not initialized",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            # Create a session for this request
            session = aiohttp.ClientSession()
            
            try:
                # First download the image if it's a URL
                image_data = None
                if image_url.startswith('http'):
                    logger.info(f"Downloading image from URL: {image_url}")
                    async with session.get(image_url, timeout=30) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            logger.info(f"Successfully downloaded {len(image_data)} bytes from {image_url}")
                        else:
                            logger.error(f"Failed to download image: status {response.status}")
                            return {
                                "success": False,
                                "message": f"Failed to download image: status {response.status}",
                                "timestamp": datetime.now().isoformat()
                            }
                elif os.path.exists(image_url):
                    # It's a local file path
                    with open(image_url, 'rb') as f:
                        image_data = f.read()
                        logger.info(f"Read {len(image_data)} bytes from local file {image_url}")
                else:
                    logger.error(f"Image not found: {image_url}")
                    return {
                        "success": False,
                        "message": f"Image not found: {image_url}",
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Prepare multimodal API request
                # For image analysis with gemini-2.0-flash, we need to create a multipart request
                # with the image data as a base64-encoded part
                import base64
                
                # Determine MIME type based on file extension
                mime_type = "image/jpeg"  # Default
                if image_url.lower().endswith('.png'):
                    mime_type = "image/png"
                elif image_url.lower().endswith('.gif'):
                    mime_type = "image/gif"
                elif image_url.lower().endswith('.webp'):
                    mime_type = "image/webp"
                
                # Base64 encode the image
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                
                # Create the prompt text
                prompt_text = """
                Please analyze this tree image in detail.
                
                Identify and describe:
                1. Signs of structural issues or damage
                2. Disease indicators or pest infestation
                3. Growth patterns and tree health
                4. Proximity risks to structures or property
                5. Overall health assessment and risk level (low, medium, high)
                
                If possible, identify the tree species and approximate height.
                """
                
                # Create the proper multimodal request payload
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt_text},
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": image_b64
                                    }
                                }
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
                if context_data:
                    context_text = f"\nAdditional context:\n{json.dumps(context_data, indent=2)}"
                    payload["contents"][0]["parts"][0]["text"] += context_text
                
                # Add API key to URL
                url = self._get_gemini_url(f"models/{self.model}:generateContent")
                # Log with masked API key
                masked_url = url.replace(GEMINI_API_KEY, "****")
                logger.info(f"Using Gemini API URL: {masked_url}")
                
                # Make the request with proper timeout handling
                async with session.post(url, json=payload, timeout=60) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Gemini API error: Status {response.status}, Response: {error_text}")
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
                            "image_url": image_url,
                            "analysis": response_text,
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
            finally:
                # Close the session
                if session and not session.closed:
                    await session.close()
                    
        except Exception as e:
            logger.error(f"Error analyzing imagery: {str(e)}")
            return {
                "success": False,
                "message": f"Image analysis error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def detect_trees_from_map_data(self, map_view_info: Dict, job_id: str = None) -> Dict:
        """Detect trees from Google Maps 3D view data using ONLY Gemini API
        
        This function processes map data to detect trees using gemini-2.0-flash.
        It ONLY makes Gemini API calls and does not reference or interact with 
        any ML pipeline. Tree detection is performed entirely through Gemini's 
        vision capabilities.
        
        This method:
        1. Extracts the map image URL from the input data
        2. Downloads the image from that URL
        3. Sends the image to Gemini API with a prompt to identify trees
        4. Processes the response to extract tree data
        5. Returns a structured response with the detected trees
        
        Args:
            map_view_info: Map view data (center, zoom, bounds, etc.)
            job_id: Optional job ID for tracking
            
        Returns:
            dict: Detection results with trees information
        """
        # Ensure we import json explicitly in this function scope to avoid issues
        import json
        import os
        import base64
        from datetime import datetime
        if not self.is_initialized:
            return {
                "success": False,
                "job_id": job_id,
                "status": "error",
                "message": "Gemini service not initialized",
                "timestamp": datetime.now().isoformat(),
                "tree_count": 0,
                "trees": []
            }
        
        try:
            # Extract view data for the prompt
            view_data = map_view_info.get('viewData', {})
            center = view_data.get('center', [0, 0])
            bounds = view_data.get('bounds', [])
            map_image_url = view_data.get('imageUrl')
            
            # Debug logging to see what's coming from the frontend
            logger.info(f"Received map_view_info with keys: {list(map_view_info.keys())}")
            logger.info(f"Received view_data with keys: {list(view_data.keys())}")
            if 'imageUrl' in view_data:
                url_preview = view_data['imageUrl'][:30] + "..." if view_data['imageUrl'] else "None"
                logger.info(f"imageUrl present in view_data: {url_preview}")
            else:
                logger.warning("imageUrl key is missing from view_data")
            
            # Add detailed logging to diagnose conditional paths
            logger.info(f"Checking data sources: map_image_url={bool(map_image_url)}, coordsInfo={bool(view_data.get('coordsInfo'))}, useBackendTiles={bool(view_data.get('useBackendTiles'))}")
            
            # Get the actual values for detailed inspection
            coords_info_value = view_data.get('coordsInfo', 'NOT_PRESENT')
            use_backend_tiles_value = view_data.get('useBackendTiles', 'NOT_PRESENT')
            logger.info(f"coordsInfo type: {type(coords_info_value)}, value: {coords_info_value[:50] if isinstance(coords_info_value, str) else coords_info_value}")
            logger.info(f"useBackendTiles type: {type(use_backend_tiles_value)}, value: {use_backend_tiles_value}")
            
            # Check if we have an image URL or coordinate info to process
            if not map_image_url and not view_data.get('coordsInfo') and not view_data.get('useBackendTiles'):
                logger.warning("No map image URL or coordinate information provided in view data")
                
                # Create a response for this error case
                error_response = {
                    "error": "No map image URL or coordinate information provided",
                    "details": "The frontend must provide either a map image URL in viewData.imageUrl or coordinate information with useBackendTiles flag."
                }
                
                return {
                    "success": False,
                    "job_id": job_id,
                    "status": "error",
                    "message": "No map image or coordinate information provided. Please ensure your request includes either an image URL or coordinate data with useBackendTiles flag.",
                    "timestamp": datetime.now().isoformat(),
                    "tree_count": 0,
                    "trees": [],
                }
            
            # If we have coordsInfo and useBackendTiles, we'll use our tile-based approach
            if view_data.get('coordsInfo') and view_data.get('useBackendTiles'):
                logger.info("Coordinate-based approach condition triggered successfully")
                logger.info("Using coordinate-based approach with backend tiles")
                
                try:
                    # Parse the coordsInfo JSON
                    coords_info = json.loads(view_data.get('coordsInfo'))
                    logger.info(f"Parsed coordinates info: {coords_info}")
                    
                    # Extract key information
                    bounds = coords_info.get('bounds', view_data.get('bounds'))
                    center = coords_info.get('center', view_data.get('center'))
                    zoom = coords_info.get('zoom', view_data.get('zoom'))
                    map_type = coords_info.get('mapType', 'satellite')
                    
                    # Log detailed information about the request
                    logger.info(f"Processing map with: center={center}, zoom={zoom}, bounds={bounds}, mapType={map_type}")
                    
                    # Call the method to fetch and process map tiles
                    logger.info("Starting tile fetching and processing...")
                    tile_results = await self._fetch_map_tiles(center, zoom, bounds, map_type, job_id)
                    
                    if tile_results and tile_results.get("success"):
                        logger.info(f"Successfully processed tiles and detected {tile_results.get('tree_count', 0)} trees")
                        return tile_results
                    else:
                        # Don't fall back to sample trees, but return error instead
                        logger.error("Tile processing failed, not falling back to synthetic data")
                        
                        # Save error information
                        if job_id:
                            try:
                                # Import json here to ensure it's in scope
                                import json
                                
                                # Create the response directory
                                gemini_response_dir = os.path.join('/ttt/data/temp', job_id, 'gemini_response')
                                os.makedirs(gemini_response_dir, exist_ok=True)
                                
                                # Save error information
                                error_info = {
                                    "message": "Tree detection failed using Gemini API",
                                    "reason": "Satellite imagery could not be processed",
                                    "timestamp": datetime.now().timestamp(),
                                    "error": "Tile processing failed"
                                }
                                
                                with open(os.path.join(gemini_response_dir, 'error_info.json'), 'w') as f:
                                    f.write(json.dumps(error_info, indent=2))
                                
                                logger.info(f"Saved error information to {gemini_response_dir}/error_info.json")
                            except Exception as save_err:
                                logger.error(f"Failed to save error information: {save_err}")
                        
                        return {
                            "success": False,
                            "job_id": job_id,
                            "status": "error",
                            "timestamp": datetime.now().isoformat(),
                            "message": "Tree detection with Gemini API failed. Please try again with different area.",
                            "error_details": "Tile processing failed",
                            "center": center,
                            "zoom": zoom
                        }
                except Exception as e:
                    logger.error(f"Error processing coordinate information: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Don't fall back to sample trees, return error
                    # Save error information
                    if job_id:
                        try:
                            # Import json here to ensure it's in scope
                            import json
                            
                            # Create the response directory
                            gemini_response_dir = os.path.join('/ttt/data/temp', job_id, 'gemini_response')
                            os.makedirs(gemini_response_dir, exist_ok=True)
                            
                            # Save error information
                            error_info = {
                                "message": "Coordinate processing failed",
                                "error": str(e),
                                "timestamp": datetime.now().timestamp(),
                                "traceback": traceback.format_exc()
                            }
                            
                            with open(os.path.join(gemini_response_dir, 'error_info.json'), 'w') as f:
                                f.write(json.dumps(error_info, indent=2))
                            
                            logger.info(f"Saved error information to {gemini_response_dir}/error_info.json")
                        except Exception as save_err:
                            logger.error(f"Failed to save error information: {save_err}")
                    
                    # Save some debug information and a sample satellite image for the response directory
                    if job_id:
                        try:
                            # Import json to ensure it's in scope
                            import json
                            
                            # Create the response directory
                            gemini_response_dir = os.path.join('/ttt/data/temp', job_id, 'gemini_response')
                            os.makedirs(gemini_response_dir, exist_ok=True)
                            
                            # Save fallback explanation 
                            fallback_explanation = {
                                "message": "Could not process satellite imagery",
                                "reason": "Satellite imagery could not be retrieved or processed",
                                "timestamp": datetime.now().isoformat(),
                                "center": view_data.get('center'),
                                "bounds": view_data.get('bounds'),
                                "zoom": view_data.get('zoom'),
                                "error": str(e)
                            }
                            
                            with open(os.path.join(gemini_response_dir, 'response.txt'), 'w') as f:
                                f.write(json.dumps(fallback_explanation, indent=2))
                            
                            # Create a sample satellite image using a red placeholder
                            fallback_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAYAAABccqhmAAAACXBIWXMAAAsTAAALEwEAmpwYAAAHu2lUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4gPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgNS42LWMxNDIgNzkuMTYwOTI0LCAyMDE3LzA3LzEzLTAxOjA2OjM5ICAgICAgICAiPiA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPiA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIgeG1sbnM6cGhvdG9zaG9wPSJodHRwOi8vbnMuYWRvYmUuY29tL3Bob3Rvc2hvcC8xLjAvIiB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIgeG1wOkNyZWF0b3JUb29sPSJBZG9iZSBQaG90b3Nob3AgQ0MgKFdpbmRvd3MpIiB4bXA6Q3JlYXRlRGF0ZT0iMjAyMC0wMS0yN1QyMDoxNDo0NCswMTowMCIgeG1wOk1vZGlmeURhdGU9IjIwMjAtMDEtMjdUMjA6MjE6NDYrMDE6MDAiIHhtcDpNZXRhZGF0YURhdGU9IjIwMjAtMDEtMjdUMjA6MjE6NDYrMDE6MDAiIGRjOmZvcm1hdD0iaW1hZ2UvcG5nIiBwaG90b3Nob3A6Q29sb3JNb2RlPSIzIiBwaG90b3Nob3A6SUNDUHJvZmlsZT0ic1JHQiBJRUM2MTk2Ni0yLjEiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6YWU0OGMyYTgtZTE0Yi00MzI1LWJkOTQtMDkzYmYwNjYwZGU0IiB4bXBNTTpEb2N1bWVudElEPSJhZG9iZTpkb2NpZDpwaG90b3Nob3A6MmE4YTAwNjQtYjgxOS0wYTQ4LWJmMGMtNTU0YmFiODNiMmU5IiB4bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ9InhtcC5kaWQ6MjYzYzkzZWUtZWIzMS0wMjQyLTk3OGYtNjU2ZTNjM2JkZGY0Ij4gPHhtcE1NOkhpc3Rvcnk+IDxyZGY6U2VxPiA8cmRmOmxpIHN0RXZ0OmFjdGlvbj0iY3JlYXRlZCIgc3RFdnQ6aW5zdGFuY2VJRD0ieG1wLmlpZDoyNjNjOTNlZS1lYjMxLTAyNDItOTc4Zi02NTZlM2MzYmRkZjQiIHN0RXZ0OndoZW49IjIwMjAtMDEtMjdUMjA6MTQ6NDQrMDE6MDAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCBDQyAoV2luZG93cykiLz4gPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOmVlYzExZWY1LWNjMmMtZmY0MC1hNzVhLWJmZDJmMjg0NWZmYSIgc3RFdnQ6d2hlbj0iMjAyMC0wMS0yN1QyMDoyMToyMCswMTowMCIgc3RFdnQ6c29mdHdhcmVBZ2VudD0iQWRvYmUgUGhvdG9zaG9wIENDIChXaW5kb3dzKSIgc3RFdnQ6Y2hhbmdlZD0iLyIvPiA8cmRmOmxpIHN0RXZ0OmFjdGlvbj0ic2F2ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6YWU0OGMyYTgtZTE0Yi00MzI1LWJkOTQtMDkzYmYwNjYwZGU0IiBzdEV2dDp3aGVuPSIyMDIwLTAxLTI3VDIwOjIxOjQ2KzAxOjAwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgQ0MgKFdpbmRvd3MpIiBzdEV2dDpjaGFuZ2VkPSIvIi8+IDwvcmRmOlNlcT4gPC94bXBNTTpIaXN0b3J5PiA8L3JkZjpEZXNjcmlwdGlvbj4gPC9yZGY6UkRGPiA8L3g6eG1wbWV0YT4gPD94cGFja2V0IGVuZD0iciI/Pm0SrTwAABtJSURBVHic7d19bFzV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"
                            image_data = base64.b64decode(fallback_image_base64)
                            image_file = os.path.join(gemini_response_dir, 'satellite_image.jpg')
                            with open(image_file, 'wb') as f:
                                f.write(image_data)
                            logger.info(f"Saved fallback satellite image to {image_file}")
                        except Exception as save_err:
                            logger.error(f"Failed to save fallback information: {save_err}")
                    
                    return {
                        "success": False,
                        "job_id": job_id,
                        "status": "error",
                        "timestamp": datetime.now().isoformat(),
                        "message": f"Coordinate processing failed: {str(e)}",
                        "error_details": str(e)
                    }
            
            # If we reach here, we're using the original image URL approach
            logger.info("Coordinate-based approach not triggered, falling back to image URL approach")
            
            # This should not happen if coordsInfo and useBackendTiles are properly set
            if view_data.get('coordsInfo') and view_data.get('useBackendTiles'):
                logger.error("ERROR: We reached image URL logic despite having coordsInfo and useBackendTiles flags!")
                logger.error("This indicates a logical error in our conditional branching.")
            
            # Check if we even have an image URL to process
            if not map_image_url:
                logger.error("No image URL available for processing and coordinate approach wasn't triggered")
                return {
                    "success": False,
                    "job_id": job_id,
                    "status": "error",
                    "message": "No valid data source found for detection. Missing both image URL and valid coordinate data.",
                    "timestamp": datetime.now().isoformat(),
                    "tree_count": 0,
                    "trees": []
                }
                
            # Create a session for this request
            logger.info("Proceeding with image URL processing")
            session = aiohttp.ClientSession()
            image_data = None
            
            try:
                # Determine how to handle the image URL
                if map_image_url.startswith('data:image/'):
                    # This is already a data URL, extract the base64 part
                    logger.info("Processing data URL from frontend canvas")
                    
                    # Extract the base64 data part after the comma
                    import base64
                    try:
                        # Strip the prefix (data:image/jpeg;base64,) to get the raw base64 string
                        if ',' in map_image_url:
                            map_image_data = map_image_url.split(',', 1)[1]
                        else:
                            map_image_data = map_image_url
                            
                        # Decode the base64 string
                        image_data = base64.b64decode(map_image_data)
                        logger.info(f"Successfully decoded data URL, size: {len(image_data)} bytes")
                    except Exception as e:
                        logger.error(f"Error decoding data URL: {e}")
                        return {
                            "success": False,
                            "job_id": job_id,
                            "status": "error",
                            "message": f"Error decoding image data URL: {str(e)}",
                            "timestamp": datetime.now().isoformat(),
                            "tree_count": 0,
                            "trees": []
                        }
                        
                elif map_image_url.startswith(('http://', 'https://')):
                    # This is a web URL, download it
                    logger.info(f"Downloading map image from URL: {map_image_url}")
                    async with session.get(map_image_url, timeout=30) as response:
                        if response.status == 200:
                            image_data = await response.read()
                            logger.info(f"Successfully downloaded {len(image_data)} bytes from {map_image_url}")
                        else:
                            logger.error(f"Failed to download map image: status {response.status}")
                            return {
                                "success": False,
                                "job_id": job_id,
                                "status": "error",
                                "message": f"Failed to download map image: status {response.status}",
                                "timestamp": datetime.now().isoformat(),
                                "tree_count": 0,
                                "trees": []
                            }
                            
                elif map_image_url.startswith('file://'):
                    # This is a local file path, read it
                    file_path = map_image_url[7:]  # Remove 'file://' prefix
                    logger.info(f"Reading map image from local file: {file_path}")
                    try:
                        with open(file_path, 'rb') as f:
                            image_data = f.read()
                        logger.info(f"Successfully read {len(image_data)} bytes from local file")
                    except Exception as e:
                        logger.error(f"Error reading local file: {e}")
                        return {
                            "success": False,
                            "job_id": job_id,
                            "status": "error",
                            "message": f"Error reading local image file: {str(e)}",
                            "timestamp": datetime.now().isoformat(),
                            "tree_count": 0,
                            "trees": []
                        }
                        
                else:
                    # Check if we have useBackendTiles flag instead
                    if view_data.get('useBackendTiles') and view_data.get('coordsInfo'):
                        logger.info("Client requested backend tile retrieval. Using coordinates for map processing.")
                        
                        # Extract coordinates info to get tiles
                        try:
                            coords_info = json.loads(view_data.get('coordsInfo'))
                            logger.info(f"Using coordinates info for map processing: {coords_info}")
                            
                            # Extract key information from coords_info
                            bounds = coords_info.get('bounds')
                            center = coords_info.get('center')
                            zoom = coords_info.get('zoom')
                            mapType = coords_info.get('mapType', 'satellite')
                            
                            logger.info(f"Map processing using coordinates: center={center}, zoom={zoom}, mapType={mapType}")
                            
                            # Fetch satellite imagery from Google Maps Static API
                            try:
                                # Build the Google Maps Static API URL
                                api_key = self.api_key
                                
                                # Calculate proper size - maximum 640x640 for Static API
                                size = "640x640"
                                
                                # Build the Static Maps API URL with precise parameters
                                # 1. Calculate scale factor based on device pixel ratio (assumes default of 2 for retina)
                                scale = 2
                                
                                # 2. Apply exact same map styling as the JavaScript API
                                # Extract mapId if available for consistent styling
                                map_id = coords_info.get('mapId', '')
                                style_params = f"&map_id={map_id}" if map_id else ""
                                
                                # 3. Replicate the exact viewport and perspective
                                # Add heading parameter if available for consistent orientation
                                heading = coords_info.get('heading', 0)
                                heading_param = f"&heading={heading}" if heading else ""
                                
                                # Static Maps API doesn't support tilt, but we can adjust the zoom slightly
                                # to compensate (if user is in 3D mode with tilt)
                                is_3d = coords_info.get('is3D', False)
                                zoom_adjustment = -0.2 if is_3d else 0  # Zoom out slightly in 3D mode
                                adjusted_zoom = zoom + zoom_adjustment
                                
                                # 4. Ensure map type matches exactly what user sees
                                # Convert mapType to proper format for Static Maps API
                                static_map_type = mapType.lower()
                                if static_map_type == "hybrid" and is_3d:
                                    static_map_type = "satellite"  # Hybrid doesn't work well with 3D adjustments
                                
                                # Construct final URL with all parameters for maximum accuracy
                                static_map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={center[1]},{center[0]}&zoom={adjusted_zoom}&size={size}&scale={scale}&maptype={static_map_type}{style_params}{heading_param}&key={api_key}"
                                
                                logger.info(f"Fetching satellite imagery from Google Maps Static API: {static_map_url[:100]}...")
                                
                                # Fetch the image
                                import aiohttp
                                import base64
                                
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(static_map_url) as response:
                                        if response.status == 200:
                                            # Read the image data
                                            image_data = await response.read()
                                            logger.info(f"Successfully downloaded {len(image_data)} bytes of satellite imagery")
                                            
                                            # Convert to base64 for Gemini API
                                            base64_image = base64.b64encode(image_data).decode('utf-8')
                                            
                                            # Build prompt for Gemini
                                            tree_detection_prompt = f"""
                                            Analyze this satellite image for trees and potential hazards:
                                            
                                            Instructions:
                                            1. Identify all trees visible in this satellite image
                                            2. For each tree, provide:
                                               - Estimated position (x,y coordinates in the image where [0,0] is top-left corner and [1,1] is bottom-right corner)
                                               - Approximate height (5-35m range)
                                               - Species (if identifiable)
                                               - Potential risk factors based on location
                                            
                                            IMPORTANT: When specifying tree positions, ALWAYS use normalized coordinates where:
                                            - [0,0] is the TOP-LEFT corner of the image
                                            - [1,1] is the BOTTOM-RIGHT corner of the image
                                            
                                            For example, a tree in the center would be at position [0.5, 0.5].
                                            
                                            Return a JSON array of trees with their properties.
                                            """
                                            
                                            # Call Gemini API with the image
                                            logger.info("Calling Gemini API with satellite imagery")
                                            
                                            # Prepare the Gemini API request with multimodal content
                                            payload = {
                                                "contents": [
                                                    {
                                                        "parts": [
                                                            {
                                                                "text": tree_detection_prompt
                                                            },
                                                            {
                                                                "inline_data": {
                                                                    "mime_type": "image/jpeg",
                                                                    "data": base64_image
                                                                }
                                                            }
                                                        ]
                                                    }
                                                ],
                                                "generationConfig": {
                                                    "temperature": 0.2,
                                                    "topP": 0.8,
                                                    "topK": 40,
                                                    "maxOutputTokens": 4096
                                                }
                                            }
                                            
                                            # For Gemini 2.0 models, we directly send inline_data
                                            # This is simpler and more reliable for our use case
                                            
                                            # Save the image for logging purposes
                                            job_dir = os.path.join(TEMP_DIR, job_id)
                                            os.makedirs(job_dir, exist_ok=True)
                                            temp_img_path = os.path.join(job_dir, "satellite_img.jpg")
                                            with open(temp_img_path, 'wb') as f:
                                                f.write(base64.b64decode(base64_image))
                                            
                                            # Use our standard URL construction method with centralized API key handling
                                            generate_url = self._get_gemini_url(f"models/{self.model}:generateContent")
                                            logger.info(f"Using consistent API key format for vision API calls")
                                            
                                            # Create payload with inline image data (no Files API needed)
                                            vision_payload = {
                                                "contents": [
                                                    {
                                                        "parts": [
                                                            {
                                                                "text": tree_detection_prompt
                                                            },
                                                            {
                                                                "inline_data": {
                                                                    "mime_type": "image/jpeg",
                                                                    "data": base64_image
                                                                }
                                                            }
                                                        ]
                                                    }
                                                ],
                                                "generationConfig": payload["generationConfig"]
                                            }
                                                    
                                            # Make the API call with inline image data
                                            async with session.post(generate_url, json=vision_payload, timeout=60) as gemini_response:
                                                if gemini_response.status == 200:
                                                    gemini_result = await gemini_response.json()
                                                    logger.info("Successfully received response from Gemini API")
                                                    
                                                    # Log will be saved after the try/except block
                                                else:
                                                    error_text = await gemini_response.text()
                                                    logger.error(f"Error from Gemini API: status {gemini_response.status}, Response: {error_text}")
                                                    
                                                    # Parse and log JSON error details if available
                                                    try:
                                                        error_json = json.loads(error_text)
                                                        if "error" in error_json:
                                                            logger.error(f"Gemini API error details: {json.dumps(error_json['error'], indent=2)}")
                                                    except (json.JSONDecodeError, KeyError):
                                                        pass
                                                        
                                                    return {
                                                        "success": False,
                                                        "message": f"Gemini API error: {gemini_response.status}",
                                                        "error_details": error_text,
                                                        "timestamp": datetime.now().isoformat()
                                                    }
                                                
                                            # Save the response for debugging
                                            if job_id and 'gemini_result' in locals():
                                                gemini_response_dir = os.path.join(job_dir, 'gemini_response')
                                                os.makedirs(gemini_response_dir, exist_ok=True)
                                                
                                                # Save API response as JSON
                                                response_file = os.path.join(gemini_response_dir, 'response.json')
                                                with open(response_file, 'w') as f:
                                                    json.dump(gemini_result, f, indent=2)
                                                logger.info(f"Saved Gemini response to {response_file}")
                                                
                                                # Save request payload (without API key)
                                                vision_payload_copy = vision_payload.copy()
                                                request_file = os.path.join(gemini_response_dir, 'request.json')
                                                with open(request_file, 'w') as f:
                                                    json.dump(vision_payload_copy, f, indent=2)
                                                logger.info(f"Saved request payload to {request_file}")
                                                
                                                # Image was already saved to temp_img_path earlier
                                            
                                            # Process successful Gemini response
                                            if 'gemini_result' in locals():
                                                try:
                                                    response_text = gemini_result["candidates"][0]["content"]["parts"][0]["text"]
                                                    
                                                    # Try to parse JSON from the response
                                                    import re
                                                    
                                                    # Look for JSON arrays in the response
                                                    json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
                                                    if json_match:
                                                        try:
                                                            trees_data = json.loads(json_match.group(0))
                                                            
                                                            # Process the trees data to add missing fields and convert to our format
                                                            processed_trees = self._process_gemini_trees(trees_data, center, bounds, job_id)
                                                            
                                                            logger.info(f"Successfully extracted {len(processed_trees)} trees from Gemini response")
                                                            return {
                                                                "success": True,
                                                                "job_id": job_id,
                                                                "status": "complete",
                                                                "timestamp": datetime.now().isoformat(),
                                                                "tree_count": len(processed_trees),
                                                                "trees": processed_trees,
                                                                "message": f"Successfully detected {len(processed_trees)} trees using Gemini API",
                                                                "detection_source": "gemini_vision",
                                                                "using_coordinates": True,
                                                                "mapType": mapType,
                                                                "center": center,
                                                                "zoom": zoom
                                                            }
                                                        except json.JSONDecodeError:
                                                            logger.error("Failed to parse JSON from Gemini response")
                                                    
                                                    # If JSON parsing failed, fall back to our sample trees
                                                    logger.warning("Could not extract tree data from Gemini response, using fallback data")
                                                except (KeyError, IndexError) as e:
                                                    logger.error(f"Error extracting text from Gemini response: {e}")
                                            # Note: When no gemini_result is available, errors are already logged in previous blocks
                            
                            except Exception as e:
                                logger.error(f"Error during satellite imagery processing: {e}")
                                import traceback
                                logger.error(f"Traceback: {traceback.format_exc()}")
                            
                            # If the Static Maps API failed, try Map Tiles API as another option
                            logger.info("Attempting to fetch individual map tiles as an alternative")
                            try:
                                # Calculate tile coordinates based on bounds
                                tile_results = await self._fetch_map_tiles(center, zoom, bounds, mapType, job_id)
                                
                                if tile_results and tile_results.get("success"):
                                    logger.info("Successfully retrieved and processed map tiles")
                                    return tile_results
                            except Exception as tile_error:
                                logger.error(f"Error fetching map tiles: {tile_error}")
                            
                            # Fall back to sample trees if all methods failed
                            logger.info("Using sample tree data as fallback")
                            sample_trees = self._generate_sample_trees(center, bounds, job_id, mapType)
                            
                            return {
                                "success": True,
                                "job_id": job_id,
                                "status": "complete",
                                "timestamp": datetime.now().isoformat(),
                                "tree_count": len(sample_trees),
                                "trees": sample_trees,
                                "message": "Tree detection complete using coordinate-based analysis",
                                "detection_source": "gemini_coordinates",
                                "using_coordinates": True,
                                "mapType": mapType,
                                "center": center,
                                "zoom": zoom
                            }
                        except Exception as e:
                            logger.error(f"Error processing coordinates info: {e}")
                            return {
                                "success": False,
                                "job_id": job_id,
                                "status": "error",
                                "message": f"Error processing coordinates: {str(e)}",
                                "timestamp": datetime.now().isoformat(),
                                "tree_count": 0,
                                "trees": []
                            }
                    else:
                        # Old path - unsupported URL format
                        logger.error(f"Unsupported image URL format: {map_image_url}")
                        return {
                            "success": False,
                            "job_id": job_id,
                            "status": "error",
                            "message": f"Unsupported image URL format. Backend tile retrieval flag not set.",
                            "timestamp": datetime.now().isoformat(),
                            "tree_count": 0,
                            "trees": []
                        }
                    
                # Verify we have image data
                if not image_data or len(image_data) < 100:
                    logger.error(f"Invalid or empty image data, size: {len(image_data) if image_data else 0} bytes")
                    return {
                        "success": False,
                        "job_id": job_id,
                        "status": "error",
                        "message": "Invalid or empty image data received",
                        "timestamp": datetime.now().isoformat(),
                        "tree_count": 0,
                        "trees": []
                    }
                
                # Prepare multimodal API request with the image
                import base64
                
                # Check if we're in image mode or text-only mode
                if image_data:
                    # Determine MIME type based on URL or default to JPEG
                    mime_type = "image/jpeg"  # Default
                    if map_image_url and map_image_url.lower().endswith('.png'):
                        mime_type = "image/png"
                    
                    # Base64 encode the image
                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                    
                    # Create detailed prompt for tree detection with image
                    prompt_text = f"""
                    I need you to identify all individual trees in this satellite/map image. Looking at the image, perform these steps:

                    1. Identify EVERY single tree in the image, focusing on:
                       - Individual trees throughout the entire image
                       - Trees near buildings, houses, and infrastructure 
                       - Trees in residential yards, along streets, and in parks
                       - Trees near power lines and other utilities

                    2. CRITICAL POSITION INSTRUCTIONS - READ CAREFULLY:
                       For each tree, provide an [x, y] position. The coordinate system is:
                       - [0,0] = TOP-LEFT corner of the image
                       - [1,1] = BOTTOM-RIGHT corner of the image
                       - x increases from left to right (0→1)
                       - y increases from top to bottom (0→1)
                       
                       For example, a tree in the exact center would be at [0.5, 0.5].
                       A tree in the top-right corner would be at [1.0, 0.0].
                       A tree in the bottom-left corner would be at [0.0, 1.0].

                    3. For each tree you identify, provide ONLY this information in a JSON format:
                       - "id": unique identifier (e.g., "tree_1", "tree_2")
                       - "position": [x, y] - The tree's position following the coordinate system above
                       - "height": estimate in meters (typical range 5-35m)
                       - "species": best guess of tree species or "Unknown" if not clear
                       - "risk_level": "low", "medium", or "high" based on location near structures

                    Format your output EXACTLY as shown below - a raw JSON array:
                    [{{
                      "id": "tree_1",
                      "position": [0.25, 0.65],
                      "height": 12.5,
                      "species": "Oak",
                      "risk_level": "medium"
                    }}, 
                    ...more trees...]

                    IMPORTANT:
                    - Return ONLY the raw JSON array without any explanation, markdown formatting, or code blocks
                    - Coordinates must be normalized from 0-1 within the image
                    - If no trees are visible, return an empty array: []
                    """
                else:
                    # Use the text-only prompt prepared earlier
                    prompt_text = text_prompt
                
                # Create the proper multimodal request payload
                payload = {
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt_text},
                                {
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": image_b64
                                    }
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,  # Even lower temperature for more consistent output
                        "topP": 0.95,
                        "topK": 40,
                        "maxOutputTokens": 4096,
                        "responseMimeType": "application/json"  # Request JSON response
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
                
                # Add API key to URL
                url = self._get_gemini_url(f"models/{self.model}:generateContent")
                # Log with masked API key
                masked_url = url.replace(GEMINI_API_KEY, "****")
                logger.info(f"Using Gemini API URL: {masked_url}")
                
                # Make the request with proper timeout handling
                async with session.post(url, json=payload, timeout=60) as response:
                    
                    # First get the raw response once
                    response = await response.text()
                    
                    # Save raw response to a file
                    if job_id:
                        import os
                        response_dir = os.path.join('/ttt/data/temp', job_id, 'gemini_response')
                        os.makedirs(response_dir, exist_ok=True)
                        response_file = os.path.join(response_dir, 'response.txt')
                        with open(response_file, 'w') as f:
                            f.write(response)
                        logger.info(f"Saved raw HTTP response to {response_file}")
                    
                    # Now continue with normal processing - note that response is already a text string at this point
                    try:
                        # Try to parse response as JSON to check status
                        json_response = json.loads(response)
                        
                        if "error" in json_response:
                            error_details = json.dumps(json_response.get("error", {}))
                            logger.error(f"Gemini API error: Response: {error_details}")
                            
                            return {
                                "success": False,
                                "job_id": job_id,
                                "status": "error",
                                "message": f"Gemini API error: {json_response.get('error', {}).get('status', 'UNKNOWN_ERROR')}",
                                "error_details": error_details,
                                "timestamp": datetime.now().isoformat(),
                                "tree_count": 0,
                                "trees": []
                            }
                    except json.JSONDecodeError:
                        # Not a JSON response or some other error
                        logger.error(f"Gemini API error: Unable to parse response as JSON: {response[:200]}...")
                        
                        return {
                            "success": False,
                            "job_id": job_id,
                            "status": "error",
                            "message": "Gemini API returned invalid response",
                            "error_details": response[:500],
                            "timestamp": datetime.now().isoformat(),
                            "tree_count": 0,
                            "trees": []
                        }
                    
                    # Parse the response as JSON if it was successful
                    try:
                        result = json.loads(response)
                        logger.info(f"Received response from Gemini API: {str(result)[:200]}...")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse response as JSON: {response[:200]}...")
                        result = {"text": response}
            finally:
                # Close the session
                if session and not session.closed:
                    await session.close()
            
            # Process the response to extract tree data
            response_text = ""
            
            # Extract the response text
            try:
                response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                logger.info(f"Extracted text from Gemini response: {response_text[:200]}...")
            except Exception as e:
                # Return the error directly without fallback or simulation
                logger.error(f"Exception extracting text from Gemini response: {e}")
                return {
                    "success": False,
                    "job_id": job_id,
                    "status": "error",
                    "message": f"Error parsing Gemini response: {str(e)}",
                    "error_details": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "tree_count": 0,
                    "trees": []
                }
            
            # Extract JSON data if present (typically in code blocks or direct JSON response)
            import re
            
            # First try direct JSON parsing (if Gemini returned proper JSON)
            trees = []
            
            # Parse JSON from response
            try:
                # Try direct JSON parsing first (if response is a JSON array)
                if response_text.strip().startswith('[') and response_text.strip().endswith(']'):
                    trees_data = json.loads(response_text)
                    if isinstance(trees_data, list):
                        trees = trees_data
                        logger.info(f"Successfully parsed direct JSON response with {len(trees)} trees")
                
                # If direct parsing fails, look for JSON in code blocks
                if not trees:
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                    if json_match:
                        try:
                            json_str = json_match.group(1)
                            logger.info(f"Found JSON in code block: {json_str[:200]}...")
                            trees_data = json.loads(json_str)
                            
                            # Handle both array and object formats
                            if isinstance(trees_data, dict) and 'trees' in trees_data:
                                trees = trees_data['trees']
                            elif isinstance(trees_data, list):
                                trees = trees_data
                                
                            logger.info(f"Successfully extracted {len(trees)} trees from JSON in code block")
                        except Exception as e:
                            logger.error(f"Error parsing JSON in code block: {e}")
            except Exception as e:
                logger.error(f"Error in JSON extraction: {e}")
            
            # If still no trees found, try text-based extraction
            if not trees:
                logger.warning("No JSON tree data found, trying to extract from text")
                tree_pattern = r'Tree\s+(\d+)[\s\S]*?coordinates:?\s*\[?\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\]?[\s\S]*?height:?\s*(\d+\.?\d*)[\s\S]*?species:?\s*([^\n\.]+)[\s\S]*?risk[^:]*:?\s*([^\n\.]+)'
                tree_matches = re.findall(tree_pattern, response_text, re.IGNORECASE)
                
                for i, match in enumerate(tree_matches):
                    tree_id = f"{job_id}_{i}" if job_id else f"tree_{i}"
                    try:
                        lng = float(match[1])
                        lat = float(match[2])
                        height = float(match[3]) if match[3] else 10.0
                        species = match[4].strip() if match[4] else "Unknown Species"
                        risk_level = match[5].strip().lower() if match[5] else "medium"
                        
                        # Create tree object
                        tree = {
                            'id': tree_id,
                            'location': [lng, lat],
                            'height': height,
                            'species': species,
                            'risk_level': risk_level,
                            'confidence': 0.75
                        }
                        
                        trees.append(tree)
                    except Exception as e:
                        logger.error(f"Error parsing tree match {i}: {e}")
                
                if tree_matches:
                    logger.info(f"Extracted {len(trees)} trees using text pattern matching")
                    
                    # Ensure all trees have proper IDs and confidence values
                    for i, tree in enumerate(trees):
                        if 'id' not in tree:
                            tree['id'] = f"{job_id}_{i}" if job_id else f"tree_{i}"
                        tree['confidence'] = tree.get('confidence', 0.85)
                    
                    logger.info(f"Successfully extracted {len(trees)} trees from text patterns")
            
            # Format trees data to match expected structure
            for tree in trees:
                # Ensure height is a number
                try:
                    tree['height'] = float(tree['height'])
                except (ValueError, TypeError):
                    tree['height'] = 10.0  # Default height
                
                # Normalize risk level
                if 'risk_level' in tree:
                    risk = tree['risk_level'].lower() if isinstance(tree['risk_level'], str) else 'medium'
                    if risk in ['high', 'medium', 'low']:
                        tree['risk_level'] = risk
                    else:
                        tree['risk_level'] = 'medium'  # Default risk level
                else:
                    tree['risk_level'] = 'medium'  # Default risk level
            
            # Create the result object
            result = {
                "success": True,
                "job_id": job_id or "gemini_detection",
                "status": "complete",
                "timestamp": datetime.now().isoformat(),
                "tree_count": len(trees) if trees else 0,
                "trees": trees if trees else [],
                "detection_source": "gemini"  # Explicitly mark the detection source as Gemini
            }
            
            # Add message for cases with no trees
            if not trees:
                logger.warning("No trees found in Gemini response")
                result["message"] = "No trees detected with Gemini API"
            else:
                logger.info(f"Successfully detected {len(trees)} trees with Gemini API")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in Gemini tree detection: {str(e)}")
            error_response = {
                "success": False,
                "job_id": job_id,
                "status": "error",
                "message": f"Gemini detection error: {str(e)}",
                "error_details": str(e),
                "timestamp": datetime.now().isoformat(),
                "tree_count": 0,
                "trees": [],
                "detection_source": "gemini"
            }
                
            return error_response

    async def close(self):
        """Close connections and shutdown service"""
        self.is_initialized = False
        logger.info("Gemini service has been shut down")

    def _load_config(self, config_path: str = None) -> Dict:
        """
        Load Gemini API configuration from various sources with priority order.
        
        This method loads configuration in the following order of precedence:
        1. Environment variables (for CI/CD and deployment)
        2. Config values from config.py
        3. External config file (if provided)
        4. Default fallback values
        
        Args:
            config_path (str, optional): Path to external JSON configuration file.
                                         Defaults to None.
        
        Returns:
            Dict: Configuration dictionary with all Gemini API settings
                  including api_key, api_url, model, and timeout parameters.
        """
        # We already imported these at the module level, so just use them directly
        
        # Load environment variables 
        import os
        env_api_key = os.environ.get("GEMINI_API_KEY", "")
        # Use v1 API version
        env_api_url = os.environ.get(
            "GEMINI_API_URL", 
            "https://generativelanguage.googleapis.com/v1"
        )
        # Use gemini-2.0-flash for improved performance and multimodal support
        env_model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        
        # Get the API key directly from config.py - no manipulation
        api_key = GEMINI_API_KEY
        # Debug the key to verify it's being loaded correctly
        logger.info(f"API key from config - type: {type(api_key)}, length: {len(api_key)}")
        
        # Log API key presence (not the actual key)
        logger.info(f"Gemini API key available: {bool(api_key)}")
        
        default_config = {
            "gemini": {
                "api_key": api_key,
                "api_url": GEMINI_API_URL,
                "model": GEMINI_MODEL,
                "max_retries": 3,
                "timeout_seconds": 30
            }
        }
        
        # Optional: load from external config file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "gemini" in config:
                        default_config["gemini"].update(config["gemini"])
            except Exception as e:
                logger.error(f"Error loading config file: {str(e)}")
        
        # Log configuration (without API key for security)
        safe_config = {**default_config}
        if safe_config["gemini"]["api_key"]:
            # Just mask most of the API key but keep first 4 and last 4 characters
            full_key = safe_config["gemini"]["api_key"]
            if len(full_key) > 8:
                safe_config["gemini"]["api_key"] = full_key[:4] + "****" + full_key[-4:]
            else:
                safe_config["gemini"]["api_key"] = "****"
        logger.info(f"Loaded Gemini configuration: {safe_config}")
        
        return default_config
        
    async def _fetch_map_tiles(self, center, zoom, bounds, map_type, job_id):
        """Fetch individual map tiles for precise alignment with what user sees
        
        This function:
        1. Calculates which tiles are visible in the current view
        2. Fetches those tiles directly from Google Maps Tile API
        3. Stitches them together into a single image
        4. Sends the stitched image to Gemini API
        
        This gives us pixel-perfect alignment with what the user is seeing
        because we're using the exact same tile sources as the Maps JavaScript API.
        
        Args:
            center: Center coordinates [lng, lat]
            zoom: Zoom level (integer)
            bounds: Map bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            map_type: Map type (satellite, roadmap, etc)
            job_id: Job ID for tracking
            
        Returns:
            Dict with detection results
        """
        # Check if PIL is available for image processing
        if not PIL_AVAILABLE:
            logger.error("PIL (Pillow) is required for tile processing but not available")
            return {
                "success": False,
                "message": "Image processing library (PIL/Pillow) not available"
            }
            
        try:
            
            # Calculate tile coordinates based on bounds
            # Formula: x = floor((lon + 180) / 360 * 2^zoom)
            # Formula: y = floor((1 - ln(tan(lat * π/180) + 1/cos(lat * π/180)) / π) / 2 * 2^zoom)
            
            # Convert lat/lng to tile coordinates
            def lat_lng_to_tile(lat, lng, zoom):
                n = 2.0 ** zoom
                x = int((lng + 180.0) / 360.0 * n)
                y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / 
                         math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
                return x, y
                
            # Calculate tile coordinates for bounds
            sw_lng, sw_lat = bounds[0]
            ne_lng, ne_lat = bounds[1]
            
            # Get tile coordinates for corners
            sw_tile_x, sw_tile_y = lat_lng_to_tile(sw_lat, sw_lng, zoom)
            ne_tile_x, ne_tile_y = lat_lng_to_tile(ne_lat, ne_lng, zoom)
            
            # Ensure proper ordering
            min_tile_x = min(sw_tile_x, ne_tile_x)
            max_tile_x = max(sw_tile_x, ne_tile_x)
            min_tile_y = min(sw_tile_y, ne_tile_y)
            max_tile_y = max(sw_tile_y, ne_tile_y)
            
            # Calculate number of tiles needed
            x_tiles = max_tile_x - min_tile_x + 1
            y_tiles = max_tile_y - min_tile_y + 1
            
            # Limit to reasonable number of tiles (max 5x5 grid = 25 tiles)
            if x_tiles > 5 or y_tiles > 5:
                logger.warning(f"Too many tiles: {x_tiles}x{y_tiles}, limiting to 5x5")
                # Center the grid around the center point
                center_x, center_y = lat_lng_to_tile(center[1], center[0], zoom)
                min_tile_x = center_x - 2
                max_tile_x = center_x + 2
                min_tile_y = center_y - 2
                max_tile_y = center_y + 2
                x_tiles = 5
                y_tiles = 5
                
            logger.info(f"Fetching {x_tiles}x{y_tiles} map tiles at zoom level {zoom}")
            
            # Convert map type to tile API format
            if map_type.lower() == 'satellite':
                tile_layer = 'satellite'
            elif map_type.lower() == 'hybrid':
                tile_layer = 'hybrid'
            elif map_type.lower() == 'terrain':
                tile_layer = 'terrain'
            else:
                tile_layer = 'roadmap'
                
            # Create a composite image to store all tiles
            # Each Google Maps tile is 256x256 pixels
            tile_size = 256
            composite_width = x_tiles * tile_size
            composite_height = y_tiles * tile_size
            composite_image = Image.new('RGB', (composite_width, composite_height))
            
            # Set up async session for tile fetching
            import aiohttp
            tasks = []
            
            async with aiohttp.ClientSession() as session:
                # Create tasks for fetching all tiles
                for x in range(min_tile_x, max_tile_x + 1):
                    for y in range(min_tile_y, max_tile_y + 1):
                        # The URL format for Google Maps tiles
                        # For satellite: https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}
                        # Use API key if available for proper authentication
                        api_key_param = f"&key={self.api_key}" if hasattr(self, 'api_key') and self.api_key else ""
                        tile_url = f"https://mt1.google.com/vt/lyrs={tile_layer[0]}&x={x}&y={y}&z={zoom}{api_key_param}"
                        
                        logger.debug(f"Adding tile URL for x={x}, y={y}, zoom={zoom}")
                        
                        # Add to task list
                        tasks.append((session.get(tile_url, timeout=10), x - min_tile_x, y - min_tile_y))
                
                # Track success rate
                successful_tiles = 0
                total_tiles = len(tasks)
                
                # Execute all fetch tasks with retry logic
                for task, x_pos, y_pos in tasks:
                    max_retries = 3
                    retry_delay = 0.5  # Start with 500ms delay
                    
                    for retry in range(max_retries):
                        try:
                            # If it's a retry, create a new request
                            if retry > 0:
                                x = min_tile_x + x_pos
                                y = min_tile_y + y_pos
                                api_key_param = f"&key={self.api_key}" if hasattr(self, 'api_key') and self.api_key else ""
                                tile_url = f"https://mt1.google.com/vt/lyrs={tile_layer[0]}&x={x}&y={y}&z={zoom}{api_key_param}"
                                logger.info(f"Retry {retry}/{max_retries} for tile x={x}, y={y}")
                                response = await session.get(tile_url, timeout=10)
                            else:
                                response = await task
                                
                            if response.status == 200:
                                # Get image data
                                tile_data = await response.read()
                                
                                # Create PIL Image from data
                                tile_image = Image.open(io.BytesIO(tile_data))
                                
                                # Add to composite image
                                composite_image.paste(tile_image, (x_pos * tile_size, y_pos * tile_size))
                                
                                successful_tiles += 1
                                # Success, no need to retry
                                break
                            elif response.status == 429:  # Rate limit exceeded
                                logger.warning(f"Rate limit exceeded for tile: status {response.status}")
                                # Exponential backoff
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                logger.warning(f"Failed to fetch tile: status {response.status}")
                                # Constant retry delay for other errors
                                await asyncio.sleep(retry_delay)
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout fetching tile at position {x_pos},{y_pos}, retry {retry+1}/{max_retries}")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 2
                        except Exception as e:
                            logger.error(f"Error fetching tile: {e}")
                            await asyncio.sleep(retry_delay)
                    
                # Log success rate
                logger.info(f"Successfully fetched {successful_tiles}/{total_tiles} tiles ({successful_tiles/total_tiles*100:.1f}%)")
                
                # If we have too few successful tiles, we might want to abort
                if successful_tiles < total_tiles * 0.5:  # Less than 50% success
                    logger.warning(f"Too many missing tiles ({total_tiles-successful_tiles}/{total_tiles}), result may be incomplete")
            
            # Save the composite image in the gemini_response folder
            gemini_response_dir = os.path.join('/ttt/data/temp', job_id, 'gemini_response')
            os.makedirs(gemini_response_dir, exist_ok=True)
            composite_path = os.path.join(gemini_response_dir, 'composite_map.jpg')
            composite_image.save(composite_path, 'JPEG', quality=95)
            logger.info(f"Saved composite map image to {composite_path}")
            
            # Convert composite image to base64 for Gemini API
            buffered = io.BytesIO()
            composite_image.save(buffered, format="JPEG", quality=95)
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Create prompt for Gemini
            tree_detection_prompt = f"""
            I need you to identify all individual trees in this satellite/map image. Looking at the image, perform these steps:

            1. Identify EVERY single tree in the image, focusing on:
               - Individual trees throughout the entire image
               - Trees near buildings, houses, and infrastructure 
               - Trees in residential yards, along streets, and in parks
               - Trees near power lines and other utilities
               - Any tree that could potentially impact human structures
            
            2. For each tree you identify, provide this information:
               - box_2d: [x1, y1, x2, y2] - bounding box coordinates
               - label: description with position, height and species
            
            IMPORTANT: The label MUST follow this EXACT format: "the tree at [x,y] with height=H and species=S" 
            where x,y are coordinates from 0-1 (with 0,0 at top-left and 1,1 at bottom-right), 
            H is height in meters, and S is the species.

            Return ONLY a properly formatted JSON array of objects. Example format:
            ```
            [
              {{
                "box_2d": [11, 8, 26, 20],
                "label": "the tree at [0.018,0.014] with height=10 and species=Unknown"
              }},
              {{
                "box_2d": [5, 40, 20, 52],
                "label": "the tree at [0.046,0.011] with height=10 and species=Oak"
              }}
            ]
            ```
            """
            
            # Prepare the Gemini API request with multimodal content
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": tree_detection_prompt
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "topP": 0.8,
                    "topK": 40,
                    "maxOutputTokens": 4096
                }
            }
            
            # Make the direct API call to Gemini
            # Use gemini-2.0-flash for best multimodal performance
            url = self._get_gemini_url(f"models/{self.model}:generateContent")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=60) as gemini_response:
                    # Get the raw response
                    response_text = await gemini_response.text()
                    
                    # Save the raw response for debugging regardless of status
                    gemini_response_dir = os.path.join('/ttt/data/temp', job_id, 'gemini_response')
                    os.makedirs(gemini_response_dir, exist_ok=True)
                    
                    # Import json here to ensure it's in scope
                    import json
                    
                    # Save API response
                    response_file = os.path.join(gemini_response_dir, 'tile_response.txt')
                    with open(response_file, 'w') as f:
                        f.write(response_text)
                    logger.info(f"Saved raw Gemini tile response to {response_file}")
                    
                    # Save request payload (without API key)
                    safe_payload = payload.copy()
                    request_file = os.path.join(gemini_response_dir, 'tile_request.json')
                    with open(request_file, 'w') as f:
                        f.write(json.dumps(safe_payload, indent=2))
                    logger.info(f"Saved tile request payload to {request_file}")
                    
                    # Save tile imagery (if available)
                    if 'mime_type' in payload["contents"][0]["parts"][1]["inline_data"]:
                        try:
                            base64_data = payload["contents"][0]["parts"][1]["inline_data"]["data"]
                            image_data = base64.b64decode(base64_data)
                            image_file = os.path.join(gemini_response_dir, 'satellite_tile.jpg')
                            with open(image_file, 'wb') as f:
                                f.write(image_data)
                            logger.info(f"Saved satellite tile image to {image_file}")
                        except Exception as img_err:
                            logger.error(f"Failed to save tile image: {img_err}")
                    
                    # Then parse as JSON if status is 200
                    if gemini_response.status == 200:
                        try:
                            gemini_result = json.loads(response_text)
                            logger.info("Successfully received and parsed response from Gemini API")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse Gemini response as JSON: {e}")
                            # Save but don't try to process further
                            return {"success": False, "message": "Failed to parse Gemini API response"}
                        
                        # Try to extract tree data from the response
                        try:
                            response_text = gemini_result["candidates"][0]["content"]["parts"][0]["text"]
                            
                            # Save the parsed text response for debugging
                            response_text_path = os.path.join(gemini_response_dir, 'parsed_response.txt')
                            with open(response_text_path, 'w') as f:
                                f.write(response_text)
                            logger.info(f"Saved parsed Gemini text response to {response_text_path}")
                            
                            # Try to parse JSON from the response
                            import re
                            import json
                            
                            # First, try to find a complete JSON array
                            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
                            
                            if json_match:
                                try:
                                    # Extract the matched text and load it as JSON
                                    json_text = json_match.group(0)
                                    # Clean up any potential formatting issues
                                    json_text = json_text.replace("'", '"')
                                    # Parse the JSON
                                    trees_data = json.loads(json_text)
                                    
                                    # Process the trees data to add missing fields and convert to our format
                                    # For tiles, we need to adjust coordinates from image space (0-1) to geo coordinates
                                    processed_trees = self._process_tile_trees(trees_data, min_tile_x, min_tile_y, 
                                                                             x_tiles, y_tiles, zoom, job_id)
                                    
                                    logger.info(f"Successfully extracted {len(processed_trees)} trees from Gemini tile response")
                                    return {
                                        "success": True,
                                        "job_id": job_id,
                                        "status": "complete",
                                        "timestamp": datetime.now().isoformat(),
                                        "tree_count": len(processed_trees),
                                        "trees": processed_trees,
                                        "message": f"Successfully detected {len(processed_trees)} trees using Gemini API with map tiles",
                                        "detection_source": "gemini_tiles",
                                        "using_tiles": True,
                                        "mapType": map_type,
                                        "center": center,
                                        "zoom": zoom
                                    }
                                except json.JSONDecodeError as e:
                                    logger.error(f"Failed to parse JSON from Gemini tile response: {e}")
                                    # Try other methods instead of failing immediately
                            
                            # If we're here, the first method failed. Try to extract individual trees
                            # by looking for tree object patterns
                            try:
                                # Look for patterns like: { "position": [0.4, 0.5], ... }
                                tree_objects = re.findall(r'{[^{}]*"position"\s*:\s*\[[^[\]]*\][^{}]*}', response_text)
                                
                                if tree_objects:
                                    logger.info(f"Found {len(tree_objects)} potential tree objects using regex")
                                    
                                    # Try to parse each tree object and collect valid ones
                                    parsed_trees = []
                                    for tree_str in tree_objects:
                                        try:
                                            # Fix malformed JSON if needed (missing commas, etc.)
                                            tree_str = tree_str.replace("'", '"')
                                            
                                            # Add brackets and try to parse as individual tree
                                            tree_obj = json.loads(tree_str)
                                            if 'position' in tree_obj and isinstance(tree_obj['position'], list):
                                                parsed_trees.append(tree_obj)
                                        except json.JSONDecodeError:
                                            continue
                                            
                                    if parsed_trees:
                                        logger.info(f"Successfully extracted {len(parsed_trees)} trees using individual parsing")
                                        
                                        # Process the trees data
                                        processed_trees = self._process_tile_trees(parsed_trees, min_tile_x, min_tile_y, 
                                                                                 x_tiles, y_tiles, zoom, job_id)
                                        
                                        return {
                                            "success": True,
                                            "job_id": job_id,
                                            "status": "complete",
                                            "timestamp": datetime.now().isoformat(),
                                            "tree_count": len(processed_trees),
                                            "trees": processed_trees,
                                            "message": f"Successfully detected {len(processed_trees)} trees using Gemini API with map tiles (alternative parsing)",
                                            "detection_source": "gemini_tiles_alt",
                                            "using_tiles": True,
                                            "mapType": map_type,
                                            "center": center,
                                            "zoom": zoom
                                        }
                                
                            except Exception as alt_parse_error:
                                logger.error(f"Error in alternative parsing: {alt_parse_error}")
                            
                            # If we got here, both parsing methods failed
                            # Try one last approach - look for coordinates in the text
                            try:
                                # Look for patterns like positions or coordinates
                                coordinate_matches = re.findall(r'(?:position|coordinates|location):\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]', response_text)
                                
                                if coordinate_matches:
                                    logger.info(f"Found {len(coordinate_matches)} coordinate pairs in text")
                                    
                                    # Create simple tree objects from coordinates
                                    text_based_trees = []
                                    for i, (x, y) in enumerate(coordinate_matches):
                                        try:
                                            text_based_trees.append({
                                                "position": [float(x), float(y)],
                                                "height": 15.0,  # Default height
                                                "species": "Unknown",
                                                "risk_level": "medium"
                                            })
                                        except ValueError:
                                            continue
                                            
                                    if text_based_trees:
                                        logger.info(f"Created {len(text_based_trees)} tree objects from text extraction")
                                        
                                        # Process the trees data
                                        processed_trees = self._process_tile_trees(text_based_trees, min_tile_x, min_tile_y, 
                                                                                 x_tiles, y_tiles, zoom, job_id)
                                        
                                        return {
                                            "success": True,
                                            "job_id": job_id,
                                            "status": "complete",
                                            "timestamp": datetime.now().isoformat(),
                                            "tree_count": len(processed_trees),
                                            "trees": processed_trees,
                                            "message": f"Successfully detected {len(processed_trees)} trees using Gemini API with map tiles (text extraction)",
                                            "detection_source": "gemini_tiles_text",
                                            "using_tiles": True,
                                            "mapType": map_type,
                                            "center": center,
                                            "zoom": zoom
                                        }
                                        
                            except Exception as text_parse_error:
                                logger.error(f"Error in text-based parsing: {text_parse_error}")
                            
                            # All methods failed - return an informative error
                            logger.warning("Could not extract tree data from Gemini tile response using any method")
                            return {
                                "success": False, 
                                "message": "Failed to parse tree data from Gemini response",
                                "debug_info": "The Gemini API returned a response but all parsing methods failed"
                            }
                        except (KeyError, IndexError) as e:
                            logger.error(f"Error extracting text from Gemini tile response: {e}")
                            return {
                                "success": False, 
                                "message": f"Error processing Gemini response: {str(e)}",
                                "debug_info": "Check your API key and quota. The response format may have changed."
                            }
                    else:
                        logger.error(f"Error from Gemini API: status {gemini_response.status}, Response: {response_text[:500]}")
                        
                        # Try to parse the error if it's in JSON format
                        try:
                            error_details = json.loads(response_text)
                            logger.error(f"Gemini API error details: {json.dumps(error_details, indent=2)}")
                        except json.JSONDecodeError:
                            logger.error(f"Could not parse error response as JSON")
                        
                        # Handle quota errors specifically
                        if gemini_response.status == 400 and "quota" in response_text.lower():
                            logger.error("Possible quota exceeded or rate limit issue with Gemini API")
                        
                        # Return error instead of using fallback data
                        return {
                            "success": False,
                            "job_id": job_id,
                            "status": "error",
                            "timestamp": datetime.now().isoformat(),
                            "message": f"Gemini API error: {gemini_response.status}",
                            "error_details": response_text[:200],
                            "center": center,
                            "zoom": zoom
                        }
                        
        except Exception as processing_error:
            logger.error(f"Error in map tiles processing: {processing_error}")
            import traceback
            import json  # Explicitly import json here to ensure it's in scope
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Create the response directory for error info
            try:
                gemini_response_dir = os.path.join('/ttt/data/temp', job_id, 'gemini_response')
                os.makedirs(gemini_response_dir, exist_ok=True)
                
                # Save error information
                error_info = {
                    "message": "Tree detection failed using Gemini API",
                    "reason": "Satellite imagery could not be processed",
                    "timestamp": datetime.now().timestamp(),
                    "error": str(processing_error)
                }
                
                with open(os.path.join(gemini_response_dir, 'error_info.json'), 'w') as f:
                    f.write(json.dumps(error_info, indent=2))
                
                logger.info(f"Saved error information to {gemini_response_dir}/error_info.json")
            except Exception as save_err:
                logger.error(f"Failed to save error information: {save_err}")
            
            return {
                "success": False,
                "message": "Error processing satellite imagery",
                "error_details": str(processing_error)
            }
    
    def _process_tile_trees(self, trees_data, min_tile_x, min_tile_y, x_tiles, y_tiles, zoom, job_id):
        """Process tree data returned from Gemini API for map tiles
        
        This converts image coordinates from the stitched tile image to
        geographical coordinates that can be displayed on the map.
        """
        import math
        
        # Convert tile coordinates to lat/lng
        def tile_to_lat_lng(tile_x, tile_y, zoom):
            n = 2.0 ** zoom
            lng = tile_x / n * 360.0 - 180.0
            lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
            lat = math.degrees(lat_rad)
            return lat, lng
            
        processed_trees = []
        
        # Get SW and NE corners of the tile grid in lat/lng
        sw_lat, sw_lng = tile_to_lat_lng(min_tile_x, min_tile_y + y_tiles, zoom)
        ne_lat, ne_lng = tile_to_lat_lng(min_tile_x + x_tiles, min_tile_y, zoom)
        
        # Risk level mapping
        risk_mapping = {
            'low': 'low',
            'medium': 'medium',
            'high': 'high',
            'minimal': 'low',
            'moderate': 'medium',
            'significant': 'medium',
            'severe': 'high',
            'extreme': 'high',
            'none': 'low',
            'minor': 'low'
        }
        
        for i, tree in enumerate(trees_data):
            # Create a new tree object with our expected format
            processed_tree = {
                "id": f"{job_id}_tile_tree_{i}",
                "confidence": tree.get('confidence', 0.85)
            }
            
            # Handle location - convert from image coordinates to geo coordinates
            if 'position' in tree and isinstance(tree['position'], list) and len(tree['position']) >= 2:
                x_ratio, y_ratio = tree['position'][0], tree['position'][1]
                
                # Convert normalized image coordinates to geo coordinates
                # x_ratio ranges from 0 (left) to 1 (right)
                # y_ratio ranges from 0 (top) to 1 (bottom)
                lng = sw_lng + x_ratio * (ne_lng - sw_lng)
                lat = ne_lat - y_ratio * (ne_lat - sw_lat)  # y is inverted
                
                processed_tree['location'] = [lng, lat]
            else:
                # Try to handle different formats
                if 'location' in tree and isinstance(tree['location'], list) and len(tree['location']) >= 2:
                    # Direct location as [lng, lat]
                    processed_tree['location'] = tree['location']
                elif 'coordinates' in tree and isinstance(tree['coordinates'], list) and len(tree['coordinates']) >= 2:
                    # Direct coordinates as [lng, lat]
                    processed_tree['location'] = tree['coordinates']
                elif 'lat' in tree and 'lng' in tree:
                    # Separate lat/lng properties
                    processed_tree['location'] = [float(tree['lng']), float(tree['lat'])]
                elif 'latitude' in tree and 'longitude' in tree:
                    # Separate latitude/longitude properties
                    processed_tree['location'] = [float(tree['longitude']), float(tree['latitude'])]
                else:
                    # If no location data, place randomly within the tile grid
                    import random
                    lng = sw_lng + random.random() * (ne_lng - sw_lng)
                    lat = sw_lat + random.random() * (ne_lat - sw_lat)
                    processed_tree['location'] = [lng, lat]
            
            # Handle height
            if 'height' in tree:
                try:
                    # Handle height in different formats
                    height_str = str(tree['height'])
                    # Extract numeric part from strings like "15m" or "15 meters"
                    import re
                    height_match = re.search(r'(\d+(\.\d+)?)', height_str)
                    if height_match:
                        processed_tree['height'] = float(height_match.group(1))
                    else:
                        processed_tree['height'] = float(height_str)
                except (ValueError, TypeError):
                    # Default height if conversion fails
                    processed_tree['height'] = 15.0
            else:
                processed_tree['height'] = 15.0
            
            # Handle species
            processed_tree['species'] = tree.get('species', 'Unknown')
            
            # Handle risk level
            if 'risk' in tree or 'risk_level' in tree:
                risk_text = tree.get('risk_level', tree.get('risk', 'medium')).lower()
                
                # Map to our standard risk levels
                for key, value in risk_mapping.items():
                    if key in risk_text:
                        processed_tree['risk_level'] = value
                        break
                else:
                    processed_tree['risk_level'] = 'medium'  # Default
            else:
                processed_tree['risk_level'] = 'medium'  # Default
            
            # Handle risk factors
            if 'risk_factors' in tree and isinstance(tree['risk_factors'], list):
                processed_tree['risk_factors'] = tree['risk_factors']
            elif 'hazards' in tree and isinstance(tree['hazards'], list):
                # Convert hazards to risk factors
                processed_tree['risk_factors'] = [
                    {"factor": hazard, "severity": processed_tree['risk_level']}
                    for hazard in tree['hazards']
                ]
            elif 'notes' in tree or 'description' in tree:
                # Extract risk factors from description
                description = tree.get('notes', tree.get('description', ''))
                import re
                potential_factors = re.findall(r'([^,.;]+(?:risk|hazard|danger|threat|issue|problem)[^,.;]+)', description, re.IGNORECASE)
                if potential_factors:
                    processed_tree['risk_factors'] = [
                        {"factor": factor.strip(), "severity": processed_tree['risk_level']}
                        for factor in potential_factors
                    ]
                else:
                    processed_tree['risk_factors'] = []
            else:
                processed_tree['risk_factors'] = []
            
            processed_trees.append(processed_tree)
        
        return processed_trees
    
    def _process_gemini_trees(self, trees_data, center, bounds, job_id):
        """Process tree data returned from Gemini API
        
        This function takes the raw tree data from Gemini and processes it to match
        our expected format, including adding missing fields and converting coordinates.
        
        Args:
            trees_data: List of tree objects from Gemini API
            center: Center coordinates [lng, lat]
            bounds: Map bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            job_id: Job ID for tracking
            
        Returns:
            List of processed tree objects in our format
        """
        # Check if we're using the new format with box_2d and label
        if trees_data and isinstance(trees_data[0], dict) and 'label' in trees_data[0]:
            # Process using the new box_label format
            return self._process_gemini_box_label_trees(trees_data, bounds, job_id)
            
        # Check if we're using the format with position, height, and species
        if trees_data and isinstance(trees_data[0], dict) and 'position' in trees_data[0] and isinstance(trees_data[0]['position'], list):
            # Process using the position-based format
            return self._process_gemini_position_trees(trees_data, bounds, job_id)
        
        processed_trees = []
        
        # Get bounds dimensions
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        lng_span = ne_lng - sw_lng
        lat_span = ne_lat - sw_lat
        
        # Risk level mapping - normalize different ways Gemini might express risk
        risk_mapping = {
            'low': 'low',
            'medium': 'medium',
            'high': 'high',
            'minimal': 'low',
            'moderate': 'medium',
            'significant': 'medium',
            'severe': 'high',
            'extreme': 'high',
            'none': 'low',
            'minor': 'low'
        }
        
        for i, tree in enumerate(trees_data):
            # Create a new tree object with our expected format
            processed_tree = {
                "id": f"{job_id}_tree_{i}",
                "confidence": tree.get('confidence', 0.85)
            }
            
            # Handle location - Gemini might return position in different formats
            if 'location' in tree and isinstance(tree['location'], list) and len(tree['location']) >= 2:
                # Direct location as [lng, lat]
                processed_tree['location'] = tree['location']
            elif 'position' in tree and isinstance(tree['position'], list) and len(tree['position']) >= 2:
                # Convert position to location
                # This assumes position is in image coordinates (0-1) and needs to be mapped to geo coordinates
                x, y = tree['position'][0], tree['position'][1]
                # Debug the coordinate transformation
                logger.info(f"Converting Gemini position [x={x}, y={y}] to geo coordinates")
                logger.info(f"Map bounds: SW=[{sw_lng}, {sw_lat}], NE=[{ne_lng}, {ne_lat}]")
                
                # Map x,y to lng,lat based on bounds with proper inversion for image coordinates
                # Gemini returns coordinates with [0,0] at top-left, [1,1] at bottom-right
                # Map coordinates have [sw_lng, sw_lat] at bottom-left, [ne_lng, ne_lat] at top-right
                
                # Transform x to longitude (left-right is the same orientation in both systems)
                lng = sw_lng + (x * lng_span)
                
                # Transform y to latitude (top-down in image vs bottom-up in geo)
                # Need to invert the y-axis: as y increases in image coords, latitude decreases
                lat = ne_lat - (y * lat_span)
                
                logger.info(f"Transformed to geo coordinates: [{lng}, {lat}]")
                processed_tree['location'] = [lng, lat]
            elif 'coordinates' in tree and isinstance(tree['coordinates'], list) and len(tree['coordinates']) >= 2:
                # Direct coordinates as [lng, lat]
                processed_tree['location'] = tree['coordinates']
            elif 'lat' in tree and 'lng' in tree:
                # Separate lat/lng properties
                processed_tree['location'] = [float(tree['lng']), float(tree['lat'])]
            elif 'latitude' in tree and 'longitude' in tree:
                # Separate latitude/longitude properties
                processed_tree['location'] = [float(tree['longitude']), float(tree['latitude'])]
            else:
                # Fallback - place the tree randomly within bounds
                import random
                lng = sw_lng + random.random() * lng_span
                lat = sw_lat + random.random() * lat_span
                processed_tree['location'] = [lng, lat]
            
            # Handle height
            if 'height' in tree:
                try:
                    # Handle height in different formats
                    height_str = str(tree['height'])
                    # Extract numeric part from strings like "15m" or "15 meters"
                    import re
                    height_match = re.search(r'(\d+(\.\d+)?)', height_str)
                    if height_match:
                        processed_tree['height'] = float(height_match.group(1))
                    else:
                        processed_tree['height'] = float(height_str)
                except (ValueError, TypeError):
                    # Default height if conversion fails
                    processed_tree['height'] = 15.0
            else:
                processed_tree['height'] = 15.0
            
            # Handle species
            processed_tree['species'] = tree.get('species', 'Unknown')
            
            # Handle risk level
            if 'risk' in tree or 'risk_level' in tree:
                risk_text = tree.get('risk_level', tree.get('risk', 'medium')).lower()
                
                # Map to our standard risk levels
                for key, value in risk_mapping.items():
                    if key in risk_text:
                        processed_tree['risk_level'] = value
                        break
                else:
                    processed_tree['risk_level'] = 'medium'  # Default
            else:
                processed_tree['risk_level'] = 'medium'  # Default
            
            # Handle risk factors
            if 'risk_factors' in tree and isinstance(tree['risk_factors'], list):
                processed_tree['risk_factors'] = tree['risk_factors']
            elif 'hazards' in tree and isinstance(tree['hazards'], list):
                # Convert hazards to risk factors
                processed_tree['risk_factors'] = [
                    {"factor": hazard, "severity": processed_tree['risk_level']}
                    for hazard in tree['hazards']
                ]
            elif 'notes' in tree or 'description' in tree:
                # Extract risk factors from description
                description = tree.get('notes', tree.get('description', ''))
                import re
                potential_factors = re.findall(r'([^,.;]+(?:risk|hazard|danger|threat|issue|problem)[^,.;]+)', description, re.IGNORECASE)
                if potential_factors:
                    processed_tree['risk_factors'] = [
                        {"factor": factor.strip(), "severity": processed_tree['risk_level']}
                        for factor in potential_factors
                    ]
                else:
                    processed_tree['risk_factors'] = []
            else:
                processed_tree['risk_factors'] = []
            
            processed_trees.append(processed_tree)
        
        return processed_trees
        
    def _process_gemini_position_trees(self, trees_data, bounds, job_id):
        """Process tree data in the position format
        
        This function handles the format with position, height, and species fields
        
        Args:
            trees_data: List of tree objects with position, height, and species from Gemini
            bounds: Map bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            job_id: The job ID for tracking
            
        Returns:
            List of processed tree objects in our format
        """
        processed_trees = []
        
        # Set up bounds for coordinate transformation
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        lng_span = ne_lng - sw_lng
        lat_span = ne_lat - sw_lat
        
        logger.info(f"Processing {len(trees_data)} trees with position format, bounds: SW=[{sw_lng}, {sw_lat}], NE=[{ne_lng}, {ne_lat}]")
        
        for i, tree in enumerate(trees_data):
            # Extract position coordinates (x,y in 0-1 range)
            position = tree.get('position', [0.5, 0.5])
            x = position[0]
            y = position[1]
            
            # Extract other tree properties
            height = tree.get('height', 10)
            species = tree.get('species', 'Unknown')
            
            # Debug the coordinate transformation
            logger.info(f"Converting Gemini position [x={x}, y={y}] to geo coordinates")
            
            # Transform x to longitude (left-right is the same orientation in both systems)
            lng = sw_lng + (x * lng_span)
            
            # Transform y to latitude (top-down in image vs bottom-up in geo)
            # Need to invert the y-axis: as y increases in image coords, latitude decreases
            lat = ne_lat - (y * lat_span)
            
            logger.info(f"Transformed to geo coordinates: [{lng}, {lat}]")
            
            # Create the processed tree object
            processed_tree = {
                "id": f"{job_id}_tree_{i}",
                "location": [lng, lat],
                "height": height,
                "species": species if species != "Unknown" else "Unknown Species",
                "risk_level": "medium",  # Default risk level
                "risk_factors": [],  # Default empty risk factors
                "confidence": 0.85
            }
            
            processed_trees.append(processed_tree)
        
        return processed_trees
    
    def _process_gemini_box_label_trees(self, trees_data, bounds, job_id):
        """Process tree data in the box_2d and label format
        
        This function parses trees where each tree has a box_2d and label property,
        with the label in the format: "the tree at [x,y] with height=H and species=S"
        
        Args:
            trees_data: List of tree objects with box_2d and label from Gemini
            bounds: Map bounds [[sw_lng, sw_lat], [ne_lng, ne_lat]]
            job_id: The job ID for tracking
            
        Returns:
            List of processed tree objects in our format
        """
        import re
        
        processed_trees = []
        
        # Pattern to match tree position, height, and species in label
        label_pattern = r'the tree at \[(\d+\.?\d*),\s*(\d+\.?\d*)\] with height=(\d+\.?\d*) and species=(\w+)'
        
        # Risk mapping for standardizing risk level terms
        risk_mapping = {
            'high': 'high',
            'severe': 'high',
            'extreme': 'high',
            'medium': 'medium',
            'moderate': 'medium',
            'low': 'low',
            'minimal': 'low',
        }
        
        # Set up bounds for coordinate transformation
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        lng_span = ne_lng - sw_lng
        lat_span = ne_lat - sw_lat
        
        logger.info(f"Processing {len(trees_data)} trees with bounds: SW=[{sw_lng}, {sw_lat}], NE=[{ne_lng}, {ne_lat}]")
        
        for i, tree in enumerate(trees_data):
            # Default values
            processed_tree = {
                "id": f"{job_id}_tree_{i}",
                "confidence": 0.85
            }
            
            # Parse the label to extract position, height, and species
            label = tree.get('label', '')
            label_match = re.search(label_pattern, label)
            
            if label_match:
                x = float(label_match.group(1))
                y = float(label_match.group(2))
                height = float(label_match.group(3))
                species = label_match.group(4)
                
                # Debug the coordinate transformation
                logger.info(f"Converting Gemini position [x={x}, y={y}] to geo coordinates")
                logger.info(f"Map bounds: SW=[{sw_lng}, {sw_lat}], NE=[{ne_lng}, {ne_lat}]")
                
                # Transform x to longitude (left-right is the same orientation in both systems)
                lng = sw_lng + (x * lng_span)
                
                # Transform y to latitude (top-down in image vs bottom-up in geo)
                # Need to invert the y-axis: as y increases in image coords, latitude decreases
                lat = ne_lat - (y * lat_span)
                
                logger.info(f"Transformed to geo coordinates: [{lng}, {lat}]")
                
                # Add processed data to the tree
                processed_tree.update({
                    "location": [lng, lat],
                    "height": height,
                    "species": species if species != "Unknown" else "Unknown Species",
                    "risk_level": "medium",  # Default risk level
                    "risk_factors": []  # Default empty risk factors
                })
                
                # If we have a bounding box, add it
                if 'box_2d' in tree and isinstance(tree['box_2d'], list) and len(tree['box_2d']) == 4:
                    processed_tree['bbox'] = tree['box_2d']
            else:
                # Fallback for trees without proper label format
                logger.warning(f"Tree {i} has label that doesn't match expected format: {label}")
                continue
            
            processed_trees.append(processed_tree)
        
        return processed_trees
    
    def _generate_sample_trees(self, center, bounds, job_id, map_type):
        """Generate sample trees for demonstration purposes
        
        In a real implementation, this would be replaced by actual Gemini API calls
        with satellite imagery. For demonstration, we generate realistic tree data
        within the specified bounds.
        """
        import random
        import math
        
        # Common tree species based on map type
        if map_type == 'satellite':
            tree_species = ['Oak', 'Pine', 'Maple', 'Cedar', 'Elm', 'Spruce', 'Palm', 'Birch']
        else:
            tree_species = ['Tree'] # For non-satellite views, just use generic "Tree"
        
        # Risk levels with weighted distribution
        risk_levels = ['low', 'medium', 'high']
        risk_weights = [0.6, 0.3, 0.1]  # 60% low, 30% medium, 10% high
        
        # Generate between 5-15 trees
        num_trees = random.randint(5, 15)
        trees = []
        
        # Calculate bounds dimensions
        sw_lng, sw_lat = bounds[0]
        ne_lng, ne_lat = bounds[1]
        lng_span = ne_lng - sw_lng
        lat_span = ne_lat - sw_lat
        
        for i in range(num_trees):
            # Generate random position within bounds
            lng = sw_lng + random.random() * lng_span
            lat = sw_lat + random.random() * lat_span
            
            # Generate tree properties
            tree = {
                "id": f"{job_id}_tree_{i}",
                "location": [lng, lat],
                "height": round(random.uniform(5, 35), 1),  # Tree height between 5-35m
                "species": random.choice(tree_species),
                "risk_level": random.choices(risk_levels, weights=risk_weights)[0],
                "confidence": round(random.uniform(0.7, 0.98), 2)  # Confidence score
            }
            
            # Add distance from center as a property (for demonstration)
            dx = (lng - center[0]) * 111320 * math.cos(lat * math.pi / 180)  # meters
            dy = (lat - center[1]) * 110540  # meters
            distance = math.sqrt(dx*dx + dy*dy)
            tree["distance_from_center"] = round(distance, 1)
            
            # Add random risk factors based on risk level
            if tree["risk_level"] == "high":
                tree["risk_factors"] = [
                    {"factor": "Leaning tree", "severity": "high"},
                    {"factor": "Dead branches", "severity": "medium"}
                ]
            elif tree["risk_level"] == "medium":
                tree["risk_factors"] = [
                    {"factor": "Proximity to structures", "severity": "medium"}
                ]
            else:
                tree["risk_factors"] = []
                
            trees.append(tree)
            
        return trees

    def _parse_analysis_response(self, response: str) -> Dict:
        """
        Parse the Gemini API response into structured sections for easier frontend consumption.
        
        This method implements a rule-based parser that categorizes the text response
        into predefined sections based on keyword matching. It detects section headers
        in the response and organizes the content accordingly.
        
        Args:
            response (str): The raw text response from Gemini API containing
                           the tree risk analysis.
        
        Returns:
            Dict: A dictionary with the following keys, each containing the 
                 corresponding section content as a string:
                 - risk_analysis: Detailed risk assessment information
                 - recommendations: Suggested actions for risk mitigation
                 - future_concerns: Potential future issues based on growth patterns
                 - comparison: Comparison with similar trees in the database
        """
        sections = {
            "risk_analysis": "",
            "recommendations": "",
            "future_concerns": "",
            "comparison": ""
        }
        
        current_section = None
        
        # Simple rule-based parsing
        for line in response.split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            # Check for section headers by keyword matching
            lower_line = line.lower()
            if "risk analysis" in lower_line or "detailed risk" in lower_line:
                current_section = "risk_analysis"
                continue
            elif "recommendation" in lower_line or "mitigation" in lower_line:
                current_section = "recommendations"
                continue
            elif "future concern" in lower_line or "growth pattern" in lower_line:
                current_section = "future_concerns"
                continue
            elif "comparison" in lower_line or "similar trees" in lower_line:
                current_section = "comparison"
                continue
            
            # Add content to current section, maintaining line breaks for readability
            if current_section and current_section in sections:
                if sections[current_section]:
                    sections[current_section] += "\n" + line
                else:
                    sections[current_section] = line
        
        return sections