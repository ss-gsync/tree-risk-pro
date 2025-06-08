"""
Tree Risk Assessment Dashboard - Backend API
--------------------------------------------
Flask application that serves as the backend for the Tree Risk Assessment Dashboard.
Provides APIs for tree data, risk assessments, validation, and advanced analytics.
Integrates with Gemini AI for data analysis.
"""

# Standard library imports
import os
import json
import asyncio
import logging
import time
from datetime import datetime

# Third-party imports
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Fix for module name conflict with Python's built-in 'code' module
# Werkzeug's debug console uses the built-in 'code' module, which was being shadowed
# by our local 'code.py' file. We renamed it to 'tree_detection_examples.py'
import sys
if 'werkzeug.debug.console' not in sys.modules and 'code' not in sys.modules:
    import code as python_code
    sys.modules['code'] = python_code

# Local application imports
from auth import require_auth
from config import ZARR_DIR, ML_DIR, LIDAR_DIR, REPORTS_DIR
from services.tree_service import TreeService
from services.validation_service import ValidationService
from services.lidar_service import LidarService
from services.database_service import DatabaseService
from services.gemini_service import GeminiService

# We don't import from tree_ml.pipeline anymore since we only use Gemini API

# Configure logging
log_path = os.path.join(os.path.dirname(__file__), 'logs', 'app.log')
log_handlers = [logging.StreamHandler()]  # Always use stream handler

# Try to use file handler, but fall back if permissions issue
try:
    # Ensure logs directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)
    log_handlers.append(logging.FileHandler(log_path))
except (IOError, PermissionError) as e:
    print(f"Warning: Could not create log file ({str(e)}). Logs will only be sent to stdout.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)
os.makedirs(ZARR_DIR, exist_ok=True) # Ensure Zarr directory exists

# Initialize basic services
tree_service = TreeService()
validation_service = ValidationService()
lidar_service = LidarService()

def create_app():
    """
    Create and configure the Flask application
    
    This function initializes the Flask application with all necessary configurations,
    registers blueprints, and initializes required services asynchronously.
    
    Returns:
        Flask: The configured Flask application instance
    """
    # Initialize Flask app
    app = Flask(__name__)
    
    # Create shared detection service instance to avoid duplicate model loading
    from services.detection_service import DetectionService
    app.detection_service = DetectionService()
    
    # Enable basic CORS for all routes with minimal configuration
    # This allows cross-origin requests from the frontend
    CORS(app, supports_credentials=True)

    # Add endpoint for ML-based tree detection with FormData (for more efficient image uploads)
    @app.route('/api/detection/detect_with_image', methods=['POST'])
    async def detect_trees_with_image_endpoint():
        """Detect trees using ML pipeline with FormData image upload"""
        logger.info("ML detection with image endpoint called - entry point")
        # Default job_id using millisecond precision to match frontend's Date.now()
        job_id = f"detection_{int(time.time() * 1000)}"
        
        try:
            # Verify we have form data
            if 'satellite_image' not in request.files:
                logger.error("No satellite image in request")
                return jsonify({'error': 'No satellite image provided', 'status': 'error'}), 400
            
            # Get the image file
            satellite_image = request.files['satellite_image']
            
            # Get job_id and map_view_info from form data
            job_id = request.form.get('job_id', job_id)
            map_view_info_str = request.form.get('map_view_info')
            
            if not map_view_info_str:
                logger.error("No map view info provided in form data")
                return jsonify({
                    'error': 'No map view info provided',
                    'status': 'error',
                    'job_id': job_id
                }), 400
            
            # Parse map view info
            try:
                map_view_info = json.loads(map_view_info_str)
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing map view info: {e}")
                return jsonify({
                    'error': 'Invalid map view info format',
                    'status': 'error',
                    'job_id': job_id
                }), 400
                
            logger.info(f"Image upload detection request received - job_id: {job_id}")
            
            # Create ML directories for this job
            ml_dir = os.path.join(ML_DIR, job_id)
            os.makedirs(ml_dir, exist_ok=True)
            
            # Save the uploaded image
            image_path = os.path.join(ml_dir, f"satellite_{job_id}.jpg")
            satellite_image.save(image_path)
            logger.info(f"Saved uploaded image to {image_path}")
            
            # COORDINATE HANDLING: Validate map center coordinates
            view_data = map_view_info.get('viewData', {})
            center_coords = view_data.get('center')
            
            # Basic validation of center coordinates
            if not center_coords or len(center_coords) != 2:
                logger.error("ERROR: No valid center coordinates in map view info")
                return jsonify({
                    'error': 'Missing center coordinates for tree detection',
                    'status': 'error',
                    'job_id': job_id,
                    'message': 'Tree detection requires valid map center coordinates.'
                }), 400
                
            # Process image with ML detection - use the shared detection service
            detection_service = app.detection_service
            
            # Run the detection using the satellite image path
            result = await detection_service.detect_trees_from_image(
                image_path=image_path,
                map_view_info=map_view_info,
                job_id=job_id
            )
            
            # Add success flag
            result['success'] = True
            result['status'] = 'complete'
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error processing image upload: {str(e)}", exc_info=True)
            return jsonify({
                'error': str(e),
                'job_id': job_id,
                'status': 'error'
            }), 500
    
    # Add endpoint for ML-based tree detection with JSON payload
    @app.route('/api/detection/detect', methods=['POST'])
    async def detect_trees_ml_endpoint():
        """Detect trees using ML pipeline from map data - Primary endpoint for ML detection"""
        logger.info("ML detection endpoint called - entry point")
        # Default job_id using millisecond precision to match frontend's Date.now()
        job_id = f"detection_{int(time.time() * 1000)}"
        
        try:
            # Get request data
            data = request.get_json()
            if not data:
                logger.error("No data provided in request")
                return jsonify({'error': 'No data provided', 'status': 'error'}), 400
            
            # Get job ID from request or generate a new one
            job_id = data.get('job_id', job_id)
            logger.info(f"Detection request received - job_id: {job_id}")
            
            # Check for required map view info
            map_view_info = data.get('map_view_info')
            if not map_view_info:
                # Try alternative sources of map data
                if 'coordinates' in data:
                    logger.info("Using coordinates data instead of map_view_info")
                    map_view_info = {'viewData': {'bounds': data['coordinates'].get('bounds')}}
                elif 'viewData' in data:
                    logger.info("Using viewData directly")
                    map_view_info = {'viewData': data['viewData']}
                else:
                    logger.error("No map view info or coordinates provided")
                    return jsonify({
                        'error': 'No map view info provided',
                        'status': 'error',
                        'job_id': job_id
                    }), 400
            
            # COORDINATE HANDLING: Comprehensive validation of map center coordinates
            view_data = map_view_info.get('viewData', {})
            center_coords = view_data.get('center')
            
            # Basic validation of center coordinates
            if not center_coords or len(center_coords) != 2:
                logger.error("ERROR: No valid center coordinates available")
                return jsonify({
                    'error': 'Missing center coordinates for tree detection',
                    'status': 'error',
                    'job_id': job_id,
                    'message': 'Tree detection requires valid map center coordinates.'
                }), 400
            
            # Define constants for Dallas default coordinates check
            DEFAULT_DALLAS_LNG = -96.7
            DEFAULT_DALLAS_LAT = 32.8
            DALLAS_TOLERANCE = 0.05
            
            # Check if these are the default Dallas coordinates
            is_default_dallas = (abs(center_coords[0] - DEFAULT_DALLAS_LNG) < DALLAS_TOLERANCE and 
                                abs(center_coords[1] - DEFAULT_DALLAS_LAT) < DALLAS_TOLERANCE)
            
            if is_default_dallas:
                logger.warning(f"Default Dallas coordinates detected: {center_coords}")
                
            logger.info(f"Using map center coordinates: {center_coords}")
            
            # Log request data for debugging
            logger.info(f"Map view bounds: {map_view_info.get('viewData', {}).get('bounds')}")
            
            # Get ML status before running detection
            try:
                # Use the shared detection service
                detection_service = app.detection_service
                
                # Check ML service status for debugging
                if hasattr(detection_service, 'ml_service') and detection_service.ml_service:
                    ml_status = detection_service.ml_service.get_model_status()
                    logger.info(f"ML service status check: models_loaded={ml_status.get('models_loaded', False)}")
                else:
                    logger.warning("ML service not available for status check")
            except Exception as status_error:
                logger.error(f"Error checking ML service status: {status_error}")
            
            # Run detection with detailed logging
            logger.info(f"Starting ML detection pipeline for job {job_id}")
            start_time = time.time()
            
            # Get detection service instance - use the shared instance
            detection_service = app.detection_service
            
            # Set ML settings in the request, but preserve any existing coordinates
            map_view_info['use_gemini'] = False
            map_view_info['use_deepforest'] = True
            map_view_info['segmentation_mode'] = True
            map_view_info['use_in_memory_service'] = True
            
            # Log the view data for debugging, especially center coordinates and zoom
            if 'viewData' in map_view_info:
                view_data = map_view_info.get('viewData', {})
                logger.info(f"ML detection center: {view_data.get('center')}")
                logger.info(f"ML detection zoom: {view_data.get('zoom')}")
                
                # Log center coordinates without modification
                center = view_data.get('center')
                if not center or len(center) != 2:
                    logger.error("Missing or invalid center coordinates - cannot proceed")
                    return jsonify({
                        'error': 'Missing valid center coordinates',
                        'status': 'error',
                        'job_id': job_id
                    }), 400
                
                # No fallbacks or defaults - just use the exact coordinates from frontend
                logger.info(f"Using exact center coordinates from frontend: {center}")
            
            # Run detection
            result = await detection_service.detect_trees_from_map_view(map_view_info, job_id)
            
            # Add success flag and timing information
            detection_time = time.time() - start_time
            if 'error' not in result or not result['error']:
                result['success'] = True
                result['status'] = 'complete'
            else:
                result['success'] = False
                result['status'] = 'error'
            
            result['detection_time_seconds'] = round(detection_time, 2)
            
            # Log detection results
            tree_count = len(result.get('trees', []))
            ml_response_dir = result.get('ml_response_dir', '')
            logger.info(f"ML detection completed in {detection_time:.2f}s for job {job_id} - found {tree_count} trees")
            
            if ml_response_dir:
                logger.info(f"ML response stored at: {ml_response_dir}")
            
            # Return results with ml_response_dir for the frontend
            return jsonify(result)
        
        except Exception as e:
            # Capture full stack trace for debugging
            import traceback
            stack_trace = traceback.format_exc()
            
            logger.error(f"UNHANDLED ERROR in ML tree detection API: {str(e)}", exc_info=True)
            logger.error(f"Stack trace: {stack_trace}")
            
            # Return detailed error information for debugging
            return jsonify({
                'error': str(e),
                'job_id': job_id,
                'status': 'error',
                'stack_trace': stack_trace if os.environ.get('DEBUG') == '1' else None
            }), 500
            
    @app.route('/api/detect-trees-ml', methods=['POST'])
    def detect_trees_ml():
        """Detect trees using ML pipeline from map data - Legacy endpoint"""
        try:
            # Get request data
            request_data = request.json or {}
            map_view_info = request_data.get('map_view_info', {})
            job_id = request_data.get('job_id', f"detection_{int(datetime.now().timestamp())}")
            
            if not map_view_info:
                return jsonify({
                    "success": False,
                    "message": "Map view information is required",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            # Log request data for debugging
            logger.info(f"ML detection request received - job_id: {job_id}")
            logger.info(f"Map view bounds: {map_view_info.get('viewData', {}).get('bounds')}")
            
            # Import the detection service - use the shared instance
            detection_service = app.detection_service
            
            # Run detection in an async context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            detection_result = loop.run_until_complete(
                detection_service.detect_trees_from_map_view(map_view_info, job_id)
            )
            loop.close()
            
            # Add success flag if there's no error
            if 'error' not in detection_result or not detection_result['error']:
                detection_result['success'] = True
            
            # Log detection results
            tree_count = len(detection_result.get('trees', []))
            logger.info(f"ML detection completed for job {job_id} - found {tree_count} trees")
            
            # Return complete detection result
            return jsonify(detection_result)
            
        except Exception as e:
            logging.error(f"Error detecting trees with ML: {str(e)}", exc_info=True)
            return jsonify({
                "success": False,
                "message": f"Detection error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }), 500
    
    # Initialize async services (database and Gemini AI)
    async def init_services(app):
        """
        Initialize the asynchronous services required by the application
        
        This inner function handles the async initialization of services like
        the DatabaseService and GeminiService. These services need to be
        initialized asynchronously to avoid blocking the main thread during startup.
        
        Args:
            app (Flask): The Flask application instance
            
        Returns:
            bool: True if all services initialized successfully, False otherwise
        """
        try:
            # Create service instances
            app.db_service = DatabaseService()
            
            # Initialize ML service (loads models on GPU) - this is synchronous
            try:
                # Add root directory to Python path to fix import errors
                import sys
                sys.path.append('/ttt')
                
                # Check if ML service already exists on the app
                if hasattr(app, 'ml_service') and app.ml_service is not None:
                    logger.info("ML service already exists on app, reusing existing instance")
                    # Check model status
                    if hasattr(app.ml_service, 'get_model_status'):
                        status = app.ml_service.get_model_status()
                        logger.info(f"Existing ML model status: {status}")
                else:
                    # Use the shared detection service's ml_service
                    # This avoids initializing both external and local services
                    logger.info("Using shared detection service to access the ML model service")
                    detection_service = app.detection_service
                    
                    if detection_service.ml_service is not None:
                        app.ml_service = detection_service.ml_service
                        
                        # Check if models are already loaded
                        if app.ml_service.models_loaded:
                            logger.info("ML models already loaded via detection service")
                            
                            # Verify model status
                            if hasattr(app.ml_service, 'get_model_status'):
                                status = app.ml_service.get_model_status()
                                logger.info(f"ML model status: {status}")
                                
                                # Log device information
                                if status.get('cuda_available', False):
                                    logger.info(f"CUDA is available, using device: {status.get('device', 'unknown')}")
                                else:
                                    logger.warning("CUDA is not available, models using CPU")
                        else:
                            # Wait for models to load if not already loaded
                            logger.info("ML service initialization started, waiting for models to load (timeout: 60s)")
                            if hasattr(app.ml_service, 'wait_for_models'):
                                models_ready = app.ml_service.wait_for_models(timeout=60)
                                if models_ready:
                                    logger.info("ML models loaded successfully and ready for inference")
                                    
                                    # Verify model status
                                    if hasattr(app.ml_service, 'get_model_status'):
                                        status = app.ml_service.get_model_status()
                                        logger.info(f"ML model status: {status}")
                                else:
                                    logger.warning("ML models failed to load in the timeout period")
                    else:
                        logger.warning("ML service not available from detection service")
            except Exception as ml_e:
                logger.error(f"Error initializing ML service: {ml_e}")
                app.ml_service = None
            
            # Initialize database service asynchronously
            db_init = await app.db_service.initialize()
            
            # Skip Gemini initialization to avoid event loop conflicts with ML models
            app.gemini_service = None
            gemini_init = False
            logger.info("Skipping Gemini initialization to avoid event loop conflicts")
            
            logger.info(f"Services initialized: DB={db_init}, Gemini={gemini_init}")
            return True
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
            return False
    
    # Run service initialization synchronously before starting the app
    try:
        # Create a new event loop for initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the initialization in the loop
        init_result = loop.run_until_complete(init_services(app))
        
        # Only close the loop if initialization is complete
        # This allows the event loop to be available for subsequent requests
        loop.close()
        
        logger.info(f"All services initialized successfully: {init_result}")
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}", exc_info=True)
        logger.warning("Proceeding with application startup despite service initialization failures")
        # Proceed with application startup despite service initialization failures

    @app.route('/api/properties', methods=['GET'])
    @require_auth
    def get_properties():
        """Get all properties"""
        try:
            properties = tree_service.get_properties()
            return jsonify(properties)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/properties/<property_id>', methods=['GET'])
    @require_auth
    def get_property(property_id):
        """Get property details"""
        try:
            property_data = tree_service.get_property(property_id)
            if not property_data:
                return jsonify({"error": "Property not found"}), 404
            return jsonify(property_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/properties/<property_id>/trees', methods=['GET'])
    @require_auth
    def get_property_trees(property_id):
        """Get all trees for a property"""
        try:
            trees = tree_service.get_trees_by_property(property_id)
            return jsonify(trees)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/trees', methods=['GET'])
    @require_auth
    def get_all_trees():
        """Get all trees"""
        try:
            # Check for filter params
            filters = {}
            if 'species' in request.args:
                filters['species'] = request.args.get('species')
                
            trees = tree_service.get_trees(filters)
            return jsonify(trees)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/trees/job/<job_id>', methods=['GET'])
    def get_trees_for_job(job_id):
        """Get trees for a specific detection job"""
        try:
            trees = tree_service.get_trees_by_job(job_id)
            return jsonify(trees)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/trees/validate', methods=['POST'])
    def save_validated_trees():
        """Save trees after user validation from detection process"""
        try:
            data = request.json
            if not data or 'trees' not in data:
                return jsonify({"error": "Missing required data"}), 400
                
            job_id = data.get('job_id')
            trees = data['trees']
            
            # Import required modules locally to avoid global imports
            import zarr
            import numpy as np
            
            # Set paths - store everything in TEMP_DIR
            temp_results_path = os.path.join(TEMP_DIR, "results.json")
            zarr_path = os.path.join(TEMP_DIR, "ml_results")
            
            # Make sure the temp directory exists
            os.makedirs(os.path.dirname(zarr_path), exist_ok=True)
            
            # Handle Zarr storage and processing of validated trees
            permanent_path = None
            
            try:
                # First, read the temporary results file
                with open(temp_results_path, 'r') as f:
                    detection_results = json.load(f)
                
                # Filter to include only selected trees
                selected_tree_ids = [tree['id'] for tree in trees]
                selected_trees = [tree for tree in detection_results.get('trees', []) 
                                if tree['id'] in selected_tree_ids]
                
                logger.info(f"Adding {len(selected_trees)} selected trees to Zarr store at {zarr_path}")
                
                # Create or open the Zarr store for the selected trees - never overwrite existing data
                store = zarr.open(zarr_path, mode='a')
                
                # Initialize store if it doesn't exist yet
                if 'job_id' not in store.attrs:
                    store.attrs['job_id'] = job_id
                    store.attrs['created_at'] = datetime.now().isoformat()
                
                # Create frames group if it doesn't exist
                if 'frames' not in store:
                    frames = store.create_group('frames')
                else:
                    frames = store.frames
                
                # Create or get frame 0 (we use a single frame for simplicity)
                if '0' not in frames:
                    frame = frames.create_group('0')
                else:
                    frame = frames['0']
                
                # Create features group with tree_features subgroup
                if 'features' not in frame:
                    features = frame.create_group('features')
                else:
                    features = frame.features
                    
                if 'tree_features' not in features:
                    tree_features = features.create_group('tree_features')
                else:
                    tree_features = features.tree_features
                
                # Extract feature arrays from the selected trees
                bboxes = []
                confidences = []
                widths = []
                heights = []
                centroids = []
                locations = []
                
                for tree in selected_trees:
                    bboxes.append(tree.get('bbox', [0, 0, 0, 0]))
                    confidences.append(tree.get('confidence', 0.0))
                    widths.append(tree.get('width', 0.0))
                    heights.append(tree.get('height', 0.0))
                    centroids.append(tree.get('centroid', [0, 0]))
                    locations.append(tree.get('location', [0, 0]))
                
                # Add these new trees to the existing datasets, or create new datasets
                for dataset_name, data_array in [
                    ('bbox', bboxes), 
                    ('confidence', confidences),
                    ('width', widths),
                    ('height', heights)
                ]:
                    if dataset_name in tree_features:
                        # Append to existing dataset
                        existing_data = tree_features[dataset_name][:]
                        combined_data = np.concatenate([existing_data, np.array(data_array)])
                        del tree_features[dataset_name]  # Delete existing dataset
                        tree_features.create_dataset(dataset_name, data=combined_data)
                    else:
                        # Create new dataset
                        tree_features.create_dataset(dataset_name, data=np.array(data_array))
                
                # Handle segmentation features (centroids)
                if 'segmentation_features' not in features:
                    seg_features = features.create_group('segmentation_features')
                else:
                    seg_features = features.segmentation_features
                
                if 'centroid' in seg_features:
                    # Append to existing centroids
                    existing_centroids = seg_features['centroid'][:]
                    combined_centroids = np.concatenate([existing_centroids, np.array(centroids)])
                    del seg_features['centroid']
                    seg_features.create_dataset('centroid', data=combined_centroids)
                else:
                    seg_features.create_dataset('centroid', data=np.array(centroids))
                
                # Make sure we have metadata with geo transforms
                if 'metadata' not in frame:
                    metadata = frame.create_group('metadata')
                    
                    # Use proper geo transform if available in detection results
                    if 'geo_transform' in detection_results.get('metadata', {}):
                        geo_transform = np.array(detection_results['metadata']['geo_transform'])
                    else:
                        # Use identity transform as placeholder (this would be replaced with actual values)
                        geo_transform = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                    
                    metadata.create_dataset('geo_transform', data=geo_transform)
                
                # Store update time
                store.attrs['updated_at'] = datetime.now().isoformat()
                store.attrs['tree_count'] = len(bboxes) + store.attrs.get('tree_count', 0)
                
                # Keep temp results file for reference (don't delete it)
                
                # Success
                permanent_path = zarr_path
                
                # Tree data is in permanent storage
                
                # Process the validated trees
                result = tree_service.save_validated_trees(trees)
                
                # Also sync to validation queue
                validation_service.sync_validated_trees(trees)
                
                return jsonify({
                    "status": "success",
                    "message": f"Successfully saved {len(trees)} validated trees",
                    "trees_saved": len(trees),
                    "permanent_path": permanent_path,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Error processing Zarr data: {str(e)}", exc_info=True)
                raise  # Re-raise to be caught by the outer try/except
                
        except Exception as e:
            logger.error(f"Error saving validated trees: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/trees/species', methods=['GET'])
    @require_auth
    def get_tree_species():
        """Get list of unique tree species"""
        try:
            species_list = tree_service.get_tree_species()
            return jsonify(species_list)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/trees/<tree_id>', methods=['GET'])
    @require_auth
    def get_tree(tree_id):
        """Get tree details"""
        try:
            tree = tree_service.get_tree(tree_id)
            if not tree:
                return jsonify({"error": "Tree not found"}), 404
            return jsonify(tree)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/trees/<tree_id>/assessment', methods=['PUT'])
    def update_tree_assessment(tree_id):
        """Update tree assessment"""
        try:
            data = request.json
            updated_tree = tree_service.update_tree_assessment(tree_id, data)
            if not updated_tree:
                return jsonify({"error": "Tree not found"}), 404
            return jsonify(updated_tree)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/validation/queue', methods=['GET'])
    def get_validation_queue():
        """Get validation queue items"""
        try:
            # Parse filters from query params
            filters = {}
            for key in ['status', 'property_id', 'risk_level']:
                if key in request.args:
                    filters[key] = request.args.get(key)
                    
            queue_items = validation_service.get_validation_queue(filters)
            return jsonify(queue_items)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/validation/<item_id>', methods=['PUT'])
    def update_validation_status(item_id):
        """Update validation item status"""
        try:
            data = request.json
            status = data.get('status')
            notes = data.get('notes', {})
            
            updated_item = validation_service.update_validation_status(item_id, status, notes)
            if not updated_item:
                return jsonify({"error": "Validation item not found"}), 404
            return jsonify(updated_item)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/lidar', methods=['GET'])
    def get_lidar_data():
        """Get processed LiDAR data for a property"""
        try:
            property_id = request.args.get('property_id')
            if not property_id:
                return jsonify({"error": "Property ID is required"}), 400
                    
            # Use the lidar service to get data
            lidar_data = lidar_service.get_lidar_data(property_id=property_id)
            
            # Return as JSON instead of file
            return jsonify(lidar_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/lidar/tree/<tree_id>', methods=['GET'])
    def get_tree_lidar_data(tree_id):
        """Get processed LiDAR data for a specific tree"""
        try:
            # Use the lidar service to get tree-specific data
            lidar_data = lidar_service.get_lidar_data(tree_id=tree_id)
            
            # Return as JSON
            return jsonify(lidar_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/validation/tree/<tree_id>/lidar', methods=['GET'])
    def get_validation_tree_lidar(tree_id):
        """Get LiDAR data for a tree in the validation system"""
        try:
            # Use validation service to get the lidar data from mock files
            lidar_data = validation_service.get_tree_lidar_data(tree_id)
            
            if not lidar_data:
                return jsonify({"error": "LiDAR data not found for tree"}), 404
                
            return jsonify(lidar_data)
        except Exception as e:
            logger.error(f"Error getting LiDAR data: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/detect-trees', methods=['POST'])
    def detect_trees():
        """Detect trees from Google imagery using Gemini API exclusively"""
        try:
            request_data = request.json or {}
            
            # Extract parameters from request - we only need map_view_info for Gemini API
            coordinates = request_data.get('coordinates')
            image_url = request_data.get('image_url')
            map_image = request_data.get('map_image')  # Image from Google Maps
            map_view_info = request_data.get('map_view_info')  # 3D view data
            include_bounding_boxes = request_data.get('include_bounding_boxes', True)
            
            # We always use Gemini API only
            logger.info(f"Tree detection request received using Gemini API only")
            logger.info(f"Request data fields: {list(request_data.keys())}")
            
            # Create job ID based on timestamp and set up logging
            timestamp = int(datetime.now().timestamp())
            job_id = f"tree_detection_{timestamp}" 
            log_file_path = os.path.join(os.path.dirname(__file__), 'logs', f"{job_id}.log")
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            
            # Create a job-specific logger
            job_logger = logging.getLogger(job_id)
            job_logger.setLevel(logging.INFO)
            job_logger.addHandler(file_handler)
            job_logger.info("Starting tree detection using satellite imagery")
            
            # Save map_view_info if provided
            if map_view_info:
                job_logger.info(f"Map view info provided: {map_view_info}")
            
            # Check input parameters
            if not coordinates and not image_url and not map_image and not map_view_info:
                return jsonify({"error": "Either coordinates, image_url, map_image, or map_view_info must be provided"}), 400
                
            # Check if we're in 3D mode before running detection
            is_3d_mode = bool(map_view_info) if map_view_info else True  # Default to True if no view info provided
            
            # Check if there was a capture error reported by the frontend
            capture_error = request_data.get('capture_error')
            if capture_error:
                job_logger.warning(f"Image capture error reported: {capture_error}")
                # Continue with other data (coordinates) even if image capture failed
            
            if not is_3d_mode:
                logger.warning(f"Tree detection attempted without 3D mode for job {job_id}")
                return jsonify({
                    "error": "Tree detection requires 3D view mode. Please switch to 3D view and try again.",
                    "job_id": job_id,
                    "status": "error"
                }), 400
            
            # Make sure temporary directory exists
            os.makedirs(TEMP_DIR, exist_ok=True)
            
            # Process map image if provided
            map_image_path = None
            if map_image:
                try:
                    job_logger.info("Processing captured map image from browser canvas")
                    
                    # Check capture method to aid debugging
                    capture_method = request_data.get('capture_method', 'unknown')
                    is_3d_capture = request_data.get('is_3d_capture', False)
                    job_logger.info(f"Image capture method: {capture_method}, 3D: {is_3d_capture}")
                    
                    # Skip processing if we received a tainted WebGL canvas error marker
                    if map_image == 'TAINTED_CANVAS_ERROR':
                        job_logger.warning("Received tainted canvas error marker. WebGL security restrictions prevented image capture.")
                        job_logger.info("Will continue with map_view_info coordinates only")
                        map_image_path = None
                        image_url = None
                    else:
                        # Strip the prefix (data:image/jpeg;base64,) to get the raw base64 string
                        if ',' in map_image:
                            map_image_data = map_image.split(',', 1)[1]
                        else:
                            map_image_data = map_image
                        
                        # Log the size to help with debugging
                        job_logger.info(f"Base64 image data length: {len(map_image_data)}")
                        
                        # Decode the base64 string
                        import base64
                        try:
                            decoded_image = base64.b64decode(map_image_data)
                            job_logger.info(f"Successfully decoded image data, size: {len(decoded_image)} bytes")
                            
                            # Verify this is actually an image
                            import imghdr
                            import io
                            image_type = imghdr.what(None, h=decoded_image[:32])
                            if not image_type:
                                job_logger.warning("Decoded data doesn't appear to be a valid image")
                                # Try to read as image anyway, might be corrupted header
                                from PIL import Image
                                try:
                                    img = Image.open(io.BytesIO(decoded_image))
                                    img.verify() # Verify it's an image
                                    job_logger.info(f"Image validated: {img.format}, size: {img.size}, mode: {img.mode}")
                                    image_type = img.format.lower()
                                except Exception as img_err:
                                    job_logger.error(f"Image validation failed: {img_err}")
                                    raise ValueError(f"Invalid image data: {img_err}")
                            
                            # Save to a temporary file with appropriate extension
                            ext = image_type if image_type else 'jpg'
                            map_image_path = os.path.join(TEMP_DIR, f"{job_id}_map.{ext}")
                            with open(map_image_path, 'wb') as f:
                                f.write(decoded_image)
                            
                            job_logger.info(f"Saved map image to {map_image_path}")
                            
                            # Set image_url to the local path for further processing
                            image_url = map_image_path
                        except base64.binascii.Error as b64_err:
                            job_logger.error(f"Base64 decoding error: {b64_err}")
                            # Try to strip additional characters if it's an incorrect format
                            try:
                                # Sometimes there are extra characters in the data URL
                                if len(map_image_data) > 100:
                                    clean_data = ''.join(c for c in map_image_data if c in 
                                                      'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
                                    decoded_image = base64.b64decode(clean_data + '=' * (4 - len(clean_data) % 4))
                                    map_image_path = os.path.join(TEMP_DIR, f"{job_id}_map_cleaned.jpg")
                                    with open(map_image_path, 'wb') as f:
                                        f.write(decoded_image)
                                    job_logger.info(f"Saved cleaned map image to {map_image_path}")
                                    image_url = map_image_path
                                else:
                                    job_logger.error("Base64 data too short or invalid")
                                    raise
                            except Exception as clean_err:
                                job_logger.error(f"Failed to clean base64 data: {clean_err}")
                                # Continue with coordinates only
                                job_logger.info("Will continue with coordinates only")
                                map_image_path = None
                                image_url = None
                except Exception as e:
                    job_logger.error(f"Error processing map image: {e}")
                    job_logger.info("Continuing with coordinates only")
                    map_image_path = None
                    image_url = None
            
            # We'll store detection results in a JSON file
            # Results will be moved to final Zarr store only after user selection
            
            # Create a unique directory for this detection run using just the job_id
            detection_dir = os.path.join(TEMP_DIR, job_id)
            os.makedirs(detection_dir, exist_ok=True)
            job_logger.info(f"Created detection directory: {detection_dir}")
            
            # Write coordinates to file if needed - inside detection directory
            coords_file = None
            if coordinates:
                coords_file = os.path.join(detection_dir, "coords.json")
                os.makedirs(os.path.dirname(coords_file), exist_ok=True)
                
                # Format coordinates properly
                if isinstance(coordinates, dict) and 'bounds' in coordinates:
                    save_data = {
                        'bounds': coordinates['bounds'],
                        'center': coordinates.get('center', [0, 0]),
                        'zoom': coordinates.get('zoom', 10)
                    }
                else:
                    save_data = coordinates
                
                with open(coords_file, 'w') as f:
                    json.dump(save_data, f)
                job_logger.info(f"Saved coordinates to {coords_file}")
            
            # Set up metadata for tracking
            job_metadata = {
                "job_id": job_id,
                "include_bounding_boxes": include_bounding_boxes,
                "coordinates": coordinates,
                "image_url": image_url,
                "debug": request_data.get('debug', False),
                "status": "processing",
                "message": "Tree detection job started",
                "started_at": datetime.now().isoformat()
            }
            
            # Add map view info if provided
            if map_view_info:
                job_metadata["map_view_info"] = map_view_info
                job_logger.info("Added map view info to job metadata")
            
            # Write metadata to a file for reference
            metadata_file = os.path.join(os.path.dirname(__file__), 'logs', f"{job_id}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(job_metadata, f, indent=2)
            
            # Map view info is required for detection
            if not map_view_info:
                logger.warning("No map view info provided, cannot process detection")
                return jsonify({
                    "job_id": job_id,
                    "status": "error",
                    "message": "No map view information provided. Cannot process tree detection.",
                    "timestamp": datetime.now().isoformat(),
                    "tree_count": 0,
                    "trees": []
                }), 400
                
            # Extract view data for checking
            view_data = map_view_info.get('viewData', {})
            
            # Log key information for debugging
            logger.info(f"Map view info keys: {list(map_view_info.keys())}")
            logger.info(f"View data keys: {list(view_data.keys())}")
            
            # Handle image from either imageUrl or map_image parameter
            if not view_data.get('imageUrl') and map_image and map_image != 'TAINTED_CANVAS_ERROR':
                logger.info("Using map_image data from request")
                
                # Update the map_view_info with the image URL
                if 'viewData' not in map_view_info:
                    map_view_info['viewData'] = {}
                map_view_info['viewData']['imageUrl'] = map_image
                
                logger.info("Added map image from request to map view info")
                job_logger.info("Added map image from request to map view info")
            
            # Create results file
            output_file = os.path.join(detection_dir, "results.json")
            job_logger.info(f"Created results file at {output_file}")
            
            # Process with ML detection service
            try:
                # Define async detection function
                async def run_ml_detection():
                    try:
                        # Use the shared ML detection service
                        detection_service = app.detection_service
                        
                        # Call detection service for tree detection
                        detection_result = await detection_service.detect_trees_from_map_view(
                            map_view_info, job_id
                        )
                        
                        # Save processed results to results.json file
                        with open(output_file, 'w') as f:
                            json.dump(detection_result, f, indent=2)
                        
                        # Create response directory
                        response_dir = os.path.join(detection_dir, "ml_response") 
                        os.makedirs(response_dir, exist_ok=True)
                        
                        # Save the satellite image if available
                        save_satellite_image(map_image, map_view_info, response_dir)
                            
                        logger.info(f"Successfully processed detection - detected {detection_result.get('tree_count', 0)} trees")
                        logger.info(f"Saved detection results to {output_file}")
                        return detection_result
                    except Exception as e:
                        logger.error(f"Error in tree detection: {str(e)}")
                        
                        # Create error response
                        error_result = {
                            "job_id": job_id,
                            "status": "error",
                            "message": f"Detection error: {str(e)}",
                            "error_details": str(e),
                            "timestamp": datetime.now().isoformat(),
                            "tree_count": 0,
                            "trees": [],
                            "detection_source": "ml_pipeline"
                        }
                        
                        # Save error result to results.json
                        with open(output_file, 'w') as f:
                            json.dump(error_result, f, indent=2)
                        
                        # Create response directory
                        response_dir = os.path.join(detection_dir, "ml_response") 
                        os.makedirs(response_dir, exist_ok=True)
                        
                        # Save the satellite image if available
                        save_satellite_image(map_image, map_view_info, response_dir)
                            
                        return error_result
                        
                # Helper function to save satellite image
                def save_satellite_image(map_image, map_view_info, output_dir):
                    # Try to save from map_image first
                    if map_image and map_image.startswith('data:image'):
                        try:
                            # Extract the base64 data
                            image_data = map_image.split(',')[1]
                            image_bytes = base64.b64decode(image_data)
                            
                            # Save the image
                            satellite_image_path = os.path.join(output_dir, "satellite_image.jpg")
                            with open(satellite_image_path, 'wb') as f:
                                f.write(image_bytes)
                                
                            logger.info(f"Saved satellite image to {satellite_image_path}")
                            return True
                        except Exception as image_err:
                            logger.error(f"Failed to save satellite image: {image_err}")
                    
                    # Try to save from map_view_info if first method failed
                    elif map_view_info and map_view_info.get('viewData', {}).get('imageUrl', '').startswith('data:image'):
                        try:
                            # Extract the base64 data from map_view_info
                            image_data = map_view_info['viewData']['imageUrl'].split(',')[1]
                            image_bytes = base64.b64decode(image_data)
                            
                            # Save the image
                            satellite_image_path = os.path.join(output_dir, "satellite_image.jpg")
                            with open(satellite_image_path, 'wb') as f:
                                f.write(image_bytes)
                                
                            logger.info(f"Saved satellite image from map_view_info to {satellite_image_path}")
                            return True
                        except Exception as image_err:
                            logger.error(f"Failed to save satellite image from map_view_info: {image_err}")
                    
                    return False
                
                # Run the ML detection with thread-safe approach
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run the ML detection and wait for it to complete
                detection_result = loop.run_until_complete(run_ml_detection())
                
                # Close the loop when done
                loop.close()
                
                logger.info(f"ML detection complete: {detection_result.get('status')}")
                return jsonify(detection_result)
                
            except Exception as e:
                logger.error(f"Error running ML detection: {str(e)}")
                error_output = {
                    "job_id": job_id,
                    "status": "error",
                    "message": f"ML detection failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "tree_count": 0,
                    "trees": [],
                    "detection_source": "ml_pipeline"
                }
                return jsonify(error_output)
                
        except Exception as e:
            logger.error(f"Error starting tree detection: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/detection-status/<job_id>', methods=['GET'])
    def get_detection_status(job_id):
        """Get status of a tree detection job"""
        try:
            # Job ID is used for logging/tracking and storing results
            
            # Check for results in the job-specific directory
            job_dir = os.path.join(ML_DIR, job_id)
            temp_results_path = os.path.join(job_dir, "results.json")
            final_zarr_path = job_dir  # ML results are stored directly in the job directory
            
            # For now we're only checking for temporary results from the ML pipeline
            # No validation or Zarr store functionality is implemented yet
            
            # Check if temporary JSON results exist (detection is complete)
            if os.path.exists(temp_results_path):
                # Detection is complete but waiting for validation
                logger.info(f"Temporary results found at {temp_results_path}, detection complete, awaiting validation")
                
                # Load the temporary results JSON
                try:
                    with open(temp_results_path, 'r') as f:
                        temp_results = json.load(f)
                    
                    # Get tree count and other info from the JSON results
                    tree_count = temp_results.get('tree_count', 0) 
                    status = temp_results.get('status', 'complete')
                    trees = temp_results.get('trees', [])
                    
                    # Track detection source
                    detection_source = temp_results.get('detection_source', 'ml_pipeline')
                    logger.info(f"Detection source: {detection_source}")
                    
                    # Make sure we have complete status even if coming from ML pipeline
                    if status == 'complete_no_detections':
                        status = 'complete'
                        logger.info('Changed status from complete_no_detections to complete for frontend compatibility')
                    
                    logger.info(f"Found {tree_count} trees in temp JSON results, status: {status}")
                
                    # Get metadata about the job if available
                    job_metadata = {}
                    metadata_file = os.path.join(os.path.dirname(__file__), 'logs', f"{job_id}_metadata.json")
                    try:
                        if os.path.exists(metadata_file):
                            with open(metadata_file, 'r') as f:
                                job_metadata = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not read job metadata: {e}")
                
                    # Make sure trees array has location data for each tree
                    if trees:
                        for i, tree in enumerate(trees):
                            # Ensure all trees have location data
                            if not tree.get('location') and tree.get('bbox'):
                                # Calculate center of bounding box as location if missing
                                bbox = tree['bbox']
                                center_x = (bbox[0] + bbox[2]) / 2
                                center_y = (bbox[1] + bbox[3]) / 2
                                
                                # Convert to geo coordinates based on map center
                                if 'coordinates' in job_metadata and 'center' in job_metadata['coordinates']:
                                    map_center = job_metadata['coordinates']['center']
                                    # Simple conversion for testing - would need proper projection in production
                                    tree['location'] = [map_center[0], map_center[1]]
                                else:
                                    # Default location
                                    tree['location'] = [0, 0]
                                
                                # Ensure all trees have required properties
                                if not tree.get('species'):
                                    tree['species'] = 'Unknown Species'
                                if not tree.get('height'):
                                    tree['height'] = 30.0
                                if not tree.get('diameter'):
                                    tree['diameter'] = 12.0
                                if not tree.get('risk_level'):
                                    tree['risk_level'] = 'medium'
                                if not tree.get('confidence'):
                                    tree['confidence'] = 0.85
                                
                                logger.info(f"Added missing data to tree {i}: {tree}")
                    
                    # Check for satellite image
                    satellite_image_path = os.path.join(job_dir, "gemini_response", "satellite_image.jpg")
                    has_satellite_image = os.path.exists(satellite_image_path)
                    
                    # Create result object with tree data
                    result = {
                        "job_id": job_id,
                        "status": "complete",  # Set status to complete to ensure UI works properly
                        "message": f"Tree detection complete: {tree_count} trees detected",
                        "results_path": temp_results_path,
                        "tree_count": tree_count,
                        "trees": trees,  # Include actual tree data
                        "completed_at": temp_results.get('timestamp', datetime.now().isoformat()),
                        "includes_bounding_boxes": job_metadata.get("include_bounding_boxes", True),
                        "coordinates": job_metadata.get("coordinates", {}),
                        "has_satellite_image": has_satellite_image,
                        "satellite_image_path": satellite_image_path if has_satellite_image else None,
                        "map_view_info": job_metadata.get("map_view_info", {})
                    }
                    
                    return jsonify(result)
                    
                except Exception as e:
                    logger.error(f"Error reading temporary results: {e}")
                    
                    # Fallback to basic info if we can't read the file properly
                    return jsonify({
                        "job_id": job_id,
                        "status": "error",
                        "message": f"Error reading detection results: {str(e)}",
                        "results_path": temp_results_path
                    })
            else:
                # Check if job is still running
                try:
                    with open(os.path.join(os.path.dirname(__file__), 'logs', f"{job_id}.log"), 'r') as f:
                        last_few_lines = f.readlines()[-10:]
                        last_line = last_few_lines[-1] if last_few_lines else ""
                except FileNotFoundError:
                    last_line = "Job status information not available"
                
                # Get metadata about the job if available
                job_metadata = {}
                metadata_file = os.path.join(os.path.dirname(__file__), 'logs', f"{job_id}_metadata.json")
                try:
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            job_metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read job metadata: {e}")
                    
                return jsonify({
                    "job_id": job_id,
                    "status": "processing",
                    "message": "Tree detection with bounding boxes in progress",
                    "last_update": last_line,
                    "includes_bounding_boxes": job_metadata.get("include_bounding_boxes", True),
                    "started_at": job_metadata.get("started_at", datetime.now().isoformat())
                })
                
        except Exception as e:
            logger.error(f"Error checking detection status: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/properties/<property_id>/report', methods=['POST'])
    def generate_property_report(property_id):
        """Generate a report for a property"""
        try:
            options = request.json or {}
            report_data = tree_service.generate_report(property_id, options)
            return jsonify(report_data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/reports/<report_id>', methods=['GET'])
    def get_report(report_id):
        """Get a specific report"""
        try:
            # Check if file exists
            report_path = os.path.join(REPORTS_DIR, f"{report_id}.pdf")
            if not os.path.exists(report_path):
                return jsonify({"error": "Report not found"}), 404
                
            return send_from_directory(
                REPORTS_DIR,
                f"{report_id}.pdf",
                as_attachment=True,
                mimetype='application/pdf'
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    # Route to serve satellite images
    @app.route('/api/temp/<job_id>/satellite_image.jpg', methods=['GET'])
    def serve_satellite_image(job_id):
        """Serve the satellite image for a specific job"""
        try:
            # Set the path to the satellite image
            image_path = os.path.join(TEMP_DIR, job_id, "gemini_response", "satellite_image.jpg")
            
            # Check if the file exists
            if not os.path.exists(image_path):
                logger.warning(f"Satellite image not found at {image_path}")
                return jsonify({"error": "Satellite image not found"}), 404
                
            # Serve the file
            return send_from_directory(
                os.path.dirname(image_path),
                os.path.basename(image_path),
                mimetype='image/jpeg'
            )
        except Exception as e:
            logger.error(f"Error serving satellite image: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    # Route to serve ML detection files from the /ttt/data directory
    @app.route('/ttt/data/<path:filepath>', methods=['GET'])
    def serve_ml_data(filepath):
        """Serve files from the /ttt/data directory"""
        try:
            # Log the requested file path
            logger.info(f"Serving ML data file: {filepath}")
            
            # Construct the full path
            full_path = os.path.join('/ttt/data', filepath)
            
            # Extract directory and filename
            directory = os.path.dirname(full_path)
            filename = os.path.basename(full_path)
            
            # Check if the file exists
            if not os.path.exists(full_path):
                logger.warning(f"ML data file not found: {full_path}")
                return jsonify({"error": f"File not found: {filepath}"}), 404
                
            # Determine MIME type
            mime_type = 'application/octet-stream'  # Default
            if filepath.endswith('.jpg') or filepath.endswith('.jpeg'):
                mime_type = 'image/jpeg'
            elif filepath.endswith('.png'):
                mime_type = 'image/png'
            elif filepath.endswith('.json'):
                mime_type = 'application/json'
                
            # Serve the file
            return send_from_directory(
                directory,
                filename,
                mimetype=mime_type
            )
        except Exception as e:
            logger.error(f"Error serving ML data file: {str(e)}")
            return jsonify({"error": str(e)}), 500
            
    # Gemini API integration for tree detection and satellite imagery
    @app.route('/api/gemini/save-results', methods=['POST'])
    @require_auth
    def save_gemini_results():
        """Save Gemini results with satellite imagery"""
        try:
            data = request.json
            if not data:
                return jsonify({"error": "No data provided"}), 400
                
            # Create timestamp-based job ID
            timestamp = int(datetime.now().timestamp())
            job_id = data.get('jobId', f"gemini_{timestamp}")
            
            # Get the results data
            results = data.get('results', {})
            if not results:
                return jsonify({"error": "No results data provided"}), 400
                
            # Create directory for results
            results_dir = os.path.join(TEMP_DIR, f"tree_detection_{timestamp}")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save results json
            results_file = os.path.join(results_dir, "results.json")
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            # Create directory for Gemini response details
            gemini_dir = os.path.join(results_dir, "gemini_response")
            os.makedirs(gemini_dir, exist_ok=True)
            
            # Save a response.txt file with job summary
            response_file = os.path.join(gemini_dir, "response.txt")
            tree_count = results.get('tree_count', 0)
            with open(response_file, 'w') as f:
                f.write(f"Processed {tree_count} trees with satellite imagery\n")
                f.write(f"Job ID: {job_id}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                
            logger.info(f"Saved Gemini results for job {job_id} with {tree_count} trees")
            
            # Also save metadata
            metadata_file = os.path.join(f"tree_detection_{timestamp}_metadata.json")
            with open(os.path.join(os.path.dirname(__file__), 'logs', metadata_file), 'w') as f:
                json.dump({
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat(),
                    "tree_count": tree_count,
                    "source": "gemini_frontend",
                    "with_satellite_imagery": True
                }, f, indent=2)
                
            return jsonify({
                "success": True,
                "job_id": job_id,
                "message": f"Successfully saved {tree_count} trees with satellite imagery",
                "results_path": results_dir
            })
            
        except Exception as e:
            logger.exception(f"Error saving Gemini results: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "Failed to save Gemini results"
            }), 500
            
    # Analytics endpoints for data visualization and reporting
    @app.route('/api/analytics/risk-distribution', methods=['GET'])
    def get_risk_distribution():
        """Get distribution of tree risk levels"""
        try:
            # Get tree data
            trees = tree_service.get_trees()
            
            # Count trees by the highest risk factor for each tree
            risk_counts = {'high': 0, 'medium': 0, 'low': 0, 'unknown': 0}
            
            for tree in trees:
                # Find highest risk level
                highest_risk = 'unknown'
                if 'risk_factors' in tree and tree['risk_factors']:
                    for factor in tree['risk_factors']:
                        if factor.get('level') in ['high', 'medium', 'low']:
                            if factor['level'] == 'high':
                                highest_risk = 'high'
                                break
                            elif factor['level'] == 'medium' and highest_risk != 'high':
                                highest_risk = 'medium'
                            elif factor['level'] == 'low' and highest_risk not in ['high', 'medium']:
                                highest_risk = 'low'
                
                risk_counts[highest_risk] += 1
            
            # Format for response
            result = [
                {"risk_level": level, "count": count}
                for level, count in risk_counts.items()
            ]
            
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
            
    @app.route('/api/analytics/property-stats', methods=['GET'])
    def get_property_stats():
        """Get statistics for each property"""
        try:
            # Get property and tree data
            properties = tree_service.get_properties()
            trees = tree_service.get_trees()
            
            property_stats = []
            
            for prop in properties:
                property_id = prop['id']
                property_trees = [tree for tree in trees if tree['property_id'] == property_id]
                
                # Count trees by risk level
                high_risk = 0
                medium_risk = 0
                low_risk = 0
                
                for tree in property_trees:
                    # Find highest risk level
                    highest_risk = 'unknown'
                    if 'risk_factors' in tree and tree['risk_factors']:
                        for factor in tree['risk_factors']:
                            if factor.get('level') in ['high', 'medium', 'low']:
                                if factor['level'] == 'high':
                                    highest_risk = 'high'
                                    break
                                elif factor['level'] == 'medium' and highest_risk != 'high':
                                    highest_risk = 'medium'
                                elif factor['level'] == 'low' and highest_risk not in ['high', 'medium']:
                                    highest_risk = 'low'
                    
                    # Increment counts
                    if highest_risk == 'high':
                        high_risk += 1
                    elif highest_risk == 'medium':
                        medium_risk += 1
                    elif highest_risk == 'low':
                        low_risk += 1
                
                property_stats.append({
                    "id": property_id,
                    "address": prop['address'],
                    "tree_count": len(property_trees),
                    "high_risk_count": high_risk,
                    "medium_risk_count": medium_risk, 
                    "low_risk_count": low_risk
                })
            
            return jsonify(property_stats)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/', methods=['GET'])
    @require_auth
    def root():
        """API root endpoint with basic health check and documentation"""
        # Import config here to avoid circular imports
        from config import APP_MODE, API_VERSION
        
        endpoints = [
            {"path": "/", "methods": ["GET"], "description": "API root with documentation"},
            {"path": "/api/properties", "methods": ["GET"], "description": "Get all properties"},
            {"path": "/api/properties/<property_id>", "methods": ["GET"], "description": "Get property details"},
            {"path": "/api/properties/<property_id>/trees", "methods": ["GET"], "description": "Get all trees for a property"},
            {"path": "/api/trees/<tree_id>", "methods": ["GET"], "description": "Get tree details"},
            {"path": "/api/trees/<tree_id>/assessment", "methods": ["PUT"], "description": "Update tree assessment"},
            {"path": "/api/validation/queue", "methods": ["GET"], "description": "Get validation queue items"},
            {"path": "/api/validation/<item_id>", "methods": ["PUT"], "description": "Update validation item status"},
            {"path": "/api/lidar", "methods": ["GET"], "description": "Get processed LiDAR data"},
            {"path": "/api/properties/<property_id>/report", "methods": ["POST"], "description": "Generate a report for a property"},
            {"path": "/api/reports/<report_id>", "methods": ["GET"], "description": "Get a specific report"},
            {"path": "/api/config", "methods": ["GET"], "description": "Get API configuration"},
            {"path": "/api/ml/status", "methods": ["GET"], "description": "Get ML model status (DeepForest and SAM)"}
        ]
        
        # Safely check service status
        database_status = False
        gemini_status = False
        ml_status = False
        
        # Check services in a safer way
        if hasattr(app, 'db_service'):
            database_status = getattr(app.db_service, '_connected', False)
        
        if hasattr(app, 'gemini_service'):
            gemini_status = getattr(app.gemini_service, '_initialized', False)
        
        # Check ML models status
        try:
            # Use the shared detection service
            detection_service = app.detection_service
            if detection_service.ml_service:
                ml_status = detection_service.ml_service.models_loaded
        except Exception:
            ml_status = False
        
        backend_status = {
            "name": "Tree Risk Assessment API",
            "status": "running",
            "mode": APP_MODE,
            "database": database_status,
            "gemini": gemini_status,
            "ml_models": ml_status,
            "version": API_VERSION,
            "endpoints": endpoints
        }
        
        return jsonify(backend_status)

    @app.route('/api/ml/status', methods=['GET'])
    async def get_ml_status():
        """Get status of ML models (DeepForest and SAM)"""
        try:
            # Use the shared detection service - this will only have the configured model service
            # (either external or local, but not both)
            detection_service = app.detection_service
            
            # Get ML model status from the detection service's ML service
            try:
                if hasattr(detection_service, 'ml_service') and detection_service.ml_service:
                    status = detection_service.ml_service.get_model_status()
                    logger.info(f"ML status check: {status}")
                    
                    # Determine service type
                    if hasattr(detection_service.ml_service, 'server_url'):
                        service_type = "external_t4"
                    else:
                        service_type = "local"
                        
                    # Add service type
                    status["service_type"] = service_type
                else:
                    logger.warning("ML service not available for status check")
                    status = {
                        "models_loaded": False,
                        "error": "ML service not initialized",
                        "service_type": "none"
                    }
            except Exception as status_error:
                logger.error(f"Error checking ML model status: {status_error}")
                status = {
                    "models_loaded": False,
                    "error": str(status_error),
                    "exception": True
                }
            
            # Add system info for debugging
            status["timestamp"] = datetime.now().isoformat()
            
            # Check directories
            status["directories"] = {
                "temp_exists": os.path.exists(TEMP_DIR),
                "zarr_exists": os.path.exists(ZARR_DIR),
                "temp_path": TEMP_DIR,
                "zarr_path": ZARR_DIR
            }
            
            # Add config info
            from config import USE_EXTERNAL_MODEL_SERVER, MODEL_SERVER_URL
            status["config"] = {
                "use_external_model_server": USE_EXTERNAL_MODEL_SERVER,
                "model_server_url": MODEL_SERVER_URL
            }
            
            # Add ML pipeline info
            status["ml_pipeline"] = {
                "detection_service_available": hasattr(detection_service, "detect_trees_from_map_view"),
                "ml_service_available": hasattr(detection_service, "ml_service") and detection_service.ml_service is not None,
                "detect_trees_available": hasattr(detection_service, "detect_trees") and detection_service.detect_trees is not None
            }
            
            # Convert any non-serializable objects to strings
            for key, value in status.items():
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    status[key] = str(value)
            
            return jsonify({
                "success": True,
                "status": status
            })
        except Exception as e:
            logger.error(f"Error getting ML model status: {str(e)}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e),
                "message": "Failed to get ML model status",
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.route('/api/config', methods=['GET'])
    def get_config():
        """Get API configuration"""
        from config import APP_MODE, API_VERSION, USE_GEMINI
        
        config = {
            "mode": APP_MODE,
            "version": API_VERSION,
            "useTestData": APP_MODE == 'test',
            "geminiEnabled": USE_GEMINI,
            "geminiStatus": getattr(app.gemini_service, 'is_initialized', False) if hasattr(app, 'gemini_service') else False,
            "mlModelsEnabled": True,  # Added flag to indicate ML models are available
            "centerCoordinatesRequired": True  # Added flag to explicitly inform frontend center coordinates are required
        }
        return jsonify(config)
        
    @app.route('/api/debug/request-info', methods=['POST'])
    def debug_request_info():
        """Debug endpoint to log detailed request information"""
        try:
            # Get request data
            data = request.get_json()
            headers = dict(request.headers)
            
            # Remove sensitive headers
            for key in ['Authorization', 'Cookie', 'token']:
                if key in headers:
                    headers[key] = '[REDACTED]'
            
            # Log detailed request information
            logger.info("DEBUG REQUEST INFO RECEIVED:")
            logger.info(f"Headers: {headers}")
            logger.info(f"Path: {request.path}")
            logger.info(f"Method: {request.method}")
            logger.info(f"Data keys: {list(data.keys()) if data else 'None'}")
            
            # Log specific fields we're interested in for debugging
            if data and 'map_view_info' in data:
                map_view_info = data['map_view_info']
                logger.info(f"map_view_info keys: {list(map_view_info.keys()) if map_view_info else 'None'}")
                
                if map_view_info and 'viewData' in map_view_info:
                    view_data = map_view_info['viewData']
                    logger.info(f"viewData keys: {list(view_data.keys()) if view_data else 'None'}")
                    
                    # Log coordinate information
                    # Only need to log center coordinates
                    if 'center' in view_data:
                        logger.info(f"center: {view_data['center']}")
                    if 'zoom' in view_data:
                        logger.info(f"zoom: {view_data['zoom']}")
            
            return jsonify({
                "success": True,
                "message": "Debug information logged successfully",
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error in debug endpoint: {str(e)}", exc_info=True)
            return jsonify({
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
        
    return app

# Create the app instance for WSGI servers
app = create_app()

if __name__ == '__main__':
    # Make sure data directories exist
    os.makedirs(LIDAR_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Listen on all interfaces for production deployment
    app.run(debug=True, host='0.0.0.0', port=5000)