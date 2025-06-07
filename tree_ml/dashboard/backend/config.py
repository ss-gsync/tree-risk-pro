# backend/config.py
import os

# Application mode
# 'development' - Development mode with local models for ML_DEV instance
# 'production' - Production mode using ML pipeline with Zarr store and optionally T4 server
APP_MODE = os.environ.get('APP_MODE', 'development')

# Data paths
BASE_DIR = '/ttt'
DATA_DIR = os.path.join(BASE_DIR, 'data')
LIDAR_DIR = os.path.join(DATA_DIR, 'lidar')
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')
ZARR_DIR = os.path.join(DATA_DIR, 'zarr')
TESTS_DIR = os.path.join(DATA_DIR, 'tests')
ML_DIR = os.path.join(DATA_DIR, 'ml')  # Use the ml directory for detection results
MOCK_DIR = os.path.join(os.path.dirname(__file__), 'mock_data')  # Path to mock data

# Ensure data directories exist
os.makedirs(LIDAR_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ZARR_DIR, exist_ok=True)  # Make sure ZARR_DIR is created
os.makedirs(ML_DIR, exist_ok=True)

# Debug mode - Enable debugging in development environment
DEBUG = os.environ.get('DEBUG', 'True' if APP_MODE == 'development' else 'False').lower() in ('true', '1', 't')

# API Configuration
API_VERSION = '0.2.3'
DEFAULT_RESPONSE_LIMIT = 100

# ML Service configuration
# Set to True to use the external T4 model server instead of local models
# Default to False for ML_DEV instance (development environment)
USE_EXTERNAL_MODEL_SERVER = os.environ.get('USE_EXTERNAL_MODEL_SERVER', 'False').lower() in ('true', '1', 't')
# URL for the external T4 model server (only used when USE_EXTERNAL_MODEL_SERVER is True)
MODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', 'http://localhost:8000')

# Generic function to get API keys from environment or .env file
def get_api_key(env_var_name, env_file_prefix=None):
    """Get API key from environment variable or .env file
    
    Args:
        env_var_name: Name of the environment variable
        env_file_prefix: Prefix in .env file (e.g., 'VITE_')
    
    Returns:
        str: API key or empty string if not found
    """
    # First check environment variable
    api_key = os.environ.get(env_var_name, '')
    
    # If not found and env_file_prefix provided, try to read from .env file
    if not api_key and env_file_prefix:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith(f'{env_file_prefix}{env_var_name}='):
                        api_key = line.split('=', 1)[1].strip()
                        break
    
    return api_key

# API configuration
GOOGLE_MAPS_API_KEY = get_api_key('GOOGLE_MAPS_API_KEY', 'VITE_')
# Get Gemini API key using our existing function
GEMINI_API_KEY = get_api_key('GEMINI_API_KEY', 'VITE_')
GEMINI_API_URL = os.environ.get('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1beta')

# IMPORTANT: Always use gemini-2.0-flash (never change to other models)
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')  # The most current stable model for image and text

# Configuration for detection methods
# ML pipeline is the primary detection method, Gemini for analytics only
USE_GEMINI = False  # Gemini only used for analytics, not for detection
USE_ML_PIPELINE = True  # Enable ML pipeline for tree detection