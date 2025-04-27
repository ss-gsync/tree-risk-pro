# backend/config.py
import os

# Application mode
# 'production' - Default mode using ML pipeline with Zarr store
APP_MODE = 'production'

# Data paths
BASE_DIR = '/ttt'
DATA_DIR = os.path.join(BASE_DIR, 'data')
LIDAR_DIR = os.path.join(DATA_DIR, 'lidar')
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')
ZARR_DIR = os.path.join(DATA_DIR, 'zarr')
TEMP_DIR = os.path.join(DATA_DIR, 'temp')  # Path for temporary detection results
MOCK_DIR = os.path.join(os.path.dirname(__file__), 'mock_data')  # Path to mock data

# Ensure data directories exist
os.makedirs(LIDAR_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
# os.makedirs(ZARR_DIR, exist_ok=True)  # Don't create ZARR_DIR automatically
os.makedirs(TEMP_DIR, exist_ok=True)

# Debug mode
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

# API Configuration
API_VERSION = '0.2.0'
DEFAULT_RESPONSE_LIMIT = 100

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
GEMINI_API_KEY = get_api_key('GEMINI_API_KEY', 'VITE_')
GEMINI_API_URL = os.environ.get('GEMINI_API_URL', 'https://generativelanguage.googleapis.com/v1')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')  # The most current stable model for image and text

# Gemini integration configuration
# ALWAYS use Gemini for tree detection without ML pipeline fallback
USE_GEMINI = True  # Force Gemini integration as the ONLY detection method
USE_ML_PIPELINE = False  # Disable ML pipeline completely