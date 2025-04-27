"""
Authentication module for the Tree Risk Dash

Provides basic authentication middleware for Flask using HTTP Basic Auth.
Uses environment variables for credentials to avoid hardcoding.
"""

from functools import wraps
from flask import request, Response
import os
import hashlib
import logging

# Get logger
logger = logging.getLogger(__name__)

# Import APP_MODE from config
from config import APP_MODE

# Default credentials (should be overridden by environment variables in production)
DEFAULT_USERNAME = "TestAdmin"
DEFAULT_PASSWORD = "trp345!"

# For production, explicitly require environment variables to be set
if APP_MODE == 'production':
    if "DASHBOARD_USERNAME" not in os.environ or "DASHBOARD_PASSWORD" not in os.environ:
        logger.warning("WARNING: Running in production without explicit credentials set in environment!")
        logger.warning("Using default credentials: TestAdmin / trp345!")
        logger.warning("Set DASHBOARD_USERNAME and DASHBOARD_PASSWORD environment variables for security.")

# Load credentials from environment variables or use defaults
USERNAME = os.environ.get("DASHBOARD_USERNAME", DEFAULT_USERNAME)
PASSWORD = os.environ.get("DASHBOARD_PASSWORD", DEFAULT_PASSWORD)

# Store password hash instead of plaintext
PASSWORD_HASH = hashlib.sha256(PASSWORD.encode()).hexdigest()

def check_auth(username, password):
    """Check if a username/password combination is valid."""
    # Hash the provided password for comparison
    provided_password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Compare username and password hash
    is_valid = username == USERNAME and provided_password_hash == PASSWORD_HASH
    
    if not is_valid:
        logger.warning(f"Failed login attempt for username: {username}")
    
    return is_valid

def authenticate():
    """Send a 401 response that enables basic auth."""
    return Response(
        'Authentication required to access the Tree Risk Dash.\n'
        'Please provide valid credentials.',
        401,
        {'WWW-Authenticate': 'Basic realm="Tree Risk Dash"'}
    )

def require_auth(f):
    """Decorator function to require HTTP Basic Authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        
        # Always enforce authentication in production mode
        if APP_MODE == 'production':
            # Check if Authorization header is present and valid
            if not auth or not check_auth(auth.username, auth.password):
                return authenticate()
        else:
            # In development/test mode, check SKIP_AUTH environment variable
            # Skip authentication if explicitly configured via environment variable
            # Authentication is disabled by default in development environments
            if os.environ.get("SKIP_AUTH", "true").lower() == "true":
                return f(*args, **kwargs)
                
            # Check if Authorization header is present and valid
            if not auth or not check_auth(auth.username, auth.password):
                return authenticate()
            
        return f(*args, **kwargs)
    return decorated