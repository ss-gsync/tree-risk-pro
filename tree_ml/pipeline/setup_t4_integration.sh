#!/bin/bash
# T4 Model Server Integration Setup Script
# =======================================
# This script updates the configuration to connect to the T4 model server

set -e

echo "============================================================"
echo "T4 Model Server Integration Setup"
echo "============================================================"

# Variables
DASHBOARD_DIR="/ttt/tree_ml/dashboard"
BACKEND_CONFIG_FILE="$DASHBOARD_DIR/backend/config.py"
MODEL_SERVER_URL=${1:-"http://t4-model-server:8000"}

# Check if files exist
if [ ! -f "$BACKEND_CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $BACKEND_CONFIG_FILE"
    exit 1
fi

echo "Setting up T4 model server integration..."

# Update config.py to use external model server
echo "Updating backend configuration to use T4 model server at $MODEL_SERVER_URL"

# Check if the config already has the MODEL_SERVER_URL variable
if grep -q "MODEL_SERVER_URL" "$BACKEND_CONFIG_FILE"; then
    # Update existing configuration
    sed -i "s|MODEL_SERVER_URL = .*|MODEL_SERVER_URL = \"$MODEL_SERVER_URL\"|g" "$BACKEND_CONFIG_FILE"
    echo "Updated existing MODEL_SERVER_URL configuration"
else
    # Add new configuration for external model server
    sed -i "/API_VERSION = .*/a \\\n# ML Service configuration\\n# Set to True to use the external T4 model server instead of local models\\nUSE_EXTERNAL_MODEL_SERVER = os.environ.get('USE_EXTERNAL_MODEL_SERVER', 'True').lower() in ('true', '1', 't')\\n# URL for the external T4 model server\\nMODEL_SERVER_URL = os.environ.get('MODEL_SERVER_URL', '$MODEL_SERVER_URL')" "$BACKEND_CONFIG_FILE"
    echo "Added T4 model server configuration to config.py"
fi

# Create an environment file to easily configure the model server URL
ENV_FILE="$DASHBOARD_DIR/.env"
echo "Creating environment file at $ENV_FILE"

# Check if .env exists and has MODEL_SERVER_URL variable
if [ -f "$ENV_FILE" ] && grep -q "MODEL_SERVER_URL" "$ENV_FILE"; then
    # Update existing configuration
    sed -i "s|MODEL_SERVER_URL=.*|MODEL_SERVER_URL=$MODEL_SERVER_URL|g" "$ENV_FILE"
    echo "Updated existing MODEL_SERVER_URL in .env file"
else
    # Add to or create .env file
    echo "MODEL_SERVER_URL=$MODEL_SERVER_URL" >> "$ENV_FILE"
    echo "USE_EXTERNAL_MODEL_SERVER=true" >> "$ENV_FILE"
    echo "Added T4 model server configuration to .env file"
fi

echo "============================================================"
echo "T4 integration setup complete!"
echo "Model server URL: $MODEL_SERVER_URL"
echo "Environment variables set in: $ENV_FILE"
echo "Configuration updated in: $BACKEND_CONFIG_FILE"
echo "============================================================"
echo "To use the T4 model server:"
echo "1. Ensure the T4 instance is running the model server"
echo "2. Make sure the model server URL is correct and accessible"
echo "3. Restart the dashboard backend to apply changes"
echo "============================================================"

exit 0