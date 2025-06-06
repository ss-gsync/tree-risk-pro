# T4 GPU Model Server Integration

This document describes the integration of a dedicated T4 GPU instance for tree detection using Grounded-SAM.

## Overview

The T4 integration allows us to offload ML inference to a dedicated GPU server while keeping the dashboard
running on a separate instance. This improves performance and allows for better resource allocation.

## Architecture

```
+---------------------+      HTTP/REST      +---------------------+
|                     |    Detection API    |                     |
|  Dashboard Server   +-------------------->+  T4 Model Server    |
|  (Compute Instance) |                     |  (GPU Instance)     |
|                     |<--------------------+                     |
+---------------------+       Results       +---------------------+
```

## Components

### 1. T4 Model Server

The T4 model server is a dedicated GPU instance running:
- Grounded-SAM model for tree detection
- FastAPI server for ML inference
- Nginx as a reverse proxy

### 2. Dashboard Integration

The dashboard integrates with the T4 model server through:
- External model service client (`external_model_service.py`)
- Configuration settings (`USE_EXTERNAL_MODEL_SERVER` and `MODEL_SERVER_URL`)
- ML service selection logic in `get_model_service()`

## Setup Instructions

### T4 Server Setup

1. **Deploy the T4 Server**
   ```
   # SSH into the T4 instance
   ssh user@t4-instance
   
   # Clone the repository
   git clone <repo-url> /ttt
   
   # Run the deployment script
   cd /ttt/tree_ml/pipeline
   sudo ./deploy_t4.sh
   ```

2. **Verify the T4 Server**
   ```
   # Check service status
   sudo systemctl status tree-detection
   
   # Check the API
   curl http://localhost/health
   ```

### Dashboard Configuration

1. **Configure the Dashboard**
   ```
   # On the dashboard server
   cd /ttt/tree_ml/pipeline
   ./setup_t4_integration.sh http://t4-instance-ip:80
   
   # Restart the dashboard backend
   systemctl restart tree-dashboard  # Or whatever your service is named
   ```

2. **Test the Integration**
   ```
   # Run the test script
   cd /ttt/tests/model_server
   ./test_external_model_service.py --image /ttt/data/tests/test_images/sample.jpg --server http://t4-instance-ip:80
   ```

## Configuration Options

The following environment variables control the T4 integration:

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_EXTERNAL_MODEL_SERVER` | `True` | Whether to use the external T4 model server |
| `MODEL_SERVER_URL` | `http://t4-model-server:8000` | URL of the T4 model server |

These can be set in the `.env` file or as environment variables.

## API Endpoints

The T4 model server exposes the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/detect` | POST | Tree detection endpoint |

## Error Handling

The integration implements comprehensive error handling with no synthetic data or fallbacks:

1. Connection failures are reported transparently
2. Server errors are passed through to the client
3. No default data is substituted on error
4. Errors include full context for debugging

## Security Considerations

- The T4 model server should be accessible only from trusted networks
- Consider adding API key authentication for production deployments
- HTTPS should be enabled for production environments

## Testing

The integration can be tested using:
- `/ttt/tests/model_server/test_external_model_service.py`

This script tests the complete flow from image input to detection results.

## Troubleshooting

Common issues:

1. **Connection Refused**
   - Check if the T4 server is running
   - Verify network connectivity and firewall settings

2. **Model Loading Failures**
   - Check T4 server logs: `journalctl -u tree-detection`
   - Verify model weights are downloaded correctly

3. **Slow Performance**
   - Check CUDA installation with `nvidia-smi`
   - Ensure CUDA libraries are correctly installed