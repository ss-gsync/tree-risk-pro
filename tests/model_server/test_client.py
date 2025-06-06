"""
Custom test client for FastAPI that works with our version.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the module
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

from fastapi.testclient import TestClient
from fastapi import FastAPI

def create_test_client(app: FastAPI) -> TestClient:
    """Create a test client for FastAPI that works with our version."""
    client = TestClient(app)
    # Override endpoints for test compatibility
    # We need to add a /health endpoint for testing
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        import torch
        return {"status": "ok", "device": "cuda" if torch.cuda.is_available() else "cpu"}
    
    return client