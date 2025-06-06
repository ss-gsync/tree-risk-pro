"""
Test script for the Gemini API integration.
This script tests basic connectivity to the Gemini API.
"""

import asyncio
import logging
import os
import sys
import json
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gemini_test')

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the GeminiService and configuration
from dashboard.backend.services.gemini_service import GeminiService
from dashboard.backend.config import GEMINI_API_KEY

async def test_direct_api_call():
    """Test direct API call to Gemini to diagnose authentication issues."""
    logger.info("Testing direct API call to Gemini v1beta API...")
    
    # API configuration - explicitly use key from config
    api_key = GEMINI_API_KEY
    logger.info(f"API key from config - type: {type(api_key)}, length: {len(api_key)}")
    logger.info(f"Using API key from config: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else '****'}")
    
    # Test v1beta endpoint for Gemini 2.0 - explicitly use v1beta
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    # Simple text-only payload
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Hello from Gemini API test! Please respond with 'Connection successful.'"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 100
        }
    }
    
    # Make the API call
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            status = response.status
            text = await response.text()
            
            logger.info(f"Direct API call status: {status}")
            logger.info(f"Direct API response: {text}")
            
            return status, text

async def test_multimodal_api():
    """Test the Gemini multimodal API with image data."""
    logger.info("Testing Gemini multimodal API...")
    
    # API configuration
    api_key = GEMINI_API_KEY
    api_version = "v1beta"  # Required for Gemini 2.0
    
    # Create a test image with PIL
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple test image
        img = Image.new('RGB', (300, 200), color=(73, 109, 137))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "Test Image for Gemini API", fill=(255, 255, 0))
        
        # Save to temporary file
        import tempfile
        import os
        import base64
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img_path = f.name
            img.save(img_path)
        
        # Read the file
        with open(img_path, 'rb') as f:
            img_data = f.read()
        os.unlink(img_path)  # Clean up
        
        # Base64 encode
        img_b64 = base64.b64encode(img_data).decode('utf-8')
        
        # Use direct multimodal API with inline data
        generate_url = f"https://generativelanguage.googleapis.com/{api_version}/models/gemini-2.0-flash:generateContent?key={api_key}"
        
        # Construct payload with inline image data
        generate_payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Describe this test image:"
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_b64
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 300
            }
        }
                
        # Make the API call
        async with aiohttp.ClientSession() as session:
            async with session.post(generate_url, json=generate_payload) as content_response:
                content_status = content_response.status
                content_text = await content_response.text()
                
                logger.info(f"Multimodal API status: {content_status}")
                if content_status == 200:
                    logger.info(f"Multimodal API response: {content_text[:200]}...")
                    return True
                else:
                    logger.error(f"Multimodal API failed: {content_status} - {content_text}")
                    return False
    except Exception as e:
        logger.error(f"Error testing multimodal API: {str(e)}")
        return False

async def test_gemini_service():
    """Test basic connectivity to the Gemini API."""
    logger.info("Initializing GeminiService...")
    service = GeminiService()
    
    # Initialize the service
    await service.initialize()
    
    # Add debug output to see service properties
    logger.info(f"GeminiService API key length: {len(service.api_key)}")
    logger.info(f"GeminiService API key first/last chars: {service.api_key[:4]}...{service.api_key[-4:]}")
    
    # Test a simple text query
    logger.info("Testing simple text query through service...")
    result = await service.query("Test connection to Gemini API. Please respond with 'Connection successful.'")
    
    # Print the result
    logger.info(f"Response success: {result['success']}")
    if result['success']:
        logger.info(f"Response content: {result['response']}")
    else:
        logger.error(f"Error message: {result['message']}")
        logger.error(f"Error details: {result.get('error_details', 'No details available')}")
    
    logger.info("Service test completed.")
    return result

async def main():
    """Run all tests."""
    # Test direct API call first
    status, text = await test_direct_api_call()
    
    # If direct API call fails, no need to test service
    if status != 200:
        logger.error("Direct API call failed, skipping service test.")
        return
    
    # Test through service
    await test_gemini_service()
    
    # Test multimodal API for image handling
    logger.info("Testing multimodal API...")
    multimodal_result = await test_multimodal_api()
    
    if multimodal_result:
        logger.info("Multimodal API test passed!")
    else:
        logger.error("Multimodal API test failed.")

if __name__ == "__main__":
    # Run the test
    asyncio.run(main())