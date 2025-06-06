"""
TTT Assessment Dashboard - Services
-----------------------------------------
This package contains the services used by the Texas Tree Transformation Assessment Dashboard.

Services:
- DatabaseService: Handles data storage and retrieval
- DetectionService: Handles tree detection using ML models
- GeolocationService: Handles geographic coordinate processing and transformations
- GeminiService: Handles integration with Google Gemini API
- LidarService: Handles LiDAR data processing
- TreeService: Handles tree data management
- ValidationService: Handles validation workflow
"""

# Import services for easier access
from .geolocation_service import GeolocationService