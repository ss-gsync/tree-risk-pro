# Tree ML Dashboard Services

This directory contains backend services for the Tree ML Dashboard application.

## ML Detection Pipeline

The ML detection pipeline is a comprehensive system for detecting trees and other objects in satellite imagery.

### Architecture Diagram

```
+-------------------+     +----------------------+     +-------------------+
|                   |     |                      |     |                   |
|  Frontend (React) +---->+ Backend API (Flask)  +---->+ ML Services       |
|                   |     |                      |     |                   |
+--------+----------+     +----------+-----------+     +---------+---------+
         |                           |                           |
         |                           |                           |
         v                           v                           v
+--------+----------+     +----------+-----------+     +---------+---------+
|                   |     |                      |     |                   |
|  User Interface   |     |  API Endpoints       |     |  Model Service    |
|  - Map View       |     |  - /detection/detect |     |  - DeepForest     |
|  - Detection UI   |     |  - /trees/*          |     |  - Grounded-SAM   |
|  - Result Display |     |  - /validation/*     |     |  - Segmentation   |
|                   |     |                      |     |                   |
+-------------------+     +----------------------+     +-------------------+
                                     |
                                     |
                          +----------v-----------+
                          |                      |
                          |  Supporting Services |
                          |  - Geolocation       |
                          |  - Validation        |
                          |  - Database          |
                          |  - Tree Service      |
                          |                      |
                          +----------+-----------+
                                     |
                                     |
                          +----------v-----------+
                          |                      |
                          |  Storage             |
                          |  - Zarr Store        |
                          |  - ML Results        |
                          |  - Detection History |
                          |                      |
                          +----------------------+
```

### Data Flow

1. **User Interaction**:
   - User navigates to a location on the map
   - User clicks "Detect Trees" button
   - Frontend captures map view coordinates and settings

2. **API Request**:
   - Frontend makes a request to `/api/detection/detect`
   - Request includes coordinates, zoom level, and detection settings
   - Optional satellite imagery may be included as base64 or FormData

3. **Backend Processing**:
   - API validates coordinates and creates a unique job ID
   - DetectionService processes the request
   - GeolocationService acquires satellite imagery if needed

4. **ML Inference**:
   - ModelService loads ML models (cached in memory)
   - Models run inference on the satellite image
   - Results include bounding boxes and segmentation masks

5. **Result Processing**:
   - DetectionService converts pixel coordinates to geographic coordinates
   - S2 geospatial indexing is applied for spatial queries
   - Results are saved to Zarr store with metadata

6. **Response to Frontend**:
   - API returns detection results with job ID
   - Frontend displays results on the map
   - User can validate, edit, or save the detection results

### Key Components

- **ModelService**: Manages ML models and performs inference
- **DetectionService**: Coordinates the detection pipeline
- **GeolocationService**: Handles coordinate systems and satellite imagery
- **S2IndexManager**: Provides geospatial indexing for efficient queries
- **TreeService**: Manages tree data and detection results

### File Structure

- `detection_service.py`: Main service for ML detection pipeline
- `ml/model_service.py`: Handles ML models and inference
- `geolocation_service.py`: Coordinates and image acquisition
- `tree_service.py`: Tree data management
- `validation_service.py`: Detection result validation

### Performance Considerations

- ML models are loaded once and kept in memory
- Progressive loading of models with status tracking
- GPU acceleration when available
- Efficient image transfer using FormData
- Zarr storage for fast retrieval of large datasets