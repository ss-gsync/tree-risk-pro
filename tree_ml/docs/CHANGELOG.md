# Changelog

All notable changes to the Tree ML project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2025-06-09

### Changed
- Replaced YOLO detection with DeepForest, SAM, and Gemini API integration
- Updated detection service to support multiple detection modes (DeepForest, Gemini, or both)
- Enhanced tree detection pipeline to use better models for aerial/satellite imagery
- Restructured model architecture to support both local and external model servers
- Simplified ML overlay implementation with integrated approach
- Improved real-time opacity and visibility controls for detection overlay
- Enhanced UI components with more consistent button styling and interaction

### Added
- DeepForest integration for tree crown detection
- SAM (Segment Anything Model) integration for tree segmentation
- Gemini Vision API integration for multimodal tree detection
- Multiple detection mode support in frontend UI
- Improved detection results with metadata and detection source tracking
- T4 GPU integration for dedicated model server deployment
- External model service client for communication with T4 server
- Configuration-based model server selection (local vs. T4)
- Comprehensive error handling with no synthetic data or fallbacks
- Setup script for T4 integration configuration
- Test suite for validating the T4 model server integration
- Detailed documentation for T4 integration architecture and setup
- Integrated ML overlay module directly in dashboard codebase
- Enhanced detection sidebar with improved overlay controls
- Immediate overlay display when detection sidebar is opened
- Improved Analytics panel with consistent UI and better visibility for buttons

### Fixed
- Resolved the issue with YOLO not detecting trees in satellite imagery by replacing it with models designed for aerial imagery
- Fixed model loading inconsistencies between CPU and GPU environments
- Improved error handling and reporting for model service failures
- Eliminated synthetic data fallbacks to properly expose underlying issues
- Fixed ML overlay opacity slider to update in real-time
- Fixed "Show Overlay" toggle functionality
- Fixed objects counter consistency across multiple detections
- Eliminated external script dependencies for ML overlay
- Fixed styling of buttons in Analytics panel for better visibility and consistency
- Fixed duplicate Analytics panel issue when clicking Analytics button
- Improved sidebar behavior to ensure proper closing of all panels when opening a new one

## [0.2.2] - 2025-05-27

### Changed
- Renamed package from `tree_risk_pro` to `tree_ml` for better clarity and simplicity
- Updated all imports and references to use the new package name
- Updated version number in package metadata
- Replaced YOLO detection with DeepForest NEON model for improved tree detection
- Integrated SAM model for high-quality tree segmentation
- Lowered confidence thresholds for detecting more potential trees
- Optimized model parameter settings for satellite imagery

### Added
- S2 geospatial indexing integration for efficient spatial queries at multiple zoom levels
- Comprehensive test scripts to verify S2 indexing functionality
- Direct model testing script to evaluate tree detection on satellite imagery
- Advanced ML testing framework for tree detection and segmentation
- Image enhancement pipeline for improved tree detection in satellite imagery
- Dual visualization approach showing detections on both original and enhanced images
- Comprehensive testing suite with detailed metrics and reports

### Fixed
- Package import issues that were causing detection service errors
- Resolved device mismatch issues between CPU and CUDA tensors
- Fixed visualization problems for both detections and segmentations
- Addressed model loading inconsistencies

### Important Findings
- S2 geospatial indexing is working correctly for all zoom levels (city, neighborhood, block, property)
- YOLO models (both custom and standard) are not detecting trees in satellite imagery
- Comprehensive testing with multiple models and extremely low confidence thresholds (down to 0.001) confirmed that YOLO is not suitable for tree detection in aerial/satellite views
- DeepForest NEON model detects significantly more trees than YOLO on satellite imagery
- Image enhancement increases detection count but introduces false positives
- Original images have fewer but more accurate detections (1-12 trees)
- Enhanced images detect more trees (32-60) with moderate false positives (30-40%)
- Double-enhanced images detect even more trees (63-99) but with high false positives (50-70%)
- SAM provides high-quality segmentation for all detected objects
- SAM segmentation produces consistently high confidence scores (0.78-0.99)
- Common false positives include baseball fields, buildings, and road features
- The testing approach was refactored to remove synthetic data generation and fallbacks to properly expose the underlying issues

### Next Steps
- Investigate specialized models for tree detection in satellite imagery (e.g., DeepForest)
- Consider integrating Gemini API-based analysis with S2 indexing for a more effective solution
- Explore fine-tuning ML models specifically for aerial/satellite tree detection
- Implement a secondary classification stage to filter false positives
- Explore techniques for specialized detection of urban trees
- Investigate more selective enhancement focused on vegetation-specific features

## [0.2.1] - 2025-04-10

### Added
- Initial integration of ML pipeline with dashboard components
- Basic tree detection and risk assessment functionality
- Preliminary API endpoints for tree data access

### Changed
- Improved satellite imagery processing
- Enhanced UI for displaying tree data

### Fixed
- Various bugs in the detection pipeline
- Performance issues with large datasets