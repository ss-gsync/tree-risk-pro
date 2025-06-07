# Tree ML - v0.2.3

A comprehensive platform for tree detection and risk assessment using S2 geospatial indexing, ML models, and Gemini AI.

## Project Overview

Tree ML is designed for arborists and property managers to visualize, analyze, and validate tree health and risk factors. This platform combines S2 geospatial indexing, machine learning, and AI-powered insights to provide accurate tree detection and risk assessments.

## Project Components

This repository consists of these main components:

1. **[Dashboard](/tree_ml/dashboard)** - The web-based UI for visualization and analysis
2. **[Pipeline](/tree_ml/pipeline)** - Data processing pipeline for imagery and ML models
3. **[Server](/tree_ml/server)** - Backend services for data storage and retrieval
4. **[Model Server](/tree_ml/pipeline/model_server.py)** - ML model server with T4 GPU integration

## Latest Release: v0.2.3 (2025-06-05)

### Key Changes
- Added T4 GPU integration for dedicated model server deployment
- Replaced YOLO with DeepForest, SAM, and Gemini API for improved tree detection
- Implemented external model service client for T4 server communication
- Added configuration-based selection between local and remote models
- Enhanced error handling with no synthetic data or fallbacks

### New Architecture
The v0.2.3 release introduces a significant architectural improvement with the addition of a dedicated T4 GPU model server. This allows for offloading intensive ML tasks from the dashboard server to a specialized GPU instance, improving overall system performance and scalability.

### Important Documentation
- See [T4_INTEGRATION.md](/tree_ml/docs/T4_INTEGRATION.md) for detailed information on the T4 model server setup
- Review [ML_FINDINGS.md](/tree_ml/docs/ML_FINDINGS.md) for analysis of tree detection models

## Previous Release: v0.2.2 (2025-05-27)

### Key Changes
- Renamed package from `tree_risk_pro` to `tree_ml` for better clarity and simplicity
- Added S2 geospatial indexing for efficient spatial queries at multiple zoom levels
- Comprehensive testing of ML models on satellite imagery
- Documentation of ML pipeline challenges

## Previous Release: v0.2.1 (2025-05-11)

### New Features
- Initial S2 geospatial indexing with Zarr store integration
- Validation reports linking to area reports via S2 cells
- New API endpoints for S2 cell-based report management
- Enhanced Object Report view with linked validation reports
- Improved ML overlay with persistent opacity settings

### UI Improvements
- Fixed Components/Detection sidebar functionality
- Enhanced OBJECT DETECTION badge visibility with proper z-index
- Improved sidebar panel management with event-based coordination
- Added subtle borders to Analysis section buttons
- Better header collapse state detection for UI positioning
- DOM cleanup improvements to prevent ghost elements

### Performance Enhancements
- Click-through functionality for ML overlay to improve marker interaction
- Smoother transitions between detection modes
- Better map container resizing during sidebar transitions
- Enhanced error handling for DOM operations

## Directory Structure

```
├── tree_ml/                   # Core package
│   ├── dashboard/             # Dashboard module
│   │   ├── backend/           # Python backend
│   │   │   ├── app.py         # Main Flask application
│   │   │   ├── services/      # Backend services
│   │   │   │   ├── ml/        # ML services
│   │   │   │   │   ├── model_service.py       # Local model service
│   │   │   │   │   ├── external_model_service.py # T4 model client
│   │   │   │   │   └── __init__.py            # Service selection
│   │   │   │   ├── tree_service.py      # Tree analysis 
│   │   │   │   ├── detection_service.py # Detection with S2 indexing
│   │   │   │   └── gemini_service.py    # Gemini AI integration
│   │   │   └── ...
│   │   │
│   │   └── src/               # Frontend source
│   │       ├── components/    # UI components
│   │       ├── hooks/         # Custom React hooks
│   │       ├── services/      # Frontend services
│   │       └── ...
│   │
│   ├── docs/                  # Documentation
│   │   ├── CHANGELOG.md       # Version history
│   │   ├── ML_FINDINGS.md     # ML pipeline findings
│   │   ├── T4_INTEGRATION.md  # T4 server documentation
│   │   └── ...
│   │
│   ├── pipeline/              # Data processing pipeline
│   │   ├── data_collection.py # Data collection utilities
│   │   ├── image_processing.py # Image processing utilities
│   │   ├── model_server.py    # T4 model server implementation
│   │   ├── deploy_t4.sh       # T4 deployment script
│   │   └── object_recognition.py # Object detection utilities
│   │
│   └── server/                # Data server components
│       ├── h5serv/            # HDF5 server for geospatial data
│       └── client/            # Client for interacting with data server
│
└── tests/                     # Tests
    ├── model_server/          # T4 model server tests
    │   ├── test_external_model_service.py # T4 client tests
    │   ├── test_model_server.py # Server tests
    │   └── ...
    └── ...
```

## S2 Geospatial Indexing

Tree ML uses Google's S2 geospatial indexing library for efficient spatial organization of tree data. Key features include:

- **Hierarchical Cell Levels**: Support for city (level 10), neighborhood (level 13), block (level 15), and property (level 18) zoom levels
- **Neighbor Finding**: Efficient algorithm for finding adjacent cells at any level
- **Spatial Queries**: Fast retrieval of trees based on location
- **Statistics Aggregation**: Group and summarize tree data by geographic area

## ML Pipeline and T4 Integration

The v0.2.3 release introduces significant improvements to the ML pipeline:

- **DeepForest Integration**: Replaced YOLO with DeepForest for improved tree detection in satellite imagery
- **SAM Integration**: Added Segment Anything Model for high-quality tree segmentation
- **T4 GPU Server**: Added dedicated GPU server for ML inference offloading
- **Unified Service Selection**: API-compatible local and remote model services
- **Configuration-Based Architecture**: Easily switch between local and T4 models via configuration

### T4 Model Server Benefits

- **Performance**: GPU acceleration with T4 provides up to 20x faster inference
- **Scalability**: Separate ML workloads from dashboard server
- **Resource Optimization**: Dashboard server can use less powerful hardware
- **Isolation**: ML pipeline failures don't affect dashboard stability

See [ML_FINDINGS.md](/tree_ml/docs/ML_FINDINGS.md) for detailed analysis and [T4_INTEGRATION.md](/tree_ml/docs/T4_INTEGRATION.md) for setup instructions.

## Setup and Installation

### Prerequisites

- Python 3.12+
- Node.js 18+
- Poetry for Python dependency management
- npm for JavaScript dependency management

### Backend Setup

1. Install Python dependencies:
   ```bash
   poetry install
   ```

2. Configure environment:
   ```bash
   # Create backend environment file
   cd tree_ml/dashboard/backend
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Run the backend:
   ```bash
   cd tree_ml/dashboard/backend
   poetry run python app.py
   ```

### Frontend Setup

1. Install JavaScript dependencies:
   ```bash
   cd tree_ml/dashboard
   npm install
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your Google Maps API key and other settings
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Access the dashboard at http://localhost:5173

## Main Features

- **S2 Geospatial Indexing**: Efficient spatial organization of tree data
- **Tree Detection**: Analysis of aerial imagery to detect trees (in development)
- **Risk Assessment**: Evaluation of tree health and risk factors
- **3D Visualization**: Interactive 2D/3D map visualization of trees and properties
- **Validation System**: Workflow for validating and refining detection results
- **Reporting**: Generate comprehensive tree risk assessment reports
- **Analytics**: Track and analyze tree risk data across properties

## Development Guidelines

- Use Black for Python code formatting
- Follow React best practices for frontend code
- Write comprehensive tests for new features
- Document code using docstrings and comments
- Follow semantic versioning

## Testing

The project uses three specialized testing scripts in the `/tests/ml` directory:

```bash
# Test on a single image
poetry run python tests/ml/ml_tester.py

# Process all test images in the directory
poetry run python tests/ml/ml_tester.py --all-images

# Test specific models or pipeline
poetry run python tests/ml/ml_tester.py --test {deepforest|sam|pipeline}

# Test CUDA/GPU compatibility 
poetry run python tests/ml/cuda_tester.py

# Alternative batch processor with additional options
poetry run python tests/ml/image_processor.py
```

Test results are stored in `data/tests/ml_test_results/` with a consistent structure. Results include shared model/pipeline output directories for comparison and individual directories for each processed image with detection results and visualizations.

See [tests/ml/README.md](/tests/ml/README.md) for comprehensive documentation of the testing framework.

## Changelog

See [CHANGELOG.md](/tree_ml/docs/CHANGELOG.md) for a complete list of changes between versions.

## License

Copyright © 2025 Texas Tree Transformations, LLC. All rights reserved.