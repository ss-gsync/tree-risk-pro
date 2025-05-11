# Tree Risk Pro - v0.2.1

A comprehensive platform for assessing tree risks using LiDAR data, aerial imagery, and Gemini AI.

## Project Overview

Tree Risk Pro is designed for arborists and property managers to visualize, analyze, and validate tree health and risk factors. This platform combines advanced LiDAR processing, machine learning, and AI-powered insights to provide accurate tree risk assessments.

## Project Components

This repository consists of these main components:

1. **[Dashboard](/tree_risk_pro/dashboard)** - The web-based UI for visualization and analysis
2. **[Pipeline](/tree_risk_pro/pipeline)** - Data processing pipeline for LiDAR and imagery
3. **[Server](/tree_risk_pro/server)** - Backend services for data storage and retrieval

## Latest Release: v0.2.1 (2025-05-11)

### New Features
- S2 geospatial indexing with Zarr store integration
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

## Previous Release: v0.2.0 (2025-04-26)

### UI Improvements
- Better header layout with consistent spacing
- Settings button now in header for easier access
- Added visual separators in sidebars
- Renamed "Save to Database" to "Save for Review"
- Fixed sidebar navigation

### Map Functionality
- 3D map state now preserved between views
- Improved 3D/2D toggle
- Fixed map sizing when using sidebars

### Backend Updates
- Added thorough code documentation
- Better Gemini AI integration
- Enhanced error handling
- Improved state management

## Access Information

Our production instance at https://34.125.120.78 uses basic authentication:
- Username: `TestAdmin`
- Password: `trp345!`

See [DEPLOYMENT.md](/tree_risk_pro/dashboard/DEPLOYMENT.md) for detailed setup instructions.

This project uses a modern stack with:

- **Frontend**: React + Vite with Redux for state management
- **Backend**: Flask Python API with RESTful endpoints
- **UI Components**: shadcn/UI with Tailwind CSS
- **Packaging**: Poetry for Python dependencies, npm for JavaScript

### Directory Structure

```
tree_risk_pro/
│
├── tree_risk_pro/                 # Core package
│   ├── dashboard/                 # Dashboard module
│   │   ├── backend/               # Python backend
│   │   │   ├── app.py             # Main Flask application
│   │   │   ├── services/          # Backend services
│   │   │   │   ├── tree_service.py        # Tree analysis 
│   │   │   │   ├── lidar_service.py       # LiDAR processing
│   │   │   │   └── gemini_service.py      # Gemini AI integration
│   │   │   └── ...
│   │   │
│   │   └── src/                   # Frontend source
│   │       ├── components/        # UI components
│   │       ├── hooks/             # Custom React hooks
│   │       ├── services/          # Frontend services
│   │       └── ...
│   │
│   ├── pipeline/                  # Data processing pipeline
│   │   ├── data_collection.py
│   │   ├── image_processing.py
│   │   └── object_recognition.py
│   │
│   └── server/                    # Data server components
│       ├── h5serv/                # HDF5 server for geospatial data
│       └── client/                # Client for interacting with data server
│
├── scripts/                       # Utility scripts
├── docs/                          # Documentation
└── tests/                         # Tests
```

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
   cd tree_risk_pro/dashboard/backend
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Run the backend:
   ```bash
   cd tree_risk_pro/dashboard/backend
   poetry run python app.py
   ```

### Frontend Setup

1. Install JavaScript dependencies:
   ```bash
   cd tree_risk_pro/dashboard
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

- **Tree Detection**: Analyze LiDAR and aerial imagery to detect and measure trees
- **Risk Assessment**: Evaluate tree health and risk factors using multiple data sources
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

## Changelog

See [CHANGELOG.md](./CHANGELOG.md) for a complete list of changes between versions.

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

## License

Copyright © 2025 Texas Tree Transformations, LLC. All rights reserved.