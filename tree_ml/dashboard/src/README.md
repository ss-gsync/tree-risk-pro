# Tree Risk Assessment Dashboard

A comprehensive web application for assessing tree risks on properties using LiDAR data and aerial imagery. The dashboard allows arborists and property managers to visualize, analyze, and validate tree risk assessments.

## Project Architecture

This project uses a modern stack with:

- **Frontend**: React + Vite with Redux for state management
- **Backend**: Express.js (development) / Flask (production) for API endpoints
- **Maps**: Google Maps Platform for spatial visualization
- **Packaging**: npm for JavaScript dependencies, Poetry for Python dependencies

### Directory Structure

```
dashboard/
│
├── backend/                          # Server directory
│   ├── app.py                        # Main Flask application (production)
│   ├── server.js                     # Express.js server (development)
│   ├── data/                         # Data storage directory 
│   │   └── mock/                     # Mock data for development
│   │       ├── properties.json       # Property data
│   │       ├── trees.json            # Tree data
│   │       └── validation_queue.json # Validation items
│   └── services/                     # Backend services
│       ├── __init__.py
│       ├── tree_service.py           # Tree analysis service
│       ├── validation_service.py     # Report validation service
│       ├── lidar_service.py          # LiDAR processing service
│       ├── database_service.py       # Database service
│       ├── gemini_service.py         # Google Gemini AI integration 
│       └── bigquery_service.py       # BigQuery analytics service
│
├── src/                              # Frontend source
│   ├── main.jsx                      # Application entry point
│   ├── App.jsx                       # Main App component
│   ├── index.css                     # Global styles
│   │
│   ├── components/                   # UI components
│   │   ├── analytics/                # Analytics components
│   │   │   └── AnalyticsDashboard.jsx # Main analytics dashboard
│   │   ├── assessment/               # Assessment components
│   │   │   ├── AssessmentPanel.jsx   # Property assessment panel
│   │   │   ├── TreeAnalysis/         # Tree analysis components
│   │   │   └── Validation/           # Validation components 
│   │   ├── common/                   # Shared UI components
│   │   │   ├── Layout/               # Layout components
│   │   │   └── Loading/              # Loading indicators
│   │   ├── ui/                       # UI component library
│   │   └── visualization/            # Map and visualization
│   │       └── MapView/              # Map components
│   │
│   ├── store/                        # Redux store
│   │   ├── index.js                  # Store configuration
│   │   └── middleware/               # Redux middleware
│   │
│   ├── features/                     # Redux features
│   │   └── map/                      # Map state management
│   │       └── mapSlice.js           # Map state slice
│   │
│   ├── services/                     # Frontend services
│   │   ├── api/                      # API clients
│   │   │   ├── apiService.js         # Core API service
│   │   │   ├── analyticsApi.js       # Analytics API
│   │   │   ├── reportApi.js          # Report API
│   │   │   └── treeApi.js            # Tree API
│   │   └── processing/               # Data processing
│   │       └── lidarProcessing.js    # LiDAR processing
│   │
│   └── hooks/                        # Custom React hooks
│       ├── useLidarData.js           # LiDAR data hook
│       ├── useTreeAnalysis.js        # Tree analysis hook
│       └── useValidation.js          # Validation hook
│
├── public/                           # Static assets
├── .env.template                     # Environment template
└── package.json                      # JavaScript dependencies
```

## Features

### Map Visualization

- Interactive Google Maps with property boundaries
- LiDAR data visualization
- Tree and property markers with risk indicators
- Layer management for different data views
- Focused visualization of selected properties

### Tree Analysis

- Detailed tree information display
- Risk assessment calculation
- Visual indicators for different risk levels
- Support for multiple risk factors per tree

### Validation System

- Validation queue for risk assessments
- Approval/rejection workflow
- Notes and documentation for each validation
- Report generation with recommendations

### Analytics Dashboard

- Risk distribution visualization
- Property statistics
- Time-based analysis of assessments
- Interactive charts and graphs

## Setup and Installation

### Prerequisites

- Node.js 16+ (for frontend and development backend)
- Python 3.9+ (for production backend)
- Google Maps Platform API key
- npm or yarn (for JavaScript dependency management)

### Frontend Setup

1. Install JavaScript dependencies:
   ```bash
   npm install
   ```

2. Create environment configuration:
   ```bash
   cp .env.template .env
   ```
   
3. Update the `.env` file with your Google Maps API key:
   ```
   VITE_GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
   ```

### Development Backend Setup

1. Install Node.js dependencies for the backend:
   ```bash
   cd backend
   npm install
   ```

2. Create mock data directories:
   ```bash
   mkdir -p backend/data/mock
   ```

3. Start the development backend server:
   ```bash
   node server.js
   ```

### Production Backend Setup

1. Install Python dependencies (with Poetry or pip):
   ```bash
   cd backend
   
   # With Poetry
   poetry install
   
   # Or with pip
   pip install -r requirements.txt
   ```

2. Start the Flask backend:
   ```bash
   python app.py
   ```

## Development

### Running the Frontend

```bash
npm run dev
```

This starts the development server on http://localhost:5173

### Running the Development Backend

```bash
cd backend
node server.js
```

This starts the Express backend on http://localhost:5000

## API Endpoints

### Properties

- `GET /api/properties` - Get all properties
- `GET /api/properties/{id}` - Get property details
- `GET /api/properties/{id}/trees` - Get all trees for a property

### Trees

- `GET /api/trees` - Get all trees
- `GET /api/trees/{id}` - Get tree details
- `PUT /api/trees/{id}/assessment` - Update tree assessment

### Validation

- `GET /api/validation/queue` - Get validation queue
- `PUT /api/validation/{id}` - Update validation status

### LiDAR

- `GET /api/lidar/property/{id}` - Get LiDAR data for property

### Analytics

- `GET /api/analytics/risk-distribution` - Get risk distribution analytics
- `GET /api/analytics/property-stats` - Get property statistics
- `GET /api/analytics/map/data` - Get geospatial analytics data

## Building for Production

### Frontend Build

```bash
npm run build
```

This creates optimized production files in the `dist` directory.

### Backend Deployment

For production deployment with Flask, consider using Gunicorn with Nginx:

```bash
gunicorn -w 4 -b 127.0.0.1:5000 app:app
```

## Testing

### Frontend Tests

```bash
npm run test
```

### Backend Tests

```bash
cd backend
pytest
```

## Custom UI Components

The project uses a simplified implementation of shadcn/ui components, which are built on top of Tailwind CSS. These components provide a consistent look and feel across the application while maintaining high performance.

## Acknowledgements

- Google Maps Platform for mapping capabilities
- React and Redux for frontend architecture
- Express.js and Flask for backend APIs
- Recharts for data visualization