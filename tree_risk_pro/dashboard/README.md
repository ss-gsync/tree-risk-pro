# Tree Risk Pro Dashboard - Beta v0.2

An application for tree risk assessment using LiDAR data, aerial imagery, and Gemini AI. Designed for arborists and property managers to visualize, analyze, and validate tree health and risk factors.

## Beta v0.2 Updates

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

🔒 **Access Information**: Our beta instance at https://34.125.120.78 uses basic authentication with credentials: Username: `TestAdmin`, Password: `trp345!`. See [DEPLOYMENT.md](./DEPLOYMENT.md) for setup instructions.

## Project Architecture

This project uses a modern stack with:

- **Frontend**: React + Vite with Redux for state management
- **Backend**: Flask Python API with RESTful endpoints
- **UI Components**: shadcn/UI with Tailwind CSS
- **Packaging**: Poetry for Python dependencies, npm for JavaScript

### Directory Structure

```
dash/
│
├── backend/                          # Python backend directory
│   ├── app.py                        # Main Flask application
│   ├── data/                         # Data storage directory 
│   └── services/
│       ├── __init__.py
│       ├── tree_service.py           # Tree analysis service
│       ├── validation_service.py     # Report validation service
│       └── lidar_service.py          # LiDAR processing service
│
├── src/                              # Frontend source
│   ├── main.jsx                      # Application entry point
│   ├── App.jsx                       # Main App component
│   ├── index.css                     # Global styles
│   │
│   ├── components/                   # UI components
│   │   ├── common/                   # Shared UI components
│   │   ├── ui/                       # shadcn/UI components
│   │   ├── visualization/            # Map and visualization
│   │   └── assessment/               # Assessment components
│   │
│   ├── lib/                          # Utility functions
│   ├── store/                        # Redux store
│   ├── features/                     # Redux features
│   ├── services/                     # Frontend services
│   └── hooks/                        # Custom React hooks
│
├── public/                           # Static assets
├── components.json                   # shadcn/ui configuration
├── jsconfig.json                     # JavaScript path aliases
├── .env.example                      # Environment template
├── pyproject.toml                    # Python dependencies
├── package.json                      # JavaScript dependencies
├── tailwind.config.js                # Tailwind CSS configuration
└── vite.config.js                    # Vite configuration
```

## Setup and Installation

### Prerequisites

- Node.js 16+ (for frontend)
- Python 3.9+ (for backend)
- Poetry (for Python dependency management)
- npm or yarn (for JavaScript dependency management)

### Frontend Setup

1. Install JavaScript dependencies:
   ```bash
   npm install
   ```

2. Install required additional packages:
   ```bash
   # Install Tailwind CSS and its dependencies
   npm install -D tailwindcss postcss autoprefixer
   
   # Install React Router
   npm install react-router-dom
   
   # Install utilities for shadcn/UI
   npm install clsx tailwind-merge
   ```

3. Create environment configuration:
   ```bash
   cp .env.example .env
   ```
   
4. Update the `.env` file with the following:
   ```
   # Required for map functionality
   VITE_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
   VITE_GOOGLE_MAPS_MAP_ID=your_map_id
   
   # API URL for development (default)
   VITE_API_URL=http://localhost:5000
   
   # API URL for production usage
   # VITE_API_URL=https://34.125.120.78
   ```

### Setting up shadcn/UI

1. Configure path aliases in vite.config.js:
   ```javascript
   import { defineConfig } from 'vite'
   import react from '@vitejs/plugin-react'
   import path from 'path'

   export default defineConfig({
     plugins: [react()],
     resolve: {
       alias: {
         '@': path.resolve(__dirname, './src'),
       },
     },
   })
   ```

2. Create a jsconfig.json file for path resolution:
   ```json
   {
     "compilerOptions": {
       "baseUrl": ".",
       "paths": {
         "@/*": ["./src/*"]
       }
     }
   }
   ```

3. Initialize shadcn/UI:
   ```bash
   npx shadcn@latest init
   ```
   
4. Add UI components as needed:
   ```bash
   npx shadcn@latest add card
   # Add other components as needed (button, dropdown, etc.)
   ```

### Backend Setup

1. Install Python dependencies using Poetry:
   ```bash
   cd backend
   poetry install
   ```

2. Create data directories:
   ```bash
   mkdir -p backend/data/lidar
   mkdir -p backend/data/reports
   ```

## Development

### Running the Frontend

```bash
npm run dev
```

This starts the development server on http://localhost:5173

### Running the Backend

There are two backend options:

1. **Python Flask Backend** (Main backend with all features):
   ```bash
   cd backend
   poetry install
   poetry run python app.py
   ```

2. **Node.js Express Backend** (For development with mock data):
   ```bash
   cd backend
   npm install
   npm run dev
   ```

Both options start the backend on http://localhost:5000

**Note:** When running locally, you may see Google Maps API errors about "The map is initialized without a valid Map ID." To fix this:
1. Make sure your `.env` file includes a valid Google Maps API key
2. Add `VITE_GOOGLE_MAPS_MAP_ID=your_map_id` to your `.env` file

## Troubleshooting

### Common Issues

1. **Path resolution problems**
   - Ensure the `@/` alias is properly configured in both `vite.config.js` and `jsconfig.json`
   - Check if imports are using the correct path format (e.g., `@/components/ui/card`)

2. **Component import issues**
   - Make sure all imported components exist in the expected location
   - For loading components, check that the imports in index.js match the actual filenames

3. **Tailwind CSS not applying**
   - Verify that the Tailwind directives are present in `src/index.css`
   - Check that the `tailwind.config.js` includes the correct content paths

4. **shadcn/UI component errors**
   - Ensure you're using the latest version of shadcn packages
   - Run `npx shadcn@latest add [component-name]` to add missing components

5. **Google Maps API errors**
   - If you see "The map is initialized without a valid Map ID" error:
     - Get a valid Map ID from Google Cloud Console (Maps Platform > Map Management)
     - Add `VITE_GOOGLE_MAPS_MAP_ID=your_map_id` to your `.env` file
     - Make sure `VITE_GOOGLE_MAPS_API_KEY` is also set correctly
     - Restart your development server

## Features

### Map Visualization

- Interactive map with property boundaries
- LiDAR data visualization
- Risk zone highlighting
- Layer management for different data views

### Tree Analysis

- Detailed tree information display
- Risk assessment calculation
- Historical growth analysis
- Automatic detection of high-risk conditions

### Validation System

- Validation queue for risk assessments
- Approval/rejection workflow
- Notes and documentation for each validation
- Report generation

## API Reference

### Properties API
- `GET /api/properties` - List all properties
- `GET /api/properties/{id}` - Property details
- `GET /api/properties/{id}/trees` - Trees on a property

### Tree Management
- `GET /api/trees/{id}` - Tree details
- `PUT /api/trees/{id}/assessment` - Update assessments

### Validation API
- `GET /api/validation/queue` - Validation queue
- `PUT /api/validation/{id}` - Update status

### Report Generation
- `POST /api/properties/{id}/report` - Create report
- `GET /api/reports/{id}` - Download report

## Security Implementation

### Authentication
- Basic authentication on all endpoints
- Current credentials: TestAdmin/trp345!
- 8-hour session timeout
- Rate limiting on failed attempts

### API Protections
- Request timeout limits
- Proper error status codes
- Secure token handling
- Network security measures

See [DEPLOYMENT.md](./DEPLOYMENT.md) for security setup details.

## Building for Production

### Frontend Build

```bash
npm run build
```

This creates optimized production files in the `dist` directory.

### Backend Deployment

For production deployment, consider using Gunicorn with Nginx:

```bash
export APP_MODE=production  # Enable production mode with TestAdmin/trp345! credentials
export SKIP_AUTH=false      # Enforce authentication
poetry run gunicorn -w 4 -b 127.0.0.1:5000 app:app
```

## Testing

### Frontend Tests

```bash
npm run test
```

### Backend Tests

```bash
cd backend
poetry run pytest
```

## Acknowledgements

- Mapbox for mapping capabilities
- React and Redux for frontend framework
- shadcn/UI for component library
- Tailwind CSS for styling
- Flask for backend API