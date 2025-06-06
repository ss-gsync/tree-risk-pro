# Tree Detection Components

This directory contains the components used for tree detection and visualization in the application.

## Component Structure

- `DetectionMode.jsx`: The main component that handles the tree detection mode UI
- `MLOverlay.jsx`: Legacy component that renders the ML detection overlay on the map
- `StandaloneDetectionOverlay.jsx`: **NEW** Standalone visualization system that works without React lifecycle constraints
- `TreeDetection.jsx`: Component that provides controls for object detection
- `TreeInfo.jsx`: Component that displays information about a detected tree
- `TreeTagForm.jsx`: Form for tagging trees with metadata
- `index.js`: Exports all components for easier importing

## Detection Visualization

We've implemented a completely standalone detection visualization system that solves issues with the display of detection results:

### StandaloneDetectionOverlay

A completely standalone detection visualization system that works without requiring React component mounting. This solves issues with visualization not appearing properly or components disappearing.

Key features:
- Works independently of React's component lifecycle
- Can be called directly as a function
- Multiple fallback methods for finding the map container
- Handles both normalized and absolute coordinates
- Supports different object classes (trees, buildings, power lines)
- Configurable opacity and appearance
- Shows count indicators and success notifications

Usage:
```jsx
// Import as a function
import { renderDetectionOverlay } from './StandaloneDetectionOverlay';

// Use directly with detection data
renderDetectionOverlay(detectionData, {
  opacity: 0.8,
  classes: { trees: true, buildings: true, powerLines: false }
});

// Or as a React component
import StandaloneDetectionOverlay from './StandaloneDetectionOverlay';

<StandaloneDetectionOverlay
  detectionData={data}
  opacity={0.8}
  classes={{ trees: true, buildings, true, powerLines: false }}
  visible={true}
/>
```

## Coordinate Handling

**IMPORTANT**: Coordinates must be passed in the correct format to ensure accurate tree detection:

1. **UserCoordinates**: The backend strictly requires `userCoordinates` in the API requests. These should be:
   - In [longitude, latitude] format
   - Formatted to 6 decimal places to match what's displayed in the UI
   - The exact coordinates shown in the LocationInfo component

2. **No Defaults or Fallbacks**: The system will NOT fall back to default coordinates:
   - If `userCoordinates` are missing, detection will fail
   - Default Dallas coordinates (32.8, -96.7) are rejected

3. **Data Flow**:
   - `MapControls.jsx` captures user's current view coordinates
   - Coordinates are formatted to 6 decimal places
   - They're passed in `mapViewInfo.viewData.userCoordinates`
   - Backend validation ensures these coordinates are present before proceeding

## How the Visualization System Works

The detection visualization system has been completely rewritten to avoid issues with React component mounting/unmounting:

1. **Direct DOM Rendering**: Instead of using React's virtual DOM, the system directly creates and manages DOM elements for visualization.

2. **Multiple Fallbacks**: The system has multiple methods for finding the map container, ensuring visualization always appears.

3. **Global Data Access**: Detection results are stored globally (window.mlDetectionData) so they can be accessed from anywhere.

4. **Event-Based Communication**: Uses custom events to communicate between components and the visualization system.

## Troubleshooting

If visualization doesn't appear:
- Check browser console for errors
- Verify that detection data is available (check network tab for API responses)
- Try refreshing the page and running detection again
- Make sure you're in 3D mode with sufficient zoom level

## Avoiding Duplicate Code

There's duplicate detection code in the codebase:

1. `/components/assessment/Detection/TreeDetection.jsx` - older implementation
2. `/components/visualization/MapView/DetectionMode.jsx` - duplicate of Detection/DetectionMode.jsx 
3. `/components/visualization/Detection/` - current main implementation

**Always use the components in this directory (`/components/visualization/Detection/`)** and avoid using duplicates from other locations to prevent confusion.