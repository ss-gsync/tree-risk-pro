// src/features/map/mapSlice.js

import { createSlice } from '@reduxjs/toolkit';

// Default center coordinates for North Dallas area (near White Rock Lake Park)
const DEFAULT_CENTER = [-96.7800, 32.8600]; 
const DEFAULT_ZOOM = 10;

const initialState = {
  center: DEFAULT_CENTER,
  zoom: DEFAULT_ZOOM,
  activeBasemap: 'roadmap', // 'roadmap', 'satellite', 'hybrid', or 'terrain'
  visibleLayers: ['properties', 'trees'], // Default visible layers
  selectedFeature: null, // Selected tree, property, etc.
  loadingLayers: false,
  mapBounds: null, // Current map bounds for data fetching
  mapInitialized: false, // Flag for map initialization
  filterSettings: {
    riskLevel: 'all',
    treeHeight: 'all',
    species: '',
    dateRange: null
  }
};

const mapSlice = createSlice({
  name: 'map',
  initialState,
  reducers: {
    setMapView: (state, action) => {
      const { center, zoom } = action.payload;
      if (center) state.center = center;
      if (zoom !== undefined) state.zoom = zoom;
    },
    
    setActiveBasemap: (state, action) => {
      state.activeBasemap = action.payload;
    },
    
    toggleLayer: (state, action) => {
      const layerId = action.payload;
      
      if (state.visibleLayers.includes(layerId)) {
        state.visibleLayers = state.visibleLayers.filter(id => id !== layerId);
      } else {
        state.visibleLayers.push(layerId);
      }
    },
    
    setVisibleLayers: (state, action) => {
      state.visibleLayers = action.payload;
    },
    
    setSelectedFeature: (state, action) => {
      state.selectedFeature = action.payload;
    },
    
    clearSelectedFeature: (state) => {
      state.selectedFeature = null;
    },
    
    setLayersLoading: (state, action) => {
      state.loadingLayers = action.payload;
    },
    
    setMapBounds: (state, action) => {
      state.mapBounds = action.payload;
    },
    
    setMapInstance: (state, action) => {
      // We don't store the actual map instance in Redux state
      // as it's not serializable, but we set a flag that it's been initialized
      state.mapInitialized = true;
    },
    
    resetMapView: (state) => {
      state.center = DEFAULT_CENTER;
      state.zoom = DEFAULT_ZOOM;
    },
    
    setFilterSettings: (state, action) => {
      state.filterSettings = {
        ...state.filterSettings,
        ...action.payload
      };
    },
    
    resetFilters: (state) => {
      state.filterSettings = initialState.filterSettings;
    }
  }
});

export const {
  setMapView,
  setActiveBasemap,
  toggleLayer,
  setVisibleLayers,
  setSelectedFeature,
  clearSelectedFeature,
  setLayersLoading,
  setMapBounds,
  setMapInstance,
  resetMapView,
  setFilterSettings,
  resetFilters
} = mapSlice.actions;

export default mapSlice.reducer;