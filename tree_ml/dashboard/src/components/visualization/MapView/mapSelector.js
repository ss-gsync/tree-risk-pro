// src/components/visualization/MapView/mapSelector.js

// Basic selectors
export const selectMapCenter = (state) => state.map.center;
export const selectMapZoom = (state) => state.map.zoom;
export const selectVisibleLayers = (state) => state.map.visibleLayers;
export const selectSelectedProperty = (state) => state.map.selectedProperty;
export const selectActiveBasemap = (state) => state.map.activeBasemap;

// Derived selectors
export const selectIsLayerVisible = (state, layerId) => 
  state.map.visibleLayers.includes(layerId);

export const selectMapView = (state) => ({
  center: state.map.center,
  zoom: state.map.zoom,
});