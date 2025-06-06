// src/store/index.js
import { configureStore } from '@reduxjs/toolkit';
import mapReducer from '../components/visualization/MapView/mapSlice';

// Create the Redux store
export const store = configureStore({
  reducer: {
    map: mapReducer,
    // Add additional reducers as needed
    // trees: treesReducer,
    // properties: propertiesReducer,
    // validation: validationReducer,
    // analytics: analyticsReducer
  },
  middleware: (getDefaultMiddleware) => 
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore non-serializable data in certain actions
        ignoredActions: ['map/setSelectedFeature'],
        ignoredPaths: ['map.selectedFeature']
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
});

export default store;