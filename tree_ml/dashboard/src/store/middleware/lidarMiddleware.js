// src/store/middleware/lidarMiddleware.js

import { processLidarData } from '../../services/processing/lidarProcessing';

// Middleware to handle LiDAR data processing
const lidarMiddleware = store => next => async action => {
  // Process the action first
  const result = next(action);
  
  // Check if this is a LiDAR data loading action
  if (action.type === 'LOAD_LIDAR_DATA') {
    try {
      // Fetch LiDAR data
      const response = await fetch(import.meta.env.VITE_LIDAR_ENDPOINT || '/api/lidar');
      const rawData = await response.arrayBuffer();
      
      // Process the data
      const processedData = await processLidarData(rawData);
      
      // Dispatch success action with processed data
      store.dispatch({
        type: 'LIDAR_DATA_LOADED',
        payload: processedData
      });
    } catch (error) {
      // Dispatch error action
      store.dispatch({
        type: 'LIDAR_DATA_ERROR',
        payload: error.message
      });
    }
  }
  
  return result;
};

export default lidarMiddleware;