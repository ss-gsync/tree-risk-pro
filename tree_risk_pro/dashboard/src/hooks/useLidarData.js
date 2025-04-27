// src/hooks/useLidarData.js

import { useState, useEffect } from 'react';
import { LidarService } from '../services/api/apiService';
import { processLidarData } from '../services/processing/lidarProcessing';

export const useLidarData = (propertyId = null) => {
  const [lidarData, setLidarData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [processingProgress, setProcessingProgress] = useState(0);

  useEffect(() => {
    if (!propertyId) return;

    const fetchLidarData = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        // Simulate processing progress for UI feedback
        let progress = 0;
        const progressInterval = setInterval(() => {
          progress += 10;
          setProcessingProgress(Math.min(progress, 90));
          if (progress >= 90) clearInterval(progressInterval);
        }, 300);
        
        // Fetch lidar data from the API
        const data = await LidarService.getLidarData(propertyId);
        
        // In a real implementation, we might need to process the raw data further
        // const processedData = await processLidarData(data);
        // setLidarData(processedData);
        
        setLidarData(data);
        setProcessingProgress(100);
        
        clearInterval(progressInterval);
      } catch (err) {
        console.error('Error fetching LiDAR data:', err);
        setError(err.message || 'Failed to load LiDAR data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchLidarData();
  }, [propertyId]);

  // Function to manually reload LiDAR data
  const reloadLidarData = async () => {
    if (!propertyId) return;
    
    try {
      setIsLoading(true);
      setError(null);
      setProcessingProgress(0);
      
      // Simulate processing progress
      let progress = 0;
      const progressInterval = setInterval(() => {
        progress += 15;
        setProcessingProgress(Math.min(progress, 90));
        if (progress >= 90) clearInterval(progressInterval);
      }, 200);
      
      // Fetch and process the data
      const data = await LidarService.getLidarData(propertyId);
      setLidarData(data);
      
      clearInterval(progressInterval);
      setProcessingProgress(100);
    } catch (err) {
      console.error('Error reloading LiDAR data:', err);
      setError(err.message || 'Failed to reload LiDAR data');
    } finally {
      setIsLoading(false);
    }
  };

  return { 
    lidarData, 
    isLoading, 
    error,
    processingProgress,
    reloadLidarData
  };
};