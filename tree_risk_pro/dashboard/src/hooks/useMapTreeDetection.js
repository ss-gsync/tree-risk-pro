// src/hooks/useMapTreeDetection.js

import { useState, useCallback, useEffect } from 'react';
import { detectTreesFromMapData } from '../services/api/treeApi';

/**
 * Custom hook for detecting trees from map data using ML pipeline
 * 
 * This hook provides functionality to detect trees from a given map view
 * using the ML detection service, with state management for loading, results,
 * and error handling.
 * 
 * @returns {Object} - Tree detection state and methods
 */
export const useMapTreeDetection = () => {
  // State for detection process
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionError, setDetectionError] = useState(null);
  const [detectedTrees, setDetectedTrees] = useState([]);
  const [detectionInfo, setDetectionInfo] = useState(null);
  const [isMLEnabled, setIsMLEnabled] = useState(true); // Default to true since we're using ML pipeline

  // Check if ML pipeline is enabled from settings
  useEffect(() => {
    try {
      const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
      // Default to true if not specified - ML pipeline is our primary detection method
      const useML = settings?.mlSettings?.useMLPipeline !== false;
      setIsMLEnabled(useML);
    } catch (e) {
      console.error('Error loading ML settings:', e);
      setIsMLEnabled(true); // Default to enabled
    }
  }, []);

  // Listen for settings changes
  useEffect(() => {
    const handleStorageChange = () => {
      try {
        const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
        // Default to true if not specified
        const useML = settings?.mlSettings?.useMLPipeline !== false;
        setIsMLEnabled(useML);
      } catch (e) {
        console.error('Error updating ML settings:', e);
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  /**
   * Detect trees from map data
   * 
   * @param {Object} mapViewInfo - Information about the current map view
   * @returns {Promise<Array>} - Array of detected trees
   */
  const detectTrees = useCallback(async (mapViewInfo) => {
    if (!isMLEnabled) {
      setDetectionError('ML tree detection is disabled in settings.');
      return [];
    }

    try {
      setIsDetecting(true);
      setDetectionError(null);
      
      if (!mapViewInfo || (!mapViewInfo.viewData && !mapViewInfo.bounds)) {
        throw new Error('Map view information is required for tree detection');
      }
      
      // Call the ML detection API
      const result = await detectTreesFromMapData(mapViewInfo);
      
      if (result.success || !result.error) {
        const trees = result.trees || [];
        setDetectedTrees(trees);
        setDetectionInfo({
          timestamp: result.timestamp || new Date().toISOString(),
          tree_count: result.tree_count || trees.length,
          center: result.center || mapViewInfo.viewData?.center,
          job_id: result.job_id,
          status: result.status || 'complete',
          ml_response_dir: result.ml_response_dir,
          detection_time: result.detection_time
        });
        
        // Auto-save functionality for detection results
        try {
          const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
          if (settings.autoSaveDetectionResults !== false) {
            // Save detection results to local storage with a timestamp key
            const key = `tree_detection_${Date.now()}`;
            localStorage.setItem(key, JSON.stringify({
              trees,
              info: {
                timestamp: result.timestamp || new Date().toISOString(),
                tree_count: result.tree_count || trees.length,
                center: result.center || mapViewInfo.viewData?.center,
                job_id: result.job_id,
                status: result.status || 'complete',
                viewData: mapViewInfo.viewData,
                detectionMethod: 'ml_pipeline',
                ml_response_dir: result.ml_response_dir,
                detection_time: result.detection_time
              }
            }));
            
            // Keep track of saved detections
            const savedDetections = JSON.parse(localStorage.getItem('saved_detections') || '[]');
            savedDetections.push(key);
            // Limit to 10 most recent
            if (savedDetections.length > 10) {
              const oldestKey = savedDetections.shift();
              localStorage.removeItem(oldestKey);
            }
            localStorage.setItem('saved_detections', JSON.stringify(savedDetections));
          }
        } catch (e) {
          console.error('Error saving detection results:', e);
        }
        
        return trees;
      } else {
        throw new Error(result.message || result.error || 'Tree detection failed');
      }
    } catch (error) {
      console.error('Error detecting trees with ML pipeline:', error);
      setDetectionError(error.message || 'Tree detection failed');
      return [];
    } finally {
      setIsDetecting(false);
    }
  }, [isMLEnabled]);

  /**
   * Load a previously saved detection result
   * 
   * @param {string} key - The storage key for the saved detection
   * @returns {boolean} - Success status
   */
  const loadSavedDetection = useCallback((key) => {
    try {
      const savedData = localStorage.getItem(key);
      if (!savedData) return false;
      
      const data = JSON.parse(savedData);
      setDetectedTrees(data.trees || []);
      setDetectionInfo(data.info || null);
      return true;
    } catch (e) {
      console.error('Error loading saved detection:', e);
      return false;
    }
  }, []);

  /**
   * Get a list of saved detections
   * 
   * @returns {Array} - List of saved detection keys and metadata
   */
  const getSavedDetections = useCallback(() => {
    try {
      const savedKeys = JSON.parse(localStorage.getItem('saved_detections') || '[]');
      return savedKeys.map(key => {
        try {
          const savedData = JSON.parse(localStorage.getItem(key) || '{}');
          return {
            key,
            timestamp: savedData.info?.timestamp || key.split('_').pop(),
            tree_count: savedData.info?.tree_count || (savedData.trees || []).length,
            center: savedData.info?.center
          };
        } catch (e) {
          return { key, error: true };
        }
      }).filter(item => !item.error);
    } catch (e) {
      console.error('Error getting saved detections:', e);
      return [];
    }
  }, []);

  /**
   * Clear the current detection results
   */
  const clearDetection = useCallback(() => {
    setDetectedTrees([]);
    setDetectionInfo(null);
    setDetectionError(null);
  }, []);

  return {
    detectTrees,
    clearDetection,
    loadSavedDetection,
    getSavedDetections,
    isDetecting,
    detectionError,
    detectedTrees,
    detectionInfo,
    isMLEnabled
  };
};

export default useMapTreeDetection;