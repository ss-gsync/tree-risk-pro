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
  const [useGemini, setUseGemini] = useState(false); // Whether to use Gemini for detection
  const [useDeepForest, setUseDeepForest] = useState(true); // Whether to use DeepForest

  // Check if ML pipeline and which models are enabled from settings
  useEffect(() => {
    try {
      const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
      // Default to true if not specified - ML pipeline is our primary detection method
      const useML = settings?.mlSettings?.useMLPipeline !== false;
      setIsMLEnabled(useML);
      
      // Get model settings
      setUseGemini(settings?.mlSettings?.useGemini === true || settings?.geminiSettings?.useGeminiForDetection === true);
      setUseDeepForest(settings?.mlSettings?.useDeepForest !== false);
    } catch (e) {
      console.error('Error loading ML settings:', e);
      setIsMLEnabled(true); // Default to enabled
      setUseDeepForest(true); // Default to DeepForest enabled
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
        
        // Get model settings
        setUseGemini(settings?.mlSettings?.useGemini === true || settings?.geminiSettings?.useGeminiForDetection === true);
        setUseDeepForest(settings?.mlSettings?.useDeepForest !== false);
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
   * @param {Object} options - Detection options
   * @param {Function} progressCallback - Optional callback for progress updates
   * @returns {Promise<Array>} - Array of detected trees
   */
  const detectTrees = useCallback(async (mapViewInfo, options = {}, progressCallback = null) => {
    if (!isMLEnabled) {
      setDetectionError('ML tree detection is disabled in settings.');
      return [];
    }

    try {
      setIsDetecting(true);
      setDetectionError(null);
      
      // Report initial progress
      if (progressCallback) {
        progressCallback({
          status: 'starting',
          progress: 0,
          message: 'Initializing tree detection'
        });
      }
      
      // Generate a consistent job ID that can be tracked
      const jobId = options.jobId || `detection_${Date.now()}`;
      
      // Prepare settings for detection
      const detectionSettings = {
        useGemini: options.useGemini ?? useGemini,
        useDeepForest: options.useDeepForest ?? useDeepForest,
        includeSegmentation: options.includeSegmentation ?? true
      };
      
      // Determine detection method for logging
      let detectionMethod = 'ml_pipeline';
      if (detectionSettings.useGemini && !detectionSettings.useDeepForest) {
        detectionMethod = 'gemini';
      } else if (detectionSettings.useGemini && detectionSettings.useDeepForest) {
        detectionMethod = 'ml_pipeline_with_gemini';
      }
      
      // Progress update: preparing API call
      if (progressCallback) {
        progressCallback({
          status: 'preparing',
          progress: 10, 
          message: 'Preparing detection request'
        });
      }
      
      // Add detection settings to mapViewInfo
      const enhancedMapViewInfo = {
        ...mapViewInfo,
        use_gemini: detectionSettings.useGemini,
        use_deepforest: detectionSettings.useDeepForest,
        include_segmentation: detectionSettings.includeSegmentation,
        job_id: jobId
      };
      
      // Progress update: API call initiated
      if (progressCallback) {
        progressCallback({
          status: 'requesting',
          progress: 20,
          message: 'Sending detection request to server'
        });
      }
      
      // Call the ML detection API
      const result = await detectTreesFromMapData(enhancedMapViewInfo);
      
      // Progress update: processing results
      if (progressCallback) {
        progressCallback({
          status: 'processing',
          progress: 80,
          message: 'Processing detection results'
        });
      }
      
      if (result.success || !result.error) {
        const trees = result.trees || [];
        
        // Save results to state
        setDetectedTrees(trees);
        setDetectionInfo({
          timestamp: result.timestamp || new Date().toISOString(),
          tree_count: result.tree_count || trees.length,
          coordinates: mapViewInfo.viewData?.center,
          job_id: result.job_id || jobId,
          status: result.status || 'complete',
          ml_response_dir: result.ml_response_dir,
          detection_time: result.detection_time,
          mode: result.mode || detectionMethod,
          method: detectionMethod,
          summary: result.summary || null
        });
        
        // Final progress update: complete
        if (progressCallback) {
          progressCallback({
            status: 'complete',
            progress: 100,
            message: `Detection complete: found ${trees.length} trees`
          });
        }
        
        // Auto-save detection results if enabled
        saveDetectionResults(trees, result, mapViewInfo, jobId, detectionMethod);
        
        return trees;
      } else {
        throw new Error(result.message || result.error || 'Tree detection failed');
      }
    } catch (error) {
      console.error('Error detecting trees with ML pipeline:', error);
      setDetectionError(error.message || 'Tree detection failed');
      
      // Progress update: error
      if (progressCallback) {
        progressCallback({
          status: 'error',
          progress: 0,
          message: error.message || 'Tree detection failed'
        });
      }
      
      return [];
    } finally {
      setIsDetecting(false);
    }
  }, [isMLEnabled, useGemini, useDeepForest]);
  
  /**
   * Helper function to save detection results to localStorage
   */
  const saveDetectionResults = useCallback((trees, result, mapViewInfo, jobId, detectionMethod) => {
    try {
      const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
      if (settings.autoSaveDetectionResults !== false) {
        // Create storage key with timestamp
        const key = `tree_detection_${Date.now()}`;
        
        // Store complete results with metadata
        localStorage.setItem(key, JSON.stringify({
          trees,
          info: {
            timestamp: result.timestamp || new Date().toISOString(),
            tree_count: result.tree_count || trees.length,
            coordinates: mapViewInfo.viewData?.center,
            job_id: jobId,
            status: result.status || 'complete',
            viewData: mapViewInfo.viewData,
            detectionMethod: detectionMethod,
            ml_response_dir: result.ml_response_dir,
            detection_time: result.detection_time
          }
        }));
        
        // Update saved detections index
        const savedDetections = JSON.parse(localStorage.getItem('saved_detections') || '[]');
        savedDetections.push(key);
        
        // Limit to 10 most recent
        if (savedDetections.length > 10) {
          const oldestKey = savedDetections.shift();
          localStorage.removeItem(oldestKey);
        }
        
        // Save updated index
        localStorage.setItem('saved_detections', JSON.stringify(savedDetections));
      }
    } catch (e) {
      console.error('Error saving detection results:', e);
    }
  }, []);

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
    isMLEnabled,
    useGemini,
    useDeepForest,
    setUseGemini,
    setUseDeepForest
  };
};

export default useMapTreeDetection;