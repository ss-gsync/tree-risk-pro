// src/services/api/geminiService.js

/**
 * Gemini API service for tree analysis
 * Note: This service only provides functions to wrap backend Gemini API calls,
 * avoiding direct calls to Gemini from the frontend for security and CORS reasons.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL === undefined
  ? 'http://localhost:5000'
  : (import.meta.env.VITE_API_URL === '' ? '' : import.meta.env.VITE_API_URL);

/**
 * Analyze a tree using Gemini AI (through the backend)
 * @param {Object} treeData - The tree data to analyze
 * @returns {Promise<Object>} - Analysis results
 */
export const analyzeTreeWithGemini = async (treeData) => {
  try {
    // If we have a tree ID, use the backend API endpoint
    if (treeData.id) {
      const response = await fetch(`${API_BASE_URL}/api/analytics/tree/${treeData.id}/analyze`);
      
      if (!response.ok) {
        throw new Error(`Tree analysis failed: ${response.status}`);
      }
      
      return await response.json();
    } else {
      // If we don't have a tree ID, send the full tree data for analysis
      const response = await fetch(`${API_BASE_URL}/api/analytics/tree/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(treeData),
      });
      
      if (!response.ok) {
        throw new Error(`Tree analysis failed: ${response.status}`);
      }
      
      return await response.json();
    }
  } catch (error) {
    console.error('Error analyzing tree with Gemini:', error);
    throw error;
  }
};

/**
 * Detect trees from map data using Gemini AI
 * @param {Object} mapViewInfo - Information about the current map view
 * @returns {Promise<Object>} - Detected trees data
 */
export const detectTreesFromMapData = async (mapViewInfo) => {
  try {
    // Create a unique job ID for this detection task
    const jobId = `gemini_${Date.now()}`;
    
    // Call the backend Gemini detection endpoint
    const response = await fetch(`${API_BASE_URL}/api/analytics/gemini/detect-trees`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        map_view_info: mapViewInfo,
        job_id: jobId
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Tree detection failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error detecting trees with Gemini:', error);
    throw error;
  }
};

// Export the Gemini API service functions
export default {
  analyzeTree: analyzeTreeWithGemini,
  detectTrees: detectTreesFromMapData
};