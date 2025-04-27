// src/services/api/analyticsApi.js

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

/**
 * Fetch analytics data from the BigQuery integration
 * @param {string} queryType - Type of query to run
 * @param {Object} params - Optional query parameters
 * @returns {Promise<Object>} - Query results
 */
export const fetchAnalytics = async (queryType, params = {}) => {
  try {
    // Build query string from parameters
    const queryParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      queryParams.append(key, value);
    });
    
    const queryString = queryParams.toString() ? `?${queryParams.toString()}` : '';
    const url = `${API_BASE_URL}/api/analytics/query/${queryType}${queryString}`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Analytics query failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching analytics:', error);
    throw error;
  }
};

/**
 * Analyze a tree using Gemini AI
 * @param {string} treeId - ID of the tree to analyze
 * @returns {Promise<Object>} - Analysis results
 */
export const analyzeTreeWithGemini = async (treeId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analytics/tree/${treeId}/analyze`);
    
    if (!response.ok) {
      throw new Error(`Tree analysis failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error analyzing tree:', error);
    throw error;
  }
};

/**
 * Generate a property report using Gemini AI
 * @param {string} propertyId - ID of the property
 * @param {Object} options - Report generation options
 * @returns {Promise<Object>} - Generated report
 */
export const generatePropertyReport = async (propertyId, options = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analytics/property/${propertyId}/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options),
    });
    
    if (!response.ok) {
      throw new Error(`Report generation failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error generating report:', error);
    throw error;
  }
};

/**
 * Sync local data to BigQuery for analytics
 * @param {string} entityType - Type of entity to sync ('trees', etc.)
 * @param {number} limit - Maximum number of records to sync
 * @returns {Promise<Object>} - Sync results
 */
export const syncToBigQuery = async (entityType = 'trees', limit = 1000) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analytics/sync/bigquery`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ entityType, limit }),
    });
    
    if (!response.ok) {
      throw new Error(`BigQuery sync failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error syncing to BigQuery:', error);
    throw error;
  }
};

/**
 * Get geospatial data for the map with analytics overlay
 * @param {Object} bounds - Map bounds {north, south, east, west}
 * @param {Object} filters - Optional filters for the data
 * @returns {Promise<Object>} - GeoJSON data for the map
 */
export const getMapAnalyticsData = async (bounds, filters = {}) => {
  try {
    // Build query string from bounds and filters
    const queryParams = new URLSearchParams({
      north: bounds.north,
      south: bounds.south,
      east: bounds.east,
      west: bounds.west,
      ...filters
    });
    
    const url = `${API_BASE_URL}/api/analytics/map/data?${queryParams.toString()}`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Map data query failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching map analytics data:', error);
    throw error;
  }
};