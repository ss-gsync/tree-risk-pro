// src/services/api/treeApi.js

const API_BASE_URL = import.meta.env.VITE_API_URL === undefined
  ? 'http://localhost:5000'
  : (import.meta.env.VITE_API_URL === '' ? '' : import.meta.env.VITE_API_URL);

/**
 * Fetch all trees for a property
 * @param {string} propertyId - ID of the property
 * @returns {Promise<Array>} - Array of tree data
 */
export const getTreesByProperty = async (propertyId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/properties/${propertyId}/trees`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch trees: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching trees:', error);
    throw error;
  }
};

/**
 * Fetch detailed data for a specific tree
 * @param {string} treeId - ID of the tree
 * @returns {Promise<Object>} - Tree data including risk assessment
 */
export const getTreeDetails = async (treeId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/trees/${treeId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch tree details: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching tree details:', error);
    throw error;
  }
};

/**
 * Update tree risk assessment data
 * @param {string} treeId - ID of the tree
 * @param {Object} assessmentData - Updated assessment data
 * @returns {Promise<Object>} - Updated tree data
 */
export const updateTreeAssessment = async (treeId, assessmentData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/trees/${treeId}/assessment`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(assessmentData),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to update tree assessment: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error updating tree assessment:', error);
    throw error;
  }
};

/**
 * Flag a tree for urgent attention
 * @param {string} treeId - ID of the tree
 * @param {Object} flagData - Flag reason and details
 * @returns {Promise<Object>} - Updated tree data
 */
export const flagTree = async (treeId, flagData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/trees/${treeId}/flag`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(flagData),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to flag tree: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error flagging tree:', error);
    throw error;
  }
};