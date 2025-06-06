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

/**
 * Detect trees from map data using ML pipeline
 * @param {Object} mapViewInfo - Information about the current map view
 * @returns {Promise<Object>} - Detected trees data
 */
export const detectTreesFromMapData = async (mapViewInfo) => {
  try {
    // Create a unique job ID for this detection task
    const jobId = mapViewInfo.job_id || `ml_detect_${Date.now()}`;
    
    // Standardize map coordinates
    const standardizedMapInfo = standardizeMapCoordinates(mapViewInfo);
    
    // Check if we have a satellite image in base64 format
    const hasImageData = standardizedMapInfo.viewData?.imageUrl?.startsWith('data:image');
    
    // Use FormData for more efficient uploads when we have image data
    if (hasImageData) {
      return await detectTreesWithFormData(standardizedMapInfo, jobId);
    } else {
      // Fall back to JSON if no image data is present
      return await detectTreesWithJson(standardizedMapInfo, jobId);
    }
  } catch (error) {
    console.error('Error detecting trees with ML pipeline:', error);
    throw error;
  }
};

/**
 * Detect trees using FormData to efficiently handle image uploads
 * @param {Object} mapInfo - Map view information with satellite image
 * @param {string} jobId - Unique job ID
 * @returns {Promise<Object>} - Detection results
 */
const detectTreesWithFormData = async (mapInfo, jobId) => {
  try {
    // Create a FormData object for multipart/form-data submission
    const formData = new FormData();
    
    // Extract the base64 image data
    const imageUrl = mapInfo.viewData.imageUrl;
    const imageData = imageUrl.split(',')[1]; // Remove data:image/jpeg;base64, prefix
    
    // Convert base64 to Blob
    const byteCharacters = atob(imageData);
    const byteArrays = [];
    
    for (let offset = 0; offset < byteCharacters.length; offset += 1024) {
      const slice = byteCharacters.slice(offset, offset + 1024);
      
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        byteNumbers[i] = slice.charCodeAt(i);
      }
      
      const byteArray = new Uint8Array(byteNumbers);
      byteArrays.push(byteArray);
    }
    
    // Create a Blob from the byte arrays
    const blob = new Blob(byteArrays, { type: 'image/jpeg' });
    
    // Create a copy of map info without the large image data
    const mapInfoWithoutImage = {
      ...mapInfo,
      viewData: { ...mapInfo.viewData }
    };
    delete mapInfoWithoutImage.viewData.imageUrl;
    
    // Add job ID to metadata
    mapInfoWithoutImage.job_id = jobId;
    
    // Add the data to the FormData
    formData.append('satellite_image', blob, `satellite_${jobId}.jpg`);
    formData.append('map_view_info', JSON.stringify(mapInfoWithoutImage));
    formData.append('job_id', jobId);
    
    // Call the backend endpoint with FormData
    const response = await fetch(`${API_BASE_URL}/api/detection/detect_with_image`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`Tree detection failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error in detectTreesWithFormData:', error);
    throw error;
  }
};

/**
 * Detect trees using standard JSON payload (no image data)
 * @param {Object} mapInfo - Map view information
 * @param {string} jobId - Unique job ID
 * @returns {Promise<Object>} - Detection results
 */
const detectTreesWithJson = async (mapInfo, jobId) => {
  try {
    // Call the backend ML detection endpoint with JSON
    const response = await fetch(`${API_BASE_URL}/api/detection/detect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        map_view_info: mapInfo,
        job_id: jobId
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Tree detection failed: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error in detectTreesWithJson:', error);
    throw error;
  }
};

/**
 * Standardizes map coordinates to ensure consistency between frontend and backend
 * @param {Object} mapInfo - Original map view information object
 * @returns {Object} - Standardized map information
 */
const standardizeMapCoordinates = (mapInfo) => {
  if (!mapInfo) {
    throw new Error('Map information is required for tree detection');
  }
  
  // Create a new object to avoid mutating the original
  const standardized = {
    ...mapInfo,
    viewData: { ...(mapInfo.viewData || {}) }
  };
  
  // Ensure viewData exists
  if (!standardized.viewData) {
    standardized.viewData = {};
  }
  
  // Single source of truth for coordinates
  // Priority: userCoordinates > center > mapControls.coordinates
  let coordinates = null;
  
  if (standardized.viewData.userCoordinates && 
      Array.isArray(standardized.viewData.userCoordinates) && 
      standardized.viewData.userCoordinates.length === 2) {
    // Use userCoordinates as the primary source
    coordinates = standardized.viewData.userCoordinates;
    console.log('Using userCoordinates as the primary source for detection', coordinates);
  } else if (standardized.viewData.center && 
             Array.isArray(standardized.viewData.center) && 
             standardized.viewData.center.length === 2) {
    // Fallback to center coordinates
    coordinates = standardized.viewData.center;
    console.log('Falling back to center coordinates for detection', coordinates);
  } else if (standardized.mapControls && 
             standardized.mapControls.coordinates && 
             Array.isArray(standardized.mapControls.coordinates) && 
             standardized.mapControls.coordinates.length === 2) {
    // Last resort: mapControls.coordinates
    coordinates = standardized.mapControls.coordinates;
    console.log('Using mapControls coordinates as last resort for detection', coordinates);
  }
  
  if (!coordinates) {
    console.error('ERROR: No valid coordinates found in map data');
    throw new Error('Missing valid coordinates for tree detection. Please try again.');
  }
  
  // Standardize coordinate fields to ensure backend can find them
  standardized.viewData.center = coordinates;
  standardized.viewData.userCoordinates = coordinates;
  
  // Ensure zoom level is consistent
  if (!standardized.viewData.zoom && 
      standardized.mapControls && 
      standardized.mapControls.zoom) {
    standardized.viewData.zoom = standardized.mapControls.zoom;
  }
  
  // Provide minimal debugging log
  console.log('Standardized coordinates for detection:', {
    coordinates,
    zoom: standardized.viewData.zoom
  });
  
  return standardized;
};