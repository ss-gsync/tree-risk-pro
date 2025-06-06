// src/services/api/reportApi.js

const API_BASE_URL = import.meta.env.VITE_API_URL === undefined
  ? 'http://localhost:5000'
  : (import.meta.env.VITE_API_URL === '' ? '' : import.meta.env.VITE_API_URL);

/**
 * Fetch validation queue items
 * @param {Object} filters - Optional filters for the queue
 * @returns {Promise<Array>} - Array of validation items
 */
export const getValidationQueue = async (filters = {}) => {
  try {
    const queryParams = new URLSearchParams(filters).toString();
    const url = `${API_BASE_URL}/api/validation/queue${queryParams ? `?${queryParams}` : ''}`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch validation queue: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching validation queue:', error);
    throw error;
  }
};

/**
 * Update validation item status
 * @param {string} itemId - ID of the validation item
 * @param {string} status - New status (approved, rejected, pending)
 * @param {Object} notes - Optional notes for the validation
 * @returns {Promise<Object>} - Updated validation item
 */
export const updateValidationStatus = async (itemId, status, notes = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/validation/${itemId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ status, notes }),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to update validation status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error updating validation status:', error);
    throw error;
  }
};

/**
 * Generate a report for a property
 * @param {string} propertyId - ID of the property
 * @param {Object} options - Report generation options
 * @returns {Promise<Object>} - Report data including download URL
 */
export const generateReport = async (propertyId, options = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/properties/${propertyId}/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(options),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to generate report: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error generating report:', error);
    throw error;
  }
};

/**
 * Get a list of previously generated reports
 * @param {string} propertyId - Optional property ID to filter reports
 * @returns {Promise<Array>} - Array of report metadata
 */
export const getReports = async (propertyId = null) => {
  try {
    const url = propertyId 
      ? `${API_BASE_URL}/api/properties/${propertyId}/reports` 
      : `${API_BASE_URL}/api/reports`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch reports: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching reports:', error);
    throw error;
  }
};

/**
 * Get validation reports for a specific S2 cell region
 * @param {string} s2CellToken - The S2 cell token for geospatial filtering
 * @returns {Promise<Array>} - Array of validation reports in that region
 */
export const getValidationReportsByS2Cell = async (s2CellToken) => {
  try {
    const url = `${API_BASE_URL}/api/reports/s2/${s2CellToken}`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch validation reports by S2 cell: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching validation reports by S2 cell:', error);
    throw error;
  }
};

/**
 * Link a validation report to an area report
 * @param {string} validationReportId - ID of the validation report
 * @param {string} areaReportId - ID of the area report to link to
 * @returns {Promise<Object>} - Link result
 */
export const linkValidationToAreaReport = async (validationReportId, areaReportId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/reports/${areaReportId}/link`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ 
        validationReportId,
        link_type: 's2_region'
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Failed to link validation to area report: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error linking validation to area report:', error);
    throw error;
  }
};

/**
 * Get all validation reports linked to an area report
 * @param {string} areaReportId - ID of the area report
 * @returns {Promise<Array>} - Array of linked validation reports
 */
export const getLinkedValidationReports = async (areaReportId) => {
  try {
    const url = `${API_BASE_URL}/api/reports/${areaReportId}/validations`;
    
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch linked validation reports: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching linked validation reports:', error);
    throw error;
  }
};