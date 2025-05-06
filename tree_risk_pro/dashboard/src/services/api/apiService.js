// src/services/api/apiService.js

/**
 * Core API service for interacting with the backend
 * 
 * For production with Nginx:
 * - Set VITE_API_URL="" in .env to use relative URLs (/api/...)
 * 
 * For development:
 * - Set VITE_API_URL="http://localhost:5000" in .env
 */
// Using strict equality to check for undefined specifically
// This ensures empty string is used as-is for relative URLs
const API_BASE_URL = import.meta.env.VITE_API_URL === undefined 
  ? 'http://localhost:5000' 
  : (import.meta.env.VITE_API_URL === '' ? '' : import.meta.env.VITE_API_URL);

console.log('API_BASE_URL:', API_BASE_URL);

/**
 * Helper function to build proper API URLs
 * - When API_BASE_URL is empty, returns /api/path (relative URL)
 * - When API_BASE_URL is set, returns full URL: http://example.com/api/path
 */
const buildApiUrl = (path) => {
  // Make sure path has leading slash
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  
  // Return proper URL format
  return API_BASE_URL === '' 
    ? normalizedPath  // Relative URL for production
    : `${API_BASE_URL}${normalizedPath}`; // Full URL for development
};

/**
 * Fetch wrapper with authentication and error handling
 * 
 * - Automatically adds authentication headers from localStorage
 * - Handles 401 errors by redirecting to login
 * - Parses JSON responses and provides error information
 */
const fetchWithErrorHandling = async (url, options = {}) => {
  // If the URL already includes http:// or https://, use it as-is
  // Otherwise, process it with our buildApiUrl function
  const resolvedUrl = url.startsWith('http') ? url : buildApiUrl(url);
  
  console.log('Fetching from URL:', resolvedUrl);
  try {
    // Get authentication token
    const authToken = localStorage.getItem('auth');
    
    // Prepare request headers
    const headers = {
      'Content-Type': 'application/json',
      ...options.headers,
    };
    
    // Add auth header when token is available
    if (authToken) {
      headers['Authorization'] = authToken;
    }
    
    // Execute request
    const response = await fetch(resolvedUrl, {
      ...options,
      headers,
    });

    console.log('Response status:', response.status);
    
    // Handle authentication failures
    if (response.status === 401) {
      localStorage.removeItem('auth');
      console.error('Authentication error - redirecting to login');
      window.location.href = window.location.pathname;
      throw new Error('Authentication required');
    }
    
    // Handle other error responses
    if (!response.ok) {
      try {
        const errorData = await response.json();
        console.error('API Error Data:', errorData);
        
        // Extract the most meaningful error message
        const errorMessage = errorData.error || errorData.message || `HTTP error ${response.status}`;
        throw new Error(errorMessage);
      } catch (parseError) {
        // If we can't parse the JSON, just use the status text
        throw new Error(`Request failed: ${response.statusText || `HTTP error ${response.status}`}`);
      }
    }

    // Parse and return successful response
    const data = await response.json();
    return data;
  } catch (error) {
    console.error(`API Error: ${error.message}`);
    throw error;
  }
};

// We'll determine if we're in test mode from the backend's config
let appConfig = {
  useTestData: false, // Default to production mode
  mode: 'production',
  version: '1.0.0'
};

// Fetch config from backend on init
const initializeConfig = async () => {
  try {
    const config = await fetchWithErrorHandling('api/config');
    appConfig = config;
    console.log('App config loaded:', appConfig);
    return config;
  } catch (error) {
    console.error('Failed to load app config from backend:', error);
    // Keep using default test mode if we can't reach the backend
    return appConfig;
  }
};

// Initialize config right away
initializeConfig();

// Function to check if we're using test data
const isTestMode = () => appConfig.useTestData === true;

// Cache for API responses to avoid repeated requests
const apiCache = {
  properties: null,
  trees: null,
  validation_queue: null
};

/**
 * Property API methods
 */
export const PropertyService = {
  // Get all properties
  getProperties: async () => {
    if (!apiCache.properties) {
      apiCache.properties = await fetchWithErrorHandling('api/properties');
    }
    return apiCache.properties;
  },
  
  // Get a single property by ID
  getProperty: async (propertyId) => {
    // If we already have all properties cached, use that
    if (apiCache.properties) {
      const property = apiCache.properties.find(p => p.id === propertyId);
      if (property) return property;
    }
    
    return fetchWithErrorHandling(`api/properties/${propertyId}`);
  },
  
  // Get trees for a property
  getPropertyTrees: async (propertyId) => {
    // Try to use cached trees if available
    if (apiCache.trees) {
      return apiCache.trees.filter(tree => tree.property_id === propertyId);
    }
    
    return fetchWithErrorHandling(`api/properties/${propertyId}/trees`);
  }
};

/**
 * Tree API methods
 */
export const TreeService = {
  // Get all trees
  getAllTrees: async (filters = {}) => {
    // Don't use cache if filters are provided
    if (!apiCache.trees || Object.keys(filters).length > 0) {
      // Build query params if filters are provided
      const queryParams = new URLSearchParams();
      if (filters.species) {
        queryParams.append('species', filters.species);
      }
      
      const url = `${API_BASE_URL}/api/trees${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
      
      // If no filters, cache the result
      const result = await fetchWithErrorHandling(url);
      if (Object.keys(filters).length === 0) {
        apiCache.trees = result;
      }
      return result;
    }
    
    return apiCache.trees;
  },
  
  // Get a single tree by ID
  getTree: async (treeId) => {
    // Try to use cached trees if available
    if (apiCache.trees) {
      const tree = apiCache.trees.find(t => t.id === treeId);
      if (tree) return tree;
    }
    
    return fetchWithErrorHandling(`${API_BASE_URL}/api/trees/${treeId}`);
  },
  
  // Get trees for a specific area detection
  getTreesForArea: async (areaId) => {
    return fetchWithErrorHandling(`${API_BASE_URL}/api/trees/area/${areaId}`);
  },
  
  // Get list of all unique tree species
  getTreeSpecies: async () => {
    return fetchWithErrorHandling(`${API_BASE_URL}/api/trees/species`);
  },
  
  // Update tree assessment
  updateTreeAssessment: async (treeId, assessmentData) => {
    // Clear cache on updates
    apiCache.trees = null;
    
    return fetchWithErrorHandling(
      `${API_BASE_URL}/api/trees/${treeId}/assessment`,
      {
        method: 'PUT',
        body: JSON.stringify(assessmentData)
      }
    );
  },
  
  // Save validated trees from tree detection process
  saveValidatedTrees: async (areaId, trees) => {
    // Clear cache on updates
    apiCache.trees = null;
    
    return fetchWithErrorHandling(
      `${API_BASE_URL}/api/trees/validate`,
      {
        method: 'POST',
        body: JSON.stringify({
          area_id: areaId,
          trees: trees
        })
      }
    );
  }
};

/**
 * Validation API methods
 */
export const ValidationService = {
  // Get validation queue
  getValidationQueue: async (filters = {}) => {
    // Don't cache filtered requests
    if (Object.keys(filters).length > 0 || !apiCache.validation_queue) {
      const queryParams = new URLSearchParams();
      Object.entries(filters).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          queryParams.append(key, value);
        }
      });
      
      const url = `${API_BASE_URL}/api/validation/queue${queryParams.toString() ? `?${queryParams.toString()}` : ''}`;
      
      // If no filters, cache the result
      const result = await fetchWithErrorHandling(url);
      if (Object.keys(filters).length === 0) {
        apiCache.validation_queue = result;
      }
      
      return result;
    }
    
    return apiCache.validation_queue;
  },
  
  // Update validation status
  updateValidationStatus: async (itemId, status, notes = {}) => {
    // Clear cache on updates
    apiCache.validation_queue = null;
    
    return fetchWithErrorHandling(
      `${API_BASE_URL}/api/validation/${itemId}`,
      {
        method: 'PUT',
        body: JSON.stringify({ status, notes })
      }
    );
  },
  
  // Get LiDAR data for a specific tree in the validation system
  getTreeLidarData: async (treeId) => {
    return fetchWithErrorHandling(`${API_BASE_URL}/api/validation/tree/${treeId}/lidar`);
  }
};

/**
 * LiDAR API methods
 */
export const LidarService = {
  // Get LiDAR data for a property
  getLidarData: async (propertyId) => {
    const queryParams = new URLSearchParams();
    queryParams.append('property_id', propertyId);
    
    const url = `${API_BASE_URL}/api/lidar?${queryParams.toString()}`;
    return fetchWithErrorHandling(url);
  },
  
  // Get detailed LiDAR data for a specific tree
  getTreeLidarData: async (treeId) => {
    return fetchWithErrorHandling(`${API_BASE_URL}/api/lidar/tree/${treeId}`);
  }
};

/**
 * Analytics API methods
 */
export const AnalyticsService = {
  // Get risk distribution
  getRiskDistribution: async () => {
    return fetchWithErrorHandling(`${API_BASE_URL}/api/analytics/risk-distribution`);
  },
  
  // Get property stats
  getPropertyStats: async () => {
    return fetchWithErrorHandling(`${API_BASE_URL}/api/analytics/property-stats`);
  }
};

/**
 * Detection API methods for tree detection using ML pipeline or Gemini AI
 */
export const DetectionService = {
  // Detect trees in an image or area using traditional ML pipeline or Gemini
  detectTrees: async (data) => {
    // Ensure boolean values are explicitly set as booleans
    const cleanData = {...data};
    
    // Debug logging for view data
    if (cleanData.map_view_info && cleanData.map_view_info.viewData) {
      const viewData = cleanData.map_view_info.viewData;
      console.log("DetectionService - viewData keys:", Object.keys(viewData));
      
      // Check if we're using the new backend tiles approach
      if (viewData.useBackendTiles && viewData.coordsInfo) {
        console.log("DetectionService - Using backend tile retrieval with coordinates info");
        // Don't log the full coords string as it could be large
        try {
          const previewLength = Math.min(100, viewData.coordsInfo.length);
          console.log("DetectionService - Coordinates info available:", 
            viewData.coordsInfo.substring(0, previewLength) + (previewLength < viewData.coordsInfo.length ? "..." : ""));
        } catch (e) {
          console.warn("DetectionService - Error processing coordinates info:", e);
        }
      } 
      // Legacy direct image URL approach
      else if (viewData.imageUrl) {
        const imageUrl = viewData.imageUrl;
        const isValidUrl = typeof imageUrl === 'string' && imageUrl.length > 1000;
        const urlPreview = typeof imageUrl === 'string'
          ? `${imageUrl.substring(0, 30)}... (length: ${imageUrl.length})` 
          : 'Not a string';
          
        console.log("DetectionService - imageUrl present:", urlPreview);
        console.log("DetectionService - imageUrl appears valid:", isValidUrl);
        
        if (!isValidUrl) {
          console.warn("DetectionService - imageUrl present but may be invalid (too short)");
        }
      } else {
        console.log("DetectionService - Using coordinates-only mode for detection");
      }
    } else {
      console.error("DetectionService - map_view_info or viewData missing - detection may not work as expected");
    }
    
    // Always use Gemini for tree detection (no longer optional)
    cleanData.use_gemini_for_tree_detection = true;
    console.log('API call - Gemini detection:', cleanData.use_gemini_for_tree_detection);
    
    // Also explicitly set dashboard_request flag
    cleanData.dashboard_request = true;
    
    // IMPORTANT DEBUG: Log the exact structure being sent to the backend
    console.log("FINAL REQUEST DATA:", JSON.stringify({
      map_view_info_keys: cleanData.map_view_info ? Object.keys(cleanData.map_view_info) : 'MISSING',
      viewData_keys: cleanData.map_view_info?.viewData ? Object.keys(cleanData.map_view_info.viewData) : 'MISSING',
      has_imageUrl: cleanData.map_view_info?.viewData?.imageUrl ? true : false,
      imageUrl_length: cleanData.map_view_info?.viewData?.imageUrl ? cleanData.map_view_info.viewData.imageUrl.length : 0,
      map_image_length: cleanData.map_image ? cleanData.map_image.length : 0
    }));
    
    try {
      const result = await fetchWithErrorHandling(
        `${API_BASE_URL}/api/detect-trees`,
        {
          method: 'POST',
          body: JSON.stringify(cleanData)
        }
      );
      
      console.log("Tree detection API result:", result);
      
      // Dispatch custom event with detection results
      if (result && typeof window !== 'undefined') {
        // Ensure result is a proper object (handle string response error)
        let processedResult = result;
        
        // Check if we received a string instead of an object (error case)
        if (typeof result === 'string') {
          console.error("Received string result instead of object:", result);
          processedResult = {
            job_id: null,
            status: 'error',
            message: `API returned unexpected string: ${result.substring(0, 100)}...`,
            trees: [],
            tree_count: 0
          };
        }
        
        // Extract fields, using proper null checks and fallback values
        const event = new CustomEvent('treeDetectionResult', {
          detail: {
            jobId: processedResult.job_id || null,
            status: processedResult.status || 'error',
            message: processedResult.message || processedResult.error || '',
            trees: Array.isArray(processedResult.trees) ? processedResult.trees : [],
            treeCount: processedResult.tree_count || 0
          }
        });
        
        console.log("Dispatching treeDetectionResult event with detected trees:", 
          Array.isArray(processedResult.trees) ? processedResult.trees.length : 0);
        window.dispatchEvent(event);
      }
      
      return result;
    } catch (error) {
      console.error("Tree detection API error:", error);
      
      // Still dispatch event with error information
      if (typeof window !== 'undefined') {
        const event = new CustomEvent('treeDetectionResult', {
          detail: {
            jobId: null,
            status: 'error',
            message: error.message || 'Failed to detect trees',
            trees: [],
            treeCount: 0
          }
        });
        
        window.dispatchEvent(event);
      }
      
      throw error;
    }
  },
  
  // Detect trees using Gemini AI from map view information
  detectTreesWithGemini: async (mapViewInfo, jobId) => {
    return fetchWithErrorHandling(
      `${API_BASE_URL}/api/analytics/gemini/detect-trees`,
      {
        method: 'POST',
        body: JSON.stringify({
          map_view_info: mapViewInfo,
          job_id: jobId || `gemini_${Date.now()}`
        })
      }
    );
  },
  
  // For backwards compatibility - no longer needed with synchronous approach
  // Will be removed in future versions
  getDetectionStatus: async (jobId) => {
    try {
      const result = await fetchWithErrorHandling(`${API_BASE_URL}/api/detection-status/${jobId}`);
      
      // Handle string response error case
      if (typeof result === 'string') {
        console.error("API returned string instead of object:", result);
        return {
          status: 'error',
          message: 'Unexpected API response format',
          tree_count: 0,
          trees: []
        };
      }
      
      // Dispatch event with detection results
      if (result && result.status === 'complete' && Array.isArray(result.trees) && result.trees.length > 0) {
        console.log(`Detection status check for job ${jobId} found ${result.trees.length} trees`);
        
        // Dispatch custom event with detection results
        if (typeof window !== 'undefined') {
          const event = new CustomEvent('treeDetectionResult', {
            detail: {
              jobId: jobId,
              status: 'complete',
              message: result.message || '',
              trees: result.trees || [],
              treeCount: result.tree_count || 0
            }
          });
          
          console.log("Dispatching treeDetectionResult event from status check:", 
            result.trees.length);
          window.dispatchEvent(event);
        }
      } else if (result && result.status === 'error') {
        // Just log the error without dispatching to avoid duplicate alerts
        console.error(`Detection error for job ${jobId}:`, result.message || 'Unknown error');
      }
      
      return result;
    } catch (error) {
      console.error(`Error checking detection status for job ${jobId}:`, error);
      return {
        status: 'error',
        message: `Error checking status: ${error.message}`,
        tree_count: 0,
        trees: []
      };
    }
  }
};

// Export a default object with all services
export default {
  property: PropertyService,
  tree: TreeService,
  validation: ValidationService,
  lidar: LidarService,
  analytics: AnalyticsService,
  detection: DetectionService
};