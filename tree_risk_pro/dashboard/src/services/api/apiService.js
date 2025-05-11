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
    console.log('Getting property data for ID:', propertyId);
    
    // MOCK DATA: Use hardcoded mock data for development
    try {
      // Try to fetch from API endpoint first (for production)
      if (!isTestMode()) {
        try {
          console.log('Attempting to fetch property from API endpoint');
          const response = await fetchWithErrorHandling(`api/properties/${propertyId}`);
          if (response && response.id) {
            console.log('Successfully fetched property data from API');
            return response;
          }
        } catch (apiError) {
          console.warn('API endpoint fetch failed, falling back to mock data:', apiError);
        }
      }
      
      // Hardcoded mock property data
      const mockProperties = [
        {
          "id": "prop-001",
          "address": "2082 Hillcrest Avenue, Dallas, TX 75230",
          "owner": "John Smith",
          "size": 0.5,
          "type": "Residential",
          "center": [-96.80609035002175, 32.87768646372364]
        },
        {
          "id": "prop-002",
          "address": "6121 Preston Hollow Road, Dallas, TX 75230",
          "owner": "Sarah Johnson",
          "size": 0.7,
          "type": "Commercial",
          "center": [-96.75513069600189, 32.894160920551954]
        },
        {
          "id": "prop-003",
          "address": "4367 Twin Hills Avenue, Dallas, TX 75230",
          "owner": "Michael Davis",
          "size": 0.3,
          "type": "Commercial",
          "center": [-96.72320761908205, 32.85882025867078]
        },
        {
          "id": "prop-004",
          "address": "8350 Forest Lane, Dallas, TX 75243",
          "owner": "Jennifer Wilson",
          "size": 0.4,
          "type": "Residential",
          "center": [-96.75909035002175, 32.90768646372364]
        },
        {
          "id": "prop-005",
          "address": "9200 White Rock Trail, Dallas, TX 75238",
          "owner": "Robert Thompson",
          "size": 0.6,
          "type": "Residential",
          "center": [-96.73209035002175, 32.88268646372364]
        }
      ];
      
      // Find the property with the matching ID
      const property = mockProperties.find(p => p.id === propertyId);
      
      if (property) {
        console.log('Found mock property:', property);
        return property;
      }
      
      // If property not found and it has a specific format, try to extract info from the ID
      if (propertyId && propertyId.startsWith('property_area_')) {
        const areaId = propertyId.replace('property_area_', '');
        return {
          id: propertyId,
          address: `Property from Detection ${areaId}`,
          owner: 'Unknown Owner',
          type: 'Residential',
          size: '0.25 acres',
          coordinates: [-96.7, 32.8]
        };
      }
      
      // If property not found, create a generic mock property with the ID
      return {
        id: propertyId,
        address: '123 Mock Street, Dallas, TX',
        owner: 'Mock Owner',
        type: 'Residential',
        size: '0.25 acres',
        coordinates: [-96.7, 32.8]
      };
    } catch (error) {
      console.error('Error retrieving mock property:', error);
      // Return a fallback property object instead of null to prevent further errors
      return {
        id: propertyId || 'unknown',
        address: 'Error retrieving property data',
        owner: 'Unknown',
        type: 'Unknown',
        size: 'Unknown',
        coordinates: [-96.7, 32.8]
      };
    }
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
  // Cache the validation queue result with a timestamp
  _validationCache: {
    data: null,
    timestamp: 0,
    inProgress: false
  },
  
  // Get validation queue with caching to prevent repeated API calls
  getValidationQueue: async (filters = {}) => {
    // If a request is already in progress, wait for it to complete
    if (ValidationService._validationCache.inProgress) {
      // Wait a bit and return the cache when it's ready
      await new Promise(resolve => setTimeout(resolve, 50));
      return ValidationService._validationCache.data || [];
    }
    
    // If we have a recent cache (< 5 seconds old) and no filters, use it
    const now = Date.now();
    const cacheAge = now - ValidationService._validationCache.timestamp;
    if (
      ValidationService._validationCache.data && 
      cacheAge < 5000 && 
      Object.keys(filters).length === 0
    ) {
      return ValidationService._validationCache.data;
    }
    
    // Mark that a request is in progress to prevent duplicate calls
    ValidationService._validationCache.inProgress = true;
    
    // MOCK DATA: Use hardcoded mock data for development
    try {
      // First try to get data from global api cache
      if (apiCache.validation_queue && Object.keys(filters).length === 0) {
        // Update our specific validation cache
        ValidationService._validationCache.data = apiCache.validation_queue;
        ValidationService._validationCache.timestamp = now;
        ValidationService._validationCache.inProgress = false;
        return apiCache.validation_queue;
      }
      
      // Use multiple fallback methods to ensure we can load mock data
      let mockData = [];
      
      // Method 1: Try to fetch from API endpoint first (for production)
      try {
        if (!isTestMode()) {
          const response = await fetchWithErrorHandling('api/validation/queue');
          if (response && Array.isArray(response)) {
            mockData = response;
          }
        }
      } catch (apiError) {
        // API endpoint fetch failed, falling back to mock data
      }
      
      // Method 2: If API fetch failed or we're in test mode, use hardcoded mock data
      if (mockData.length === 0) {
        // Use hardcoded mock data when no real data is available
        mockData = [
          {
            "id": "val-001",
            "tree_id": "tree-001",
            "property_id": "prop-001",
            "tree_species": "Live Oak",
            "tree_height": 45,
            "tree_diameter": 24,
            "tree_condition": "Mature",
            "canopy_density": "Moderate",
            "leaning_angle": "5° SE",
            "lidar_height": 46.5,
            "proximity": "Near power lines",
            "property_type": "Residential",
            "location": {
              "description": "2082 Hillcrest Avenue, Dallas, TX 75230",
              "coordinates": [
                -96.80542964970164,
                32.87769319534349
              ]
            },
            "status": "pending",
            "created_at": "2025-03-08T09:30:00.000Z",
            "updated_at": "2025-03-12T14:07:41.913Z",
            "riskFactors": [
              {
                "type": "split_branch",
                "level": "high",
                "description": "Large split limb (15+ ft) detected"
              },
              {
                "type": "structure_proximity",
                "level": "medium",
                "description": "Canopy overhanging structure"
              },
              {
                "type": "unbalanced_canopy",
                "level": "medium",
                "description": "Asymmetrical canopy development"
              }
            ]
          },
          {
            "id": "val-002",
            "tree_id": "tree-003",
            "property_id": "prop-002",
            "tree_species": "Bald Cypress",
            "tree_height": 60,
            "tree_diameter": 30,
            "tree_condition": "Stressed",
            "canopy_density": "Sparse",
            "leaning_angle": "18° NW",
            "lidar_height": 61.2,
            "proximity": "Near roadway",
            "property_type": "Commercial",
            "location": {
              "description": "6121 Preston Hollow Road, Dallas, TX 75230",
              "coordinates": [
                -96.755,
                32.8937
              ]
            },
            "status": "pending",
            "created_at": "2025-03-09T14:15:00.000Z",
            "updated_at": "2025-03-12T14:07:41.913Z",
            "riskFactors": [
              {
                "type": "lean",
                "level": "medium",
                "description": "Tree leaning 15-25 degrees"
              },
              {
                "type": "root_exposure",
                "level": "medium",
                "description": "Root system partially exposed"
              },
              {
                "type": "canopy_dieback",
                "level": "medium",
                "description": "Upper canopy dieback detected"
              }
            ]
          },
          {
            "id": "val-003",
            "tree_id": "tree-002",
            "property_id": "prop-001",
            "tree_species": "Chinese Pistache",
            "tree_height": 35,
            "tree_diameter": 18,
            "tree_condition": "Healthy",
            "canopy_density": "Dense",
            "leaning_angle": "No Lean",
            "lidar_height": 34.8,
            "proximity": "Near structure",
            "property_type": "Residential",
            "location": {
              "description": "2082 Hillcrest Avenue, Dallas, TX 75230",
              "coordinates": [
                -96.80607053808008,
                32.87694806481281
              ]
            },
            "status": "pending",
            "created_at": "2025-03-10T11:45:00.000Z",
            "updated_at": "2025-03-12T14:07:41.913Z",
            "riskFactors": [
              {
                "type": "foliage_discoloration",
                "level": "low",
                "description": "Minor canopy discoloration detected"
              },
              {
                "type": "canopy_thinning",
                "level": "low",
                "description": "Slight canopy thinning pattern"
              }
            ],
            "inspector_notes": "Tree appears to be responding well to treatment. LiDAR scan indicates healthy overall structure."
          },
          {
            "id": "val-004",
            "tree_id": "tree-005",
            "property_id": "prop-004",
            "tree_species": "Southern Magnolia",
            "tree_height": 38,
            "tree_diameter": 20,
            "tree_condition": "Diseased",
            "canopy_density": "Dense",
            "leaning_angle": "No Lean",
            "lidar_height": 37.5,
            "proximity": "Near fence",
            "property_type": "Residential",
            "location": {
              "description": "8350 Forest Lane, Dallas, TX 75243",
              "coordinates": [
                -96.7595,
                32.9075
              ]
            },
            "status": "pending",
            "created_at": "2025-03-01T13:25:00.000Z",
            "updated_at": "2025-03-05T09:15:30.000Z",
            "riskFactors": [
              {
                "type": "disease",
                "level": "high",
                "description": "Possible fungal infection affecting crown"
              }
            ],
            "inspector_id": "INS-27",
            "inspector_name": "Robert Chen",
            "recommendations": "Treatment with fungicide required. LiDAR shows 20% crown thinning indicative of disease progression. Schedule follow-up inspection in 3 months.",
            "follow_up_date": "2025-06-05"
          },
          {
            "id": "val-005",
            "tree_id": "tree-004",
            "property_id": "prop-003",
            "tree_species": "Red Oak",
            "tree_height": 55,
            "tree_diameter": 28,
            "tree_condition": "Damaged",
            "canopy_density": "Moderate",
            "leaning_angle": "3° SW",
            "lidar_height": 54.5,
            "proximity": "Near construction",
            "property_type": "Commercial",
            "location": {
              "description": "4367 Twin Hills Avenue, Dallas, TX 75230",
              "coordinates": [
                -96.7235,
                32.8585
              ]
            },
            "status": "pending",
            "created_at": "2025-03-11T10:10:00.000Z",
            "updated_at": "2025-03-12T09:30:15.000Z",
            "riskFactors": [
              {
                "type": "root_damage",
                "level": "high",
                "description": "Exposed roots with signs of damage from recent construction"
              },
              {
                "type": "trunk_damage",
                "level": "medium",
                "description": "Bark damage on north side of trunk"
              }
            ],
            "inspector_id": "INS-14",
            "inspector_name": "Lisa Wong",
            "inspection_scheduled": "2025-03-14T13:00:00.000Z",
            "priority": "high",
            "reported_by": "Neighbor"
          }
        ];
        
        // Add a few more mock data entries to ensure there's enough data
        mockData.push(
          {
            "id": "val-007",
            "tree_id": "tree-006",
            "property_id": "prop-005",
            "tree_species": "Cedar Elm",
            "tree_height": 42,
            "tree_diameter": 22,
            "tree_condition": "Mature",
            "canopy_density": "Moderate",
            "leaning_angle": "No Lean",
            "lidar_height": 41.5,
            "proximity": "Near structure",
            "property_type": "Residential",
            "location": {
              "description": "9200 White Rock Trail, Dallas, TX 75238",
              "coordinates": [
                -96.7325,
                32.8825
              ]
            },
            "status": "pending",
            "created_at": "2025-03-11T09:15:00.000Z",
            "updated_at": "2025-03-12T14:07:41.913Z",
            "riskFactors": [
              {
                "type": "cavity",
                "level": "medium",
                "description": "Small cavity in main trunk"
              }
            ]
          },
          {
            "id": "val-008",
            "tree_id": "tree-007",
            "property_id": "prop-005",
            "tree_species": "Pecan",
            "tree_height": 50,
            "tree_diameter": 26,
            "tree_condition": "Healthy",
            "canopy_density": "Moderate",
            "leaning_angle": "2° NE",
            "lidar_height": 49.5,
            "proximity": "Near structure",
            "property_type": "Residential",
            "location": {
              "description": "9200 White Rock Trail, Dallas, TX 75238",
              "coordinates": [
                -96.7315,
                32.8815
              ]
            },
            "status": "pending",
            "created_at": "2025-03-10T16:30:00.000Z",
            "updated_at": "2025-03-12T14:07:41.913Z",
            "riskFactors": [
              {
                "type": "deadwood",
                "level": "low",
                "description": "Minor deadwood in upper canopy"
              }
            ]
          }
        );
      }
      
      // Filter the data based on the provided filters
      let filteredData = [...mockData];
      
      if (filters.status && filters.status !== 'all') {
        filteredData = filteredData.filter(item => item.status === filters.status);
      }
      
      if (filters.riskLevel && filters.riskLevel !== 'all') {
        filteredData = filteredData.filter(item => {
          // Check if any of the risk factors match the requested level
          if (filters.riskLevel === 'high') {
            return item.riskFactors && item.riskFactors.some(factor => factor.level === 'high');
          } else if (filters.riskLevel === 'medium') {
            return item.riskFactors && 
                  item.riskFactors.some(factor => factor.level === 'medium') && 
                  !item.riskFactors.some(factor => factor.level === 'high');
          } else if (filters.riskLevel === 'low') {
            return item.riskFactors && 
                  item.riskFactors.some(factor => factor.level === 'low') && 
                  !item.riskFactors.some(factor => factor.level === 'medium' || factor.level === 'high');
          }
          return true;
        });
      }
      
      // Cache the result if unfiltered
      if (Object.keys(filters).length === 0) {
        apiCache.validation_queue = filteredData;
        
        // Update our specific validation cache
        ValidationService._validationCache.data = filteredData;
        ValidationService._validationCache.timestamp = Date.now();
      }
      
      // Reset the in-progress flag
      ValidationService._validationCache.inProgress = false;
      
      return filteredData;
    } catch (error) {
      console.error('Error in validation queue:', error);
      return [];
    }
  },
  
  // Update validation status
  updateValidationStatus: async (itemId, status, notes = {}) => {
    // Clear all caches on updates
    apiCache.validation_queue = null;
    ValidationService._validationCache.data = null;
    ValidationService._validationCache.timestamp = 0;
    
    // MOCK IMPLEMENTATION: Update the validation item in the cache
    try {
      // For mock implementation, just update the status in memory
      console.log(`Updating validation status for item ${itemId} to ${status}`);
      
      // Try to get the validation data from the cache
      if (apiCache.validation_queue) {
        // Find the validation item in the cache
        const itemIndex = apiCache.validation_queue.findIndex(item => item.id === itemId);
        if (itemIndex >= 0) {
          // Create an updated item
          const updatedItem = {
            ...apiCache.validation_queue[itemIndex],
            status: status,
            updated_at: new Date().toISOString()
          };
          
          // Add notes if provided
          if (notes && Object.keys(notes).length > 0) {
            updatedItem.notes = {
              ...(updatedItem.notes || {}),
              ...notes
            };
          }
          
          // Update the cache
          apiCache.validation_queue[itemIndex] = updatedItem;
          console.log('Updated validation item in cache');
          
          // Return the updated item
          return updatedItem;
        }
      }
      
      // If production mode, try to use the API
      if (!isTestMode()) {
        return fetchWithErrorHandling(
          `${API_BASE_URL}/api/validation/${itemId}`,
          {
            method: 'PUT',
            body: JSON.stringify({ status, notes })
          }
        );
      }
      
      // Return a mock updated item if we couldn't find it in the cache
      return {
        id: itemId,
        status: status,
        updated_at: new Date().toISOString(),
        notes: notes
      };
    } catch (error) {
      console.error('Error updating validation status:', error);
      throw error;
    }
  },
  
  // Save detailed validation data for a tree
  saveValidationData: async (itemId, validationData) => {
    // Clear cache on updates
    apiCache.validation_queue = null;
    
    // Mock implementation for testing
    console.log('Saving validation data for item:', itemId, validationData);
    
    try {
      // In production, use the API
      if (!isTestMode()) {
        return fetchWithErrorHandling(
          `${API_BASE_URL}/api/validation/${itemId}/data`,
          {
            method: 'POST',
            body: JSON.stringify(validationData)
          }
        );
      }
      
      // In test mode, just return the data
      return {
        success: true,
        id: itemId,
        validation_date: validationData.validation_date,
        message: 'Validation data saved successfully'
      };
    } catch (error) {
      console.error('Error saving validation data:', error);
      throw error;
    }
  },
  
  // Get LiDAR data for a specific tree in the validation system
  getTreeLidarData: async (treeId) => {
    // First try the actual API endpoint if not in test mode
    if (!isTestMode()) {
      try {
        const response = await fetchWithErrorHandling(`${API_BASE_URL}/api/validation/tree/${treeId}/lidar`);
        return response;
      } catch (error) {
        console.warn('Error fetching LiDAR data from API, falling back to mock data:', error);
      }
    }
    
    // Mock LiDAR data for testing
    console.log('Returning mock LiDAR data for tree ID:', treeId);
    
    // Generate some variation based on tree ID
    let pointCount = 12500 + (parseInt(treeId.replace(/\D/g, '')) % 10) * 1000;
    let canopyVolume = 450 + (parseInt(treeId.replace(/\D/g, '')) % 20) * 10;
    let trunkVolume = 85 + (parseInt(treeId.replace(/\D/g, '')) % 10) * 5;
    
    return {
      scan_id: `scan_${treeId}_${Date.now()}`,
      tree_id: treeId,
      scan_date: "2025-03-01",
      scan_type: "Aerial LiDAR",
      point_count: pointCount,
      point_density: "25 pts/m²",
      trunk_volume: trunkVolume,
      canopy_volume: canopyVolume,
      canopy_area: canopyVolume / 2.5,
      biomass_estimate: trunkVolume * 7.5,
      carbon_sequestration: trunkVolume * 3.67,
      thumbnail_url: null, // No image for mock data
      scan_url: null, // No 3D model for mock data
      risk_indicators: {
        lean: {
          detected: treeId.includes('3') || treeId.includes('7'),
          severity: treeId.includes('3') ? 'medium' : 'low',
          angle: treeId.includes('3') ? "15°" : "7°",
          direction: "Northwest"
        },
        cavity: {
          detected: treeId.includes('5'),
          severity: 'high',
          location: "Main trunk, 8ft from ground"
        },
        deadwood: {
          detected: treeId.includes('1') || treeId.includes('4'),
          severity: 'medium',
          location: "Upper canopy, multiple branches"
        }
      }
    };
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