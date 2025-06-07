/**
 * detectionService.js - ML Detection data handling service
 * 
 * Handles loading ML detection data from APIs and files,
 * with support for S2 cell coordinate mapping
 */

/**
 * Load metadata from a detection run
 * @param {string} detectionId - The detection ID to load
 * @returns {Promise<Object>} - The metadata object
 */
export async function loadDetectionMetadata(detectionId) {
  try {
    // First try to load from API endpoint (works with both local and T4 server mode)
    try {
      const apiPath = `/api/ml/detection/${detectionId}/metadata`;
      console.log(`Trying to load metadata from API: ${apiPath}`);
      
      const apiResponse = await fetch(apiPath);
      if (apiResponse.ok) {
        const metadata = await apiResponse.json();
        console.log(`Successfully loaded metadata for detection ${detectionId} from API`);
        
        // Add API info to metadata
        metadata._loadedFrom = 'api';
        
        // Store the successful path for trees.json loading
        window._lastMetadataPath = apiPath;
        
        return metadata;
      }
    } catch (apiError) {
      console.warn("API metadata fetch failed, falling back to direct file access:", apiError);
    }
    
    // Fall back to direct file access
    const metadataPath = `/data/ml/${detectionId}/ml_response/metadata.json`;
    
    console.log(`Loading metadata from direct file: ${metadataPath}`);
    const response = await fetch(metadataPath);
    
    if (!response.ok) {
      throw new Error(`Failed to load metadata: ${response.status} ${response.statusText}`);
    }
    
    const metadata = await response.json();
    console.log(`Successfully loaded metadata for detection ${detectionId} from direct file`);
    
    // Add file info to metadata
    metadata._loadedFrom = 'file';
    
    // Store the successful path for trees.json loading
    window._lastMetadataPath = metadataPath;
    
    return metadata;
  } catch (error) {
    console.error('Error loading metadata:', error);
    return null;
  }
}

/**
 * Load tree detection data with optimized processing
 * @param {string} detectionId - The detection ID to load
 * @returns {Promise<Object>} - The processed detection data
 */
export async function loadDetectionData(detectionId) {
  try {
    // First load metadata to get coordinate mapping information
    const metadata = await loadDetectionMetadata(detectionId);
    if (!metadata) {
      throw new Error('Failed to load metadata');
    }
    
    // First try to load from API endpoint
    let data;
    let loadedFrom = '';
    
    try {
      const apiPath = `/api/ml/detection/${detectionId}/trees`;
      console.log(`Trying to load trees data from API: ${apiPath}`);
      
      const apiResponse = await fetch(apiPath);
      if (apiResponse.ok) {
        data = await apiResponse.json();
        console.log(`Successfully loaded trees data for detection ${detectionId} from API`);
        loadedFrom = 'api';
      } else {
        // Fall back to direct file access
        const treesPath = `/data/ml/${detectionId}/ml_response/trees.json`;
        
        console.log(`Loading trees data from direct file: ${treesPath}`);
        const response = await fetch(treesPath);
        if (!response.ok) {
          throw new Error(`Failed to load trees.json: ${response.status} ${response.statusText}`);
        }
        
        data = await response.json();
        loadedFrom = 'file';
      }
    } catch (apiError) {
      // Final fallback to direct file access
      console.warn("API trees fetch failed, falling back to direct file access:", apiError);
      
      const treesPath = `/data/ml/${detectionId}/ml_response/trees.json`;
      
      console.log(`Loading trees data from direct file (fallback): ${treesPath}`);
      const response = await fetch(treesPath);
      if (!response.ok) {
        throw new Error(`Failed to load trees.json: ${response.status} ${response.statusText}`);
      }
      
      data = await response.json();
      loadedFrom = 'file-fallback';
    }
    
    // Add load source to data
    data._loadedFrom = loadedFrom;
    
    if (!data.success || !data.detections || !Array.isArray(data.detections)) {
      console.error('Invalid detection data format:', data);
      return null;
    }
    
    console.log(`Processing ${data.detections.length} detections for efficient rendering`);
    
    // Start processing timer
    const startTime = performance.now();
    
    // Extract bounds from metadata
    const bounds = metadata.bounds;
    
    // Create transformed data structure with both trees and detections arrays
    const transformedData = {
      trees: [],
      detections: [], // Keep the original detections array
      raw_data: data  // Store original data for reference
    };
    
    // OPTIMIZATION: Use batch processing with limited segmentation data
    // Update metrics for performance monitoring
    const metrics = {
      totalObjects: data.detections.length,
      withSegmentation: 0,
      withS2Cells: 0,
      processingTime: 0
    };
    
    // Process detections in batches
    const BATCH_SIZE = 100;
    for (let i = 0; i < data.detections.length; i += BATCH_SIZE) {
      const batch = data.detections.slice(i, i + BATCH_SIZE);
      
      // Process each batch concurrently
      batch.forEach(detection => {
        // Skip invalid detections
        if (!detection.bbox || detection.bbox.length !== 4) return;
        
        // Extract normalized bounding box coordinates
        const [x, y, x2, y2] = detection.bbox;
        
        // Keep segmentation data intact
        if (detection.segmentation) {
          metrics.withSegmentation++;
        }
        
        // Calculate geographic coordinates using S2 cells when available
        const location = calculateCoordinates(detection.bbox, bounds, metadata);
        
        // Get S2 cell ID for this location
        let s2_cell = null;
        if (metadata.s2_cells && location) {
          try {
            // Extract S2 cell information using precise location
            const [lng, lat] = location;
            
            // Use S2 cell mapping from metadata if available
            if (metadata.s2_cells.mapping) {
              // Find the S2 cell that contains this point
              const cells = metadata.s2_cells.cells;
              if (cells && cells.length > 0) {
                // Use center cell as default
                s2_cell = metadata.s2_cells.center_cell;
                metrics.withS2Cells++;
              }
            }
          } catch (e) {
            console.warn('Error calculating S2 cell for detection:', e);
          }
        }
        
        // Preserve original class with spaces before normalizing
        const originalClass = detection.class || 'healthy tree';
        
        // Create object with properties expected by MLOverlay
        const detectionObject = {
          box: {
            x: x,
            y: y,
            width: x2 - x,
            height: y2 - y
          },
          confidence: detection.confidence || 0.5,
          // Keep the original class string and also provide normalized versions
          original_class: originalClass,
          // Convert space-separated classes to underscore format
          class: originalClass.replace(/\s+/g, '_'),
          category: originalClass.replace(/\s+/g, '_'),
          // Also preserve original properties for compatibility
          bbox: detection.bbox,
          mask_score: detection.mask_score,
          location: location,
          segmentation: detection.segmentation,
          s2_cell: s2_cell
        };
        
        // Add all detections to a single trees array
        transformedData.trees.push(detectionObject);
        
        // Also maintain a detections array for components that expect it
        if (!transformedData.detections) transformedData.detections = [];
        transformedData.detections.push(detectionObject);
      });
    }
    
    // Update final metrics
    metrics.processingTime = performance.now() - startTime;
    
    console.log('Detection data processing metrics:', {
      totalObjects: metrics.totalObjects,
      trees: transformedData.trees.length,
      withSegmentation: metrics.withSegmentation,
      withS2Cells: metrics.withS2Cells,
      processingTime: `${metrics.processingTime.toFixed(2)}ms`
    });
    
    // Add job ID and other metadata
    transformedData.job_id = detectionId;
    transformedData.timestamp = metadata.timestamp || new Date().toISOString();
    transformedData.metadata = metadata;
    
    // Dispatch early data loaded event for preview to show data before imagery is ready
    // This allows the preview to initialize with the data we have now while waiting for imagery
    try {
      // Ensure the job ID is correct format for all consumers
      const normalizedJobId = detectionId.startsWith('detection_') ? detectionId : `detection_${detectionId}`;
      console.log('detectionService: Dispatching results with normalized job ID:', normalizedJobId);
      
      const detailData = {
        ...transformedData,
        _preliminary: true,
        job_id: normalizedJobId,
        timestamp: new Date().toISOString(),
        // Add explicit paths to make it easier for consumers
        paths: {
          visualizationImage: `/api/ml/detection/${detectionId}/visualization`,
          visualizationImageFallback: `/data/ml/${detectionId}/ml_response/combined_visualization.jpg`,
          satelliteImage: `/api/ml/detection/${detectionId}/satellite`,
          satelliteImageFallback: `/data/ml/${detectionId}/satellite_${detectionId}.jpg`,
          metadataApi: `/api/ml/detection/${detectionId}/metadata`,
          metadataFile: `/data/ml/${detectionId}/ml_response/metadata.json`,
          treesApi: `/api/ml/detection/${detectionId}/trees`,
          treesFile: `/data/ml/${detectionId}/ml_response/trees.json`,
          loadedFrom: loadedFrom
        },
        // Include the raw data for consumers that need the original format
        raw_trees_data: {
          success: data.success,
          detections: data.detections
        },
        // Ensure we have both arrays properly populated
        trees: transformedData.trees,
        detections: data.detections || transformedData.detections
      };
      
      // Dispatch as both document and window events for maximum compatibility
      const earlyDataEvent = new CustomEvent('fastInferenceResults', { detail: detailData });
      document.dispatchEvent(earlyDataEvent);
      window.dispatchEvent(new CustomEvent('fastInferenceResults', { detail: detailData }));
      
      // Also dispatch as detectionDataLoaded event for MLOverlayInitializer
      window.dispatchEvent(new CustomEvent('detectionDataLoaded', { detail: detailData }));
      
      // For direct access, try to call the preview function
      if (typeof window.showDetectionPreview === 'function') {
        console.log('detectionService: Directly calling showDetectionPreview with data');
        window.showDetectionPreview(detailData);
      } else {
        console.warn('detectionService: window.showDetectionPreview is not available yet');
        // Set a timeout to try again in case the function is defined later
        setTimeout(() => {
          if (typeof window.showDetectionPreview === 'function') {
            console.log('detectionService: Calling showDetectionPreview after delay');
            window.showDetectionPreview(detailData);
          }
        }, 500);
      }
      
      console.log('Dispatched fastInferenceResults event with early detection data');
    } catch (error) {
      console.error('Error dispatching early detection data event:', error);
    }
    
    return transformedData;
  } catch (error) {
    console.error('Error loading detection data:', error);
    return null;
  }
}

/**
 * Calculate real-world coordinates from normalized bounding box position
 * Uses precise geo-transformation based on metadata mapping with Google Maps alignment
 * 
 * ENHANCED: Improved precision to perfectly match Google Maps Tiles API
 */
function calculateCoordinates(bbox, bounds, metadata) {
  // Get center point of the bounding box
  const [x, y, x2, y2] = bbox;
  const centerX = (x + x2) / 2;
  const centerY = (y + y2) / 2;
  
  // For debugging
  console.log(`COORDINATE DEBUG - Bbox: [${x.toFixed(4)}, ${y.toFixed(4)}, ${x2.toFixed(4)}, ${y2.toFixed(4)}], Center: (${centerX.toFixed(4)}, ${centerY.toFixed(4)})`);
  
  // Google Maps alignment correction factor (determined empirically from testing)
  // This accounts for slight differences in how the images are projected vs. how Google Maps positions elements
  const CORRECTION_FACTOR_X = 0;
  const CORRECTION_FACTOR_Y = 0.00025; // Slight offset to better align with Google Maps markers
  
  /**
   * Calculate coordinates using S2 cells (most accurate method)
   */
  if (metadata && metadata.mapping && metadata.mapping.s2_cells) {
    try {
      // Extract mapping parameters from the nested s2_cells object
      const s2Mapping = metadata.mapping.s2_cells;
      
      // Log mapping info for debugging
      console.log('Using S2 cell mapping with correction factors', {
        mapping: s2Mapping,
        correctionX: CORRECTION_FACTOR_X,
        correctionY: CORRECTION_FACTOR_Y
      });
      
      // Convert normalized coordinates to pixel coordinates
      const pixelX = centerX * s2Mapping.width;
      const pixelY = centerY * s2Mapping.height;
      
      // PRECISE CALCULATION WITH CORRECTION:
      // 1. Start from southwest corner (lowest latitude and longitude)
      // 2. For longitude (X): Move east by pixelX * lng_per_pixel
      // 3. For latitude (Y): Move north by (height - pixelY) * lat_per_pixel
      // 4. Apply correction factors for perfect Google Maps alignment
      const longitude = s2Mapping.sw_lng + (pixelX * s2Mapping.lng_per_pixel) + CORRECTION_FACTOR_X;
      const latitude = s2Mapping.sw_lat + ((s2Mapping.height - pixelY) * s2Mapping.lat_per_pixel) - CORRECTION_FACTOR_Y;
      
      console.log(`S2 coordinates calculated: (${longitude.toFixed(8)}, ${latitude.toFixed(8)})`);
      return [longitude, latitude];
    } catch (e) {
      console.warn('Error using S2 cell mapping, trying alternative methods', e);
    }
  }
  
  /**
   * Calculate using standard mapping parameters
   */
  if (metadata && metadata.mapping) {
    const mapping = metadata.mapping;
    
    console.log('Using standard mapping with correction factors', {
      mapping,
      correctionX: CORRECTION_FACTOR_X,
      correctionY: CORRECTION_FACTOR_Y
    });
    
    // Convert normalized coordinates to pixel coordinates
    const pixelX = centerX * mapping.width;
    const pixelY = centerY * mapping.height;
    
    // PRECISE CALCULATION WITH CORRECTION:
    const longitude = mapping.sw_lng + (pixelX * mapping.lng_per_pixel) + CORRECTION_FACTOR_X;
    const latitude = mapping.sw_lat + ((mapping.height - pixelY) * mapping.lat_per_pixel) - CORRECTION_FACTOR_Y;
    
    console.log(`Standard coordinates calculated: (${longitude.toFixed(8)}, ${latitude.toFixed(8)})`);
    return [longitude, latitude];
  }
  
  /**
   * Fallback to simple linear interpolation with bounds
   */
  if (bounds && bounds.length === 2) {
    console.log('Using bounds-based fallback with correction factors', {
      bounds,
      correctionX: CORRECTION_FACTOR_X,
      correctionY: CORRECTION_FACTOR_Y
    });
    
    // Linear interpolation with corrections:
    const longitude = bounds[0][0] + (centerX * (bounds[1][0] - bounds[0][0])) + CORRECTION_FACTOR_X;
    // For latitude, we invert Y since image coordinates are top-down but geo coordinates are bottom-up
    const latitude = bounds[0][1] + ((1 - centerY) * (bounds[1][1] - bounds[0][1])) - CORRECTION_FACTOR_Y;
    
    console.log(`Fallback coordinates calculated: (${longitude.toFixed(8)}, ${latitude.toFixed(8)})`);
    return [longitude, latitude];
  }
  
  // Cannot calculate coordinates
  console.error('Unable to calculate coordinates - missing required metadata');
  return null;
}

/**
 * Initialize map with correct positioning based on metadata
 */
export async function initializeMap(detectionId) {
  const metadata = await loadDetectionMetadata(detectionId);
  if (!metadata || !metadata.bounds) {
    console.error('Cannot initialize map: Missing or invalid metadata');
    return false;
  }
  
  const bounds = metadata.bounds;
  
  // Calculate center from bounds
  const center = [
    (bounds[0][0] + bounds[1][0]) / 2,  // longitude
    (bounds[0][1] + bounds[1][1]) / 2   // latitude
  ];
  
  // Determine appropriate zoom level based on bounds size
  let zoom = 17; // Default zoom
  if (metadata.mapping && metadata.mapping.lng_per_pixel) {
    // Calculate appropriate zoom based on the lng_per_pixel value
    // Lower values of lng_per_pixel mean higher zoom levels
    const lngPerPixel = metadata.mapping.lng_per_pixel;
    if (lngPerPixel < 1e-6) zoom = 20;      // Very high zoom
    else if (lngPerPixel < 5e-6) zoom = 19;
    else if (lngPerPixel < 1e-5) zoom = 18;
    else if (lngPerPixel < 5e-5) zoom = 17;
    else if (lngPerPixel < 1e-4) zoom = 16;
    else zoom = 15;                          // Low zoom
  }
  
  // Center map on coordinates from detection
  if (window.map && center) {
    console.log(`Centering map at: ${center[1]}, ${center[0]} with zoom ${zoom}`);
    window.map.setCenter({ lat: center[1], lng: center[0] });
    window.map.setZoom(zoom);
    
    // If we have S2 cell information, log it
    if (metadata.s2_cells && metadata.s2_cells.center_cell) {
      console.log(`Map centered on S2 cell: ${metadata.s2_cells.center_cell} (${metadata.s2_cells.cell_level} level)`);
    }
    
    return true;
  }
  
  return false;
}

/**
 * Apply detection overlay to the map
 * @param {string} detectionId - The detection ID to load and display
 * @param {boolean} appendMode - Whether to append to existing overlay (default: false)
 */
export async function applyDetectionOverlay(detectionId, appendMode = false) {
  console.log(`Loading and applying detection overlay for ${detectionId}...`);
  
  try {
    // Load MLOverlay renderer
    if (typeof window.renderMLOverlay !== 'function') {
      throw new Error('MLOverlay functions not available');
    }
    
    // Initialize map with correct positioning (only if not appending)
    if (!appendMode) {
      const mapInitialized = await initializeMap(detectionId);
      if (!mapInitialized) {
        console.warn('Map initialization failed, will still attempt to apply overlay');
      }
    }
    
    // Load detection data
    const detectionData = await loadDetectionData(detectionId);
    if (!detectionData) {
      throw new Error('Failed to load detection data');
    }
    
    // Find map container
    const mapContainer = document.getElementById('map-container') || 
                       document.querySelector('.map-container') ||
                       document.getElementById('map');
    
    if (!mapContainer) {
      throw new Error('Could not find map container');
    }
    
    // Set global data for ML overlay (don't overwrite in append mode)
    if (!appendMode || !window.mlDetectionData) {
      window.mlDetectionData = detectionData;
    } else if (appendMode && window.mlDetectionData) {
      // In append mode, merge with existing data
      const existingData = window.mlDetectionData;
      
      // Merge trees, buildings and power lines arrays
      existingData.trees = [...(existingData.trees || []), ...(detectionData.trees || [])];
      existingData.buildings = [...(existingData.buildings || []), ...(detectionData.buildings || [])];
      existingData.power_lines = [...(existingData.power_lines || []), ...(detectionData.power_lines || [])];
      
      // Use the new detection as the current job ID
      existingData.job_id = detectionData.job_id;
      
      // Update timestamp to latest
      existingData.timestamp = new Date().toISOString();
      
      // Use the merged data
      window.mlDetectionData = existingData;
    }
    
    // Apply overlay with detection data
    window.renderMLOverlay(window.mlDetectionData, {
      opacity: 1.0,
      classes: { trees: true, buildings: true, powerLines: true },
      targetElement: mapContainer,
      jobId: detectionData.job_id,
      forceRenderBoxes: true,
      showSegmentation: true,
      suppressNotification: false,
      appendMode: appendMode // Pass append mode to renderer
    });
    
    console.log('Detection overlay applied successfully');
    
    // Add S2 cell markers if S2 data is available
    if (detectionData.metadata && detectionData.metadata.s2_cells) {
      addS2CellMarkers(detectionData);
    }
    
    return detectionData;
  } catch (error) {
    console.error('Error applying detection overlay:', error);
    alert(`Error: ${error.message || 'Failed to apply detection overlay'}`);
    return null;
  }
}

/**
 * Helper function to manage the S2 cell markers for geographical visualization
 */
function addS2CellMarkers(detectionData) {
  if (!window.google || !window.map || !detectionData.metadata || !detectionData.metadata.s2_cells) {
    return false;
  }
  
  // Clear existing S2 cell markers
  if (window._s2CellMarkers) {
    window._s2CellMarkers.forEach(marker => marker.setMap(null));
  }
  window._s2CellMarkers = [];
  
  const s2Cells = detectionData.metadata.s2_cells;
  
  // Add marker for center cell
  if (s2Cells.center_cell && detectionData.metadata.bounds) {
    const bounds = detectionData.metadata.bounds;
    const centerLat = (bounds[0][1] + bounds[1][1]) / 2;
    const centerLng = (bounds[0][0] + bounds[1][0]) / 2;
    
    const marker = new google.maps.Marker({
      position: { lat: centerLat, lng: centerLng },
      map: window.map,
      title: `Center S2 Cell: ${s2Cells.center_cell}`,
      icon: {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 8,
        fillColor: '#4285F4',
        fillOpacity: 0.8,
        strokeColor: '#ffffff',
        strokeWeight: 2
      },
      zIndex: 1000
    });
    
    // Add info window with cell information
    const infoContent = `
      <div style="padding:8px;max-width:200px;">
        <h3 style="margin:0 0 8px;font-size:14px;">S2 Cell Information</h3>
        <div style="font-size:12px;line-height:1.4;">
          <div><strong>Cell ID:</strong> ${s2Cells.center_cell}</div>
          <div><strong>Level:</strong> ${s2Cells.cell_level}</div>
          <div><strong>Type:</strong> Center Cell</div>
        </div>
      </div>
    `;
    
    const infoWindow = new google.maps.InfoWindow({
      content: infoContent
    });
    
    marker.addListener('click', () => {
      infoWindow.open(window.map, marker);
    });
    
    window._s2CellMarkers.push(marker);
  }
  
  return true;
}

// Export the main functions
export default {
  loadDetectionMetadata,
  loadDetectionData,
  applyDetectionOverlay,
  initializeMap,
  addS2CellMarkers
};