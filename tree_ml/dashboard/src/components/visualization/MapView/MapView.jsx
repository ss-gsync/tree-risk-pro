// src/components/visualization/MapView/MapView.jsx

import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle, lazy, Suspense } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { ErrorBoundary } from '../../../App';
import { setMapView, setSelectedFeature } from './mapSlice';
import { useGoogleMapsApi } from '../../../hooks/useGoogleMapsApi';
import { store } from '../../../store';
import { TreeService } from '../../../services/api/apiService';
import DetectionMode from '../Detection/DetectionMode';
import * as MLOverlayModule from '../Detection/MLOverlay';
import MLOverlayInitializer from '../Detection/MLOverlayInitializer';
import LocationInfo from './LocationInfo';

// Get the renderMLOverlay function from the module
const { renderMLOverlay } = MLOverlayModule;

// Direct rendering component for detection overlay
const DirectDetectionOverlay = ({ validationData }) => {
  // Use effect to render detection overlay when validationData changes
  React.useEffect(() => {
    // Only proceed if we have validationData
    if (!validationData) return;
    
    console.log("DirectDetectionOverlay: Rendering with validation data:", {
      hasResponseDir: !!validationData.ml_response_dir,
      hasTreesArray: !!validationData.trees,
      treeCount: validationData.trees?.length,
      timestamp: Date.now()
    });
    
    // Normalize data for visualization
    const normalizeData = () => {
      // First try to use trees directly from validationData
      if (validationData.trees && validationData.trees.length > 0) {
        // Process trees to ensure boxes are properly structured
        const processedTrees = validationData.trees.map(tree => {
          if (!tree.box && tree.bounds) {
            // Use bounds if available
            tree.box = {
              x: tree.bounds[0],
              y: tree.bounds[1],
              width: tree.bounds[2] - tree.bounds[0],
              height: tree.bounds[3] - tree.bounds[1]
            };
          } else if (!tree.box && tree.x !== undefined && tree.y !== undefined) {
            // Use x,y coordinates if available
            tree.box = {
              x: tree.x,
              y: tree.y,
              width: tree.width || 0.05,
              height: tree.height || 0.05
            };
          } else if (tree.box && typeof tree.box === 'string') {
            // Parse string box
            try {
              tree.box = JSON.parse(tree.box);
            } catch (e) {
              console.error("DirectDetectionOverlay: Failed to parse box string", tree.box);
              tree.box = { x: 0.5, y: 0.5, width: 0.05, height: 0.05 };
            }
          } else if (!tree.box) {
            // Create default box
            console.warn("DirectDetectionOverlay: Tree missing box, creating default");
            tree.box = { x: 0.5, y: 0.5, width: 0.05, height: 0.05 };
          }
          return tree;
        });
        
        return {
          trees: processedTrees,
          buildings: validationData.buildings || [],
          power_lines: validationData.powerLines || []
        };
      }
      
      // Fallback to any data source we can find
      return window.mlDetectionData || null;
    };
    
    // Get the normalized data
    const detectionData = normalizeData();
    
    // Debug the final data we're using
    console.log("DirectDetectionOverlay: Processed data", {
      treeCount: detectionData?.trees?.length || 0,
      buildingCount: detectionData?.buildings?.length || 0,
      powerLineCount: detectionData?.power_lines?.length || 0,
      hasBoxes: detectionData?.trees?.[0]?.box ? true : false
    });
    
    // Function to render detection boxes directly without depending on StandaloneDetectionOverlay
    const renderDirectly = (container, data) => {
      if (!container || !data || !data.trees || data.trees.length === 0) {
        console.error("DirectDetectionOverlay: Cannot render directly - missing container or data");
        return false;
      }
      
      console.log("DirectDetectionOverlay: Rendering directly with container:", 
        container.id || container.className || "unnamed");
      
      // Create overlay container
      const overlay = document.createElement('div');
      overlay.id = 'direct-detection-overlay-' + Date.now();
      overlay.style.position = 'absolute';
      overlay.style.top = '0';
      overlay.style.left = '0';
      overlay.style.width = '100%';
      overlay.style.height = '100%';
      overlay.style.pointerEvents = 'none';
      overlay.style.zIndex = '9999'; // Very high z-index to ensure visibility
      
      // Add border for debugging
      overlay.style.border = '4px solid rgba(255, 0, 0, 0.2)';
      
      // Remove any existing overlays
      const existingOverlays = container.querySelectorAll('[id^="direct-detection-overlay"]');
      existingOverlays.forEach(el => el.parentNode.removeChild(el));
      
      // Add boxes for trees
      data.trees.forEach((tree, index) => {
        if (!tree.box) {
          console.warn("DirectDetectionOverlay: Tree missing box data:", tree);
          return;
        }
        
        // Create box element
        const box = document.createElement('div');
        box.className = 'detection-box tree-box';
        box.style.position = 'absolute';
        box.style.left = (tree.box.x * 100) + '%';
        box.style.top = (tree.box.y * 100) + '%';
        box.style.width = (tree.box.width * 100) + '%';
        box.style.height = (tree.box.height * 100) + '%';
        box.style.border = '5px solid rgba(40, 167, 69, 0.9)';
        box.style.outline = '2px solid white';
        box.style.backgroundColor = 'rgba(40, 167, 69, 0.15)';
        box.style.boxSizing = 'border-box';
        box.style.zIndex = '10000';
        box.style.pointerEvents = 'none';
        box.style.boxShadow = '0 0 8px rgba(40, 167, 69, 0.7)';
        
        // Add label to some trees
        if (index % 3 === 0) {
          const label = document.createElement('div');
          label.style.position = 'absolute';
          label.style.top = '-22px';
          label.style.left = '0';
          label.style.backgroundColor = 'rgba(40, 167, 69, 0.9)';
          label.style.color = 'white';
          label.style.padding = '3px 8px';
          label.style.fontSize = '12px';
          label.style.fontWeight = 'bold';
          label.style.borderRadius = '4px';
          label.style.whiteSpace = 'nowrap';
          label.style.boxShadow = '0 1px 3px rgba(0,0,0,0.3)';
          label.style.zIndex = '10001';
          
          const confidence = tree.confidence ? ` ${Math.round(tree.confidence * 100)}%` : '';
          label.textContent = 'Tree' + confidence;
          
          box.appendChild(label);
        }
        
        overlay.appendChild(box);
      });
      
      // Add counter
      const counter = document.createElement('div');
      counter.style.position = 'absolute';
      counter.style.bottom = '20px';
      counter.style.right = '20px';
      counter.style.backgroundColor = 'rgba(0, 0, 0, 0.75)';
      counter.style.color = 'white';
      counter.style.padding = '10px 16px';
      counter.style.borderRadius = '6px';
      counter.style.fontSize = '14px';
      counter.style.fontWeight = 'bold';
      counter.style.zIndex = '10001';
      counter.style.boxShadow = '0 4px 10px rgba(0,0,0,0.3)';
      counter.textContent = `${data.trees.length} objects detected`;
      
      overlay.appendChild(counter);
      
      // Add to container
      container.appendChild(overlay);
      
      // Show success notification
      const notification = document.createElement('div');
      notification.style.position = 'fixed';
      notification.style.top = '20px';
      notification.style.left = '50%';
      notification.style.transform = 'translateX(-50%)';
      notification.style.backgroundColor = 'rgba(16, 185, 129, 0.95)';
      notification.style.color = 'white';
      notification.style.padding = '12px 20px';
      notification.style.borderRadius = '6px';
      notification.style.zIndex = '10000';
      notification.style.fontSize = '15px';
      notification.style.fontWeight = 'bold';
      notification.style.boxShadow = '0 4px 12px rgba(0,0,0,0.2)';
      notification.textContent = `Detection complete: ${data.trees.length} objects found`;
      
      document.body.appendChild(notification);
      
      setTimeout(() => {
        if (notification.parentNode) {
          notification.parentNode.removeChild(notification);
        }
      }, 4000);
      
      console.log("DirectDetectionOverlay: Direct rendering completed with", data.trees.length, "objects");
      return true;
    };
    
    // This function triggers the center preview without collapsing sidebar
    const createDetectionPreview = (data) => {
      // Trigger center pane preview to show detection results
      window.dispatchEvent(new CustomEvent('showCenterPanePreview', {
        detail: { data }
      }));
      
      // Return false to prevent any legacy DOM manipulation
      return false;
    };
    
    // Simplified rendering to use center preview only
    const attemptRendering = () => {
      console.log("DirectDetectionOverlay: Using center pane preview for detection results", {
        hasData: !!detectionData,
        treeCount: detectionData?.trees?.length || 0,
        buildingCount: detectionData?.buildings?.length || 0,
        powerLineCount: detectionData?.power_lines?.length || 0,
        hasDetections: !!detectionData?.detections,
        detectionCount: detectionData?.detections?.length || 0
      });
      
      // Make sure the center preview is displayed
      console.log("Showing center preview for detection results");
      
      // Create temporary notification to confirm event dispatch
      const notify = document.createElement('div');
      notify.style.position = 'fixed';
      notify.style.top = '40px';
      notify.style.left = '50%';
      notify.style.transform = 'translateX(-50%)';
      notify.style.backgroundColor = 'rgba(0, 128, 255, 0.8)';
      notify.style.color = 'white';
      notify.style.padding = '10px 20px';
      notify.style.borderRadius = '4px';
      notify.style.zIndex = '99999';
      notify.style.fontWeight = 'bold';
      notify.textContent = `Dispatching events for ${detectionData?.detections?.length || detectionData?.trees?.length || 0} objects`;
      document.body.appendChild(notify);
      
      setTimeout(() => {
        if (notify.parentNode) {
          notify.parentNode.removeChild(notify);
        }
      }, 5000);
      
      // DIRECT APPROACH: If renderMLOverlay is available, call it directly
      if (typeof window.renderMLOverlay === 'function' && detectionData) {
        console.log("DirectDetectionOverlay: Using direct renderMLOverlay call for immediate rendering");
        
        try {
          // Prepare data in the format MLOverlay expects
          let mlData = detectionData;
          
          // If we have detections array in the right format, ensure proper formatting
          if (detectionData.detections && Array.isArray(detectionData.detections)) {
            console.log(`DirectDetectionOverlay: Found ${detectionData.detections.length} detections, using directly`);
            // We already have the right format, just add any missing fields
            if (!mlData.job_id && validationData && validationData.jobId) {
              mlData.job_id = validationData.jobId;
            }
          } 
          // Otherwise, convert trees array to detections format if needed
          else if (detectionData.trees && Array.isArray(detectionData.trees)) {
            console.log(`DirectDetectionOverlay: Converting ${detectionData.trees.length} trees to detections format`);
            
            // Create detections array from trees
            const detections = detectionData.trees.map(tree => {
              // Create detection object with required fields
              return {
                bbox: tree.box ? [tree.box.x, tree.box.y, tree.box.x + tree.box.width, tree.box.y + tree.box.height] : 
                      tree.bbox ? tree.bbox : [0.4, 0.4, 0.6, 0.6],
                confidence: tree.confidence || 0.9,
                class: tree.class || tree.species || 'Healthy Tree'
              };
            });
            
            // Create new data object with detections array
            mlData = {
              detections: detections,
              job_id: validationData?.jobId || 'direct-render',
              metadata: {
                job_id: validationData?.jobId || 'direct-render',
                detection_count: detections.length
              }
            };
          }
          
          // Call the global function directly
          window.renderMLOverlay(mlData, {
            opacity: 1.0,
            forceRenderBoxes: true,
            jobId: validationData?.jobId || 'direct-render',
            debug: true
          });
          
          console.log("DirectDetectionOverlay: Direct call to renderDetectionOverlay completed");
        } catch (e) {
          console.error("Error directly calling renderDetectionOverlay:", e);
        }
      }
      
      // Use React-based center preview component
      window.dispatchEvent(new CustomEvent('showCenterPanePreview', {
        detail: { data: detectionData }
      }));
      
      // Also dispatch mlDetectionDataLoaded event for MLOverlay
      window.dispatchEvent(new CustomEvent('mlDetectionDataLoaded', {
        detail: { 
          data: detectionData,
          jobId: validationData?.jobId || 'direct-render',
          source: 'DirectDetectionOverlay'
        }
      }));
      
      // Also dispatch the detectionDataAvailable event to ensure ML overlay gets the data
      window.dispatchEvent(new CustomEvent('detectionDataAvailable', {
        detail: { data: detectionData }
      }));
      
      // Explicitly handle custom marker placement
      window.dispatchEvent(new CustomEvent('useCustomMarkerPlacement', {
        detail: {
          treeData: detectionData.detections || detectionData.trees || [],
          jobId: validationData?.jobId || 'direct-render',
          filters: {
            trees: true,
            buildings: true,
            powerLines: true
          },
          showLabels: true,
          method: 'DirectDetectionOverlay'
        }
      }));
      
      // Force renderer update with a more aggressive approach
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('showCenterPanePreview', {
          detail: { data: detectionData, forceUpdate: true }
        }));
      }, 500);
      
      return true;
    };
    
    // First attempt immediately
    const success = attemptRendering();
    
    // If not successful, retry after a delay
    if (!success) {
      console.log("DirectDetectionOverlay: First attempt failed, will retry after delay");
      setTimeout(attemptRendering, 1000);
      
      // And one more retry with longer delay
      setTimeout(attemptRendering, 3000);
    }
    
    // Clean up on unmount
    return () => {
      // Clean up existing overlays
      document.querySelectorAll('[id^="direct-detection-overlay"]').forEach(el => {
        if (el.parentNode) el.parentNode.removeChild(el);
      });
    };
  }, [validationData]);
  
  // Component doesn't render anything visible in the React tree
  return null;
};

// Import 3D viewer components lazily for better performance
const CesiumViewer = lazy(() => import('./CesiumViewer'));
const GoogleMaps3DViewer = lazy(() => import('./GoogleMaps3DViewer'));

const MapView = forwardRef(({ onDataLoaded, headerState }, ref) => {
  // Core state and refs
  const mapContainer = useRef(null);
  const map = useRef(null);
  const cesiumViewerRef = useRef(null);
  const googleMaps3DViewerRef = useRef(null);
  const markersRef = useRef([]);
  const dispatch = useDispatch();
  
  // Data state
  const [trees, setTrees] = useState([]);
  const [properties, setProperties] = useState([]);
  const [detectedTrees, setDetectedTrees] = useState([]);
  
  // Map configuration
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "";
  const mapId = import.meta.env.VITE_GOOGLE_MAPS_MAP_ID || "";
  const { isLoaded, loadError } = useGoogleMapsApi(
    apiKey,
    ['marker', 'drawing'],
    true // Enable WebGL for 3D
  );
  
  // UI state
  const [headerCollapsed, setHeaderCollapsed] = useState(true);
  const [isRightSidebarCollapsed, setIsRightSidebarCollapsed] = useState(false);
  const [isValidationMode, setIsValidationMode] = useState(false);
  const [validationData, setValidationData] = useState(null);
  
  // 3D state
  const [is3DMode, setIs3DMode] = useState(false);
  const [is3DSupported, setIs3DSupported] = useState(false);
  const [mapTilt, setMapTilt] = useState(0);
  const [mapHeading, setMapHeading] = useState(0);
  const [map3DApi, setMap3DApi] = useState('cesium');
  
  // Get map center and zoom from Redux
  const { center, zoom, activeBasemap, visibleLayers } = useSelector((state) => state.map);

  // Update header state when provided externally
  useEffect(() => {
    if (headerState !== undefined) {
      setHeaderCollapsed(headerState);
      window.dispatchEvent(new CustomEvent('headerCollapse', {
        detail: { collapsed: headerState }
      }));
    }
  }, [headerState]);
  
  // Load settings from localStorage
  const [settings, setSettings] = useState(() => {
    const defaults = {
      defaultView: '2d',
      defaultMapType: 'roadmap',
      showHighRiskByDefault: false,
      mapSettings: {
        showLabels: true,
        showTerrain: false
      }
    };
    
    try {
      const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
      if (savedSettings) {
        return { ...defaults, ...JSON.parse(savedSettings) };
      }
    } catch (e) {
      console.error("Error loading settings:", e);
    }
    
    return defaults;
  });
  
  // Listen for settings changes
  useEffect(() => {
    const handleStorageChange = () => {
      try {
        const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
        if (savedSettings) {
          const newSettings = JSON.parse(savedSettings);
          setSettings(prevSettings => ({
            ...prevSettings,
            ...newSettings
          }));
          
          // Apply satellite label setting if needed
          if (map.current && isLoaded && window.google && window.google.maps) {
            const currentMapType = map.current.getMapTypeId();
            if (currentMapType === window.google.maps.MapTypeId.SATELLITE || 
                currentMapType === window.google.maps.MapTypeId.HYBRID) {
              map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
            }
          }
        }
      } catch (e) {
        console.error("Error updating settings from storage:", e);
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, [isLoaded]);
  
  // Initialize Google Maps
  const initializeGoogleMap = () => {
    if (!isLoaded || !mapContainer.current) return;
    
    // Expose mapRef globally for MLOverlay to use
    window.mapRef = map;
    
    // If map already exists, just trigger resize and return
    if (map.current) {
      window.google.maps.event.trigger(map.current, 'resize');
      return;
    }
    
    try {
      console.log("Initializing Google Maps...");
      
      // 1. Check WebGL support for 3D
      try {
        const webGLSupported = window.google.maps.WebGLOverlayView !== undefined;
        setIs3DSupported(webGLSupported);
        console.log("WebGL for 3D maps is supported:", webGLSupported);
      } catch (e) {
        setIs3DSupported(false);
        console.warn("WebGL for 3D maps is not supported:", e);
      }
      
      // 2. Set up map center - use coordinates from Redux store without falling back to defaults
      // If no valid coordinates are available, just don't initialize the map yet
      if (!center || center.length !== 2 || !center[0] || !center[1]) {
        console.error("No valid map center coordinates available in Redux store");
        return; // Exit initialization if no valid coordinates
      }
      const mapCenter = { lat: center[1], lng: center[0] };
      
      // 3. Set hybrid mode as default
      localStorage.setItem('currentMapType', 'hybrid');
      window.currentMapType = 'hybrid';
      
      // 4. Create map instance
      map.current = new window.google.maps.Map(mapContainer.current, {
        center: mapCenter,
        zoom: zoom || 13,
        mapTypeId: window.google.maps.MapTypeId.HYBRID,
        zoomControl: true,
        mapTypeControl: false,
        scaleControl: true,
        streetViewControl: false,
        rotateControl: true,
        fullscreenControl: true,
        tilt: is3DMode ? 45 : 0,
        heading: 0
      });
      
      // 5. Store map instance globally for other components
      window.googleMapsInstance = map.current;
      window._mapReady = true;
      window._googleMap = map.current;
      window._map = map.current;
      
      // 6. Set up idle listener to update Redux state with current coordinates
      map.current.addListener('idle', () => {
        if (map.current && map.current.getCenter) {
          const center = map.current.getCenter();
          const currentCenter = [
            center.lng(),
            center.lat()
          ];
          const currentZoom = map.current.getZoom();
          
          // Get current state from Redux store
          const state = store.getState();
          const storeCenter = state.map.center;
          const storeZoom = state.map.zoom;
          
          // Only update Redux if coordinates or zoom have changed significantly
          // This prevents infinite loops caused by minor floating-point differences
          const coordsChanged = 
            !storeCenter || 
            Math.abs(storeCenter[0] - currentCenter[0]) > 0.0001 || 
            Math.abs(storeCenter[1] - currentCenter[1]) > 0.0001;
            
          const zoomChanged = storeZoom !== currentZoom;
          
          if (coordsChanged || zoomChanged) {
            // Update Redux state with current map position
            dispatch(setMapView({ center: currentCenter, zoom: currentZoom }));
            
            // Log center coordinates for debugging
            console.log("Map center updated:", currentCenter);
            
            // CRITICAL FIX: Dispatch mapIdle event for LocationInfo.jsx to update window.mapCoordinates
            // This is essential for tree detection to use correct coordinates
            window.dispatchEvent(new CustomEvent('mapIdle'));
          }
        }
      });
      
      // 7. Handle map clicks to clean up info windows
      map.current.addListener('click', (event) => {
        if (event.placeId) return; // Skip if clicking on a place marker
        
        // Close and clean up info windows
        markersRef.current.forEach(item => {
          if (item instanceof window.google.maps.InfoWindow) {
            item.close();
          }
        });
        
        // Remove closed info windows from reference array
        markersRef.current = markersRef.current.filter(
          item => !(item instanceof window.google.maps.InfoWindow)
        );
      });
      
      // 8. Dispatch map ready event for other components
      window.dispatchEvent(new CustomEvent('googleMapReady', {
        detail: { mapInstance: map.current }
      }));
      
      // Expose the map instance globally for easier access by MLOverlay
      window.googleMapsInstance = map.current;
      
      // CRITICAL FIX: Also expose mapRef more directly to ensure MLOverlay can find it
      window.mapRef = map;
      window._mapRef = map;
      window.mapWrapper = document.getElementById('map-wrapper');
      window._googleMap = map.current;
      
      console.log('MapView: Exposed map instance and references globally for better accessibility');
      
      console.log("Google Maps successfully initialized with center:", mapCenter);
    } catch (error) {
      console.error("Error initializing Google Maps:", error);
    }
  };
  
  // Initialize Google Maps when API is loaded
  useEffect(() => {
    if (!map.current && isLoaded && window.google && window.google.maps) {
      initializeGoogleMap();
    }
  }, [isLoaded]);
  
  // Clean up global references when component unmounts
  useEffect(() => {
    return () => {
      if (window.googleMapsInstance) {
        window.googleMapsInstance = null;
      }
    };
  }, []);
  
  // Log errors when Maps API fails to load
  useEffect(() => {
    if (loadError) {
      console.error('Error loading Google Maps API:', loadError);
    }
  }, [loadError]);
  
  // Update map when center or zoom changes in Redux store
  useEffect(() => {
    if (map.current && isLoaded && center && center.length === 2) {
      map.current.setCenter({ lat: center[1], lng: center[0] });
      map.current.setZoom(zoom);
    }
  }, [center, zoom, isLoaded]);
  
  // Always enforce HYBRID view to ensure labels appear
  useEffect(() => {
    const interval = setInterval(() => {
      try {
        if (window.google && window.google.maps && map.current) {
          const currentMapType = map.current.getMapTypeId();
          
          if (localStorage.getItem('currentMapType') === 'hybrid' || 
              window.currentMapType === 'hybrid') {
            if (currentMapType !== window.google.maps.MapTypeId.HYBRID) {
              map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
            }
          }
        }
      } catch (e) {
        // Ignore errors
      }
    }, 1000);
    
    return () => clearInterval(interval);
  }, [isLoaded]);
  
  // Handle marker clicks to show tree info with enhanced info window
  const handleMarkerClick = (marker, tree, markerColor, source = 'database') => {
    if (!map.current || !window.google || !tree) return;
    
    try {
      // Close any existing info windows
      markersRef.current.forEach(item => {
        if (item instanceof window.google.maps.InfoWindow) {
          item.close();
        }
      });
      
      // Determine risk level text and color
      let riskLevelText = 'Low Risk';
      let riskLevelColor = '#2ECC40';
      
      if (tree.risk_level === 'high' || (tree.risk_factors && tree.risk_factors.some(rf => rf.level === 'high'))) {
        riskLevelText = 'High Risk';
        riskLevelColor = '#FF4136';
      } else if (tree.risk_level === 'medium' || (tree.risk_factors && tree.risk_factors.some(rf => rf.level === 'medium'))) {
        riskLevelText = 'Medium Risk';
        riskLevelColor = '#FF851B';
      }
      
      // Create enhanced info window with tree details and controls
      const infoWindow = new window.google.maps.InfoWindow({
        content: `
          <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 12px; max-width: 320px;">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
              <div style="width: 12px; height: 12px; border-radius: 50%; background-color: ${markerColor}; margin-right: 8px;"></div>
              <div style="font-weight: 600; font-size: 16px;">${tree.species || 'Unknown Tree'}</div>
            </div>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
              <div>
                <div style="font-size: 12px; color: #666; margin-bottom: 2px;">Height</div>
                <div style="font-weight: 500; font-size: 14px;">${tree.height || 'Unknown'} m</div>
              </div>
              
              <div>
                <div style="font-size: 12px; color: #666; margin-bottom: 2px;">Risk Level</div>
                <div style="display: inline-block; font-size: 13px; font-weight: 600; color: ${riskLevelColor}; background-color: ${riskLevelColor}15; padding: 3px 8px; border-radius: 4px;">
                  ${riskLevelText}
                </div>
              </div>
              ${tree.confidence ? 
                `<div style="text-align: right;">
                  <div style="font-size: 12px; color: #666; margin-bottom: 2px;">Confidence</div>
                  <div style="font-weight: 500; font-size: 13px;">${(tree.confidence * 100).toFixed(0)}%</div>
                </div>` : ''}
            </div>
            
            <div style="border-top: 1px solid #eee; padding-top: 10px; margin-top: 8px;">
              <div style="display: flex; gap: 8px; justify-content: space-between;">
                <button id="infowindow-edit-${tree.id || Date.now()}" style="flex: 1; padding: 6px; background: #f1f5f9; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center;">
                  <span style="margin-right: 4px;">✏️</span> Edit
                </button>
                <button id="infowindow-zoom-${tree.id || Date.now()}" style="flex: 1; padding: 6px; background: #f1f5f9; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center;">
                  <span style="margin-right: 4px;">🔍</span> Zoom
                </button>
                <button id="infowindow-remove-${tree.id || Date.now()}" style="flex: 1; padding: 6px; background: #fee2e2; border: none; border-radius: 4px; cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center;">
                  <span style="margin-right: 4px;">🗑️</span> Remove
                </button>
              </div>
            </div>
          </div>
        `
      });
      
      infoWindow.open(map.current, marker);
      
      // Add InfoWindow to markers array for later reference
      markersRef.current.push(infoWindow);
      
      // Add event listeners to the buttons after the info window is displayed
      window.google.maps.event.addListener(infoWindow, 'domready', () => {
        // Edit button
        document.getElementById(`infowindow-edit-${tree.id || Date.now()}`)?.addEventListener('click', () => {
          infoWindow.close();
          // Use the existing edit metadata function
          if (typeof handleEditMetadata === 'function') {
            handleEditMetadata(tree);
          } else {
            // Fallback if the function isn't defined in scope
            // Create a simple form for editing tree metadata
            const form = document.createElement('div');
            form.className = 'tree-metadata-form';
            form.style.position = 'fixed';
            form.style.top = '50%';
            form.style.left = '50%';
            form.style.transform = 'translate(-50%, -50%)';
            form.style.backgroundColor = 'white';
            form.style.padding = '20px';
            form.style.borderRadius = '8px';
            form.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
            form.style.zIndex = '2000';
            form.style.minWidth = '300px';
            
            // Add form content
            form.innerHTML = `
              <h3 style="margin-top:0;font-size:18px;margin-bottom:16px">Edit Tree Metadata</h3>
              <div style="margin-bottom:12px">
                <label style="display:block;margin-bottom:4px;font-weight:500">Species</label>
                <input type="text" id="tree-species" value="${tree.species || ''}" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:4px">
              </div>
              <div style="margin-bottom:12px">
                <label style="display:block;margin-bottom:4px;font-weight:500">Height (m)</label>
                <input type="number" id="tree-height" value="${tree.height || ''}" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:4px">
              </div>
              <div style="margin-bottom:12px">
                <label style="display:block;margin-bottom:4px;font-weight:500">Risk Level</label>
                <select id="tree-risk" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:4px">
                  <option value="low" ${tree.risk_level === 'low' ? 'selected' : ''}>Low Risk</option>
                  <option value="medium" ${tree.risk_level === 'medium' ? 'selected' : ''}>Medium Risk</option>
                  <option value="high" ${tree.risk_level === 'high' ? 'selected' : ''}>High Risk</option>
                </select>
              </div>
              <div style="display:flex;justify-content:flex-end;margin-top:16px">
                <button id="cancel-edit" style="padding:8px 16px;background:#f1f5f9;border:none;border-radius:4px;margin-right:8px;cursor:pointer">Cancel</button>
                <button id="save-edit" style="padding:8px 16px;background:#3b82f6;color:white;border:none;border-radius:4px;cursor:pointer">Save Changes</button>
              </div>
            `;
            
            document.body.appendChild(form);
            
            // Add event listeners
            document.getElementById('cancel-edit').addEventListener('click', () => {
              document.body.removeChild(form);
            });
            
            document.getElementById('save-edit').addEventListener('click', () => {
              // Update tree metadata
              const newSpecies = document.getElementById('tree-species').value;
              const newHeight = document.getElementById('tree-height').value;
              const newRiskLevel = document.getElementById('tree-risk').value;
              
              tree.species = newSpecies;
              tree.height = newHeight ? parseFloat(newHeight) : undefined;
              tree.risk_level = newRiskLevel;
              
              // Update marker title
              marker.setTitle(`${tree.species || 'Unknown Tree'} - ${tree.height || 'Unknown'}m`);
              
              // Update label content if it exists
              if (tree.label) {
                tree.label.setContent(`<div style="background-color: ${markerColor}; color: white; padding: 3px 6px; border-radius: 4px; font-weight: bold; font-size: 10px; white-space: nowrap;">${tree.species || 'Tree'}</div>`);
              }
              
              // Dispatch event for metadata update
              window.dispatchEvent(new CustomEvent('treeMetadataUpdated', {
                detail: { tree, source }
              }));
              
              // Remove form
              document.body.removeChild(form);
              
              // Show notification
              const notification = document.createElement('div');
              notification.style.position = 'fixed';
              notification.style.top = '70px';
              notification.style.left = '50%';
              notification.style.transform = 'translateX(-50%)';
              notification.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
              notification.style.color = 'white';
              notification.style.padding = '8px 16px';
              notification.style.borderRadius = '4px';
              notification.style.zIndex = '1000';
              notification.style.fontSize = '14px';
              notification.style.fontWeight = '500';
              notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
              notification.textContent = 'Tree metadata updated';
              document.body.appendChild(notification);
              
              // Remove notification after 1.5 seconds
              setTimeout(() => {
                if (notification.parentNode) {
                  notification.parentNode.removeChild(notification);
                }
              }, 1500);
            });
          }
        });
        
        // Zoom button
        document.getElementById(`infowindow-zoom-${tree.id || Date.now()}`)?.addEventListener('click', () => {
          infoWindow.close();
          map.current.panTo(marker.getPosition());
          map.current.setZoom(Math.max(map.current.getZoom(), 18)); // Zoom in if currently zoomed out
        });
        
        // Remove button
        document.getElementById(`infowindow-remove-${tree.id || Date.now()}`)?.addEventListener('click', () => {
          infoWindow.close();
          
          // Ask for confirmation
          const confirmed = window.confirm('Are you sure you want to remove this marker?');
          if (!confirmed) return;
          
          // Remove the marker from the map
          marker.setMap(null);
          
          // Remove the label if it exists
          if (tree.label) {
            tree.label.close();
          }
          
          // Remove the rectangle if it exists
          if (tree.rectangle) {
            tree.rectangle.setMap(null);
          }
          
          // Remove from markersRef array
          markersRef.current = markersRef.current.filter(item => item !== marker && item !== tree.rectangle);
          
          // Dispatch event for marker removal
          window.dispatchEvent(new CustomEvent('treeMarkerRemoved', {
            detail: { tree, source }
          }));
          
          // Show notification
          const notification = document.createElement('div');
          notification.style.position = 'fixed';
          notification.style.top = '70px';
          notification.style.left = '50%';
          notification.style.transform = 'translateX(-50%)';
          notification.style.backgroundColor = 'rgba(239, 68, 68, 0.9)';
          notification.style.color = 'white';
          notification.style.padding = '8px 16px';
          notification.style.borderRadius = '4px';
          notification.style.zIndex = '1000';
          notification.style.fontSize = '14px';
          notification.style.fontWeight = '500';
          notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
          notification.textContent = 'Marker removed';
          document.body.appendChild(notification);
          
          // Remove notification after 1.5 seconds
          setTimeout(() => {
            if (notification.parentNode) {
              notification.parentNode.removeChild(notification);
            }
          }, 1500);
        });
      });
      
      // Update selected feature in Redux store
      dispatch(setSelectedFeature(tree));
      
      // Notify other components about the selection
      window.dispatchEvent(new CustomEvent('treeSelected', {
        detail: { tree }
      }));
    } catch (error) {
      console.error('Error handling marker click:', error);
    }
  };
  
  // Handle saving validated trees
  const handleSaveValidatedTrees = async (trees) => {
    if (!validationData || !validationData.areaId) {
      alert('Missing area ID for validation');
      return;
    }
    
    try {
      const { areaId } = validationData;
      const saveResult = await TreeService.saveValidatedTrees(areaId, trees);
      
      setIsValidationMode(false);
      setValidationData(null);
      setDetectedTrees([]);
      
      alert(`Successfully saved ${trees.length} validated trees.`);
      
      window.dispatchEvent(new CustomEvent('refreshMapMarkers'));
    } catch (error) {
      console.error('Error saving validated trees:', error);
      alert('Failed to save trees: ' + error.message);
    }
  };
  
  // Handle exiting validation mode
  const handleExitValidationMode = () => {
    setIsValidationMode(false);
    setValidationData(null);
    setDetectedTrees([]);
    
    // Clear all detection markers
    markersRef.current.forEach(item => {
      try {
        if (item.setMap) {
          item.setMap(null);
        } else if (item instanceof window.google.maps.InfoWindow) {
          item.close();
        }
      } catch (e) {
        console.warn("Error clearing marker:", e);
      }
    });
    
    // Reset markers array
    markersRef.current = [];
  };
  
  // Function to toggle 3D mode
  const toggle3DMode = () => {
    if (!is3DSupported) return;
    
    try {
      const newMode = !is3DMode;
      console.log(`Toggling 3D mode: ${is3DMode} → ${newMode}`);
      
      if (newMode) {
        // Going from 2D to 3D
        if (map.current) {
          // Capture current position from 2D map
          const mapCenter = map.current.getCenter();
          const position = {
            center: [mapCenter.lng(), mapCenter.lat()],
            zoom: map.current.getZoom()
          };
          
          // Store in Redux for the 3D view to use
          dispatch(setMapView({
            center: position.center,
            zoom: position.zoom
          }));
          
          // Use hybrid map type in 3D
          map.current.setMapTypeId(google.maps.MapTypeId.HYBRID);
          map.current.setTilt(0);
          setMapTilt(45);
        }
        
        window.dispatchEvent(new CustomEvent('mapModeChanged', { 
          detail: { mode: '3D', tilt: 45 } 
        }));
      } else {
        // Going from 3D to 2D
        const state = store.getState();
        const { center, zoom } = state.map;
        
        if (map.current && center && zoom) {
          // Force map to refresh
          window.google.maps.event.trigger(map.current, 'resize');
          
          // Set the position
          map.current.setCenter({ lat: center[1], lng: center[0] });
          map.current.setZoom(zoom);
          
          // Default to hybrid view
          map.current.setMapTypeId(google.maps.MapTypeId.HYBRID);
          window.currentMapType = 'hybrid';
          
          // Reset tilt for 2D view
          map.current.setTilt(0);
          setMapTilt(0);
        }
        
        window.dispatchEvent(new CustomEvent('mapModeChanged', { 
          detail: { mode: '2D', tilt: 0 } 
        }));
      }
      
      setIs3DMode(newMode);
      
      // Set a global flag for 3D mode that can be checked anywhere
      window.is3DMode = newMode;
      console.log(`Set global is3DMode flag to ${newMode}`);
      
      // Update global map instance reference
      if (map.current) {
        window.googleMapsInstance = map.current;
      }
    } catch (error) {
      console.error('Error toggling 3D mode:', error);
      setIs3DSupported(false);
    }
  };
  
  // Handle window resize and sidebar changes
  useEffect(() => {
    let resizeTimeout;
    
    const handleResize = () => {
      clearTimeout(resizeTimeout);
      
      resizeTimeout = setTimeout(() => {
        if (map.current && window.google && window.google.maps) {
          window.google.maps.event.trigger(map.current, 'resize');
        }
      }, 150);
    };
    
    const handleHeaderCollapse = (event) => {
      setHeaderCollapsed(event.detail.collapsed);
    };
    
    // Add event listeners
    window.addEventListener('validationQueueToggle', handleResize);
    window.addEventListener('validationSidebarToggle', handleResize);
    window.addEventListener('leftSidebarToggle', handleResize);
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    window.addEventListener('resize', handleResize);
    
    // Set up ResizeObserver to monitor map container and header size changes
    const resizeObserver = new ResizeObserver(() => {
      handleResize();
      
      // Update sidebar positions based on header height
      const sidebarElements = document.querySelectorAll('[id$="-sidebar"]');
      const header = document.querySelector('header');
      const headerHeight = header ? header.offsetHeight : 64;
      
      sidebarElements.forEach(sidebar => {
        if (sidebar) {
          try {
            sidebar.style.top = `${headerHeight}px`;
          } catch (e) {
            console.warn("Could not update sidebar position:", e);
          }
        }
      });
    });
    
    // Observe the map container and header
    const mapContainer = document.getElementById('map-container');
    if (mapContainer) {
      resizeObserver.observe(mapContainer);
    }
    
    const header = document.querySelector('header');
    if (header) {
      resizeObserver.observe(header);
    }
    
    // Initial resize
    handleResize();
    
    // Clean up
    return () => {
      clearTimeout(resizeTimeout);
      window.removeEventListener('validationQueueToggle', handleResize);
      window.removeEventListener('validationSidebarToggle', handleResize);
      window.removeEventListener('leftSidebarToggle', handleResize);
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
      window.removeEventListener('resize', handleResize);
      
      if (mapContainer) {
        resizeObserver.unobserve(mapContainer);
      }
      
      if (header) {
        resizeObserver.unobserve(header);
      }
      
      resizeObserver.disconnect();
    };
  }, []);
  
  // Listen for viewGeminiResponse events
  useEffect(() => {
    const handleViewGeminiResponse = (event) => {
      const { jobId, responsePath } = event.detail;
      
      try {
        console.log(`Viewing Gemini response for job ${jobId} at ${responsePath}`);
        
        alert(`Gemini Response Files for Job ${jobId}:\n\n` +
              `To view these files, open your terminal and run:\n` +
              `ls -la ${responsePath}\n\n` +
              `To view the response.txt file, run:\n` +
              `cat ${responsePath}/response.txt\n\n` +
              `To view the satellite image, check:\n` +
              `${responsePath}/satellite_image.jpg or composite_map.jpg`);
      } catch (error) {
        console.error("Error viewing Gemini response:", error);
        alert(`Error viewing Gemini response: ${error.message}`);
      }
    };
    
    window.addEventListener('viewGeminiResponse', handleViewGeminiResponse);
    
    return () => {
      window.removeEventListener('viewGeminiResponse', handleViewGeminiResponse);
    };
  }, []);
  
  // Listen for tree validation mode entry
  useEffect(() => {
    const handleEnterValidationMode = async (event) => {
      if (!event.detail) {
        console.log("Entering validation mode without details");
        setIsValidationMode(true);
        return;
      }
      
      // Log the event details to help diagnose issues
      console.log("enterDetectionMode event received with details:", {
        jobId: event.detail.jobId || event.detail.detectionJobId,
        ml_response_dir: event.detail.ml_response_dir,
        visualizationPath: event.detail.visualizationPath,
        treeCount: event.detail.treeCount,
        hasTreesArray: !!event.detail.trees,
        treesLength: event.detail.trees?.length,
        timestamp: event.detail.timestamp
      });
      
      // Update header state if provided
      if (event.detail.headerCollapsed !== undefined) {
        setHeaderCollapsed(event.detail.headerCollapsed);
        
        window.dispatchEvent(new CustomEvent('headerCollapse', {
          detail: { collapsed: event.detail.headerCollapsed }
        }));
      }
      
      // Check for initialDetectionOnly flag
      if (event.detail.initialDetectionOnly) {
        console.log("Entering detection mode without triggering object detection");
        
        setIsValidationMode(true);
        
        // Force detection mode state to be visible
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('setDetectionModeState', {
            detail: { active: true }
          }));
        }, 10);
        
        // Initialize with empty data
        setDetectedTrees([]);
        setValidationData({
          jobId: 'pending_detection',
          mode: 'detection',
          treeCount: 0,
          source: 'sidebar',
          useSatelliteImagery: true,
          headerCollapsed: event.detail.headerCollapsed !== undefined ? 
            event.detail.headerCollapsed : headerCollapsed
        });
        return;
      }
      
      // Handle sidebar validation mode
      if (event.detail.source === 'sidebar') {
        console.log("Entering validation mode from sidebar click");
        setIsValidationMode(true);
        return;
      }
      
      // In 3D mode, don't do any tree loading
      if (is3DMode) {
        console.log("Entering object detection validation mode in 3D view");
        setIsValidationMode(true);
        return;
      }
      
      if (!map.current || !isLoaded) {
        console.error("Map not ready for validation mode");
        return;
      }
      
      try {
        // Consolidate jobId - use either jobId or detectionJobId, prefer jobId
        const jobId = event.detail.jobId || event.detail.detectionJobId;
        const { 
          areaId, 
          treeCount, 
          trees, 
          ml_response_dir, 
          visualizationPath 
        } = event.detail;
        
        console.log(`Entering object detection validation mode for job ${jobId} with ${treeCount} trees`);
        console.log(`ML response directory: ${ml_response_dir || 'not provided'}`);
        
        // Validate the ML response directory
        if (!ml_response_dir) {
          console.warn("ML response directory not provided in event - visualization may not work");
        }
        
        // Update UI state first
        setIsValidationMode(true);
        
        // Create validation data object with all necessary information
        const newValidationData = {
          jobId: jobId, // Use jobId as single identifier
          areaId: areaId,
          treeCount: trees?.length || treeCount || 0,
          source: event.detail.source || 'detection',
          ml_response_dir: ml_response_dir, // This is the critical field for ML visualization
          visualizationPath: visualizationPath, // Backup visualization path
          timestamp: event.detail.timestamp || Date.now() // For debugging
        };
        
        // Use trees from event or leave undefined to be fetched later
        if (trees && trees.length > 0) {
          console.log(`Using ${trees.length} objects provided in event`);
          setDetectedTrees(trees);
          newValidationData.trees = trees;
        } else {
          console.log(`No objects provided in event, will need to fetch them`);
        }
        
        // Set the validation data which will trigger the MLOverlay to be rendered
        setValidationData(newValidationData);
        
        // Debug validation data
        console.log("Validation data set:", newValidationData);
        
        // CRITICAL FIX: Explicitly dispatch openTreeDetection event to ensure ML overlay is visible
        // and shows bounding boxes from trees.json when the Detection sidebar is opened
        window.dispatchEvent(new CustomEvent('openTreeDetection', {
          detail: {
            sidebarInitialization: true,
            initialVisibility: true,
            source: 'sidebar_button',
            jobId: jobId,
            data: {
              trees: trees || []
            }
          }
        }));
        
        // Store data globally for other components to access
        window.mlOverlaySettings = {
          ...(window.mlOverlaySettings || {}),
          showOverlay: true,
          opacity: 0.7,
          showSegmentation: true
        };
        window.detectionShowOverlay = true;
        
        // Also store the data globally for other components to access
        if (trees && trees.length > 0) {
          window.mlDetectionData = {
            job_id: jobId,
            trees: trees,
            metadata: {
              job_id: jobId,
              ml_response_dir: ml_response_dir
            }
          };
          
          // Broadcast that detection data is available
          window.dispatchEvent(new CustomEvent('detectionDataAvailable', {
            detail: { 
              data: window.mlDetectionData,
              source: 'enterDetectionMode'
            }
          }));
        }
        
        // If ML results available, notify that visualization is being displayed
        if (ml_response_dir) {
          console.log(`ML visualization being displayed from: ${ml_response_dir}`);
          
          // Show a notification to the user
          const notification = document.createElement('div');
          notification.style.position = 'absolute';
          notification.style.top = '120px';
          notification.style.left = '50%';
          notification.style.transform = 'translateX(-50%)';
          notification.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
          notification.style.color = 'white';
          notification.style.padding = '8px 16px';
          notification.style.borderRadius = '4px';
          notification.style.zIndex = '1000';
          notification.style.fontSize = '14px';
          notification.style.fontWeight = '500';
          notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
          notification.textContent = `Displaying ${trees?.length || treeCount || 0} objects from detection`;
          document.body.appendChild(notification);
          
          // Remove notification after 3 seconds
          setTimeout(() => {
            if (notification.parentNode) {
              notification.parentNode.removeChild(notification);
            }
          }, 3000);
        }
      } catch (error) {
        console.error("Error entering validation mode:", error);
      }
    };
    
    // Listen for the validation mode event
    window.addEventListener('enterDetectionMode', handleEnterValidationMode);
    // Also listen for the alternate name for backward compatibility
    window.addEventListener('enterValidationMode', handleEnterValidationMode);
    
    // Listen for updates to the visualization path
    const handleVisualizationPathUpdate = (event) => {
      if (event.detail && event.detail.visualizationPath && validationData) {
        console.log(`Updating visualization path to: ${event.detail.visualizationPath}`);
        setValidationData(prev => ({
          ...prev,
          visualizationPath: event.detail.visualizationPath
        }));
      }
    };
    window.addEventListener('updateVisualizationPath', handleVisualizationPathUpdate);
    
    return () => {
      window.removeEventListener('enterDetectionMode', handleEnterValidationMode);
      window.removeEventListener('enterValidationMode', handleEnterValidationMode);
      window.removeEventListener('updateVisualizationPath', handleVisualizationPathUpdate);
    };
  }, [isLoaded, is3DMode, headerCollapsed, validationData]);
  
  // Expose map and methods to parent via ref
  useImperativeHandle(ref, () => ({
    // Basic map controls
    getMap: () => map.current,
    getMarkers: () => markersRef.current,
    fitBounds: (bounds) => {
      if (map.current && bounds) {
        map.current.fitBounds(bounds);
      }
    },
    panTo: (coords) => {
      if (map.current && coords) {
        map.current.panTo(coords);
      }
    },
    setZoom: (zoom) => {
      if (map.current && zoom) {
        map.current.setZoom(zoom);
      }
    },
    
    // 3D controls
    toggle3DMode,
    getMapMode: () => ({
      is3DMode,
      is3DSupported,
      tilt: mapTilt,
      heading: mapHeading
    }),
    
    // Tree handling
    handleMarkerClick,
    
    // Method to get map view data for tree detection
    captureCurrentView: () => {
      if (!map.current) {
        console.error("Map reference not available");
        return Promise.reject(new Error("Map reference not available"));
      }
      
      return new Promise((resolve, reject) => {
        try {
          // Check if Google Maps is loaded
          if (!window.google || !window.google.maps) {
            console.error("Google Maps API not loaded");
            return reject(new Error("Google Maps API not loaded"));
          }
          
          try {
            const mapBounds = map.current.getBounds();
            const center = map.current.getCenter();
            const zoom = map.current.getZoom();
            const heading = map.current.getHeading ? map.current.getHeading() : 0;
            const tilt = map.current.getTilt ? map.current.getTilt() : 0;
            const is3D = tilt > 0;
            const mapTypeId = map.current.getMapTypeId ? map.current.getMapTypeId() : 'satellite';
            
            // Format data about the current view
            const viewData = {
              bounds: [
                [mapBounds.getSouthWest().lng(), mapBounds.getSouthWest().lat()],
                [mapBounds.getNorthEast().lng(), mapBounds.getNorthEast().lat()]
              ],
              center: [center.lng(), center.lat()],
              // CRITICAL: Also include userCoordinates for backend validation
              userCoordinates: [center.lng(), center.lat()],
              zoom: zoom,
              heading: heading,
              tilt: tilt,
              is3D: is3D
            };
            
            // Get map container dimensions
            const mapContainer = document.getElementById('map-container');
            if (!mapContainer) {
              console.error("Map container element not found");
              return reject(new Error("Map container element not found"));
            }
            
            // Get the dimensions of the visible map
            const mapWidth = mapContainer.offsetWidth;
            const mapHeight = mapContainer.offsetHeight;
            console.log(`Map dimensions: ${mapWidth}x${mapHeight}`);
            
            viewData.mapWidth = mapWidth;
            viewData.mapHeight = mapHeight;
            
            // Get additional coordinates for backend tile retrieval
            try {
              const mapType = map.current.getMapTypeId();
              const isSatellite = mapType === 'satellite' || mapType === 'hybrid';
              
              // Create a position object with detailed information
              const position = {
                bounds: viewData.bounds,
                center: viewData.center,
                zoom: zoom,
                isSatellite: isSatellite,
                mapType: mapType,
                timestamp: Date.now(),
                mapWidth: mapWidth,
                mapHeight: mapHeight
              };
              
              viewData.coordsInfo = JSON.stringify(position);
            } catch (coordsError) {
              console.warn("Error getting detailed map coordinates:", coordsError);
            }
            
            // Create the result object with comprehensive location data
            const result = {
              viewData: {
                ...viewData,
                useBackendTiles: true
              },
              is3D: is3D,
              mapType: mapTypeId,
              timestamp: Date.now()
            };
            
            console.log("Captured map view with coordinates:", viewData.center);
            resolve(result);
          } catch (mapError) {
            console.error("Error capturing map data:", mapError);
            
            // There should be no fallback - if we can't get current map data, we should fail
            return reject(new Error("Failed to capture current map view - unable to proceed with detection"));
          }
        } catch (criticalError) {
          console.error("Critical error in captureCurrentView:", criticalError);
          reject(criticalError);
        }
      });
    },
    
    // Function to create and display bounding boxes for AI-detected trees
    renderDetectedTrees: (treeData, source = 'ai') => {
      const mapInstance = map.current;
      if (!mapInstance || !window.google || !treeData || treeData.length === 0) {
        console.warn("Cannot render tree detection: missing map, Google Maps API, or tree data");
        return false;
      }
      
      console.log(`Rendering ${treeData.length} trees from ${source} detection`);
      
      // Check if there are existing markers
      if (markersRef.current.length > 0) {
        const confirmClear = window.confirm(
          `There are ${markersRef.current.length} existing tree markers on the map. ` +
          `Would you like to clear them before adding the ${treeData.length} newly detected trees?`
        );
        
        if (confirmClear) {
          // Clear existing markers
          markersRef.current.forEach(item => {
            try {
              if (item.setMap) {
                item.setMap(null);
              } else if (item instanceof window.google.maps.InfoWindow) {
                item.close();
              }
            } catch (error) {
              console.warn("Error clearing marker:", error);
            }
          });
          
          // Clear the markers array
          markersRef.current = [];
        }
      }
      
      // For each tree, create a marker
      const newMarkers = [];
      
      treeData.forEach((tree, index) => {
        try {
          if (!tree.location || !Array.isArray(tree.location) || tree.location.length < 2) {
            console.warn(`Tree ${index} missing valid location data:`, tree);
            return;
          }
          
          // Extract location
          const [lng, lat] = tree.location;
          
          // Determine color based on risk level and source
          let markerColor, strokeColor, fillOpacity, strokeWeight, markerScale;
          
          if (source === 'ai') {
            markerScale = 10;
            strokeWeight = 2;
            fillOpacity = 0.4;
            
            // Color based on risk level
            if (tree.risk_level === 'high') {
              markerColor = '#FF4136';
              strokeColor = '#FF0000';
            } else if (tree.risk_level === 'medium') {
              markerColor = '#FF851B';
              strokeColor = '#FF6600';
            } else {
              markerColor = '#2ECC40';
              strokeColor = '#20A030';
            }
          } else {
            markerScale = 8;
            strokeWeight = 1;
            fillOpacity = 0.8;
            
            if (tree.risk_level === 'high') {
              markerColor = '#BB0000';
              strokeColor = 'white';
            } else if (tree.risk_level === 'medium') {
              markerColor = '#DD7700';
              strokeColor = 'white';
            } else {
              markerColor = '#19A030';
              strokeColor = 'white';
            }
          }
          
          // Create a marker at the tree location with enhanced visibility and metadata
          const marker = new window.google.maps.Marker({
            position: { lat, lng },
            map: mapInstance,
            title: `${tree.species || 'Unknown Tree'} - ${tree.height || 'Unknown'}m`,
            draggable: true,
            icon: {
              path: window.google.maps.SymbolPath.CIRCLE,
              fillColor: markerColor,
              fillOpacity: fillOpacity,
              strokeColor: strokeColor,
              strokeWeight: strokeWeight,
              scale: markerScale
            },
            clickable: true,
            zIndex: 100
          });
          
          // Add a label to make markers more visible
          const label = new window.google.maps.InfoWindow({
            content: `<div style="background-color: ${markerColor}; color: white; padding: 3px 6px; border-radius: 4px; font-weight: bold; font-size: 10px; white-space: nowrap;">${tree.species || 'Tree'}</div>`,
            pixelOffset: new window.google.maps.Size(0, -30),
            disableAutoPan: true
          });
          
          // Show label by default for better visibility
          label.open(mapInstance, marker);
          
          // Store reference to label in tree object
          tree.label = label;
          
          // Add custom context menu for marker operations
          const createContextMenu = (marker, tree) => {
            // Create context menu elements
            const contextMenu = document.createElement('div');
            contextMenu.className = 'marker-context-menu';
            contextMenu.style.position = 'absolute';
            contextMenu.style.backgroundColor = 'white';
            contextMenu.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
            contextMenu.style.borderRadius = '4px';
            contextMenu.style.padding = '8px 0';
            contextMenu.style.zIndex = '1000';
            contextMenu.style.display = 'none';
            
            // Create menu items
            const menuItems = [
              { text: 'Edit Metadata', icon: '✏️', action: () => handleEditMetadata(tree) },
              { text: 'Remove Marker', icon: '🗑️', action: () => handleRemoveMarker(marker, tree) },
              { text: 'Toggle Label', icon: '🏷️', action: () => toggleLabel(marker, tree) },
              { text: 'Center on Map', icon: '🎯', action: () => centerOnMarker(marker) }
            ];
            
            menuItems.forEach(item => {
              const menuItem = document.createElement('div');
              menuItem.className = 'marker-context-menu-item';
              menuItem.style.padding = '8px 16px';
              menuItem.style.cursor = 'pointer';
              menuItem.style.fontSize = '14px';
              menuItem.style.display = 'flex';
              menuItem.style.alignItems = 'center';
              menuItem.innerHTML = `<span style="margin-right:8px">${item.icon}</span> ${item.text}`;
              
              menuItem.addEventListener('mouseover', () => {
                menuItem.style.backgroundColor = '#f0f9ff';
              });
              
              menuItem.addEventListener('mouseout', () => {
                menuItem.style.backgroundColor = 'transparent';
              });
              
              menuItem.addEventListener('click', (e) => {
                e.stopPropagation();
                contextMenu.style.display = 'none';
                item.action();
              });
              
              contextMenu.appendChild(menuItem);
            });
            
            // Add context menu to the document body
            document.body.appendChild(contextMenu);
            
            // Add right-click event listener to the marker
            marker.addListener('rightclick', (e) => {
              // Position the context menu at the mouse position
              const projection = mapInstance.getProjection();
              const position = projection.fromLatLngToPoint(e.latLng);
              
              // Convert to screen coordinates
              const scale = Math.pow(2, mapInstance.getZoom());
              const worldPoint = new window.google.maps.Point(
                position.x * scale,
                position.y * scale
              );
              
              contextMenu.style.left = `${e.domEvent.clientX}px`;
              contextMenu.style.top = `${e.domEvent.clientY}px`;
              contextMenu.style.display = 'block';
              
              // Add a custom class to identify this menu
              contextMenu.classList.add('active-marker-menu');
              
              // Hide menu when clicking elsewhere
              const hideMenu = () => {
                contextMenu.style.display = 'none';
                document.removeEventListener('click', hideMenu);
              };
              
              // Add a delay to prevent immediate hiding
              setTimeout(() => {
                document.addEventListener('click', hideMenu);
              }, 100);
            });
            
            return contextMenu;
          };
          
          // Helper functions for marker operations
          const handleEditMetadata = (tree) => {
            // Create a simple form for editing tree metadata
            const form = document.createElement('div');
            form.className = 'tree-metadata-form';
            form.style.position = 'fixed';
            form.style.top = '50%';
            form.style.left = '50%';
            form.style.transform = 'translate(-50%, -50%)';
            form.style.backgroundColor = 'white';
            form.style.padding = '20px';
            form.style.borderRadius = '8px';
            form.style.boxShadow = '0 4px 20px rgba(0,0,0,0.3)';
            form.style.zIndex = '2000';
            form.style.minWidth = '300px';
            
            // Add form content
            form.innerHTML = `
              <h3 style="margin-top:0;font-size:18px;margin-bottom:16px">Edit Tree Metadata</h3>
              <div style="margin-bottom:12px">
                <label style="display:block;margin-bottom:4px;font-weight:500">Species</label>
                <input type="text" id="tree-species" value="${tree.species || ''}" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:4px">
              </div>
              <div style="margin-bottom:12px">
                <label style="display:block;margin-bottom:4px;font-weight:500">Height (m)</label>
                <input type="number" id="tree-height" value="${tree.height || ''}" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:4px">
              </div>
              <div style="margin-bottom:12px">
                <label style="display:block;margin-bottom:4px;font-weight:500">Risk Level</label>
                <select id="tree-risk" style="width:100%;padding:8px;border:1px solid #ddd;border-radius:4px">
                  <option value="low" ${tree.risk_level === 'low' ? 'selected' : ''}>Low Risk</option>
                  <option value="medium" ${tree.risk_level === 'medium' ? 'selected' : ''}>Medium Risk</option>
                  <option value="high" ${tree.risk_level === 'high' ? 'selected' : ''}>High Risk</option>
                </select>
              </div>
              <div style="display:flex;justify-content:flex-end;margin-top:16px">
                <button id="cancel-edit" style="padding:8px 16px;background:#f1f5f9;border:none;border-radius:4px;margin-right:8px;cursor:pointer">Cancel</button>
                <button id="save-edit" style="padding:8px 16px;background:#3b82f6;color:white;border:none;border-radius:4px;cursor:pointer">Save Changes</button>
              </div>
            `;
            
            document.body.appendChild(form);
            
            // Add event listeners
            document.getElementById('cancel-edit').addEventListener('click', () => {
              document.body.removeChild(form);
            });
            
            document.getElementById('save-edit').addEventListener('click', () => {
              // Update tree metadata
              const newSpecies = document.getElementById('tree-species').value;
              const newHeight = document.getElementById('tree-height').value;
              const newRiskLevel = document.getElementById('tree-risk').value;
              
              tree.species = newSpecies;
              tree.height = newHeight ? parseFloat(newHeight) : undefined;
              tree.risk_level = newRiskLevel;
              
              // Update marker title
              marker.setTitle(`${tree.species || 'Unknown Tree'} - ${tree.height || 'Unknown'}m`);
              
              // Update label content if it exists
              if (tree.label) {
                tree.label.setContent(`<div style="background-color: ${markerColor}; color: white; padding: 3px 6px; border-radius: 4px; font-weight: bold; font-size: 10px; white-space: nowrap;">${tree.species || 'Tree'}</div>`);
              }
              
              // Dispatch event for metadata update
              window.dispatchEvent(new CustomEvent('treeMetadataUpdated', {
                detail: { tree, source }
              }));
              
              // Remove form
              document.body.removeChild(form);
              
              // Show notification
              const notification = document.createElement('div');
              notification.style.position = 'fixed';
              notification.style.top = '70px';
              notification.style.left = '50%';
              notification.style.transform = 'translateX(-50%)';
              notification.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
              notification.style.color = 'white';
              notification.style.padding = '8px 16px';
              notification.style.borderRadius = '4px';
              notification.style.zIndex = '1000';
              notification.style.fontSize = '14px';
              notification.style.fontWeight = '500';
              notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
              notification.textContent = 'Tree metadata updated';
              document.body.appendChild(notification);
              
              // Remove notification after 1.5 seconds
              setTimeout(() => {
                if (notification.parentNode) {
                  notification.parentNode.removeChild(notification);
                }
              }, 1500);
            });
          };
          
          const handleRemoveMarker = (marker, tree) => {
            // Ask for confirmation
            const confirmed = window.confirm('Are you sure you want to remove this marker?');
            if (!confirmed) return;
            
            // Remove the marker from the map
            marker.setMap(null);
            
            // Remove the label if it exists
            if (tree.label) {
              tree.label.close();
            }
            
            // Remove the rectangle if it exists
            if (tree.rectangle) {
              tree.rectangle.setMap(null);
            }
            
            // Remove from markersRef array
            markersRef.current = markersRef.current.filter(item => item !== marker && item !== tree.rectangle);
            
            // Dispatch event for marker removal
            window.dispatchEvent(new CustomEvent('treeMarkerRemoved', {
              detail: { tree, source }
            }));
            
            // Show notification
            const notification = document.createElement('div');
            notification.style.position = 'fixed';
            notification.style.top = '70px';
            notification.style.left = '50%';
            notification.style.transform = 'translateX(-50%)';
            notification.style.backgroundColor = 'rgba(239, 68, 68, 0.9)';
            notification.style.color = 'white';
            notification.style.padding = '8px 16px';
            notification.style.borderRadius = '4px';
            notification.style.zIndex = '1000';
            notification.style.fontSize = '14px';
            notification.style.fontWeight = '500';
            notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
            notification.textContent = 'Marker removed';
            document.body.appendChild(notification);
            
            // Remove notification after 1.5 seconds
            setTimeout(() => {
              if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
              }
            }, 1500);
          };
          
          const toggleLabel = (marker, tree) => {
            if (tree.label) {
              if (tree.labelVisible === false) {
                // Show the label
                tree.label.open(mapInstance, marker);
                tree.labelVisible = true;
              } else {
                // Hide the label
                tree.label.close();
                tree.labelVisible = false;
              }
            }
          };
          
          const centerOnMarker = (marker) => {
            mapInstance.panTo(marker.getPosition());
            mapInstance.setZoom(Math.max(mapInstance.getZoom(), 18)); // Zoom in if currently zoomed out
          };
          
          // Create and attach context menu
          const contextMenu = createContextMenu(marker, tree);
          tree.contextMenu = contextMenu;
          
          // For AI-detected trees, add a bounding box
          let rectangle = null;
          if (source === 'ai') {
            // Calculate bounds - size based on tree height if available
            const boxSize = tree.height ? tree.height / 200 : 0.0001;
            const bounds = {
              north: lat + boxSize,
              south: lat - boxSize,
              east: lng + boxSize,
              west: lng - boxSize
            };
            
            // Create a rectangle for the bounding box
            rectangle = new window.google.maps.Rectangle({
              bounds: bounds,
              strokeColor: strokeColor,
              strokeOpacity: 0.8,
              strokeWeight: 2,
              fillColor: markerColor,
              fillOpacity: 0.08,
              map: mapInstance,
              clickable: true,
              draggable: true
            });
            
            // Add click listener to rectangle
            rectangle.addListener('click', () => {
              handleMarkerClick(marker, tree, markerColor, source);
            });
            
            // Add bounds_changed listener to rectangle
            rectangle.addListener('bounds_changed', () => {
              try {
                // Get the center of the rectangle
                const bounds = rectangle.getBounds();
                const center = bounds.getCenter();
                const newLat = center.lat();
                const newLng = center.lng();
                
                // Update marker position
                marker.setPosition({ lat: newLat, lng: newLng });
                
                // Update the tree location
                tree.location = [newLng, newLat];
                
                // Dispatch event for DetectionMode component
                window.dispatchEvent(new CustomEvent('treeRepositioned', {
                  detail: { tree, source, newLocation: [newLng, newLat] }
                }));
              } catch (error) {
                console.error("Error handling rectangle drag:", error);
              }
            });
            
            // Store reference to rectangle in tree object
            tree.rectangle = rectangle;
            
            // Store rectangle in markers array
            newMarkers.push(rectangle);
          }
          
          // Add click listener to marker
          marker.addListener('click', () => {
            handleMarkerClick(marker, tree, markerColor, source);
          });
          
          // Add drag end event to update tree location
          marker.addListener('dragend', (event) => {
            const newLat = event.latLng.lat();
            const newLng = event.latLng.lng();
            
            // Update tree location
            tree.location = [newLng, newLat];
            
            // Update rectangle position if it exists
            if (source === 'ai' && rectangle) {
              const boxSize = tree.height ? tree.height / 200 : 0.0001;
              const bounds = {
                north: newLat + boxSize,
                south: newLat - boxSize,
                east: newLng + boxSize,
                west: newLng - boxSize
              };
              rectangle.setBounds(bounds);
            }
            
            // Notify UI about repositioning
            window.dispatchEvent(new CustomEvent('treeRepositioned', {
              detail: { tree, source, newLocation: [newLng, newLat] }
            }));
            
            // Show notification
            const notification = document.createElement('div');
            notification.className = 'tree-repositioned-notification';
            notification.style.position = 'absolute';
            notification.style.top = '70px';
            notification.style.left = '50%';
            notification.style.transform = 'translateX(-50%)';
            notification.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
            notification.style.color = 'white';
            notification.style.padding = '8px 16px';
            notification.style.borderRadius = '4px';
            notification.style.zIndex = '1000';
            notification.style.fontSize = '14px';
            notification.style.fontWeight = '500';
            notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
            notification.textContent = 'Object position updated';
            document.body.appendChild(notification);
            
            // Remove notification after 1.5 seconds
            setTimeout(() => {
              if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
              }
            }, 1500);
          });
          
          // Store reference to marker in tree object
          tree.marker = marker;
          
          // Store marker in markers array
          newMarkers.push(marker);
        } catch (error) {
          console.error(`Error creating marker for tree ${index}:`, error);
        }
      });
      
      // Add all new markers to the markers reference array
      markersRef.current = [...markersRef.current, ...newMarkers];
      
      console.log(`Added ${newMarkers.length} new markers for ${treeData.length} ${source} trees`);
      return true;
    }
  }));
  
  return (
    <ErrorBoundary componentName="MapContainer">
      <div className="relative w-full h-full" id="map-wrapper">
        {/* MLOverlayInitializer - Always render this to ensure proper Google Maps integration */}
        <MLOverlayInitializer />
        
        {/* Google Maps Container */}
        <div 
          ref={mapContainer}
          id="map-container"
          className={`transition-all duration-300 ${is3DMode ? 'hidden' : 'block'}`}
          style={{ 
            position: 'absolute', 
            top: 0,
            right: 0,
            bottom: 0,
            left: 0
          }}
        ></div>
        
        {/* 3D Viewer */}
        {is3DMode && (
          <Suspense fallback={<div className="w-full h-full flex items-center justify-center">Loading 3D View...</div>}>
            <div id="map-3d-container" className="w-full h-full transition-all duration-300" style={{ 
              position: 'absolute', 
              top: 0, 
              right: 0, 
              bottom: 0, 
              left: 0,
              zIndex: 10
            }}>
              {map3DApi === 'cesium' ? (
                <CesiumViewer 
                  ref={cesiumViewerRef}
                  center={center}
                  zoom={zoom}
                  apiKey={apiKey}
                />
              ) : (
                <GoogleMaps3DViewer
                  ref={googleMaps3DViewerRef}
                  apiKey={apiKey}
                  center={center}
                  zoom={zoom}
                />
              )}
            </div>
          </Suspense>
        )}
        
        {/* Tree Detection Mode UI */}
        {isValidationMode && (
          <ErrorBoundary componentName="ValidationModeFragment">
            <>
              {/* DetectionMode component renders the main sidebar */}
              <ErrorBoundary componentName="DetectionMode">
                {/* ML Overlay for detection visualization - using standalone renderer */}
                {validationData && (validationData.ml_response_dir || validationData.trees) && (
                  <DirectDetectionOverlay validationData={validationData} />
                )}
                
                <DetectionMode
                  key={`detection-${Date.now()}`}
                  mapRef={map}
                  validationData={validationData}
                  detectedTrees={detectedTrees}
                  onExitValidation={handleExitValidationMode}
                  onSaveTrees={handleSaveValidatedTrees}
                  headerCollapsed={headerCollapsed}
                />
              </ErrorBoundary>
              
              {/* Object Detection Badge - removed as requested */}
              
              {/* Empty container for markers only - no sidebar needed */}
            </>
          </ErrorBoundary>
        )}
      </div>
    </ErrorBoundary>
  );
});

export default MapView;