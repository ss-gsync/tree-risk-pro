// src/components/visualization/MapView/MapView.jsx

import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle, lazy, Suspense } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Card, CardContent } from '@/components/ui/card';
import { setMapView, setSelectedFeature } from '../../../features/map/mapSlice';
import { useGoogleMapsApi } from '../../../hooks/useGoogleMapsApi';
import { store } from '../../../store';
import { TreeService, PropertyService, DetectionService } from '../../../services/api/apiService';
import Map3DToggle from './Map3DToggle';
import TreeValidationMode from './TreeValidationMode';

// Import 3D viewer components lazily for better performance
const CesiumViewer = lazy(() => import('./CesiumViewer'));
const GoogleMaps3DViewer = lazy(() => import('./GoogleMaps3DViewer'));


const MapView = forwardRef(({ onDataLoaded }, ref) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const cesiumViewerRef = useRef(null);
  const googleMaps3DViewerRef = useRef(null);
  const markersRef = useRef([]);
  const dispatch = useDispatch();
  const [trees, setTrees] = useState([]);
  const [properties, setProperties] = useState([]);
  // Load Google Maps API with your API key from environment variables
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || "";
  const mapId = import.meta.env.VITE_GOOGLE_MAPS_MAP_ID || "";
  
  // Get Google Maps loading status
  const { isLoaded, loadError } = useGoogleMapsApi(
    apiKey,
    ['marker', 'drawing'], // Libraries
    true // Enable WebGL for 3D
  );
  // Load initial settings from localStorage
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
          
          // If map is already initialized, apply the satellite label setting
          // if the current map type is satellite
          if (map.current && isLoaded && window.google && window.google.maps) {
            const currentMapType = map.current.getMapTypeId();
            if (currentMapType === window.google.maps.MapTypeId.SATELLITE || 
                currentMapType === window.google.maps.MapTypeId.HYBRID) {
              
              // Check if the label setting changed
              const showLabels = newSettings.mapSettings?.showLabels !== false;
              
              // Apply the correct map type based on the setting
              if (showLabels) {
                map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
                console.log("Updating to HYBRID mode based on settings change");
              } else {
                map.current.setMapTypeId(window.google.maps.MapTypeId.SATELLITE);
                console.log("Updating to SATELLITE mode based on settings change");
              }
            }
          }
        }
      } catch (e) {
        console.error("Error updating settings from storage:", e);
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [isLoaded]);
  
  const [is3DMode, setIs3DMode] = useState(settings.defaultView === '3d');
  const [is3DSupported, setIs3DSupported] = useState(false);
  const [mapTilt, setMapTilt] = useState(settings.defaultView === '3d' ? 45 : 0);
  const [mapHeading, setMapHeading] = useState(0);
  const [isValidationMode, setIsValidationMode] = useState(false);
  const [validationData, setValidationData] = useState(null);
  const [isRightSidebarCollapsed, setIsRightSidebarCollapsed] = useState(false);
  const [detectedTrees, setDetectedTrees] = useState([]);
  
  // Get map3DApi setting from localStorage or use default
  const [map3DApi, setMap3DApi] = useState(() => {
    const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        return settings.map3DApi || 'cesium';
      } catch (e) {
        return 'cesium';
      }
    }
    return 'cesium';
  });
  
  // Function to initialize or re-initialize Google Maps - optimized for rendering speed
  const initializeGoogleMap = () => {
    if (!isLoaded || !mapContainer.current) return;
    
    // If the map already exists, just return - we don't want to recreate it
    if (map.current) {
      // Just resize and refresh the current map instead of recreating
      window.google.maps.event.trigger(map.current, 'resize');
      return;
    }
    
    try {
      console.log("Initializing Google Maps with center:", center, "zoom:", zoom);
      
      // Create the map instance
      map.current = new window.google.maps.Map(mapContainer.current, {
        center: { lat: center[1], lng: center[0] },
        zoom: zoom || 13,
        mapId: import.meta.env.VITE_GOOGLE_MAPS_MAP_ID || "",
        disableDefaultUI: false,
        zoomControl: true,
        mapTypeControl: true,
        scaleControl: true,
        streetViewControl: true,
        rotateControl: true,
        fullscreenControl: true,
        tilt: mapTilt,
        heading: mapHeading
      });
      
      // Check if WebGL is supported for 3D maps
      try {
        const webGLSupported = window.google.maps.WebGLOverlayView !== undefined;
        console.log("WebGL for 3D maps is supported:", webGLSupported);
        setIs3DSupported(webGLSupported);
      } catch (e) {
        console.warn("WebGL for 3D maps is not supported:", e);
        setIs3DSupported(false);
      }
      
      // Once map is created, refresh markers
      if (map.current) {
        const refreshEvent = new CustomEvent('refreshMapMarkers');
        window.dispatchEvent(refreshEvent);
      }
      
      console.log("Google Maps initialized successfully");
    } catch (error) {
      console.error("Error initializing Google Maps:", error);
    }
  };
  const { center, zoom, activeBasemap, visibleLayers } = useSelector((state) => state.map);
  
  // Log initialization info once when component mounts
  useEffect(() => {
    console.log('Map component initialized, API Key available:', !!apiKey, 'Map ID available:', !!mapId);
  }, []); // Empty deps array ensures this only runs once
  
  // Effect to show clear error message when Maps fails to load
  useEffect(() => {
    if (loadError) {
      console.error('Error loading Google Maps API:', loadError);
    }
  }, [loadError]);

  // Pre-render map tiles in background to avoid lag when switching modes
  const preRenderMapTiles = () => {
    if (!map.current || !isLoaded) return;
    try {
      // Even when hidden, periodically trigger events to keep tiles fresh
      window.google.maps.event.trigger(map.current, 'resize');
    } catch (e) {}
  };
  
  // Set up background rendering to keep map ready
  useEffect(() => {
    if (is3DMode && map.current) {
      // When in 3D mode, periodically refresh the map in the background
      const interval = setInterval(preRenderMapTiles, 2000);
      return () => clearInterval(interval);
    }
  }, [is3DMode, isLoaded]);
  
  useEffect(() => {
    // Only initialize the map if it doesn't exist yet and Google Maps is loaded
    if (!map.current && isLoaded && window.google && window.google.maps) {
      try {
        const googleCenter = { lat: center[1], lng: center[0] }; // Convert [lng, lat] to {lat, lng}
        
        // Set default fallback coordinates if center is invalid
        const validCenter = (center[0] && center[1]) ? 
          { lat: center[1], lng: center[0] } : 
          { lat: 32.8766, lng: -96.8064 }; // Dallas area
        
        // Check if WebGL is supported for 3D maps
        let webGLSupported = false;
        try {
          // Check if the WebGL namespace exists
          webGLSupported = window.google.maps.WebGLOverlayView !== undefined;
          console.log("WebGL for 3D maps is supported");
          setIs3DSupported(true);
        } catch (e) {
          console.warn("WebGL for 3D maps is not supported:", e);
          setIs3DSupported(false);
        }
        
        // Determine initial map type from settings
        const mapTypeId = settings.defaultMapType.toUpperCase();
        const initialMapType = window.google.maps.MapTypeId[mapTypeId] || 
                              window.google.maps.MapTypeId.ROADMAP;
        
        console.log("Initializing map with type:", mapTypeId, "3D mode:", is3DMode);
        
        // Create map with fallbacks for errors
        map.current = new window.google.maps.Map(mapContainer.current, {
          center: validCenter,
          zoom: zoom || 13,
          mapTypeId: initialMapType,
          mapId: mapId, // Map ID from environment variables
          disableDefaultUI: false,
          zoomControl: true,
          mapTypeControl: true,
          scaleControl: true,
          streetViewControl: true,
          rotateControl: true,
          fullscreenControl: true,
          // 3D settings (support depends on API version and browser)
          tilt: is3DMode ? 45 : 0, // Initial tilt for 3D mode
          heading: 0,
          // Enable 3D building layer in the map
          // Keep UI controls minimal so we can add our own
          mapTypeControl: true,
          mapTypeControlOptions: {
            mapTypeIds: [
              google.maps.MapTypeId.ROADMAP,
              google.maps.MapTypeId.SATELLITE,
              google.maps.MapTypeId.HYBRID,
              google.maps.MapTypeId.TERRAIN
            ],
            position: google.maps.ControlPosition.TOP_LEFT,
            style: google.maps.MapTypeControlStyle.HORIZONTAL_BAR
          },
          // Custom controls
          mapTypes: google.maps.MapTypeControlStyle.HORIZONTAL_BAR,
          fullscreenControlOptions: {
            position: google.maps.ControlPosition.TOP_RIGHT
          }
        });
        
        // Check if Aerial Imagery with 3D view is supported
        if (webGLSupported) {
          // Try to enable 3D buildings if WebGL is supported
          try {
            // Initial map state - flat by default, will be changed by 3D toggle
            const initialTilt = is3DMode ? 45 : 0;
            map.current.setTilt(initialTilt);
            setMapTilt(initialTilt);
            
            // Set global state for the 3D mode
            window.is3DModeActive = is3DMode;
            
            // Set up listeners to keep the global state in sync
            window.addEventListener('mapModeChanged', function(e) {
              window.is3DModeActive = e.detail.mode === '3D';
            });
            
            console.log("3D buildings initialized with tilt:", initialTilt);
            
            // Apply high risk filter if enabled in settings
            if (settings.showHighRiskByDefault) {
              console.log("Applying high risk filter based on settings");
              
              // Set global variables to indicate high risk filter is active
              window.currentRiskFilter = 'high';
              window.highRiskFilterActive = true;
              window.showOnlyHighRiskTrees = true;
              
              // Schedule it slightly after initialization to ensure map is ready
              setTimeout(() => {
                // Create filter event
                const filterEvent = new CustomEvent('filterHighRiskOnly', { 
                  detail: { active: true } 
                });
                window.dispatchEvent(filterEvent);
              }, 500);
            }
          } catch (e) {
            console.warn("Failed to enable 3D buildings:", e);
            setIs3DSupported(false);
          }
        }
        
        console.log("Map successfully initialized with 3D support:", webGLSupported);
      } catch (error) {
        console.error("Error initializing map:", error);
        setLoadError(error);
      }

      // Handle map movement to update Redux store
      try {
        map.current.addListener('idle', () => {
          try {
            if (map.current && map.current.getCenter) {
              const center = [
                map.current.getCenter().lng(),
                map.current.getCenter().lat()
              ];
              const zoom = map.current.getZoom();
              dispatch(setMapView({ center, zoom }));
            }
          } catch (error) {
            console.warn("Error updating map view in Redux:", error);
          }
        });
      } catch (error) {
        console.warn("Error adding idle listener to map:", error);
      }
      
      // Add a click listener to the map to close InfoWindows when clicking elsewhere
      try {
        map.current.addListener('click', (event) => {
          try {
            // Check if the click was directly on the map (not on a marker or other map UI element)
            if (event.placeId) return;
            
            // Close all InfoWindows
            markersRef.current.forEach(item => {
              try {
                if (item instanceof window.google.maps.InfoWindow) {
                  item.close();
                }
              } catch (error) {
                console.warn("Error closing InfoWindow:", error);
              }
            });
            
            // Filter out closed InfoWindows
            markersRef.current = markersRef.current.filter(item => {
              try {
                return !(item instanceof window.google.maps.InfoWindow);
              } catch (error) {
                console.warn("Error filtering InfoWindows:", error);
                return true; // Keep item in array if we can't determine type
              }
            });
          } catch (error) {
            console.warn("Error in map click handler:", error);
          }
        });
      } catch (error) {
        console.warn("Error adding click listener to map:", error);
      }
    }
  }, [dispatch, isLoaded, center, zoom]);

  // Update map when center or zoom changes in Redux store
  useEffect(() => {
    if (map.current && isLoaded) {
      // Convert [lng, lat] to {lat, lng} for Google Maps
      map.current.setCenter({ lat: center[1], lng: center[0] });
      map.current.setZoom(zoom);
    }
  }, [center, zoom, isLoaded]);

  // Listen for events to change map type
  useEffect(() => {
    const handleSetMapTypeId = (event) => {
      if (map.current && isLoaded && window.google && window.google.maps) {
        try {
          const { mapTypeId } = event.detail;
          const mapId = import.meta.env.VITE_GOOGLE_MAPS_MAP_ID;
          
          console.log(`Changing map type to: ${mapTypeId}`);
          
          // Check if mapId is present and show a warning
          if (mapId && mapId.trim() !== '') {
            console.info(
              "Note: When using mapId, some custom styles may not apply correctly with map types. " +
              "This is normal Google Maps behavior."
            );
          }
          
          // Safely try to set the map type
          try {
            if (mapTypeId === 'satellite') {
              // Check if we should show labels in satellite view
              const showLabels = settings.mapSettings?.showLabels !== false;
              if (showLabels) {
                // Use HYBRID which includes labels
                map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
                console.log("Using HYBRID mode for satellite with labels");
              } else {
                // Use SATELLITE which doesn't include labels
                map.current.setMapTypeId(window.google.maps.MapTypeId.SATELLITE);
                console.log("Using SATELLITE mode without labels");
              }
            } else if (mapTypeId === 'roadmap') {
              map.current.setMapTypeId(window.google.maps.MapTypeId.ROADMAP);
            } else if (mapTypeId === 'hybrid') {
              map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
            } else if (mapTypeId === 'terrain') {
              map.current.setMapTypeId(window.google.maps.MapTypeId.TERRAIN);
            }
            
            // Trigger resize to ensure map displays correctly with new type
            setTimeout(() => {
              window.google.maps.event.trigger(map.current, 'resize');
            }, 100);
          } catch (mapTypeError) {
            console.warn("Error setting map type:", mapTypeError.message);
          }
        } catch (error) {
          console.warn("Error handling map type change:", error);
        }
      }
    };
    
    window.addEventListener('setMapTypeId', handleSetMapTypeId);
    
    return () => {
      window.removeEventListener('setMapTypeId', handleSetMapTypeId);
    };
  }, [isLoaded, settings]);

  // Handle different basemap types if needed
  useEffect(() => {
    if (map.current && isLoaded && activeBasemap) {
      switch (activeBasemap) {
        case 'satellite':
          // Check if we should show labels in satellite view
          const showLabels = settings.mapSettings?.showLabels !== false;
          if (showLabels) {
            // Use HYBRID which includes labels
            map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
            console.log("Using HYBRID mode for satellite with labels");
          } else {
            // Use SATELLITE which doesn't include labels
            map.current.setMapTypeId(window.google.maps.MapTypeId.SATELLITE);
            console.log("Using SATELLITE mode without labels");
          }
          break;
        case 'roadmap':
          map.current.setMapTypeId(window.google.maps.MapTypeId.ROADMAP);
          break;
        case 'terrain':
          map.current.setMapTypeId(window.google.maps.MapTypeId.TERRAIN);
          break;
        case 'hybrid':
          map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
          break;
        default:
          // Default satellite view should also respect the labels setting
          const showLabelsDefault = settings.mapSettings?.showLabels !== false;
          if (showLabelsDefault) {
            map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
          } else {
            map.current.setMapTypeId(window.google.maps.MapTypeId.SATELLITE);
          }
      }
    }
  }, [activeBasemap, isLoaded, settings]);

  // Function to toggle 3D mode - simplified approach with 2D map always visible
  const toggle3DMode = () => {
    if (!is3DSupported) return;
    
    try {
      const newMode = !is3DMode;
      console.log(`Toggling 3D mode: ${is3DMode} → ${newMode}`);
      
      if (newMode) {
        // Going from 2D to 3D - save current position before mounting 3D view
        if (map.current) {
          // Capture current position from 2D map
          const mapCenter = map.current.getCenter();
          const position = {
            center: [mapCenter.lng(), mapCenter.lat()],
            zoom: map.current.getZoom()
          };
          
          console.log("Toggling to 3D with position:", position);
          
          // Store in Redux for the 3D view to use when mounted
          dispatch(setMapView({
            center: position.center,
            zoom: position.zoom
          }));
          
          // Immediately set map to satellite but DON'T set tilt yet (to avoid built-in 3D)
          map.current.setMapTypeId(google.maps.MapTypeId.SATELLITE);
          
          // Keep tilt at 0 to avoid triggering Google Maps' built-in 3D mode
          map.current.setTilt(0);
          
          // Just update the state variable (without changing actual map tilt)
          setMapTilt(45);
        }
        
        // Announce 3D mode is active
        window.dispatchEvent(new CustomEvent('mapModeChanged', { 
          detail: { mode: '3D', tilt: 45 } 
        }));
      } else {
        // Going from 3D to 2D - get position from Redux (updated by Cesium)
        const state = store.getState();
        const { center, zoom } = state.map;
        
        // Apply position to the 2D map (which is always loaded)
        if (map.current && center && zoom) {
          // Force map to refresh
          window.google.maps.event.trigger(map.current, 'resize');
          
          // Set the position
          map.current.setCenter({ lat: center[1], lng: center[0] });
          map.current.setZoom(zoom);
        }
        
        // Reset map appearance for 2D viewing
        if (map.current) {
          // Return to normal map type
          if (activeBasemap) {
            map.current.setMapTypeId(google.maps.MapTypeId[activeBasemap.toUpperCase()]);
          } else {
            map.current.setMapTypeId(google.maps.MapTypeId.ROADMAP);
          }
          
          // Reset tilt to 0 for 2D view
          map.current.setTilt(0);
          setMapTilt(0);
        }
        
        // Announce 2D mode is active
        window.dispatchEvent(new CustomEvent('mapModeChanged', { 
          detail: { mode: '2D', tilt: 0 } 
        }));
      }
      
      // Update state immediately - just one flag
      setIs3DMode(newMode);
    } catch (error) {
      console.error('Error toggling 3D mode:', error);
      setIs3DSupported(false);
    }
  };
  
  // Listen for tilt changes to keep state in sync
  useEffect(() => {
    if (!map.current || !isLoaded) return;
    
    try {
      const tiltChangedListener = map.current.addListener('tilt_changed', () => {
        if (map.current) {
          const newTilt = map.current.getTilt();
          setMapTilt(newTilt);
          setIs3DMode(newTilt > 0);
        }
      });
      
      const headingChangedListener = map.current.addListener('heading_changed', () => {
        if (map.current) {
          setMapHeading(map.current.getHeading());
        }
      });
      
      return () => {
        if (google && google.maps) {
          google.maps.event.removeListener(tiltChangedListener);
          google.maps.event.removeListener(headingChangedListener);
        }
      };
    } catch (error) {
      console.warn('Error setting up tilt/heading listeners:', error);
    }
  }, [map.current, isLoaded]);
  
  /**
   * Event listener for navigation events that require map adjustments
   * 
   * When users navigate back to the Map view from other views (like Settings),
   * we need to ensure the map renders correctly by triggering resize events.
   * This helps prevent visual glitches after navigation.
   */
  useEffect(() => {
    const handleNavigateTo = (event) => {
      // When navigating back to Map view, ensure proper rendering
      if (event.detail && event.detail.view === 'Map') {
        // Force map resize to ensure proper rendering
        setTimeout(() => {
          if (map.current) {
            window.google.maps.event.trigger(map.current, 'resize');
          }
          window.dispatchEvent(new Event('resize'));
        }, 100);
      }
    };
    
    window.addEventListener('navigateTo', handleNavigateTo);
    
    return () => {
      window.removeEventListener('navigateTo', handleNavigateTo);
    };
  }, []);
  
  // Listen for settings changes in localStorage and custom API change events
  useEffect(() => {
    // Handler for storage changes (from other tabs/windows)
    const handleStorageChange = () => {
      const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
      if (savedSettings) {
        try {
          const settings = JSON.parse(savedSettings);
          setMap3DApi(settings.map3DApi || 'cesium');
        } catch (e) {
          console.error('Error parsing settings:', e);
        }
      }
    };
    
    // Handler for direct API change events (from Settings panel)
    const handleApiChange = (event) => {
      const { newApi, previousApi } = event.detail;
      console.log(`3D API changing from ${previousApi} to ${newApi}`);
      
      // Update the map3DApi state
      setMap3DApi(newApi);
      
      // If we're in 3D mode, we'll need to re-enter it with the new API
      // This event is dispatched after exiting 3D mode if needed
    };
    
    // Handler for forced 2D mode (for API switching)
    const handleForced2D = (event) => {
      if (event.detail && event.detail.force2D && is3DMode) {
        console.log("Forcing exit from 3D mode to apply API change");
        toggle3DMode(); // Exit 3D mode
      }
    };
    
    /**
     * Handler for 3D toggle events from sidebar or settings
     * 
     * This handler responds to requestToggle3DViewType events, which are used to:
     * 1. Toggle between 2D and 3D views (when coming back from Settings)
     * 2. Optionally change the 3D API if specified
     * 3. Ensure 3D state matches the requested state
     */
    const handleToggle3DViewType = (event) => {
      if (event.detail) {
        // Extract the show3D flag and optional map3DApi
        const { show3D, map3DApi: requestedApi } = event.detail;
        
        // If the API is specified, update it first
        if (requestedApi && requestedApi !== map3DApi) {
          console.log(`Changing 3D API to ${requestedApi} before toggling view`);
          setMap3DApi(requestedApi);
        }
        
        // Check if we need to toggle the view (only toggle if state doesn't match)
        if (show3D !== is3DMode) {
          console.log(`Toggling 3D mode to: ${show3D}`);
          toggle3DMode();
        }
      }
    };
    
    // Add event listeners
    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('map3DApiChanged', handleApiChange);
    window.addEventListener('requestToggle3D', handleForced2D);
    window.addEventListener('requestToggle3DViewType', handleToggle3DViewType);
    
    // Load settings on mount
    handleStorageChange();
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('map3DApiChanged', handleApiChange);
      window.removeEventListener('requestToggle3D', handleForced2D);
      window.removeEventListener('requestToggle3DViewType', handleToggle3DViewType);
    };
  }, [is3DMode, map3DApi, toggle3DMode]);

  // Make the map ref globally available for Gemini integration
  useEffect(() => {
    // Store the map reference globally
    window.mapRef = ref;
    
    return () => {
      // Clean up global reference when component unmounts
      window.mapRef = null;
    };
  }, [ref]);
  
  // Expose map and methods to parent via ref
  useImperativeHandle(ref, () => ({
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
    handleMarkerClick: (marker, tree, markerColor) => {
      handleMarkerClick(marker, tree, markerColor);
    },
    toggle3DMode: () => {
      toggle3DMode();
    },
    getMapMode: () => ({
      is3DMode,
      is3DSupported,
      tilt: mapTilt,
      heading: mapHeading
    }),
    
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
          
          // Get basic map view data - this should always work with a valid map
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
              zoom: zoom,
              heading: heading,
              tilt: tilt,
              is3D: is3D
            };
            
            // Get additional coordinates for backend tile retrieval
            let coordsInfo = null;
            try {
              // Get the tile coordinates that are currently visible
              const mapType = map.current.getMapTypeId();
              const isSatellite = mapType === 'satellite' || mapType === 'hybrid';
              
              // Create a position object with detailed information
              const position = {
                bounds: [
                  [mapBounds.getSouthWest().lng(), mapBounds.getSouthWest().lat()],
                  [mapBounds.getNorthEast().lng(), mapBounds.getNorthEast().lat()]
                ],
                center: [center.lng(), center.lat()],
                zoom: zoom,
                isSatellite: isSatellite,
                mapType: mapType,
                timestamp: Date.now()
              };
              
              coordsInfo = JSON.stringify(position);
              console.log("Successfully retrieved map coordinates");
            } catch (coordsError) {
              console.warn("Error getting detailed map coordinates:", coordsError);
            }
            
            // Create the result object with comprehensive location data
            const result = {
              viewData: {
                ...viewData,
                coordsInfo: coordsInfo,
                useBackendTiles: true
              },
              is3D: is3D,
              mapType: mapTypeId,
              timestamp: Date.now()
            };
            
            // Resolve with the coordinate information
            resolve(result);
          } catch (mapError) {
            console.error("Error capturing map data:", mapError);
            
            // Fallback to using Redux state
            const mapState = store.getState().map;
            const errorResult = { 
              captureError: "Error capturing map data",
              viewData: {
                center: mapState.center,
                zoom: mapState.zoom,
                bounds: [
                  [mapState.center[0] - 0.003, mapState.center[1] - 0.003],
                  [mapState.center[0] + 0.003, mapState.center[1] + 0.003]
                ],
                useBackendTiles: true
              },
              is3D: is3DMode,
              mapType: 'satellite'
            };
            
            console.log("Returning fallback error result with map state");
            resolve(errorResult);
          }
        } catch (criticalError) {
          console.error("Critical error in captureCurrentView:", criticalError);
          reject(criticalError);
        }
      });
    },
    showHighRiskOnly: () => {
      if (map.current && window.google) {
        console.log("Creating high risk markers only");
        
        // Clear the markers array after the markers have been removed from the map
        markersRef.current = [];
        
        // Add tree markers if the trees layer is visible
        if (trees.length > 0) {
          trees.forEach(tree => {
            // Skip non-high risk trees
            if (!tree.risk_factors || !tree.risk_factors.some(rf => rf.level === 'high')) {
              return;
            }
            
            if (tree.location) {
              const [lng, lat] = tree.location;
              
              // Create marker with standard marker approach
              const marker = new window.google.maps.Marker({
                position: { lat, lng },
                map: map.current,
                title: `${tree.species} - ${tree.height}ft`,
                icon: {
                  path: window.google.maps.SymbolPath.CIRCLE,
                  fillColor: 'red',
                  fillOpacity: 0.8,
                  strokeColor: 'white',
                  strokeWeight: 1,
                  scale: 8
                }
              });
              
              // Add click listener to marker
              marker.addListener('click', () => {
                handleMarkerClick(marker, tree, 'red');
              });
              
              // Add marker to markers array for later reference
              markersRef.current.push(marker);
            }
          });
        }
      }
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
        // Create confirmation dialog
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
      
      // For each tree, create a rectangle and marker with distinctive styling
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
          let markerColor, strokeColor, fillOpacity, strokeWeight, markerShape, markerScale;
          
          if (source === 'ai') {
            // AI-detected trees - use square markers with dashed border
            markerShape = window.google.maps.SymbolPath.CIRCLE;
            markerScale = 10;
            strokeWeight = 2;
            fillOpacity = 0.4;
            
            // Color based on risk level
            if (tree.risk_level === 'high') {
              markerColor = '#FF4136'; // Bright red
              strokeColor = '#FF0000'; // Red border
            } else if (tree.risk_level === 'medium') {
              markerColor = '#FF851B'; // Orange
              strokeColor = '#FF6600'; // Dark orange border
            } else {
              markerColor = '#2ECC40'; // Light green
              strokeColor = '#20A030'; // Slightly darker green border
            }
          } else {
            // Validation queue trees - use circular markers with solid border
            markerShape = window.google.maps.SymbolPath.CIRCLE;
            markerScale = 8;
            strokeWeight = 1;
            fillOpacity = 0.8;
            
            // More muted colors for validation trees
            if (tree.risk_level === 'high') {
              markerColor = '#BB0000'; // Dark red
              strokeColor = 'white';
            } else if (tree.risk_level === 'medium') {
              markerColor = '#DD7700'; // Dark orange
              strokeColor = 'white';
            } else {
              markerColor = '#19A030'; // Darker green for validation queue trees
              strokeColor = 'white';
            }
          }
          
          // Create a marker at the tree location
          const marker = new window.google.maps.Marker({
            position: { lat, lng },
            map: mapInstance,
            title: `${tree.species || 'Unknown Tree'} - ${tree.height || 'Unknown'}m`,
            icon: {
              path: markerShape,
              fillColor: markerColor,
              fillOpacity: fillOpacity,
              strokeColor: strokeColor,
              strokeWeight: strokeWeight,
              scale: markerScale
            }
          });
          
          // For AI-detected trees, add a distinctive bounding box
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
            const rectangle = new window.google.maps.Rectangle({
              bounds: bounds,
              strokeColor: strokeColor,
              strokeOpacity: 0.8,
              strokeWeight: 2,
              fillColor: markerColor,
              fillOpacity: 0.08,
              map: mapInstance,
              clickable: true,
              draggable: false
            });
            
            // Add click listener to rectangle
            rectangle.addListener('click', () => {
              handleMarkerClick(marker, tree, markerColor, source);
            });
            
            // Store rectangle
            newMarkers.push(rectangle);
          }
          
          // Add click listener to marker
          marker.addListener('click', () => {
            handleMarkerClick(marker, tree, markerColor, source);
          });
          
          // Store marker
          newMarkers.push(marker);
          
        } catch (error) {
          console.error(`Error creating marker for tree ${index}:`, error);
        }
      });
      
      // Add all new markers to the markers reference array
      markersRef.current = [...markersRef.current, ...newMarkers];
      
      console.log(`Added ${newMarkers.length} new markers for ${treeData.length} ${source} trees`);
      return true;
    },
    filterTrees: (riskFilter, treeSpecies) => {
      if (!map.current || !window.google) return;
      
      console.log("Filtering trees by risk:", riskFilter, "and species:", treeSpecies);
      
      // Remove all existing markers from the map
      markersRef.current.forEach(marker => {
        try {
          if (marker instanceof window.google.maps.Marker) {
            marker.setMap(null);
          } else if (marker instanceof window.google.maps.InfoWindow) {
            marker.close();
          }
        } catch (error) {
          console.warn("Error removing marker:", error);
        }
      });
      
      // Clear the markers array after the markers have been removed from the map
      markersRef.current = [];
      
      // Set global filters for other components to reference
      window.currentRiskFilter = riskFilter;
      window.currentSpeciesFilter = treeSpecies;
      window.showOnlyHighRiskTrees = riskFilter !== 'all';
      window.highRiskFilterActive = riskFilter === 'high' || riskFilter === 'high_medium';
      
      // Add tree markers if the trees layer is visible
      if (trees.length > 0) {
        trees.forEach(tree => {
          // Apply risk level filter
          if (riskFilter !== 'all') {
            if (riskFilter === 'high_medium' && (!tree.risk_factors || 
                !tree.risk_factors.some(rf => rf.level === 'high' || rf.level === 'medium'))) {
              return;
            } else if (riskFilter === 'high' && (!tree.risk_factors || !tree.risk_factors.some(rf => rf.level === 'high'))) {
              return;
            } else if (riskFilter === 'medium' && (!tree.risk_factors || !tree.risk_factors.some(rf => rf.level === 'medium'))) {
              return;
            } else if (riskFilter === 'low' && (!tree.risk_factors || !tree.risk_factors.some(rf => rf.level === 'low'))) {
              return;
            }
          }
          
          // Apply species filter
          if (treeSpecies && tree.species !== treeSpecies) {
            return;
          }
          
          if (tree.location) {
            const [lng, lat] = tree.location;
            
            // Determine marker color based on risk level
            let markerColor = 'green';
            if (tree.risk_factors && tree.risk_factors.length > 0) {
              const highestRisk = tree.risk_factors.reduce((highest, current) => {
                if (current.level === 'high') return 'high';
                if (current.level === 'medium' && highest !== 'high') return 'medium';
                return highest;
              }, 'low');
              
              markerColor = highestRisk === 'high' ? 'red' : (highestRisk === 'medium' ? 'orange' : 'green');
            }
            
            // Create marker with standard marker approach
            const marker = new window.google.maps.Marker({
              position: { lat, lng },
              map: map.current,
              title: `${tree.species} - ${tree.height}ft`,
              icon: {
                path: window.google.maps.SymbolPath.CIRCLE,
                fillColor: markerColor,
                fillOpacity: 0.8,
                strokeColor: 'white',
                strokeWeight: 1,
                scale: 8
              }
            });
            
            // Add click listener to marker
            marker.addListener('click', () => {
              handleMarkerClick(marker, tree, markerColor);
            });
            
            // Add marker to markers array for later reference
            markersRef.current.push(marker);
          }
        });
      }
    }
  }));

  // Add effect to listen for entering tree validation mode
  useEffect(() => {
    const handleEnterValidationMode = async (event) => {
      if (!event.detail) {
        console.log("Entering validation mode without details");
        setIsValidationMode(true);
        return;
      }
      
      // Check for initialDetectionOnly flag - just open the sidebar without running detection
      if (event.detail.initialDetectionOnly) {
        console.log("Entering detection mode without triggering tree detection");
        setIsValidationMode(true);
        
        // Initialize with empty data since we'll run detection later
        setDetectedTrees([]);
        setValidationData({
          jobId: 'pending_detection',
          mode: 'detection',
          treeCount: 0,
          source: 'sidebar',
          useSatelliteImagery: true
        });
        return;
      }
      
      // Handle the case when triggered from sidebar with source='sidebar'
      if (event.detail.source === 'sidebar') {
        console.log("Entering validation mode from sidebar click");
        setIsValidationMode(true);
        return; // We don't need to do any further processing
      }
      
      // In 3D mode, don't do any tree loading or bounds changing
      if (is3DMode) {
        console.log("Entering tree validation mode in 3D view");
        setIsValidationMode(true);
        return;
      }
      
      if (!map.current || !isLoaded) return;
      
      try {
        const { detectionJobId, areaId, treeCount, trees, bounds } = event.detail;
        
        console.log(`Entering tree validation mode for detection ${detectionJobId} with ${treeCount} trees`);
        console.log("Received trees in event:", trees);
        
        // Update UI state first to enter validation mode immediately
        setIsValidationMode(true);
        
        // We have two possibilities:
        // 1. Trees were included directly in the event detail (preferred)
        // 2. Need to fetch trees from the backend
        
        if (trees && trees.length > 0) {
          // Use trees from the event detail
          console.log(`Using ${trees.length} trees from event detail`);
          setDetectedTrees(trees);
          setValidationData({
            jobId: detectionJobId,
            areaId: areaId,
            treeCount: trees.length,
            bounds: bounds
          });
        } else {
          // Fetch detection results and trees
          console.log("No trees in event detail, fetching from backend");
          const detectionResult = await DetectionService.getDetectionStatus(detectionJobId);
          
          if (detectionResult.status !== 'complete' && 
              detectionResult.status !== 'validated' && 
              detectionResult.status !== 'complete_no_detections') {
            console.error('Detection is not complete yet, status:', detectionResult.status);
            return;
          }
          
          // Use trees from detection result if available
          if (detectionResult.trees && detectionResult.trees.length > 0) {
            console.log(`Using ${detectionResult.trees.length} trees from detection result`);
            setDetectedTrees(detectionResult.trees);
          } else {
            // Last resort - fetch trees for area
            console.log("No trees in detection result, fetching from area endpoint");
            const treesForArea = await TreeService.getTreesForArea(areaId);
            setDetectedTrees(treesForArea);
          }
          
          setValidationData({
            jobId: detectionJobId,
            areaId: areaId,
            treeCount: detectionResult.tree_count || treeCount,
            bounds: bounds,
            detectionResult: detectionResult
          });
        }
        
        // After a brief delay to ensure the UI has updated, fit map to detection bounds or tree positions
        setTimeout(() => {
          try {
            if (bounds && bounds.length === 2) {
              const sw = new window.google.maps.LatLng(bounds[0][1], bounds[0][0]);
              const ne = new window.google.maps.LatLng(bounds[1][1], bounds[1][0]);
              const boundingBox = new window.google.maps.LatLngBounds(sw, ne);
              map.current.fitBounds(boundingBox);
              console.log("Fit map to bounds:", bounds);
            } else if (trees && trees.length > 0) {
              // Create bounds from tree positions
              const treeBounds = new window.google.maps.LatLngBounds();
              let treeCount = 0;
              
              trees.forEach(tree => {
                if (tree.location && tree.location.length === 2) {
                  treeBounds.extend(new window.google.maps.LatLng(tree.location[1], tree.location[0]));
                  treeCount++;
                }
              });
              
              if (treeCount > 0) {
                map.current.fitBounds(treeBounds);
                console.log(`Fit map to bounds of ${treeCount} trees`);
              }
            }
          } catch (error) {
            console.error("Error fitting map to bounds:", error);
          }
        }, 200);
      } catch (error) {
        console.error('Error entering validation mode:', error);
        alert('Failed to enter validation mode: ' + error.message);
      }
    };
    
    // Add handler for map capture events for tree detection
    const handleMapCapture = async (event) => {
      console.log("Handling map capture request for tree detection");
      
      try {
        // Capture map view data, whether in 2D or 3D mode
        const captureResult = await ref.current.captureCurrentView();
        console.log("Map capture result:", captureResult);
        
        // Check if the function returned a valid result
        if (captureResult && event.detail && event.detail.requestData) {
          // Add the viewData for the view (contains map center, zoom, bounds, etc.)
          event.detail.requestData.map_view_info = captureResult;
          
          // For Gemini API, check if the image URL was captured successfully
          if (captureResult.viewData && captureResult.viewData.imageUrl) {
            const imageUrl = captureResult.viewData.imageUrl;
            console.log("Image URL successfully captured for Gemini API", {
              length: imageUrl.length,
              sample: imageUrl.substring(0, 30) + "..."
            });
            
            // IMPORTANT: The backend looks for map_image first, so always set it
            // regardless of URL format. The backend knows how to handle data: URLs
            event.detail.requestData.map_image = imageUrl;
            
            // Flag that we've included image data
            event.detail.requestData.has_image_data = true;
            event.detail.requestData.capture_method = is3DMode ? "3d_viewport" : "2d_map";
            
            // Set debugging info to track image transmission
            event.detail.requestData._debug_image_size = imageUrl.length;
            
            // Log complete request for debugging
            console.log("Tree detection request prepared with image data");
          } else {
            console.warn("No image URL captured - Gemini API may not work correctly");
            event.detail.requestData.capture_error = "Failed to capture image data";
            event.detail.requestData.has_image_data = false;
          }
          
          console.log("Added view data to detection request", event.detail.requestData);
        }
      } catch (error) {
        console.error("Error capturing view:", error);
        
        // Still add basic view info even if capture failed
        if (event.detail && event.detail.requestData) {
          event.detail.requestData.capture_error = error.message;
          
          // Add minimal view info
          if (!event.detail.requestData.map_view_info) {
            event.detail.requestData.map_view_info = {
              viewData: {
                center: center,
                zoom: zoom,
                is3D: is3DMode
              },
              is3D: is3DMode
            };
          }
        }
      }
    };
    
    // Handler for tree detection results
    const handleTreeDetectionResult = async (event) => {
      const { jobId, status, trees } = event.detail;
      
      console.log(`Received tree detection result for job ${jobId}:`, {
        status,
        treeCount: trees ? trees.length : 0
      });
      
      if (status === 'complete' && trees && trees.length > 0) {
        // Use our new renderDetectedTrees function to show the trees with bounding boxes
        if (ref.current && ref.current.renderDetectedTrees) {
          console.log(`Rendering ${trees.length} detected trees with bounding boxes`);
          ref.current.renderDetectedTrees(trees, 'ai');
        } else {
          console.warn("renderDetectedTrees function not available");
        }
      } else if (status === 'error') {
        // Just log the error but don't show a second alert (the service API already shows one)
        console.error("Tree detection failed:", event.detail.message);
      }
    };
    
    // Handler for Feature Selection button from Analytics menu
    const handleOpenFeatureSelection = (event) => {
      console.log("Feature Selection requested", event?.detail);
      const mode = event?.detail?.mode || 'feature_selection';
      const clearExisting = event?.detail?.clearExisting === true;
      const requestedTab = event?.detail?.tab; // Get the requested tab if any
      
      // If clearExisting is set, ensure we clear any existing validation data
      if (clearExisting) {
        console.log("Clearing existing validation data");
        setDetectedTrees([]);
        setValidationData(null);
      }
      
      // Set validation mode with the right context
      setIsValidationMode(true);
      
      // Fetch available trees or use previously detected trees
      const availableTrees = detectedTrees.length > 0 && !clearExisting ? 
        detectedTrees : trees;
        
      if (availableTrees.length > 0) {
        console.log(`Using ${availableTrees.length} available trees for Feature Selection, mode: ${mode}`);
        setDetectedTrees(availableTrees);
        
        // Set validation data with information for the validation process
        setValidationData({
          jobId: mode,
          areaId: "current_view",
          treeCount: availableTrees.length,
          bounds: null,
          mode: "feature_selection",
          tab: requestedTab // Pass the requested tab if specified
        });
        
        // If we have bounds for these trees, fit map to them
        if (map.current) {
          try {
            const bounds = new window.google.maps.LatLngBounds();
            let treesWithLocation = 0;
            
            availableTrees.forEach(tree => {
              if (tree.location && tree.location.length === 2) {
                bounds.extend(new window.google.maps.LatLng(tree.location[1], tree.location[0]));
                treesWithLocation++;
              }
            });
            
            if (treesWithLocation > 0) {
              map.current.fitBounds(bounds);
              console.log(`Fit map to bounds of ${treesWithLocation} trees`);
            }
          } catch (error) {
            console.error("Error fitting map to tree bounds:", error);
          }
        }
      } else {
        console.log("No trees available for Feature Selection, opening empty view");
        // Set empty validation data to start fresh
        setValidationData({
          jobId: `${mode}_empty`,
          areaId: "current_view",
          treeCount: 0,
          bounds: null,
          mode: "feature_selection",
          tab: requestedTab // Pass the requested tab if specified
        });
      }
    };
    
    // Generic function to handle map resize after sidebar toggle
    const handleMapResize = (isCollapsed) => {
      console.log("Sidebar state changed:", isCollapsed ? "collapsed" : "expanded", "3D Mode:", is3DMode);
      
      // For 3D mode, we need to handle resize differently
      if (is3DMode) {
        // Trigger resize on the window for 3D viewers
        window.dispatchEvent(new Event('resize'));
        
        // Schedule a few resizes to ensure 3D view properly adjusts
        const resizeSequence = [100, 300, 500];
        resizeSequence.forEach(delay => {
          window.setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
          }, delay);
        });
        
        return;
      }
      
      // 2D mode with Google Maps
      if (window.google && map.current) {
        // Multi-step approach to ensure smooth map resizing:
        
        // 1. First trigger a resize immediately for quick response
        window.google.maps.event.trigger(map.current, 'resize');
        
        // 2. Schedule a series of resize events during and after the CSS transition
        const resizeSequence = [50, 150, 250, 350, 450];
        
        resizeSequence.forEach(delay => {
          window.setTimeout(() => {
            // Trigger resize on both window and map
            window.dispatchEvent(new Event('resize'));
            if (map.current) {
              window.google.maps.event.trigger(map.current, 'resize');
              
              // Recenter to maintain focus
              if (map.current.getCenter) {
                const currentCenter = map.current.getCenter();
                map.current.setCenter(currentCenter);
              }
            }
          }, delay);
        });
        
        // 3. After the transition is fully complete, do a final resize and minor zoom adjustment
        // to force Google Maps to redraw all tiles correctly
        window.setTimeout(() => {
          if (map.current) {
            window.google.maps.event.trigger(map.current, 'resize');
            
            // Slightly adjust zoom to force complete redraw
            const zoom = map.current.getZoom();
            map.current.setZoom(zoom - 0.001);
            setTimeout(() => {
              if (map.current) {
                map.current.setZoom(zoom);
                
                // Final recenter
                const currentCenter = map.current.getCenter();
                map.current.setCenter(currentCenter);
                
                console.log("Map resize complete after sidebar toggle");
              }
            }, 50);
          }
        }, 500);
      }
    };
    
    // Listen for tree validation sidebar toggle events
    const handleValidationSidebarToggle = (event) => {
      const isCollapsed = event.detail.collapsed;
      const source = event.detail.source || 'unknown';
      
      console.log(`TreeValidationMode sidebar toggle: ${isCollapsed ? 'collapsed' : 'expanded'} (source: ${source})`);
      setIsRightSidebarCollapsed(isCollapsed);
      
      // Resize the map
      handleMapResize(isCollapsed);
    };
    
    // Listen for validation queue sidebar toggle events
    const handleValidationQueueToggle = (event) => {
      const isCollapsed = event.detail.collapsed;
      const source = event.detail.source || 'unknown';
      
      console.log(`ValidationQueue sidebar toggle: ${isCollapsed ? 'collapsed' : 'expanded'} (source: ${source})`);
      
      // We don't need to update the isRightSidebarCollapsed state here since
      // this sidebar is not managed by the MapView component
      
      // Resize the map
      handleMapResize(isCollapsed);
    };
    
    // Handle explicit exit validation mode
    const handleExitValidationMode = () => {
      console.log("Explicitly exiting validation mode");
      setIsValidationMode(false);
      setValidationData(null);
      setDetectedTrees([]);
    };
    
    window.addEventListener('exitValidationMode', handleExitValidationMode);
    window.addEventListener('enterTreeValidationMode', handleEnterValidationMode);
    window.addEventListener('captureMapViewForDetection', handleMapCapture);
    window.addEventListener('treeDetectionResult', handleTreeDetectionResult);
    window.addEventListener('openFeatureSelection', handleOpenFeatureSelection);
    window.addEventListener('validationSidebarToggle', handleValidationSidebarToggle);
    window.addEventListener('validationQueueToggle', handleValidationQueueToggle);
    
    return () => {
      window.removeEventListener('exitValidationMode', handleExitValidationMode);
      window.removeEventListener('enterTreeValidationMode', handleEnterValidationMode);
      window.removeEventListener('captureMapViewForDetection', handleMapCapture);
      window.removeEventListener('treeDetectionResult', handleTreeDetectionResult);
      window.removeEventListener('openFeatureSelection', handleOpenFeatureSelection);
      window.removeEventListener('validationSidebarToggle', handleValidationSidebarToggle);
      window.removeEventListener('validationQueueToggle', handleValidationQueueToggle);
    };
  }, [isLoaded, map.current, is3DMode]);

  // Exit validation mode
  const handleExitValidationMode = (event) => {
    // Get the source of the exit request if available
    const source = event?.detail?.source || 'unknown';
    const target = event?.detail?.target || 'unknown';
    const clearExisting = event?.detail?.clearExisting === true;
    
    console.log(`Exiting validation mode from source: ${source}, target: ${target}, clearExisting: ${clearExisting}`);
    
    // For Imagery button, always fully reset validation mode
    if (target === 'imagery' || clearExisting || source === 'unknown') {
      setIsValidationMode(false);
      setValidationData(null);
      setDetectedTrees([]);
      
      // Refresh map markers
      const refreshEvent = new CustomEvent('refreshMapMarkers');
      window.dispatchEvent(refreshEvent);
      
      console.log("Full exit - clearing all validation data for Imagery view");
    } else {
      // For transitions between modes, just update the validation mode 
      // without clearing detected trees data
      setIsValidationMode(false);
      
      console.log("Partial exit - keeping detected trees data for mode switch");
    }
  };
  
  // Handle marker click
  const handleMarkerClick = (marker, tree, markerColor) => {
    // Close any open InfoWindows first
    markersRef.current.forEach(item => {
      if (item instanceof window.google.maps.InfoWindow) {
        item.close();
      }
    });
    
    // Create risk level text and color
    let riskLevelText = "Low Risk";
    let riskLevelColor = "#19A030"; // Default to low risk green
    
    if (tree.risk_level === 'high') {
      riskLevelText = "High Risk";
      riskLevelColor = "#FF4136";
    } else if (tree.risk_level === 'medium') {
      riskLevelText = "Medium Risk";
      riskLevelColor = "#FF851B";
    }
    
    // Create and open InfoWindow with elegant styling
    const infoWindow = new window.google.maps.InfoWindow({
      content: `
        <div style="min-width: 240px; padding: 10px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
          <h3 style="margin-top: 0; margin-bottom: 10px; font-weight: 600; font-size: 16px; color: #333; border-bottom: 1px solid #eee; padding-bottom: 8px;">${tree.species || 'Unknown Tree'}</h3>
          
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
            <div>
              <div style="font-size: 12px; color: #666; margin-bottom: 2px;">Height</div>
              <div style="font-weight: 500;">${tree.height || 'Unknown'} ft</div>
            </div>
            <div>
              <div style="font-size: 12px; color: #666; margin-bottom: 2px;">Diameter</div>
              <div style="font-weight: 500;">${tree.diameter || 'Unknown'} in</div>
            </div>
          </div>
          
          <div style="display: flex; align-items: center; margin-top: 10px;">
            <div style="flex-grow: 1;">
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
        </div>
      `
    });
    
    infoWindow.open(map.current, marker);
    
    // Add InfoWindow to markers array for later reference
    markersRef.current.push(infoWindow);
    
    // Update selected feature in Redux store
    dispatch(setSelectedFeature(tree));
  };
  
  // Handle saving validated trees
  const handleSaveValidatedTrees = async (trees) => {
    if (!validationData || !validationData.areaId) {
      alert('Missing area ID for validation');
      return;
    }
    
    try {
      // Get area ID from validation data
      const { areaId } = validationData;
      
      // Save trees
      const saveResult = await TreeService.saveValidatedTrees(areaId, trees);
      
      // Exit validation mode 
      setIsValidationMode(false);
      setValidationData(null);
      setDetectedTrees([]);
      
      // Show success message
      alert(`Successfully saved ${trees.length} validated trees.`);
      
      // Refresh map markers
      const refreshEvent = new CustomEvent('refreshMapMarkers');
      window.dispatchEvent(refreshEvent);
      
    } catch (error) {
      console.error('Error saving validated trees:', error);
      alert('Failed to save trees: ' + error.message);
    }
  };

  // Listen for sidebar changes to resize map with debounced handling
  useEffect(() => {
    let resizeTimeout;
    let resizeCount = 0;
    
    const handleResize = (event) => {
      clearTimeout(resizeTimeout);
      
      // Increment resize count for logging
      resizeCount++;
      
      // Debounce resize events to prevent too many triggers
      resizeTimeout = setTimeout(() => {
        if (map.current && window.google && window.google.maps) {
          try {
            window.google.maps.event.trigger(map.current, 'resize');
            console.log("Map resize complete after sidebar toggle", resizeCount > 1 ? resizeCount : '');
            resizeCount = 0;
          } catch (e) {
            console.warn("Error triggering map resize:", e);
          }
        }
      }, 150); // Delay resize to let transitions complete
    };
    
    // Add event listeners for sidebar toggles
    window.addEventListener('validationQueueToggle', handleResize);
    window.addEventListener('validationSidebarToggle', handleResize);
    window.addEventListener('leftSidebarToggle', handleResize);
    
    // Also listen for window resize events
    window.addEventListener('resize', handleResize);
    
    // Set up ResizeObserver to monitor map container size changes
    const resizeObserver = new ResizeObserver(entries => {
      handleResize();
    });
    
    // Observe the map container for size changes
    const mapContainer = document.getElementById('map-container');
    if (mapContainer) {
      resizeObserver.observe(mapContainer);
    }
    
    // Initial resize after component mounts
    handleResize();
    
    // Clean up
    return () => {
      clearTimeout(resizeTimeout);
      window.removeEventListener('validationQueueToggle', handleResize);
      window.removeEventListener('validationSidebarToggle', handleResize);
      window.removeEventListener('leftSidebarToggle', handleResize);
      window.removeEventListener('resize', handleResize);
      
      // Disconnect resize observer
      if (mapContainer) {
        resizeObserver.unobserve(mapContainer);
      }
      resizeObserver.disconnect();
    };
  }, []);

  // Listen for viewGeminiResponse events
  useEffect(() => {
    const handleViewGeminiResponse = (event) => {
      const { jobId, responsePath } = event.detail;
      
      // Create a fetch request to get the list of files
      try {
        console.log(`Viewing Gemini response for job ${jobId} at ${responsePath}`);
        
        // Get the window.location string so we can show instructions
        const baseUrl = window.location.origin;
        
        // Show an alert with instructions
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

  return (
    <div className={`relative w-full h-full transition-all duration-300`}>
      {/* Google Maps Container - padding changes when sidebar is visible */}
      <div 
        ref={mapContainer}
        className={`w-full h-full transition-all duration-300 ${is3DMode ? 'hidden' : 'block'}`}
        style={{ 
          position: 'absolute', 
          top: 0, 
          right: 0, 
          bottom: 0, 
          left: 0 
        }}
      ></div>
      
      {/* Removed the background overlay as it wasn't enhancing visibility */}
      
      {/* 3D View Toggle has been moved to the sidebar to avoid duplication */}
      
      {/* 3D Viewer based on selected API */}
      {is3DMode && (
        <Suspense fallback={<div className="w-full h-full flex items-center justify-center">Loading 3D View...</div>}>
          <div className="w-full h-full transition-all duration-300" style={{ 
            position: 'absolute', 
            top: 0, 
            right: 0, 
            bottom: 0, 
            left: 0 
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
      
      {/* Tree Validation Mode */}
      {isValidationMode && (
        <TreeValidationMode
          mapRef={map}
          validationData={validationData}
          detectedTrees={detectedTrees}
          onExitValidation={handleExitValidationMode}
          onSaveTrees={handleSaveValidatedTrees}
        />
      )}
    </div>
  );
});

export default MapView;