// src/components/visualization/MapView/MapView.jsx

import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle, lazy, Suspense } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { ErrorBoundary } from '../../../App';
import { Card, CardContent } from '@/components/ui/card';
import { setMapView, setSelectedFeature } from '../../../features/map/mapSlice';
import { useGoogleMapsApi } from '../../../hooks/useGoogleMapsApi';
import { store } from '../../../store';
import { TreeService, PropertyService, DetectionService } from '../../../services/api/apiService';
// Map3DToggle has been moved to MapControls
import DetectionMode from './DetectionMode';

// Import 3D viewer components lazily for better performance
const CesiumViewer = lazy(() => import('./CesiumViewer'));
const GoogleMaps3DViewer = lazy(() => import('./GoogleMaps3DViewer'));


const MapView = forwardRef(({ onDataLoaded, headerState }, ref) => {
  // Override the local header collapsed state if headerState is provided
  useEffect(() => {
    if (headerState !== undefined) {
      setHeaderCollapsed(headerState);
      
      // Dispatch header collapse event for sidebar components to pick up
      window.dispatchEvent(new CustomEvent('headerCollapse', {
        detail: { 
          collapsed: headerState
        }
      }));
    }
  }, [headerState]);
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
                map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
                console.log("Updating to HYBRID mode (no labels) based on settings change");
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
  const [headerCollapsed, setHeaderCollapsed] = useState(true); // Track header state
  
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
        disableDefaultUI: true, // Disable all default UI controls
        zoomControl: true, // Selectively enable only the controls we want
        mapTypeControl: false, // Disable Map/Satellite buttons 
        scaleControl: true,
        streetViewControl: false, // Disable street view
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
  
  // IMPORTANT: ALWAYS enforce HYBRID view to make sure labels appear
  useEffect(() => {
    // Set up interval that forces HYBRID view for satellite
    const interval = setInterval(() => {
      try {
        if (window.google && window.google.maps && map.current) {
          // Force resize
          window.google.maps.event.trigger(map.current, 'resize');
          
          // Get current map type
          const currentMapType = map.current.getMapTypeId();
          
          // ALWAYS ENFORCE HYBRID for labels - critical fix
          if (localStorage.getItem('currentMapType') === 'hybrid' || 
              window.currentMapType === 'hybrid') {
            if (currentMapType !== window.google.maps.MapTypeId.HYBRID) {
              console.log("FORCING HYBRID mode for labels");
              map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
            }
          }
        }
      } catch (e) {
        // Ignore errors
      }
    }, 1000); // Higher frequency to ensure labels appear
    
    return () => clearInterval(interval);
  }, [isLoaded]);
  
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
        
        // Always use hybrid view regardless of settings or saved preferences
        console.log("Initializing map with FORCED HYBRID view, 3D mode:", is3DMode);
        
        // Force hybrid view in localStorage to ensure consistency
        try {
          // Store current map type directly to localStorage with explicit HYBRID value
          localStorage.setItem('currentMapType', 'hybrid');
          
          // Create global flags to ensure hybrid view
          window.FORCE_MAP_VIEW = 'hybrid';
          window.currentMapType = 'hybrid';
          
          // Explicitly tell Google Maps API to use HYBRID mode
          if (window.google && window.google.maps) {
            const HYBRID = window.google.maps.MapTypeId.HYBRID;
            console.log("Explicitly setting global HYBRID preference");
          }
          
          // Update settings if they exist
          const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
          if (savedSettings) {
            try {
              const settings = JSON.parse(savedSettings);
              if (settings) {
                // Ensure hybrid view settings are set
                settings.defaultMapType = 'hybrid';
                settings.mapTypeId = 'hybrid';
                
                // Explicitly ensure labels are enabled for hybrid view
                settings.mapSettings = settings.mapSettings || {};
                settings.mapSettings.showLabels = true;
                
                localStorage.setItem('treeRiskDashboardSettings', JSON.stringify(settings));
                
                // Broadcast the settings update with forceApply flag
                window.dispatchEvent(new CustomEvent('settingsUpdated', {
                  detail: { 
                    settings: settings,
                    source: 'mapInit',
                    forceApply: true
                  }
                }));
              }
            } catch (e) {
              console.error("Error parsing settings JSON:", e);
              
              // Create new settings if parsing failed
              const defaultSettings = { 
                defaultMapType: 'hybrid', 
                mapTypeId: 'hybrid',
                mapSettings: { showLabels: true } 
              };
              localStorage.setItem('treeRiskDashboardSettings', JSON.stringify(defaultSettings));
            }
          } else {
            // Create new settings if none exist
            const defaultSettings = { 
              defaultMapType: 'hybrid', 
              mapTypeId: 'hybrid',
              mapSettings: { showLabels: true } 
            };
            localStorage.setItem('treeRiskDashboardSettings', JSON.stringify(defaultSettings));
          }
        } catch (e) {
          console.error("Error saving hybrid view preference:", e);
        }
        
        // Create map - ALWAYS FORCE HYBRID VIEW AT INITIALIZATION
        map.current = new window.google.maps.Map(mapContainer.current, {
          center: validCenter,
          zoom: zoom || 13,
          mapTypeId: window.google.maps.MapTypeId.HYBRID, // ALWAYS USE HYBRID
          mapId: mapId, // Map ID from environment variables
          disableDefaultUI: false, // Enable default UI temporarily to ensure styles load
          zoomControl: true, // Selectively enable only the controls we want
          mapTypeControl: false, // DISABLE DEFAULT CONTROLS TO AVOID CONFLICTS
          scaleControl: true,
          streetViewControl: false, // Disable street view
          rotateControl: true,
          fullscreenControl: true,
          // 3D settings (support depends on API version and browser)
          tilt: is3DMode ? 45 : 0, // Initial tilt for 3D mode
          heading: 0,
          // No longer needed as we've disabled mapTypeControl
          /*mapTypeControlOptions: {
            mapTypeIds: [
              google.maps.MapTypeId.ROADMAP,
              google.maps.MapTypeId.SATELLITE,
              google.maps.MapTypeId.HYBRID,
              google.maps.MapTypeId.TERRAIN
            ],
            position: google.maps.ControlPosition.TOP_LEFT,
            style: google.maps.MapTypeControlStyle.HORIZONTAL_BAR
          },*/
          // Custom controls
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
        // Don't try to set loadError since it's from the useGoogleMapsApi hook
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

  // Add event listeners for marker placement and location markers
  useEffect(() => {
    // Event handler for adding a marker with address at the current location
    const handleAddLocationMarker = (event) => {
      if (!map.current || !isLoaded || !window.google) return;
      
      try {
        const { position, address } = event.detail;
        const [lng, lat] = position;
        
        console.log(`Adding location marker at [${lng}, ${lat}] with address: ${address}`);
        
        // Remove all existing markers first
        markersRef.current.forEach(item => {
          if (item.setMap) {
            item.setMap(null);
          }
        });
        
        // Clear the markers array
        markersRef.current = [];
        
        // Create a marker at the specified location
        const marker = new window.google.maps.Marker({
          position: { lat, lng },
          map: map.current,
          animation: window.google.maps.Animation.DROP,
          title: address || `Location [${lat.toFixed(6)}, ${lng.toFixed(6)}]`
        });
        
        // Add an info window with the address if available
        if (address) {
          const infoWindow = new window.google.maps.InfoWindow({
            content: `<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 8px;">
              <p style="margin: 0; font-weight: 500;">${address}</p>
              <p style="margin: 4px 0 0; font-size: 12px; color: #666;">
                [${lat.toFixed(6)}, ${lng.toFixed(6)}]
              </p>
            </div>`
          });
          
          // Open the info window
          infoWindow.open(map.current, marker);
          
          // Add event listener to open info window when marker is clicked
          marker.addListener('click', () => {
            infoWindow.open(map.current, marker);
          });
          
          // Add info window to markers array
          markersRef.current.push(infoWindow);
        }
        
        // Add marker to markers array
        markersRef.current.push(marker);
        
      } catch (error) {
        console.error('Error adding location marker:', error);
      }
    };
    
    // Simple event handler for adding a center marker (without address)
    const handleAddCenterMarker = (event) => {
      if (!map.current || !isLoaded || !window.google) return;
      
      try {
        const { position } = event.detail;
        const [lng, lat] = position;
        
        console.log(`Adding center marker at [${lng}, ${lat}]`);
        
        // Remove all existing markers first
        markersRef.current.forEach(item => {
          if (item.setMap) {
            item.setMap(null);
          }
        });
        
        // Clear the markers array
        markersRef.current = [];
        
        // Create a marker at the specified location
        const marker = new window.google.maps.Marker({
          position: { lat, lng },
          map: map.current,
          animation: window.google.maps.Animation.DROP
        });
        
        // Add marker to markers array
        markersRef.current.push(marker);
        
      } catch (error) {
        console.error('Error adding center marker:', error);
      }
    };
    
    // Add event listeners
    window.addEventListener('addLocationMarker', handleAddLocationMarker);
    window.addEventListener('addCenterMarker', handleAddCenterMarker);
    
    // Clean up
    return () => {
      window.removeEventListener('addLocationMarker', handleAddLocationMarker);
      window.removeEventListener('addCenterMarker', handleAddCenterMarker);
    };
  }, [isLoaded, map.current]);
  
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
                // Always use HYBRID even when labels disabled to maintain consistency
                map.current.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
                console.log("Using HYBRID mode (labels preference ignored)");
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

  // Completely remove auto map type switching - let the toggle button handle it directly
  useEffect(() => {
    // We're purposely NOT implementing any automatic map type switching here
    // This is because it was conflicting with the direct toggle button control
    
    if (map.current && isLoaded && activeBasemap) {
      console.log(`BasemapEffect - Active basemap changed to: ${activeBasemap}`);
      // ONLY do a resize, but let the user's direct choice control the map view
      setTimeout(() => {
        window.google.maps.event.trigger(map.current, 'resize');
      }, 100);
    }
  }, [activeBasemap, isLoaded]);

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
          
          // Use the current map type or default to HYBRID when entering 3D
          const currentMapType = localStorage.getItem('currentMapType') || 'hybrid';
          console.log(`Preserving current map type when entering 3D: ${currentMapType}`);
          
          // Convert to Google Maps type
          let googleMapType;
          switch (currentMapType) {
            case 'map':
              googleMapType = google.maps.MapTypeId.ROADMAP;
              break;
            case 'satellite':
              googleMapType = google.maps.MapTypeId.SATELLITE;
              break;
            case 'hybrid':
            default:
              googleMapType = google.maps.MapTypeId.HYBRID;
              break;
          }
          
          // Set the map type
          map.current.setMapTypeId(googleMapType);
          
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
          // Return to last selected map type or default to HYBRID
          const savedMapType = localStorage.getItem('currentMapType') || 'hybrid';
          console.log(`Restoring saved map type from 3D to 2D: ${savedMapType}`);
          
          if (savedMapType === 'map') {
            map.current.setMapTypeId(google.maps.MapTypeId.ROADMAP);
          } else if (savedMapType === 'satellite') {
            map.current.setMapTypeId(google.maps.MapTypeId.SATELLITE);
          } else {
            // Default to hybrid
            map.current.setMapTypeId(google.maps.MapTypeId.HYBRID);
          }
          
          // Also set the global state to match
          window.currentMapType = savedMapType;
          
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
      
      // Debug window.google.maps availability to ensure the API is properly loaded
      console.log("Google Maps API availability check:", {
        googleDefined: typeof window.google !== 'undefined',
        mapsApiDefined: window.google && typeof window.google.maps !== 'undefined',
        markerAvailable: window.google && window.google.maps && typeof window.google.maps.Marker !== 'undefined'
      });
      
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
            draggable: true, // Make all markers draggable for repositioning
            icon: {
              path: markerShape,
              fillColor: markerColor,
              fillOpacity: fillOpacity,
              strokeColor: strokeColor,
              strokeWeight: strokeWeight,
              scale: markerScale
            },
            clickable: true,
            raiseOnDrag: true, // Raise above other markers when dragging
            optimized: false, // Disable marker optimization for better interaction
            zIndex: 100 // Ensure marker appears above overlay (overlay z-index is 5)
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
              draggable: true // Make the rectangle draggable too
            });
            
            // Add click listener to rectangle
            rectangle.addListener('click', () => {
              handleMarkerClick(marker, tree, markerColor, source);
            });
            
            // Add drag end listener to rectangle
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
          
          // Simplified dragstart event
          marker.addListener('dragstart', () => {
            console.log("Marker drag started");
            // Make sure marker stays on top during drag
            marker.setZIndex(200); // Higher z-index during drag, but still manageable
          });
          
          // Add drag end event to update tree location
          marker.addListener('dragend', (event) => {
            const newLat = event.latLng.lat();
            const newLng = event.latLng.lng();
            
            // Update tree location in the tree data
            tree.location = [newLng, newLat];
            
            // Update marker position
            marker.setPosition({ lat: newLat, lng: newLng });
            
            // Update bounding box position if it exists
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
            
            // Dispatch an event to notify the UI that a tree was repositioned
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
      
      // If header state is provided in the event, update our local state
      if (event.detail.headerCollapsed !== undefined) {
        console.log("Updating header state from validation mode event:", event.detail.headerCollapsed);
        setHeaderCollapsed(event.detail.headerCollapsed);
        
        // Also dispatch the event for other components
        window.dispatchEvent(new CustomEvent('headerCollapse', {
          detail: { 
            collapsed: event.detail.headerCollapsed
          }
        }));
      }
      
      // Check for initialDetectionOnly flag - just open the sidebar without running detection
      if (event.detail.initialDetectionOnly) {
        console.log("Entering detection mode without triggering tree detection");
        
        // IMPORTANT: Set validation mode first thing
        setIsValidationMode(true);
        
        // Force detection mode state to be visible
        setTimeout(() => {
          window.dispatchEvent(new CustomEvent('setDetectionModeState', {
            detail: { active: true }
          }));
        }, 10);
        
        // Initialize with empty data since we'll run detection later
        setDetectedTrees([]);
        setValidationData({
          jobId: 'pending_detection',
          mode: 'detection',
          treeCount: 0,
          source: 'sidebar',
          useSatelliteImagery: true,
          // Include header state in validation data
          headerCollapsed: event.detail.headerCollapsed !== undefined ? 
            event.detail.headerCollapsed : headerCollapsed
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
      
      // IMPORTANT: First ensure that any existing Detection sidebar is properly closed
      setIsValidationMode(false); // First set state to false to fully remove Detection component
      
      // Also force close using the event - this will trigger internal cleanup
      window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
        detail: { 
          source: 'feature_selection', 
          forceRemove: true 
        }
      }));
      
      // Reset map container width to prevent grey areas
      const mapContainer = document.querySelector('#map-container');
      if (mapContainer) {
        // Reset right property to zero to allow the Database sidebar to set its own width
        mapContainer.style.right = '0px';
      }
      
      // If clearExisting is set, ensure we clear any existing validation data
      if (clearExisting) {
        console.log("Clearing existing validation data");
        setDetectedTrees([]);
        setValidationData(null);
      }
      
      // Short delay before setting validation mode again to ensure clean state
      setTimeout(() => {
        // Set validation mode with the right context
        setIsValidationMode(true);
      }, 100);
      
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
      
      // Update map container margins based on sidebar state
      const mapContainer = document.getElementById('map-container');
      if (mapContainer) {
        if (isCollapsed) {
          mapContainer.style.right = '0';
        } else {
          mapContainer.style.right = '384px';
        }
      }
      
      // Notify the map via rightPanelToggle for responsive layout
      window.dispatchEvent(new CustomEvent('rightPanelToggle', {
        detail: {
          isOpen: !isCollapsed,
          panelWidth: 384,
          panelType: 'validation'
        }
      }));
      
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
    const handleExitValidationMode = (event) => {
      const source = event?.detail?.source || 'unknown';
      const forceRemove = event?.detail?.forceRemove === true;
      const target = event?.detail?.target || 'unknown';
      
      console.log("Explicitly exiting validation mode from source:", source, "target:", target, "with forceRemove:", forceRemove);
      
      // ALWAYS do a complete cleanup
      setIsValidationMode(false);
      setValidationData(null);
      setDetectedTrees([]);
      
      // Always reset map container width to prevent visual artifacts
      const mapContainer = document.getElementById('map-container');
      if (mapContainer) {
        mapContainer.style.right = '0px';
      }
      
      // Force sidebar close by dispatching event directly to DetectionMode component
      window.dispatchEvent(new CustomEvent('setDetectionModeState', {
        detail: { active: false, forceCollapse: true }
      }));
      
      // Dispatch an event to notify other components
      window.dispatchEvent(new CustomEvent('validationModeExited', {
        detail: { source, target, forceRemove: true }
      }));
      
      // Force resize to ensure proper rendering after cleanup
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
        if (map.current && window.google && window.google.maps) {
          window.google.maps.event.trigger(map.current, 'resize');
        }
      }, 50);
    };
    
    // Common function to run ML-based object detection 
    const runObjectDetection = async (mode) => {
      try {
        console.log(`Tree ${mode} requested - Using ML pipeline`);
        
        // Verify ref and captureCurrentView are available
        if (!ref.current || !ref.current.captureCurrentView) {
          throw new Error("Map reference is not available");
        }
        
        // Capture current map view
        const mapViewInfo = await ref.current.captureCurrentView();
        
        // Create a unique job ID for this detection task
        const jobId = `ml_${mode}_${Date.now()}`;
        console.log(`Created job ID: ${jobId} for ${mode} task`);
        
        // Add mode-specific parameters to the map view info
        const requestParams = {
          ...mapViewInfo
        };
        
        // Add mode-specific flags
        if (mode === 'segmentation') {
          requestParams.segmentation_mode = true;
        }
        
        // Import the API service
        const { detectTreesFromMapData } = await import('../../../services/api/treeApi');
        
        // Show loading indicator
        const loadingToast = document.createElement('div');
        loadingToast.className = 'loading-indicator-toast';
        loadingToast.style.position = 'absolute';
        loadingToast.style.top = '70px';
        loadingToast.style.left = '50%';
        loadingToast.style.transform = 'translateX(-50%)';
        loadingToast.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
        loadingToast.style.color = 'white';
        loadingToast.style.padding = '8px 16px';
        loadingToast.style.borderRadius = '4px';
        loadingToast.style.zIndex = '9999';
        loadingToast.style.fontSize = '14px';
        loadingToast.style.fontWeight = '500';
        loadingToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
        loadingToast.textContent = `Running ${mode} using ML pipeline...`;
        document.body.appendChild(loadingToast);
        
        try {
          // Run the ML detection via API
          const result = await detectTreesFromMapData({
            map_view_info: requestParams,
            job_id: jobId
          });
          
          // Remove loading indicator
          if (loadingToast.parentNode) {
            loadingToast.parentNode.removeChild(loadingToast);
          }
          
          // Show success notification
          const successToast = document.createElement('div');
          successToast.className = 'success-toast';
          successToast.style.position = 'absolute';
          successToast.style.top = '70px';
          successToast.style.left = '50%';
          successToast.style.transform = 'translateX(-50%)';
          successToast.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
          successToast.style.color = 'white';
          successToast.style.padding = '8px 16px';
          successToast.style.borderRadius = '4px';
          successToast.style.zIndex = '9999';
          successToast.style.fontSize = '14px';
          successToast.style.fontWeight = '500';
          successToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
          
          const treeCount = result.trees?.length || 0;
          successToast.textContent = `${mode.charAt(0).toUpperCase() + mode.slice(1)} complete: Found ${treeCount} trees`;
          document.body.appendChild(successToast);
          
          // Remove notification after 2.5 seconds
          setTimeout(() => {
            if (successToast.parentNode) {
              successToast.parentNode.removeChild(successToast);
            }
          }, 2500);
          
          // Create a custom event to handle the results
          window.dispatchEvent(new CustomEvent('treeDetectionResult', {
            detail: {
              jobId: result.job_id,
              status: result.error ? 'error' : 'complete',
              trees: result.trees || [],
              message: result.error || "",
              ml_response_dir: result.ml_response_dir || "",
              mode: mode
            }
          }));
          
          // Enter validation mode with the detected trees
          window.dispatchEvent(new CustomEvent('enterTreeValidationMode', {
            detail: {
              detectionJobId: result.job_id,
              trees: result.trees || [],
              treeCount: treeCount,
              source: 'ml_pipeline',
              mode: mode
            }
          }));
          
          return result;
        } catch (error) {
          // Remove loading indicator in case of error
          if (loadingToast.parentNode) {
            loadingToast.parentNode.removeChild(loadingToast);
          }
          throw error;
        }
      } catch (error) {
        console.error(`Error running ${mode}:`, error);
        alert(`Error in ${mode}: ${error.message}`);
        return null;
      }
    };
    
    // Handler for ML tree detection request from the Detection button
    const handleRequestTreeDetection = async (event) => {
      console.log("Handling tree detection request from source:", event?.detail?.source || 'unknown');
      
      try {
        // First make sure validation mode is activated to show the debug indicator and sidebar
        setIsValidationMode(true);
        
        // Apply header state from event if provided
        if (event?.detail?.headerCollapsed !== undefined) {
          setHeaderCollapsed(event.detail.headerCollapsed);
          
          // Also dispatch event for other components
          window.dispatchEvent(new CustomEvent('headerCollapse', {
            detail: { collapsed: event.detail.headerCollapsed }
          }));
        }
        
        // Make sure map container is adjusted for sidebar
        const mapContainer = document.getElementById('map-container');
        if (mapContainer) {
          mapContainer.style.right = '384px';
          
          // Directly add "detection-sidebar" visible class 
          // to force sidebar visibility if not present
          const existingSidebar = document.querySelector('.detection-sidebar');
          if (!existingSidebar) {
            console.log("Creating detection sidebar container");
            const sidebarContainer = document.createElement('div');
            sidebarContainer.className = 'detection-sidebar';
            sidebarContainer.style.position = 'fixed';
            sidebarContainer.style.top = headerCollapsed ? '40px' : '64px';
            sidebarContainer.style.right = '0';
            sidebarContainer.style.width = '384px';
            sidebarContainer.style.bottom = '0';
            sidebarContainer.style.background = 'white';
            sidebarContainer.style.zIndex = '100';
            sidebarContainer.style.boxShadow = '-2px 0 5px rgba(0,0,0,0.1)';
            sidebarContainer.style.overflow = 'auto';
            sidebarContainer.style.transition = 'all 0.3s ease';
            document.body.appendChild(sidebarContainer);
          }
        }
        
        // Ensure the detection mode state is set to active, even if the component isn't mounted yet
        window.dispatchEvent(new CustomEvent('setDetectionModeState', {
          detail: { active: true }
        }));
        
        // Initialize with data to make sidebar visible
        setValidationData({
          jobId: 'pending_detection',
          mode: 'detection',
          treeCount: 0,
          source: 'sidebar',
          useSatelliteImagery: true,
          headerCollapsed: event?.detail?.headerCollapsed !== undefined ? 
            event.detail.headerCollapsed : headerCollapsed
        });
        
        // Then run the actual detection
        await runObjectDetection('detection');
      } catch (error) {
        console.error("Error in handleRequestTreeDetection:", error);
        // Still try to run detection even if UI setup fails
        await runObjectDetection('detection');
      }
    };
    
    // Handler for ML tree segmentation request from the Segmentation button
    const handleRequestSegmentation = async (event) => {
      await runObjectDetection('segmentation');
    };
    
    window.addEventListener('exitValidationMode', handleExitValidationMode);
    window.addEventListener('enterTreeValidationMode', handleEnterValidationMode);
    window.addEventListener('captureMapViewForDetection', handleMapCapture);
    window.addEventListener('treeDetectionResult', handleTreeDetectionResult);
    window.addEventListener('openFeatureSelection', handleOpenFeatureSelection);
    window.addEventListener('validationSidebarToggle', handleValidationSidebarToggle);
    window.addEventListener('validationQueueToggle', handleValidationQueueToggle);
    window.addEventListener('requestTreeDetection', handleRequestTreeDetection);
    window.addEventListener('requestSegmentation', handleRequestSegmentation);
    
    // Improved sidebar responsiveness with robust resize handling
    const handleLeftSidebarToggle = (event) => {
      const { collapsed } = event.detail;
      console.log("Left sidebar toggle event received:", collapsed);
      
      // Store current map center and zoom BEFORE container size changes
      let savedCenter, savedZoom;
      try {
        if (map.current && map.current.getCenter && map.current.getZoom) {
          savedCenter = map.current.getCenter();
          savedZoom = map.current.getZoom();
          console.log("Saved map position before resize:", savedCenter.toString(), "zoom:", savedZoom);
        }
      } catch (e) {
        console.error("Error saving map position:", e);
      }
      
      // Update map container position with animated transition
      const mapContainer = document.getElementById('map-container');
      if (mapContainer) {
        const sidebarWidth = collapsed ? 40 : 256;
        
        // Hide map while repositioning to prevent flashing
        mapContainer.style.visibility = 'hidden';
        
        // Check for any existing right panels to maintain proper sizing
        const rightPanel = document.querySelector('.detection-sidebar, .feature-selection-sidebar, .tree-inventory-sidebar, .validation-sidebar');
        const rightPanelWidth = rightPanel ? rightPanel.getBoundingClientRect().width : 0;
        
        // Apply CSS with more comprehensive styling
        mapContainer.style.cssText = `
          position: absolute !important;
          top: 0 !important;
          bottom: 0 !important;
          transition: right 0.3s ease-out, left 0.3s ease-out !important;
          left: ${collapsed ? '40px' : '256px'} !important;
          right: ${rightPanelWidth > 0 ? `${rightPanelWidth}px` : '0'} !important;
          width: auto !important;
          height: 100% !important;
          visibility: hidden !important;
          z-index: 5 !important;
        `;
        
        // Schedule multiple resize events during and after the CSS transition
        const resizePoints = [50, 100, 200, 300, 400, 500];
        
        // Create a sequence of resize events
        resizePoints.forEach(delay => {
          setTimeout(() => {
            try {
              // Force resize through multiple mechanisms
              window.dispatchEvent(new Event('resize'));
              
              if (map.current && window.google && window.google.maps) {
                // Trigger Google Maps internal resize
                window.google.maps.event.trigger(map.current, 'resize');
                
                // Restore original center if we have one
                if (savedCenter && map.current.setCenter) {
                  map.current.setCenter(savedCenter);
                  if (savedZoom) map.current.setZoom(savedZoom);
                }
                
                // Make map visible at the final resize point
                if (delay === 300) {
                  mapContainer.style.visibility = 'visible';
                  console.log(`Map container made visible after ${delay}ms resize`);
                }
              }
            } catch (e) {
              console.error(`Error in resize sequence at ${delay}ms:`, e);
            }
          }, delay);
        });
      }
    };
    
    // Enhanced right panel toggle handler with improved resize sequence
    const handleRightPanelToggle = (event) => {
      const { isOpen, panelWidth = 384, panelType = 'unknown' } = event.detail;
      console.log(`Right panel toggle: ${isOpen ? 'open' : 'closed'}, type: ${panelType}, width: ${panelWidth}px`);
      
      // Store map state before size changes
      let savedCenter, savedZoom;
      try {
        if (map.current && map.current.getCenter && map.current.getZoom) {
          savedCenter = map.current.getCenter();
          savedZoom = map.current.getZoom();
        }
      } catch (e) {
        console.error("Error saving map state:", e);
      }
      
      // Get current left position to maintain it during right panel changes
      const mapContainer = document.getElementById('map-container');
      if (mapContainer) {
        // Determine left position based on sidebar collapsed state
        const sidebarElement = document.querySelector('[class*="sidebar"]');
        const isSidebarCollapsed = sidebarElement && sidebarElement.classList.contains('collapsed');
        const leftPosition = isSidebarCollapsed ? '40px' : '256px';
        
        // Hide map while repositioning to prevent flashing
        mapContainer.style.visibility = 'hidden';
        
        // Get the current left sidebar state
        const leftSidebar = document.querySelector('.sidebar, [class*="sidebar"]');
        const leftOffset = leftSidebar ? 
          (leftSidebar.classList.contains('collapsed') ? '40px' : '256px') : 
          '0';
          
        // Apply comprehensive styling that respects both sidebars
        mapContainer.style.cssText = `
          position: absolute !important;
          top: 0 !important;
          bottom: 0 !important;
          transition: right 0.3s ease-out, left 0.3s ease-out !important;
          left: ${leftOffset} !important;
          right: ${isOpen ? `${panelWidth}px` : '0'} !important;
          width: auto !important;
          height: 100% !important;
          visibility: hidden !important;
          z-index: 5 !important;
        `;
        
        // Comprehensive resize sequence with position restoration
        const resizeTimes = [10, 50, 150, 250, 350, 450, 550];
        
        // Execute resize sequence
        resizeTimes.forEach(delay => {
          setTimeout(() => {
            try {
              // Window resize first
              window.dispatchEvent(new Event('resize'));
              
              if (map.current && window.google && window.google.maps) {
                // Google Maps resize trigger
                window.google.maps.event.trigger(map.current, 'resize');
                
                // Restore position
                if (savedCenter && map.current.setCenter) {
                  map.current.setCenter(savedCenter);
                  if (savedZoom) map.current.setZoom(savedZoom);
                }
                
                // Show map at 350ms which should be mid-sequence
                if (delay === 350) {
                  mapContainer.style.visibility = 'visible';
                  console.log(`Map container revealed after right panel toggle (${delay}ms)`);
                }
              }
            } catch (e) {
              console.error(`Error in right panel resize at ${delay}ms:`, e);
            }
          }, delay);
        });
      }
    };
    
    // Register responsive layout event listeners
    window.addEventListener('leftSidebarToggle', handleLeftSidebarToggle);
    window.addEventListener('rightPanelToggle', handleRightPanelToggle);
    
    // Create proper cleanup function
    const originalRemoveListeners = () => {
      window.removeEventListener('leftSidebarToggle', handleLeftSidebarToggle);
      window.removeEventListener('rightPanelToggle', handleRightPanelToggle);
    };
    
    // Initialize responsive layout based on current sidebar state
    setTimeout(() => {
      try {
        const sidebarElement = document.querySelector('[class*="sidebar"]');
        const isSidebarCollapsed = sidebarElement && (
          sidebarElement.classList.contains('collapsed') || 
          sidebarElement.getBoundingClientRect().width < 100
        );
        
        // Apply initial positioning based on sidebar state
        const mapContainer = document.getElementById('map-container');
        if (mapContainer) {
          // Check for active right panels
          const rightPanel = document.querySelector('.detection-sidebar, .feature-selection-sidebar, .tree-inventory-sidebar, .validation-sidebar');
          const rightPanelWidth = rightPanel ? rightPanel.getBoundingClientRect().width : 0;
          
          // Get proper sidebar width measurements
          const leftSidebar = document.querySelector('.sidebar, [class*="sidebar"]');
          const leftOffset = leftSidebar ? 
            (leftSidebar.classList.contains('collapsed') ? '41px' : '257px') : 
            '0';
          
          // First hide map container while positioning is applied
          mapContainer.style.visibility = 'hidden';
          
          // Apply comprehensive responsive styling with !important to override any other styles
          mapContainer.style.cssText = `
            position: absolute !important;
            top: 0 !important;
            bottom: 0 !important;
            left: ${leftOffset} !important;
            right: ${rightPanelWidth > 0 ? `${rightPanelWidth}px` : '0'} !important;
            width: auto !important;
            height: 100% !important;
            z-index: 5 !important;
            margin: 0 !important;
            padding: 0 !important;
            transition: left 0.3s ease-out, right 0.3s ease-out !important;
            visibility: hidden !important;
          `;
          
          console.log("Applied initial responsive layout for map container");
          
          // Log the current right sidebar state
          if (rightPanelWidth > 0) {
            console.log(`Map sized for right panel width: ${rightPanelWidth}px`);
          }
          
          // Force an initial resize after styles are applied
          setTimeout(() => {
            // Resize the map
            window.dispatchEvent(new Event('resize'));
            if (map.current && window.google && window.google.maps) {
              window.google.maps.event.trigger(map.current, 'resize');
              console.log("Triggered initial map resize");
            }
            
            // Show map after resize is complete
            setTimeout(() => {
              mapContainer.style.visibility = 'visible';
              console.log("Map container now visible after resize");
            }, 100);
          }, 300);
        }
      } catch (e) {
        console.error("Error initializing responsive layout:", e);
      }
    }, 200);
    
    return () => {
      // Clean up all event listeners
      window.removeEventListener('exitValidationMode', handleExitValidationMode);
      window.removeEventListener('enterTreeValidationMode', handleEnterValidationMode);
      window.removeEventListener('captureMapViewForDetection', handleMapCapture);
      window.removeEventListener('treeDetectionResult', handleTreeDetectionResult);
      window.removeEventListener('openFeatureSelection', handleOpenFeatureSelection);
      window.removeEventListener('validationSidebarToggle', handleValidationSidebarToggle);
      window.removeEventListener('validationQueueToggle', handleValidationQueueToggle);
      window.removeEventListener('requestTreeDetection', handleRequestTreeDetection);
      window.removeEventListener('requestSegmentation', handleRequestSegmentation);
      window.removeEventListener('leftSidebarToggle', handleLeftSidebarToggle);
      window.removeEventListener('rightPanelToggle', handleRightPanelToggle);
      
      // Clean up map type change listener
      if (map.current && window.google && window.google.maps) {
        window.google.maps.event.clearListeners(map.current, 'maptypeid_changed');
      }
    };
  }, [isLoaded, map.current, is3DMode]);

  // Exit validation mode with comprehensive error handling and multi-stage cleanup
  const handleExitValidationMode = (event) => {
    try {
      // Get the source of the exit request if available
      const source = event?.detail?.source || 'unknown';
      const target = event?.detail?.target || 'unknown';
      const clearExisting = event?.detail?.clearExisting === true;
      const forceRemove = event?.detail?.forceRemove === true;
      
      console.log(`Exiting validation mode from source: ${source}, target: ${target}, clearExisting: ${clearExisting}, forceRemove: ${forceRemove}`);
      
      // Notify any detection-related components that cleanup is happening
      try {
        window.dispatchEvent(new CustomEvent('validationModeCleanup', {
          detail: { initiated: true, source }
        }));
      } catch (e) {
        console.error("Error dispatching validationModeCleanup event:", e);
      }
      
      // ALWAYS do a complete cleanup of component
      setIsValidationMode(false);
      setValidationData(null);
      setDetectedTrees([]);
      
      // Always reset map container width to prevent visual artifacts
      try {
        const mapContainer = document.getElementById('map-container');
        if (mapContainer) {
          mapContainer.style.right = '0px';
        }
      } catch (e) {
        console.error("Error resetting map container width:", e);
      }
      
      // PHASE 1: Hide all elements first before attempting removal
      
      // Function to safely apply styles to elements
      const safelyApplyHidingStyles = (element) => {
        if (!element || !document.body.contains(element)) return false;
        try {
          element.style.display = 'none';
          element.style.visibility = 'hidden';
          element.style.opacity = '0';
          element.style.pointerEvents = 'none';
          element.style.width = '0px';
          element.style.zIndex = '-1';
          return true;
        } catch (err) {
          console.error("Error applying hiding styles:", err);
          return false;
        }
      };
      
      // Function to safely remove an element after a delay
      const safelyRemoveElementAfterDelay = (element, delayMs, label) => {
        if (!element || !document.body.contains(element)) return;
        
        setTimeout(() => {
          try {
            if (element.parentNode && document.body.contains(element)) {
              element.parentNode.removeChild(element);
              console.log(`Successfully removed ${label}`);
            }
          } catch (e) {
            console.error(`Error removing ${label}:`, e);
            // Last resort - try to hide it completely if removal fails
            try {
              element.style.display = 'none !important';
              element.style.visibility = 'hidden !important';
              element.style.opacity = '0 !important';
              element.style.pointerEvents = 'none !important';
              element.style.width = '0px !important';
              element.style.height = '0px !important';
              element.style.position = 'absolute !important';
              element.style.zIndex = '-9999 !important';
              element.style.overflow = 'hidden !important';
              element.style.clip = 'rect(0,0,0,0) !important';
            } catch (e2) {}
          }
        }, delayMs);
      };
      
      // Clean up detection sidebars - multiple selectors for robustness
      try {
        const sidebarSelectors = [
          '.detection-sidebar', 
          '#detection-sidebar',
          '.detection-sidebar-container',
          '#detection-sidebar-container',
          '[class*="detection-sidebar"]',
          '[id*="detection-sidebar"]'
        ];
        
        const sidebarQuery = sidebarSelectors.join(', ');
        const existingSidebars = document.querySelectorAll(sidebarQuery);
        
        console.log(`Found ${existingSidebars.length} detection sidebars to clean up`);
        
        existingSidebars.forEach((sidebar, index) => {
          if (sidebar && document.body.contains(sidebar)) {
            // Mark for complete removal
            sidebar.classList.add('detection-sidebar-removed');
            console.log(`Hiding sidebar ${index + 1}/${existingSidebars.length}`);
            
            // First hide it with CSS
            if (safelyApplyHidingStyles(sidebar)) {
              // Then remove from DOM with gradual delay to prevent simultaneous DOM operations
              safelyRemoveElementAfterDelay(sidebar, 100 + (index * 50), `detection sidebar ${index + 1}`);
            }
          }
        });
      } catch (e) {
        console.error("Error in sidebar cleanup:", e);
      }
      
      // Also clean detection container if it exists - try multiple selectors
      try {
        const containerSelectors = [
          '#detection-mode-container',
          '.detection-mode-container',
          '[id*="detection-mode"]',
          '[class*="detection-mode"]'
        ];
        
        containerSelectors.forEach(selector => {
          const containers = document.querySelectorAll(selector);
          containers.forEach((container, index) => {
            if (container && document.body.contains(container)) {
              safelyApplyHidingStyles(container);
              safelyRemoveElementAfterDelay(container, 150 + (index * 50), `detection container ${selector}`);
            }
          });
        });
      } catch (e) {
        console.error("Error cleaning detection containers:", e);
      }
      
      // Clean up detection badges with multiple selectors
      try {
        const badgeSelectors = [
          '#detection-debug', 
          '.detection-debug', 
          '[id*="detection-debug"]',
          '[class*="detection-badge"]',
          '[id*="detection-badge"]',
          '[class*="object-detection"]',
          '[id*="object-detection"]'
        ];
        
        const badgeQuery = badgeSelectors.join(', ');
        const detectionBadges = document.querySelectorAll(badgeQuery);
        
        console.log(`Found ${detectionBadges.length} detection badges to remove`);
        
        detectionBadges.forEach((badge, index) => {
          if (badge && document.body.contains(badge)) {
            // Hide it first
            safelyApplyHidingStyles(badge);
            // Remove with staggered timing
            safelyRemoveElementAfterDelay(badge, 200 + (index * 50), `detection badge ${index + 1}`);
          }
        });
      } catch (e) {
        console.error("Error in detection badge cleanup:", e);
      }
      
      // Clean up all ML overlays using multiple selectors
      try {
        const overlaySelectors = [
          '#ml-detection-overlay', 
          '.ml-detection-overlay', 
          '[id*="ml-detection"]',
          '[class*="ml-detection"]',
          '[id*="ml-overlay"]',
          '[class*="ml-overlay"]',
          '[id*="detection-overlay"]',
          '[class*="detection-overlay"]'
        ];
        
        const overlayQuery = overlaySelectors.join(', ');
        const overlays = document.querySelectorAll(overlayQuery);
        
        console.log(`Found ${overlays.length} ML overlays to remove`);
        
        overlays.forEach((overlay, index) => {
          if (overlay && document.body.contains(overlay)) {
            // Hide it first
            safelyApplyHidingStyles(overlay);
            // Remove with staggered timing
            safelyRemoveElementAfterDelay(overlay, 250 + (index * 50), `ML overlay ${index + 1}`);
          }
        });
      } catch (e) {
        console.error("Error in ML overlay cleanup:", e);
      }
      
      // PHASE 2: Additional cleanup for any remaining elements using multiple techniques
      
      // 1. Wait 300ms to let the initial removal operations complete
      setTimeout(() => {
        try {
          // 2. Try another round of removals with more aggressive selectors
          const allDetectionElements = document.querySelectorAll('[id*="detection"], [class*="detection"]');
          allDetectionElements.forEach(element => {
            if (element && document.body.contains(element)) {
              try {
                safelyApplyHidingStyles(element);
                safelyRemoveElementAfterDelay(element, 100, "remaining detection element");
              } catch (e) {}
            }
          });
          
          // 3. Dispatch a cleanup completion event
          window.dispatchEvent(new CustomEvent('validationModeCleanupComplete', {
            detail: { success: true }
          }));
          
          // 4. Refresh map markers to ensure clean state
          window.dispatchEvent(new CustomEvent('refreshMapMarkers'));
          
          // 5. Force resize to ensure proper rendering after cleanup
          window.dispatchEvent(new Event('resize'));
          if (map.current && window.google && window.google.maps) {
            window.google.maps.event.trigger(map.current, 'resize');
          }
          
          // 6. Run another resize after a bit longer delay to ensure everything has settled
          setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
          }, 300);
          
        } catch (e) {
          console.error("Error in phase 2 cleanup:", e);
        }
      }, 300);
      
      console.log("Full exit - cleared all validation data and reset container size");
    } catch (error) {
      console.error("Critical error in handleExitValidationMode:", error);
      // Emergency fallback - try to reset state variables anyway
      try {
        setIsValidationMode(false);
        setValidationData(null);
        setDetectedTrees([]);
      } catch (e) {
        console.error("Failed to reset state in emergency fallback:", e);
      }
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
    
    // Dispatch an event to notify DetectionMode about the selection
    window.dispatchEvent(new CustomEvent('treeSelected', {
      detail: { tree }
    }));
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
    
    // Handler for header collapse/expand events
    const handleHeaderCollapse = (event) => {
      setHeaderCollapsed(event.detail.collapsed);
    };
    
    // Add event listeners for sidebar toggles
    window.addEventListener('validationQueueToggle', handleResize);
    window.addEventListener('validationSidebarToggle', handleResize);
    window.addEventListener('leftSidebarToggle', handleResize);
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    // Also listen for window resize events
    window.addEventListener('resize', handleResize);
    
    // Set up ResizeObserver to monitor map container and header size changes
    const resizeObserver = new ResizeObserver(entries => {
      handleResize();
      
      // Also update all sidebar positions based on current header state
      // This ensures sidebars don't overlap the header when it resizes
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
    
    // Observe the map container for size changes
    const mapContainer = document.getElementById('map-container');
    if (mapContainer) {
      resizeObserver.observe(mapContainer);
    }
    
    // Also observe the header for size changes
    const header = document.querySelector('header');
    if (header) {
      resizeObserver.observe(header);
    }
    
    // Initial resize after component mounts
    handleResize();
    
    // Clean up
    return () => {
      clearTimeout(resizeTimeout);
      window.removeEventListener('validationQueueToggle', handleResize);
      window.removeEventListener('validationSidebarToggle', handleResize);
      window.removeEventListener('leftSidebarToggle', handleResize);
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
      window.removeEventListener('resize', handleResize);
      
      // Disconnect resize observer
      if (mapContainer) {
        resizeObserver.unobserve(mapContainer);
      }
      
      // Also unobserve header
      const header = document.querySelector('header');
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
    <ErrorBoundary componentName="MapContainer">
      <div className="relative w-full h-full" id="map-wrapper">
        {/* Google Maps Container with explicit positioning relative to visible sidebars */}
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
        
        {/* Removed the background overlay as it wasn't enhancing visibility */}
        
        {/* 3D View Toggle has been moved to the sidebar to avoid duplication */}
        
        {/* 3D Viewer based on selected API */}
        {is3DMode && (
          <Suspense fallback={<div className="w-full h-full flex items-center justify-center">Loading 3D View...</div>}>
            <div id="map-3d-container" className="w-full h-full transition-all duration-300" style={{ 
              position: 'absolute', 
              top: 0, 
              right: 0, 
              bottom: 0, 
              left: 0,
              zIndex: 10 // Ensure 3D view appears above 2D map
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
        
        {/* Tree Detection Mode and UI Elements */}
        {isValidationMode && (
          <ErrorBoundary componentName="ValidationModeFragment">
            <>
              {/* DetectionMode component renders the main sidebar */}
              <ErrorBoundary componentName="DetectionMode">
                <DetectionMode
                  key={`detection-${Date.now()}`} /* Force re-render on each open */
                  mapRef={map}
                  validationData={validationData}
                  detectedTrees={detectedTrees}
                  onExitValidation={handleExitValidationMode}
                  onSaveTrees={handleSaveValidatedTrees}
                  // Pass header state
                  headerCollapsed={headerCollapsed}
                />
              </ErrorBoundary>
              
              {/* Object Detection Badge in upper-right corner */}
              <div 
                id="detection-debug" 
                style={{
                  display: 'block', 
                  position: 'fixed', 
                  top: headerCollapsed ? '45px' : '69px', 
                  right: '389px', 
                  background: 'rgba(0,128,255,0.7)', 
                  zIndex: 200, 
                  padding: '4px 10px', 
                  fontSize: '14px', 
                  color: 'white',
                  fontWeight: 'bold',
                  borderBottomLeftRadius: '4px',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
                }}
              >
                OBJECT DETECTION
              </div>
              
              {/* Backup sidebar container to ensure visibility */}
              <div 
                className="detection-sidebar-container"
                style={{
                  position: 'fixed',
                  top: headerCollapsed ? '40px' : '64px',
                  right: '0',
                  width: '384px',
                  bottom: '0',
                  background: 'white',
                  zIndex: '50',
                  boxShadow: '-2px 0 5px rgba(0,0,0,0.1)',
                  pointerEvents: 'auto'
                }}
              />
            </>
          </ErrorBoundary>
        )}
        
      </div>
    </ErrorBoundary>
  );
});

export default MapView;