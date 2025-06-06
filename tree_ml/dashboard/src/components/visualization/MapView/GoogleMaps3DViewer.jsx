// src/components/visualization/MapView/GoogleMaps3DViewer.jsx

import React, { useEffect, useRef, forwardRef, useImperativeHandle, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { setMapView } from './mapSlice';
import { store } from '../../../store';

/**
 * GoogleMaps3DViewer - 3D map component using Google Maps JavaScript API
 */
const GoogleMaps3DViewer = forwardRef(({ apiKey, onDataLoaded }, ref) => {
  const viewerContainer = useRef(null);
  const mapInstance = useRef(null);
  const dispatch = useDispatch();
  const [isInitialized, setIsInitialized] = useState(false);
  
  const { center, zoom } = useSelector((state) => state.map);
  
  // Initialize Google Maps when the component mounts
  useEffect(() => {
    if (!viewerContainer.current || mapInstance.current) return;
    
    // Set up cleanup function to run when component unmounts
    const cleanup = () => {
      // Only clean up if we're the current instance
      if (window.googleMapsInstance === mapInstance.current) {
        console.log("Cleaning up 3D map global instance reference on unmount");
        window.googleMapsInstance = null;
      }
    };
    
    
    // Function to properly initialize the map in 3D mode
    const initMap = () => {
      if (!window.google || !window.google.maps) {
        console.error('Google Maps JavaScript API not loaded');
        return;
      }

      try {
        // Create map with 3D-specific options
        mapInstance.current = new window.google.maps.Map(viewerContainer.current, {
          center: { lat: center[1], lng: center[0] },
          zoom: zoom,
          mapTypeId: google.maps.MapTypeId.HYBRID,
          tilt: 45, // Enable 3D perspective view
          heading: 0,
          gestureHandling: 'greedy', // Make controls more responsive
          disableDefaultUI: false,
          zoomControl: true,
          mapTypeControl: true,
          scaleControl: true,
          rotateControl: true,
          fullscreenControl: true,
          streetViewControl: false // Disable street view to avoid conflicts
        });
        
        // Important post-initialization step to ensure 3D mode
        const postInitSetup = () => {
          if (!mapInstance.current) return;
          
          // Set tilt explicitly after map is loaded to force 3D mode
          mapInstance.current.setTilt(45);
          
          // Set up event listeners
          mapInstance.current.addListener('zoom_changed', () => {
            if (mapInstance.current) {
              const currentCenter = mapInstance.current.getCenter();
              const currentZoom = mapInstance.current.getZoom();
              
              if (currentCenter && currentZoom) {
                dispatch(setMapView({
                  center: [currentCenter.lng(), currentCenter.lat()],
                  zoom: currentZoom
                }));
              }
            }
          });
          
          mapInstance.current.addListener('center_changed', () => {
            if (mapInstance.current) {
              const currentCenter = mapInstance.current.getCenter();
              
              if (currentCenter) {
                dispatch(setMapView({
                  center: [currentCenter.lng(), currentCenter.lat()],
                  zoom: mapInstance.current.getZoom() || zoom
                }));
              }
            }
          });
          
          
          console.log('Google Maps 3D view initialized successfully with tilt:', mapInstance.current.getTilt());
          setIsInitialized(true);
          
          // Expose the 3D map instance globally
          window.googleMapsInstance = mapInstance.current;
          
          // Set a flag to indicate map is ready and store reference in multiple ways
          window._mapReady = true;
          window._googleMap = mapInstance.current;
          window._map = mapInstance.current;
          window._map3d = mapInstance.current;
          
          // CRITICAL FIX: Add idle listener to update global coordinate source of truth via mapIdle event
          mapInstance.current.addListener('idle', () => {
            // Dispatch mapIdle event for LocationInfo.jsx to update window.mapCoordinates
            // This is essential for tree detection to use correct coordinates in 3D mode
            window.dispatchEvent(new CustomEvent('mapIdle'));
          });
          
          // Dispatch a custom event that other components can listen for
          const mapReadyEvent = new CustomEvent('googleMapReady', {
            detail: { 
              mapInstance: mapInstance.current,
              is3D: true
            }
          });
          window.dispatchEvent(mapReadyEvent);
          
          console.log("Google Maps 3D instance exposed globally with multiple references");
          
          if (onDataLoaded) {
            onDataLoaded({ isLoaded: true });
          }
        };
        
        // Wait for the map to be idle before final setup
        google.maps.event.addListenerOnce(mapInstance.current, 'idle', postInitSetup);
        
      } catch (error) {
        console.error('Error initializing Google Maps:', error);
      }
    };
    
    // Use proper loading approach for Google Maps API
    if (window.google && window.google.maps) {
      initMap();
    } else {
      // The recommended way to load the Google Maps JavaScript API
      // Following: https://developers.google.com/maps/documentation/javascript/load-maps-js-api
      
      // Check if API is already being loaded
      if (window.googleMapsApiLoading) {
        console.log('Google Maps API is already being loaded');
        // Wait for global promise to resolve
        if (window.googleMapsApiPromise) {
          window.googleMapsApiPromise
            .then(() => {
              console.log('Google Maps API loaded from existing promise');
              initMap();
            })
            .catch(err => {
              console.error('Failed to load Google Maps API:', err);
            });
        }
        return;
      }
      
      // Mark as loading
      window.googleMapsApiLoading = true;
      
      // Create a promise to track loading
      window.googleMapsApiPromise = new Promise((resolve, reject) => {
        // Create callback function
        window.initGoogleMaps3D = () => {
          console.log('Google Maps API loaded via callback');
          window.googleMapsApiLoading = false;
          resolve();
          initMap();
        };
        
        // Load API with 'loading=async' parameter
        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&libraries=places,geometry&v=weekly&callback=initGoogleMaps3D&loading=async`;
        script.async = true;
        script.onerror = (err) => {
          console.error('Failed to load Google Maps API:', err);
          window.googleMapsApiLoading = false;
          reject(err);
        };
        
        document.head.appendChild(script);
      });
      
      // Cleanup function
      return () => {
        // Clean up API loading reference
        if (window.initGoogleMaps3D) {
          delete window.initGoogleMaps3D;
        }
        
        // Also clean up map instance reference
        cleanup();
      };
    }
    
    // Always return cleanup function regardless of initialization path
    return cleanup;
  }, [apiKey, dispatch, onDataLoaded]);
  
  // Update map when center or zoom changes in Redux store
  // Avoid unnecessary updates that could break 3D view
  useEffect(() => {
    if (!mapInstance.current || !isInitialized) return;
    
    try {
      // Get current map state
      const currentCenter = mapInstance.current.getCenter();
      const currentZoom = mapInstance.current.getZoom();
      
      if (!currentCenter) return;
      
      // Only update center if it has changed significantly
      const currentLng = currentCenter.lng();
      const currentLat = currentCenter.lat();
      const centerHasChanged = 
        Math.abs(currentLng - center[0]) > 0.0001 || 
        Math.abs(currentLat - center[1]) > 0.0001;
      
      if (centerHasChanged) {
        mapInstance.current.setCenter({ lat: center[1], lng: center[0] });
      }
      
      // Only update zoom if it has changed
      if (currentZoom !== zoom) {
        mapInstance.current.setZoom(zoom);
      }
      
    } catch (error) {
      console.error('Error updating map from Redux state:', error);
    }
  }, [center, zoom, isInitialized]);
  
  // Expose methods to parent component via ref
  useImperativeHandle(ref, () => ({
    // Get the map instance
    getMap: () => mapInstance.current,
    
    
    // Method to capture the current view for tree detection
    captureCurrentView: () => {
      return new Promise((resolve, reject) => {
        if (!mapInstance.current) {
          reject(new Error('Google Maps not initialized'));
          return;
        }
        
        try {
          const center = mapInstance.current.getCenter();
          const zoom = mapInstance.current.getZoom();
          const tilt = mapInstance.current.getTilt();
          const heading = mapInstance.current.getHeading();
          
          const viewData = {
            center: [center.lng(), center.lat()],
            zoom: zoom,
            tilt: mapInstance.current.getTilt(),
            heading: mapInstance.current.getHeading(),
            is3D: true
          };
          
          // Get projection for accurate coordinate conversion
          const projection = mapInstance.current.getProjection();
          if (projection) {
            viewData.projection = {
              worldWidth: projection.getWorldWidth()
            };
          }
          
          // Get exact bounds information
          try {
            const bounds = mapInstance.current.getBounds();
            if (bounds) {
              viewData.bounds = [
                [bounds.getSouthWest().lng(), bounds.getSouthWest().lat()],
                [bounds.getNorthEast().lng(), bounds.getNorthEast().lat()]
              ];
              
              // Get the visible region coordinates precisely
              const sw = bounds.getSouthWest();
              const ne = bounds.getNorthEast();
              const nw = new google.maps.LatLng(ne.lat(), sw.lng());
              const se = new google.maps.LatLng(sw.lat(), ne.lng());
              
              // Store all corners for more precise calculations
              viewData.corners = {
                northEast: [ne.lng(), ne.lat()],
                southWest: [sw.lng(), sw.lat()],
                northWest: [nw.lng(), nw.lat()],
                southEast: [se.lng(), se.lat()]
              };
            }
          } catch (e) {
            console.warn('Could not get map bounds:', e);
          }
          
          // Try to capture the DOM element dimensions
          try {
            if (viewerContainer.current) {
              const rect = viewerContainer.current.getBoundingClientRect();
              viewData.containerWidth = rect.width;
              viewData.containerHeight = rect.height;
            }
          } catch (e) {
            console.warn('Could not get container dimensions:', e);
          }
          
          // Try to capture the actual map canvas if possible
          try {
            if (viewerContainer.current) {
              console.log('Attempting to capture map canvas...');
              
              // Find Google Maps canvas elements
              const gmStyle = viewerContainer.current.querySelector('.gm-style');
              if (gmStyle) {
                console.log('Found .gm-style container');
                
                // Get all canvas elements
                const canvasElements = gmStyle.querySelectorAll('canvas');
                console.log(`Found ${canvasElements.length} canvas elements in Google Maps`);
                
                if (canvasElements.length > 0) {
                  // Find the largest canvas (usually the main map view)
                  let mainCanvas = null;
                  let maxArea = 0;
                  
                  // Log all canvases to help with debugging
                  for (let i = 0; i < canvasElements.length; i++) {
                    const canvas = canvasElements[i];
                    const area = canvas.width * canvas.height;
                    console.log(`Canvas ${i}: width=${canvas.width}, height=${canvas.height}, area=${area}`);
                    
                    // Pick the largest canvas by area
                    if (area > maxArea) {
                      maxArea = area;
                      mainCanvas = canvas;
                    }
                  }
                  
                  // If no canvas found, use the first one as a fallback
                  if (!mainCanvas && canvasElements.length > 0) {
                    mainCanvas = canvasElements[0];
                    console.log(`Using first canvas as fallback: width=${mainCanvas.width}, height=${mainCanvas.height}`);
                  }
                  
                  if (mainCanvas) {
                    console.log(`Selected main canvas: width=${mainCanvas.width}, height=${mainCanvas.height}`);
                    
                    try {
                      // Don't try to force rendering which can cause CORS issues
                      // Just capture the canvas directly
                      let imageUrl;
                      
                      // CREATE A MANUAL CANVAS for capture (CORS bypass) - IMPROVED VERSION
                      try {
                        console.log('Creating manual canvas for capture with explicit CORS handling');
                        
                        // Create a new canvas element for a clean capture
                        const captureCanvas = document.createElement('canvas');
                        captureCanvas.width = mainCanvas.width;
                        captureCanvas.height = mainCanvas.height;
                        
                        // Get the context with willReadFrequently set to true for better performance
                        const ctx = captureCanvas.getContext('2d', { willReadFrequently: true });
                        
                        // Fill with a background color first
                        ctx.fillStyle = '#ffffff';
                        ctx.fillRect(0, 0, captureCanvas.width, captureCanvas.height);
                        
                        // Try to handle CORS properly
                        try {
                          // Mark that we're trying to use the proper crossOrigin setting
                          mainCanvas.crossOrigin = 'anonymous';
                          
                          // Draw the main canvas onto our capture canvas
                          ctx.drawImage(mainCanvas, 0, 0);
                          console.log('Canvas drawn with crossOrigin="anonymous"');
                        } catch (corsError) {
                          console.log('CrossOrigin draw failed, trying without CORS handling', corsError);
                          
                          // Try a direct draw without CORS handling as fallback
                          ctx.drawImage(mainCanvas, 0, 0);
                        }
                        
                        // Try different image formats if one fails
                        try {
                          // Try JPEG first (smaller file size)
                          imageUrl = captureCanvas.toDataURL('image/jpeg', 0.9);
                          console.log('Canvas captured with JPEG format, length:', imageUrl.length);
                          
                          if (!imageUrl || imageUrl.length < 1000) {
                            throw new Error('JPEG capture failed or too small');
                          }
                        } catch (jpegError) {
                          console.log('JPEG capture failed, trying PNG', jpegError);
                          
                          // Try PNG as fallback (more compatible but larger)
                          imageUrl = captureCanvas.toDataURL('image/png');
                          console.log('Canvas captured with PNG format, length:', imageUrl.length);
                        }
                      } catch (e) {
                        console.log('Manual canvas capture failed completely, trying direct capture', e);
                        // Last resort: try direct capture from the main canvas
                        try {
                          imageUrl = mainCanvas.toDataURL('image/jpeg', 0.9);
                          console.log('Direct canvas capture successful with JPEG');
                        } catch (directError) {
                          console.log('Direct JPEG capture failed, trying PNG', directError);
                          imageUrl = mainCanvas.toDataURL('image/png');
                          console.log('Direct canvas capture with PNG, length:', imageUrl.length);
                        }
                      }
                      
                      if (imageUrl && (imageUrl.startsWith('data:image/jpeg;base64,') || 
                                      imageUrl.startsWith('data:image/png;base64,'))) {
                        viewData.imageUrl = imageUrl;
                        viewData.mapWidth = mainCanvas.width;
                        viewData.mapHeight = mainCanvas.height;
                        console.log('Successfully captured map canvas as data URL, length:', imageUrl.length);
                      } else {
                        console.error('Invalid data URL format returned from canvas');
                      }
                    } catch (canvasError) {
                      console.error('Error capturing canvas:', canvasError);
                    }
                  } else {
                    console.error('Failed to identify main canvas element');
                  }
                } else {
                  console.error('No canvas elements found in Google Maps container');
                }
              } else {
                console.error('Could not find .gm-style container in map');
              }
            }
          } catch (canvasError) {
            console.error('Error trying to capture map canvas:', canvasError);
          }
          
          // Function to convert pixel coordinates to LatLng
          viewData.pixelToLatLng = (x, y, width, height) => {
            try {
              if (viewerContainer.current && mapInstance.current && width && height) {
                // First normalize x,y to be relative to the container (0-1)
                const normalizedX = x / width;
                const normalizedY = y / height;
                
                // Get bounds
                const bounds = mapInstance.current.getBounds();
                const sw = bounds.getSouthWest();
                const ne = bounds.getNorthEast();
                
                // Interpolate to get lat/lng
                const lat = ne.lat() - (ne.lat() - sw.lat()) * normalizedY;
                const lng = sw.lng() + (ne.lng() - sw.lng()) * normalizedX;
                
                return [lng, lat];
              }
            } catch (e) {
              console.error('Error converting pixel to LatLng:', e);
            }
            return null;
          };
          
          resolve({ viewData });
        } catch (error) {
          console.error('Error capturing view parameters:', error);
          
          const reduxState = store.getState();
          resolve({
            viewData: {
              center: reduxState.map.center,
              zoom: reduxState.map.zoom,
              is3D: true
            }
          });
        }
      });
    },
    
    // Method to add tree markers from ML detection results
    addTreeMarkers: (trees) => {
      if (!mapInstance.current) return [];
      
      const markers = trees.map(tree => {
        if (!tree.location) return null;
        
        const [lng, lat] = tree.location;
        
        let markerColor = 'green';
        if (tree.risk_factors && tree.risk_factors.some(rf => rf.level === 'high')) {
          markerColor = 'red';
        } else if (tree.risk_factors && tree.risk_factors.some(rf => rf.level === 'medium')) {
          markerColor = 'orange';
        }
        
        const marker = new window.google.maps.Marker({
          position: { lat, lng },
          map: mapInstance.current,
          title: `${tree.species || 'Tree'} - ${tree.height || 'Unknown'}ft`,
          icon: {
            path: window.google.maps.SymbolPath.CIRCLE,
            fillColor: markerColor,
            fillOpacity: 0.8,
            strokeColor: 'white',
            strokeWeight: 1,
            scale: 8
          }
        });
        
        return marker;
      }).filter(Boolean);
      
      return markers;
    }
  }));
  
  return (
    <div className="h-full w-full relative">
      <div 
        ref={viewerContainer} 
        className="h-full w-full google-maps-container"
      />
    </div>
  );
});

export default GoogleMaps3DViewer;