import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { setMapView, setSelectedFeature } from './mapSlice';
import { store } from '../../../store';

/**
 * CesiumViewer - 3D map component using Cesium.js and Google's Photorealistic 3D Tiles
 * This component enables high-quality 3D visualization with temporally aligned RGB+geometry data
 */
const CesiumViewer = forwardRef(({ onDataLoaded, apiKey }, ref) => {
  const viewerContainer = useRef(null);
  const cesiumViewer = useRef(null);
  const tileset = useRef(null);
  const dispatch = useDispatch();
  const [isLoaded, setIsLoaded] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [treeMarkers, setTreeMarkers] = useState([]);
  
  const { center, zoom } = useSelector((state) => state.map);
  
  // Load Cesium scripts dynamically - only once
  useEffect(() => {
    // Set a flag on window to track if we've already started loading
    if (!window.cesiumLoading) {
      window.cesiumLoading = true;
      
      // If already loaded, just mark as ready
      if (window.Cesium) {
        setIsLoaded(true);
        return;
      }
      
      console.log("Loading Cesium.js libraries...");
      
      // Create script element for Cesium JS
      const script = document.createElement('script');
      script.src = "https://ajax.googleapis.com/ajax/libs/cesiumjs/1.105/Build/Cesium/Cesium.js";
      script.async = true;
      
      // Create link element for Cesium CSS
      const link = document.createElement('link');
      link.href = "https://ajax.googleapis.com/ajax/libs/cesiumjs/1.105/Build/Cesium/Widgets/widgets.css";
      link.rel = "stylesheet";
      
      // Handle loading events
      script.onload = () => {
        console.log("Cesium.js loaded successfully");
        setIsLoaded(true);
        window.cesiumLoaded = true;
      };
      
      script.onerror = (error) => {
        console.error("Error loading Cesium.js:", error);
        setLoadError("Failed to load Cesium.js library");
        window.cesiumLoading = false;
      };
      
      // Add elements to document
      document.head.appendChild(script);
      document.head.appendChild(link);
    } else if (window.cesiumLoaded) {
      // If already loaded in a previous instance
      setIsLoaded(true);
    }
    
    // No cleanup needed - we want to keep Cesium loaded
  }, []);
  
  // Initialize Cesium viewer after libraries are loaded - optimized for faster startup
  useEffect(() => {
    if (!isLoaded || !viewerContainer.current || cesiumViewer.current) return;
    
    try {
      console.log("Initializing Cesium viewer...");
      
      // Check if Cesium is available
      if (!window.Cesium) {
        throw new Error("Cesium not available");
      }
      
      // Set Cesium ion access token if needed (fallback for some base layers)
      window.Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI1YWFmZjBjMC1lZWNlLTQ3OTQtYjI1NC0xZTRiZmVjYzU0MjciLCJpZCI6MTg3ODk0LCJpYXQiOjE3MTI2MTE2NzV9.BEV3nD-dIwYIaDnLBvZO6smVjc4t-DI5BHC-ejWbUuc';
      
      // Performance optimization options
      const viewerOptions = {
        imageryProvider: false,
        baseLayerPicker: false,
        requestRenderMode: true,    // Only render new frames on changes
        maximumRenderTimeChange: 0, // Render as fast as possible
        timeline: false,
        animation: false,
        geocoder: false,
        homeButton: false,
        sceneModePicker: false,
        navigationHelpButton: false,
        infoBox: false,             // Disable infoBox to save memory
        fullscreenButton: false,    // Disable to improve startup
        useBrowserRecommendedResolution: true, // Use optimal resolution
        contextOptions: {
          webgl: {
            alpha: false,          // Improves performance
            antialias: false,      // Disable for better performance
            failIfMajorPerformanceCaveat: false,
            powerPreference: 'high-performance',
            preserveDrawingBuffer: true  // Required for toDataURL() to work with WebGL
          }
        }
      };
      
      // Create the Cesium viewer with performance-optimized settings
      cesiumViewer.current = new window.Cesium.Viewer(viewerContainer.current, viewerOptions);
      
      // Configure viewer for better performance
      const scene = cesiumViewer.current.scene;
      scene.fog.enabled = false;
      scene.globe.show = false;
      scene.skyBox.show = false;    // Disable skybox for faster rendering
      scene.sun = undefined;        // Remove sun for faster performance
      scene.moon = undefined;       // Remove moon for faster performance
      scene.skyAtmosphere = undefined; // Remove atmosphere for faster performance
      scene.backgroundColor = window.Cesium.Color.BLACK; // Simpler background
      
      // Add Google's Photorealistic 3D Tiles
      // Use the API key from props without placeholder fallback
      const apiKeyToUse = apiKey || '';
      
      // Check if API key is valid
      if (!apiKeyToUse) {
        const error = new Error('Google Maps API key is missing. 3D view will not work.');
        console.error(error);
        setLoadError(error.message);
        return;
      }
      
      // Track if the tileset loads successfully
      let tilesetLoadFailed = false;
      
      try {
        console.log("Attempting to load Google 3D Tiles with new API key");
        
        // Create the tileset with error handling
        const tilesetPromise = window.Cesium.Cesium3DTileset.fromUrl(
          `https://tile.googleapis.com/v1/3dtiles/root.json?key=${apiKeyToUse}`,
          {
            showCreditsOnScreen: true,
            maximumScreenSpaceError: 8, // Quality vs performance tradeoff (lower = higher quality)
          }
        );
        
        // Handle the promise result
        tilesetPromise.then(loadedTileset => {
          console.log("Google 3D Tiles loaded successfully");
          tileset.current = cesiumViewer.current.scene.primitives.add(loadedTileset);
        }).catch(error => {
          tilesetLoadFailed = true;
          console.error("Failed to load Google 3D Tiles:", error);
          setLoadError(`Failed to load 3D Tiles: ${error.message || "API key may not have correct permissions"}`);
          
          // Enable the globe as a fallback so the user sees something
          if (cesiumViewer.current) {
            console.log("Enabling globe as fallback");
            cesiumViewer.current.scene.globe.show = true;
            cesiumViewer.current.scene.fog.enabled = true;
          }
        });
      } catch (error) {
        tilesetLoadFailed = true;
        console.error("Error creating 3D Tileset:", error);
        setLoadError(`Error setting up 3D view: ${error.message}`);
        
        // Enable the globe as a fallback
        if (cesiumViewer.current) {
          cesiumViewer.current.scene.globe.show = true;
          cesiumViewer.current.scene.fog.enabled = true;
        }
      }
      
      // Initialize the 3D camera based on the current center and zoom
      // Keep it simple - just use the lat/long directly
      // Apply position correction to account for Cesium's view offset
      // Shift the view slightly south to compensate for Cesium's northward bias
      const longitude = center[0];
      const latitude = center[1] - 0.008; // Shift south by ~0.9km (approx. 0.008 degrees lat)
      
      // Adjusted zoom to height conversion
      // Using zoom - 1.2 to make the 3D view more magnified
      const height = 40075000 / Math.pow(2, zoom - 1.2);
      
      console.log("Initializing 3D view with position:", { center: [longitude, latitude], zoom });
      
      // Just set the view to match the current center
      cesiumViewer.current.camera.setView({
        destination: window.Cesium.Cartesian3.fromDegrees(longitude, latitude, height),
        orientation: {
          heading: window.Cesium.Math.toRadians(0),
          pitch: window.Cesium.Math.toRadians(-85), // More overhead view
          roll: 0
        }
      });
      
      // Create debounce function to prevent too many updates
      const debounce = (func, delay) => {
        let timeoutId;
        return function(...args) {
          if (timeoutId) clearTimeout(timeoutId);
          timeoutId = setTimeout(() => {
            func.apply(this, args);
          }, delay);
        };
      };
      
      // Create debounced handler for camera changes
      const handleCameraChange = debounce(() => {
        if (!cesiumViewer.current) return;
        
        // Get current camera position
        const cartographic = window.Cesium.Cartographic.fromCartesian(
          cesiumViewer.current.camera.position
        );
        
        const longitude = window.Cesium.Math.toDegrees(cartographic.longitude);
        // Add the correction factor when updating Redux to compensate for our initial shift
        const latitude = window.Cesium.Math.toDegrees(cartographic.latitude) + 0.008;
        
        // Calculate zoom from height - inverse of our 1.2 factor formula
        const height = cartographic.height;
        // Add 1.2 to match our height calculation
        const calculatedZoom = Math.log2(40075000 / height) + 1.2;
        
        // Update Redux state with the position
        dispatch(setMapView({
          center: [longitude, latitude],
          zoom: Math.min(21, Math.max(1, Math.round(calculatedZoom)))  // Use calculated zoom, clamped to valid range
        }));
      }, 300); // 300ms debounce delay
      
      // Listen for camera changes to update Redux state
      const eventRemover = cesiumViewer.current.camera.changed.addEventListener(handleCameraChange);
      
      // Store event remover function for cleanup
      cesiumViewer.current._cameraChangedEventRemover = eventRemover;
      
      // Notify parent component
      if (onDataLoaded) {
        onDataLoaded({ isLoaded: true });
      }
      
      console.log("Cesium viewer initialized successfully");
    } catch (error) {
      console.error("Error initializing Cesium viewer:", error);
      setLoadError(error.message || "Failed to initialize 3D viewer");
    }
  }, [isLoaded]); // Only depend on isLoaded to prevent re-initialization loops
  
  // We don't need to update Cesium from Redux changes after initial setup
  // This allows Cesium to operate independently (Ctrl+pan) while in 3D mode
  
  // Expose methods to parent component via ref
  // Handle visibility changes and preserve state
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (cesiumViewer.current && cesiumViewer.current.scene) {
          const isVisible = entries[0].isIntersecting;
          
          // When becoming hidden, save the current position to Redux
          if (!isVisible && cesiumViewer.current.camera) {
            const cartographic = window.Cesium.Cartographic.fromCartesian(
              cesiumViewer.current.camera.position
            );
            
            const longitude = window.Cesium.Math.toDegrees(cartographic.longitude);
            // Apply correction factor when updating position to Redux
            const latitude = window.Cesium.Math.toDegrees(cartographic.latitude) + 0.008;
            
            // Calculate zoom from height using our formula with 1.2 factor
            const height = cartographic.height;
            const calculatedZoom = Math.log2(40075000 / height) + 1.2;
            
            // Save state to Redux for 2D view to use
            dispatch(setMapView({
              center: [longitude, latitude],
              zoom: Math.min(21, Math.max(1, Math.round(calculatedZoom)))
            }));
            
            // Pause rendering to save resources
            cesiumViewer.current.scene.requestRenderMode = true;
          } else if (isVisible) {
            // When becoming visible again, update from Redux state
            const { center, zoom } = store.getState().map;
            
            if (center && zoom && cesiumViewer.current.camera) {
              // Calculate height from zoom using our formula with 1.2 factor
              const height = 40075000 / Math.pow(2, zoom - 1.2);
              
              // Update camera position with correction
              cesiumViewer.current.camera.setView({
                destination: window.Cesium.Cartesian3.fromDegrees(
                  center[0], center[1] - 0.008, height // Apply correction to match our other adjustments
                ),
                orientation: {
                  heading: window.Cesium.Math.toRadians(0),
                  pitch: window.Cesium.Math.toRadians(-85),
                  roll: 0
                }
              });
            }
            
            // Force a render
            cesiumViewer.current.scene.requestRender();
          }
        }
      },
      { threshold: 0.1 }
    );
    
    if (viewerContainer.current) {
      observer.observe(viewerContainer.current);
    }
    
    return () => {
      observer.disconnect();
    };
  }, [isLoaded, dispatch]);
  
  useImperativeHandle(ref, () => ({
    // Get the Cesium viewer instance
    getCesiumViewer: () => cesiumViewer.current,
    
    // Get the 3D tileset
    getTileset: () => tileset.current,
    
    // Method to capture current view parameters for server-side satellite imagery
    captureCurrentView: async () => {
      if (!cesiumViewer.current) {
        throw new Error("Cesium viewer not initialized");
      }
      
      try {
        console.log("Capturing view parameters for server-side satellite imagery...");
        
        // Get camera details
        const camera = cesiumViewer.current.camera;
        const cartographic = window.Cesium.Cartographic.fromCartesian(camera.position);
        
        // Get view parameters with position correction
        const longitude = window.Cesium.Math.toDegrees(cartographic.longitude);
        const latitude = window.Cesium.Math.toDegrees(cartographic.latitude) + 0.008; // Apply correction
        const heading = window.Cesium.Math.toDegrees(camera.heading);
        const pitch = window.Cesium.Math.toDegrees(camera.pitch);
        
        // Get view rectangle
        const viewRectangle = cesiumViewer.current.camera.computeViewRectangle();
        const bounds = [
          [window.Cesium.Math.toDegrees(viewRectangle.west), 
           window.Cesium.Math.toDegrees(viewRectangle.south) + 0.008], // Apply correction to south
          [window.Cesium.Math.toDegrees(viewRectangle.east), 
           window.Cesium.Math.toDegrees(viewRectangle.north) + 0.008]  // Apply correction to north
        ];
        
        // Get the current zoom level from Redux store
        const { zoom } = store.getState().map;
        
        // Get the actual canvas dimensions for exact view matching
        const canvas = cesiumViewer.current.scene.canvas;
        const viewportWidth = canvas.width;
        const viewportHeight = canvas.height;
        
        console.log(`Actual Cesium viewer dimensions: ${viewportWidth}x${viewportHeight}`);
        
        // Format coordinates to 6 decimal places for consistency
        const formattedCoords = [
          parseFloat(longitude.toFixed(6)),
          parseFloat(latitude.toFixed(6))
        ];
        
        // Create view data with necessary parameters for server-side image retrieval
        // CRITICAL: Always include userCoordinates for ML detection
        const viewData = {
          bounds: bounds,
          userCoordinates: formattedCoords, // CRITICAL: Add userCoordinates for backend
          center: formattedCoords, // Use same formatted coordinates 
          zoom: zoom, // Use current zoom 
          heading: heading,
          tilt: -pitch, // Convert to tilt (positive down)
          mapWidth: viewportWidth,
          mapHeight: viewportHeight
        };
        
        // Return just the view data - server will handle image retrieval
        return { viewData };
      } catch (error) {
        console.error("Error capturing view parameters:", error);
        // If there's an error, return just the basic view info from Redux
        const { center, zoom } = store.getState().map;
        
        // Format coordinates to 6 decimal places for consistency
        const formattedCoords = center ? [
          parseFloat(center[0].toFixed(6)),
          parseFloat(center[1].toFixed(6))
        ] : null;
        
        // Make sure we always include userCoordinates
        return {
          viewData: {
            userCoordinates: formattedCoords, // CRITICAL: Include userCoordinates
            center: formattedCoords,
            zoom: zoom
          }
        };
      }
    },
    
    // Add tree markers from ML detection results
    addTreeMarkers: (trees) => {
      if (!cesiumViewer.current) return;
      
      // Clear existing markers
      treeMarkers.forEach(marker => {
        cesiumViewer.current.entities.remove(marker);
      });
      
      // Create new markers
      const newMarkers = [];
      
      trees.forEach(tree => {
        if (!tree.location) return;
        
        const [lng, lat] = tree.location;
        let height = tree.height || 10; // Default height if not provided
        
        // Determine color based on risk level
        let color = window.Cesium.Color.GREEN;
        if (tree.risk_factors && tree.risk_factors.some(rf => rf.level === 'high')) {
          color = window.Cesium.Color.RED;
        } else if (tree.risk_factors && tree.risk_factors.some(rf => rf.level === 'medium')) {
          color = window.Cesium.Color.ORANGE;
        }
        
        // Create entities for the tree
        const entities = [];
        
        // Create a box entity at the tree location
        const boxEntity = cesiumViewer.current.entities.add({
          name: `Tree ${tree.id} - ${tree.species || 'Unknown'}`,
          position: window.Cesium.Cartesian3.fromDegrees(lng, lat, height / 2),
          box: {
            dimensions: new window.Cesium.Cartesian3(3, 3, height),
            material: color.withAlpha(0.7),
            outline: true,
            outlineColor: window.Cesium.Color.WHITE
          },
          description: `
            <h3>${tree.species || 'Tree'}</h3>
            <p>Height: ${tree.height}ft</p>
            <p>Location: ${lat.toFixed(6)}, ${lng.toFixed(6)}</p>
            ${tree.risk_factors ? 
              `<p>Risk Factors: ${tree.risk_factors.map(rf => rf.description).join(', ')}</p>` 
              : ''}
          `
        });
        
        entities.push(boxEntity);
        
        // If segmentation contour is available, visualize it
        if (tree.segmentation_contour && tree.segmentation_contour.length > 0) {
          console.log("Displaying segmentation contour for tree", tree.id);
          
          // Create a polyline from the contour points
          // Need to transform from image coordinates to geographic coordinates
          try {
            // Use the location as base coordinates
            const center = [lng, lat];
            
            // Scale factor for contour points (adjust as needed)
            const scale = 0.00005; // Roughly 5 meters per 100 pixels
            
            // Create hierarchy for polygon from contour points
            const positions = [];
            
            // Map contour points to geographic coordinates
            tree.segmentation_contour.forEach(point => {
              // Adjust point relative to center
              const relX = (point[0] - 320) * scale; // Assuming image center is at 320,240
              const relY = (point[1] - 240) * scale;
              
              // Add to position array
              positions.push(window.Cesium.Cartesian3.fromDegrees(
                center[0] + relX,
                center[1] + relY,
                1 // Just above ground
              ));
            });
            
            // Add the first point again to close the polygon
            if (positions.length > 0) {
              positions.push(positions[0]);
            }
            
            // Create polygon entity for segmentation
            const polygonEntity = cesiumViewer.current.entities.add({
              polyline: {
                positions: positions,
                width: 2,
                material: window.Cesium.Color.YELLOW.withAlpha(0.8),
                clampToGround: true
              }
            });
            
            entities.push(polygonEntity);
          } catch (error) {
            console.error("Error creating segmentation visualization:", error);
          }
        }
        
        // Add all entities to the markers array
        newMarkers.push(...entities);
      });
      
      setTreeMarkers(newMarkers);
      
      return newMarkers;
    }
  }));
  
  // Minimal cleanup - since we're keeping the component mounted
  useEffect(() => {
    return () => {
      // Only do minimal cleanup since we're not fully destroying the viewer
      if (cesiumViewer.current) {
        try {
          // Just pause rendering when not visible
          if (cesiumViewer.current.scene) {
            cesiumViewer.current.scene.requestRenderMode = true;
          }
          
          // Keep the event listener attached - we'll need it when visible again
        } catch (error) {
          console.error("Error pausing Cesium viewer:", error);
        }
      }
    };
  }, []);
  
  if (loadError) {
    return (
      <div className="h-full w-full flex items-center justify-center">
        <div className="text-center p-4">
          <h3 className="text-lg font-medium text-red-600 mb-2">Error loading 3D viewer</h3>
          <p className="text-gray-600">{loadError}</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="h-full w-full relative">
      {!isLoaded ? (
        <div className="flex items-center justify-center h-full w-full">
          <div className="text-center">
            <p className="text-gray-600">Loading 3D viewer...</p>
          </div>
        </div>
      ) : (
        <>
          <div 
            ref={viewerContainer} 
            className="h-full w-full cesium-container"
            style={{ position: 'relative' }}
          />
          
          {/* Additional UI elements can be added here */}
        </>
      )}
    </div>
  );
});

export default CesiumViewer;