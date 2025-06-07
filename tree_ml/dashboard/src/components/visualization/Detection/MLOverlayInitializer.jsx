// src/components/visualization/Detection/MLOverlayInitializer.jsx
//
// This component ensures that MLOverlay.js is properly initialized
// and handles the Google Maps API dependency correctly.
//
// Enhanced version with performance optimizations and improved initialization

import React, { useEffect, useState } from 'react';
// Import MLOverlay with named imports
import * as MLOverlayModule from './MLOverlay';

/**
 * MLOverlayInitializer Component
 * 
 * This component handles the proper initialization of the MLOverlay.js 
 * functionality with Google Maps API. It ensures that the overlay is
 * only initialized after the Google Maps API is fully loaded.
 * 
 * Enhanced with:
 * 1. Visibility state management
 * 2. Efficient overlay toggling
 * 3. Performance monitoring
 * 4. Error handling
 * 5. Mock data initialization for testing
 */
const MLOverlayInitializer = () => {
  const [initialized, setInitialized] = useState(false);
  
  // Setup ML Overlay initialization
  useEffect(() => {
    console.log('MLOverlayInitializer: Starting initialization...');
    
    // Define a function to attempt initialization with a fallback mechanism
    const attemptInitialization = () => {
      // Directly use the integrated MLOverlay.js module
      console.log('MLOverlayInitializer: Using integrated MLOverlay.js module');
      
      const initializeWithIntegratedModule = () => {
        return new Promise((resolve, reject) => {
          try {
            console.log('MLOverlayInitializer: Setting up global functions from MLOverlay module');
            
            // Expose the functions from MLOverlayModule globally
            window.MLDetectionOverlay = MLOverlayModule.MLDetectionOverlay;
            window.renderMLOverlay = MLOverlayModule.renderMLOverlay;
            window.removeMLOverlay = MLOverlayModule.removeMLOverlay;
            window.updateMLOverlayOpacity = MLOverlayModule.updateMLOverlayOpacity;
            window.updateMLOverlayClasses = MLOverlayModule.updateMLOverlayClasses;
            window.toggleMLOverlayVisibility = (visible) => {
              console.log(`MLOverlayInitializer: toggleMLOverlayVisibility called with visible=${visible}`);
              if (window._mlDetectionOverlay && window._mlDetectionOverlay.div) {
                window._mlDetectionOverlay.div.style.display = visible ? 'block' : 'none';
                return true;
              }
              return false;
            };
            
            // Verify that the functions are available
            if (typeof window.MLDetectionOverlay === 'function' && 
                typeof window.renderMLOverlay === 'function' &&
                typeof window.updateMLOverlayOpacity === 'function') {
              console.log('MLOverlayInitializer: MLOverlay functions are now available globally');
              resolve(true);
            } else {
              console.warn('MLOverlayInitializer: Module loaded but functions not properly exported');
              reject(new Error('Module functions not available'));
            }
          } catch (err) {
            console.error('MLOverlayInitializer: Error initializing with integrated module', err);
            reject(err);
          }
        });
      };
      
      // Initialize with the integrated module
      initializeWithIntegratedModule()
        .then(() => {
          console.log('MLOverlayInitializer: Successfully initialized with integrated module');
          setInitialized(true);
          initializeWithExistingData();
          return true;
        })
        .catch(() => {
          console.warn('MLOverlayInitializer: Failed to initialize with integrated module, falling back to standard methods');
          
          // Continue with standard initialization methods
          // 1. Try the new unified function from index.js
          if (typeof window.ensureMLOverlayInitialized === 'function') {
            console.log('MLOverlayInitializer: Found ensureMLOverlayInitialized function in window');
            const isInitialized = window.ensureMLOverlayInitialized();
            
            if (isInitialized) {
              console.log('MLOverlayInitializer: Successfully initialized via window.ensureMLOverlayInitialized');
              setInitialized(true);
              initializeWithExistingData();
              return true;
            }
          }
          
          // 2. Try the module's function
          if (typeof MLOverlayModule.ensureInitialized === 'function') {
            console.log('MLOverlayInitializer: Found ensureInitialized method in module');
            
            // Check initialization status
            const isInitialized = MLOverlayModule.ensureInitialized();
            
            // If already initialized, just log success and update state
            if (isInitialized) {
              console.log('MLOverlayInitializer: MLOverlay already initialized via module method');
              setInitialized(true);
              
              // Ensure we also initialize the overlay with any existing data
              initializeWithExistingData();
              return true;
            }
          } 
          
          // 3. Check for global ensureInitialized function
          if (typeof window.ensureInitialized === 'function') {
            console.log('MLOverlayInitializer: Found global ensureInitialized function');
            
            // Use the global function instead
            const isInitialized = window.ensureInitialized();
            
            if (isInitialized) {
              console.log('MLOverlayInitializer: MLOverlay initialized via global function');
              setInitialized(true);
              
              // Ensure we also initialize the overlay with any existing data
              initializeWithExistingData();
              return true;
            }
          }
          
          // 4. Check window.treeDetection
          if (window.treeDetection && typeof window.treeDetection.ensureInitialized === 'function') {
            console.log('MLOverlayInitializer: Found ensureInitialized in window.treeDetection');
            const isInitialized = window.treeDetection.ensureInitialized();
            
            if (isInitialized) {
              console.log('MLOverlayInitializer: MLOverlay initialized via treeDetection.ensureInitialized');
              setInitialized(true);
              initializeWithExistingData();
              return true;
            }
          }
          
          // 5. Try direct initialization of the class as a last resort
          console.log('MLOverlayInitializer: Attempting direct initialization');
          
          // Access the module's default export directly
          if (MLOverlayModule && typeof MLOverlayModule === 'object') {
            // Store MLOverlay functions on window for easier access
            window.renderMLOverlay = MLOverlayModule.renderMLOverlay;
            window.removeMLOverlay = MLOverlayModule.removeMLOverlay;
            window.updateMLOverlayOpacity = MLOverlayModule.updateMLOverlayOpacity;
            window.updateMLOverlayClasses = MLOverlayModule.updateMLOverlayClasses;
            
            // Mark as initialized
            setInitialized(true);
            initializeWithExistingData();
            return true;
          }
          
          // If all methods fail, return false
          return false;
        });
      
      // Return true to indicate we're handling initialization asynchronously
      return true;
    };
    
    // Try immediate initialization
    if (attemptInitialization()) {
      return; // Already initialized successfully
    }
    
    console.log('MLOverlayInitializer: Initial initialization attempt failed, setting up event listeners');
    
    // First, try window.google check with interval
    const checkInterval = setInterval(() => {
      // Check if Google Maps is available now
      if (typeof google !== 'undefined' && google.maps && google.maps.OverlayView) {
        console.log('MLOverlayInitializer: Google Maps API detected, attempting initialization');
        
        if (attemptInitialization()) {
          clearInterval(checkInterval);
        }
      }
    }, 500);
    
    // Clear interval after 10 seconds max to prevent memory leaks
    setTimeout(() => {
      clearInterval(checkInterval);
    }, 10000);
    
    // Also listen for explicit initialization events
    const handleMapsApiInitialized = () => {
      console.log('MLOverlayInitializer: Maps API initialized event received');
      clearInterval(checkInterval);
      
      // Wait a short delay to ensure full initialization
      setTimeout(() => {
        attemptInitialization();
      }, 100);
    };
    
    // Listen for custom map initialization events
    window.addEventListener('mapsApiInitialized', handleMapsApiInitialized);
    window.addEventListener('googleMapReady', handleMapsApiInitialized);
    window.addEventListener('mapInitialized', handleMapsApiInitialized);
    
    // Clean up event listeners
    return () => {
      clearInterval(checkInterval);
      window.removeEventListener('mapsApiInitialized', handleMapsApiInitialized);
      window.removeEventListener('googleMapReady', handleMapsApiInitialized);
      window.removeEventListener('mapInitialized', handleMapsApiInitialized);
    };
  }, []);
  
  // Function to initialize overlay with existing data - make it visible by default
  const initializeWithExistingData = () => {
    // CRITICAL FIX: Make overlay visible by default when Detection button is clicked
    console.log('MLOverlayInitializer: Setting overlay to visible by default');
    
    // ALWAYS default to showing the overlay for better UX - users should see something when they click Detection
    const shouldShowOverlay = true; // Force visible always
    
    // NEVER clear detection data - preserve all markers and detection results
    if (window.mlDetectionData) {
      console.log('MLOverlayInitializer: Preserving existing detection data');
      // Just ensure we have a backup copy if needed
      window._savedMlDetectionData = window.mlDetectionData;
    }
    
    // Make overlay visible by default - this is the key fix
    console.log('MLOverlayInitializer: Setting overlay to visible state');
    window.mlOverlaySettings = {
      ...(window.mlOverlaySettings || {}),
      showOverlay: true,
      initialVisibility: true,
      pendingButtonTrigger: false  // Don't wait for explicit trigger
    };
    window.detectionShowOverlay = true;
    
    // Dispatch event to make sure overlay is immediately visible
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
        detail: {
          showOverlay: true,
          showSegmentation: window.mlOverlaySettings?.showSegmentation !== false,
          opacity: window.mlOverlaySettings?.opacity || 0.7,
          source: 'initialization'
        }
      }));
    }, 100);
    
    // Try to make the overlay visible if it exists
    if (window._mlDetectionOverlay && window._mlDetectionOverlay.div) {
      console.log('MLOverlayInitializer: Making existing overlay div visible');
      window._mlDetectionOverlay.div.style.display = 'block';
    } else {
      // Create an overlay even if we don't have data - this is critical for first click to work
      const mapInstance = window.map || window.googleMapsInstance || window._googleMap;
      
      if (mapInstance && typeof window.renderMLOverlay === 'function') {
        try {
          if (window.mlDetectionData) {
            // If we have data, use it
            console.log('MLOverlayInitializer: Creating overlay with existing data');
            window.renderMLOverlay(
              mapInstance,
              window.mlDetectionData,
              {
                opacity: window.mlOverlaySettings?.opacity || 0.7,
                showSegmentation: window.mlOverlaySettings?.showSegmentation !== false,
                forceRenderBoxes: true
              }
            );
          } else {
            // No data yet - create empty placeholder overlay
            console.log('MLOverlayInitializer: Creating placeholder overlay with no data');
            window.renderMLOverlay(
              mapInstance,
              { trees: [], metadata: window.mapViewInfo?.viewData || {} },
              {
                opacity: window.mlOverlaySettings?.opacity || 0.7,
                showSegmentation: false,
                placeholderMode: true
              }
            );
          }
        } catch (e) {
          console.error('MLOverlayInitializer: Error creating overlay:', e);
        }
      } else {
        console.warn('MLOverlayInitializer: Map or renderMLOverlay not available yet');
      }
    }
    
    // Listen for openTreeDetection events to handle button triggers
    const handleOpenTreeDetection = (event) => {
      // Always ensure the overlay is visible when detection is triggered
      console.log('MLOverlayInitializer: Detection triggered, ensuring overlay is visible');
      
      // Set the global flag for showing the overlay
      window.detectionShowOverlay = true;
      window.mlOverlaySettings = {
        ...(window.mlOverlaySettings || {}),
        showOverlay: true,
        pendingButtonTrigger: false
      };
      
      // Make sure we have any saved data available
      if (!window.mlDetectionData && window._savedMlDetectionData) {
        window.mlDetectionData = window._savedMlDetectionData;
      }
      
      // Make overlay visible if it exists but is hidden
      if (window._mlDetectionOverlay && window._mlDetectionOverlay.div) {
        window._mlDetectionOverlay.div.style.display = 'block';
      }
    };
    
    // Ensure listener is registered
    window.addEventListener('openTreeDetection', handleOpenTreeDetection);
    
    // We only need to create mock data in development mode for testing
    if (process.env.NODE_ENV === 'development') {
      const isTestMode = localStorage.getItem('enable_ml_test_mode') === 'true';
      if (isTestMode) {
        console.log('MLOverlayInitializer: Test mode enabled, creating mock data');
        createMockDataIfNeeded();
      }
    }
    
    // Return the event cleaner function
    return () => {
      window.removeEventListener('openTreeDetection', handleOpenTreeDetection);
    };
  };
  
  // Placeholder for test data handling
  const createMockDataIfNeeded = () => {
    // Mock data creation has been disabled
    console.log('MLOverlayInitializer: Mock data creation disabled');
    return;
  };
  
  // Store these event handlers in refs to maintain references between hooks
  const handlersRef = React.useRef({
    handleOverlaySettingsChange: null,
    handleDetectionDataUpdate: null,
    cleanupInitialization: null
  });
  
  // Setup visibility toggle handling
  useEffect(() => {
    if (!initialized) return;
    
    // Handle visibility changes from other components
    const handleOverlaySettingsChange = (event) => {
      const { showOverlay, showSegmentation, opacity } = event.detail;
      
      console.log(`MLOverlayInitializer: Received overlay settings change: visible=${showOverlay}, segmentation=${showSegmentation}, opacity=${opacity}`);
      
      // Update MLOverlay visibility if it exists
      if (window._mlDetectionOverlay) {
        if (showOverlay === false) {
          // Hide the overlay
          if (window._mlDetectionOverlay.div) {
            window._mlDetectionOverlay.div.style.display = 'none';
          }
        } else if (showOverlay === true) {
          // Show the overlay
          if (window._mlDetectionOverlay.div) {
            window._mlDetectionOverlay.div.style.display = 'block';
          } else if (typeof MLOverlayModule.renderMLOverlay === 'function' && 
                     (window.mlDetectionData || window.detectionData)) {
            // Try to create the overlay if it doesn't exist but should
            console.log('MLOverlayInitializer: Recreating overlay because it was toggled on but no div exists');
            
            const mapInstance = window.map || window.googleMapsInstance || window._googleMap;
            if (mapInstance) {
              MLOverlayModule.renderMLOverlay(
                mapInstance,
                window.mlDetectionData || window.detectionData,
                {
                  opacity: opacity !== undefined ? opacity : 0.7,
                  showSegmentation: showSegmentation !== undefined ? showSegmentation : true,
                  forceRenderBoxes: true
                }
              );
            }
          }
        }
        
        // Update opacity if specified - enhanced with more robust handling
        if (opacity !== undefined) {
          console.log(`MLOverlayInitializer: Handling opacity change to ${opacity}`);
          
          // Update global settings for consistency
          window.mlOverlaySettings = {
            ...(window.mlOverlaySettings || {}),
            opacity: opacity
          };
          
          // Save to localStorage for persistence
          try {
            localStorage.setItem('ml-overlay-opacity', opacity.toString());
          } catch (e) {
            console.error("Error saving opacity to localStorage:", e);
          }
          
          // APPROACH 1: Use the module function if available
          if (typeof MLOverlayModule.updateMLOverlayOpacity === 'function') {
            MLOverlayModule.updateMLOverlayOpacity(opacity);
          }
          
          // APPROACH 2: Use the global function if available
          if (typeof window.updateMLOverlayOpacity === 'function') {
            window.updateMLOverlayOpacity(opacity);
          }
          
          // APPROACH 3: Update the MLDetectionOverlay instance directly
          if (window._mlDetectionOverlay) {
            // Use updateOpacity method if available
            if (typeof window._mlDetectionOverlay.updateOpacity === 'function') {
              window._mlDetectionOverlay.updateOpacity(opacity);
            } 
            // Direct access to div if method not available
            else if (window._mlDetectionOverlay.div) {
              window._mlDetectionOverlay.div.style.opacity = opacity.toString();
            }
          }
          
          // APPROACH 4: Update DOM elements directly as fallback
          const overlayElement = document.getElementById('ml-detection-overlay');
          if (overlayElement) {
            overlayElement.style.opacity = opacity.toString();
            
            // Also update background color for any tint elements
            const tintElements = overlayElement.querySelectorAll('.tint-layer') || [];
            if (tintElements.length > 0) {
              tintElements.forEach(tint => {
                tint.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
              });
            } else {
              // If no tint elements, update the overlay background directly
              overlayElement.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
            }
          }
          
          // APPROACH 5: Update segmentation masks directly
          const masks = document.querySelectorAll('.segmentation-mask');
          if (masks.length > 0) {
            masks.forEach(mask => {
              mask.style.opacity = Math.min(opacity * 1.2, 0.8); // Adjust for better visibility
            });
          }
          
          // Dispatch event for other components
          window.dispatchEvent(new CustomEvent('mlOverlayOpacityUpdated', {
            detail: { 
              opacity: opacity,
              source: 'ml_overlay_initializer',
              timestamp: Date.now()
            }
          }));
        }
        
        // Update segmentation if specified
        if (showSegmentation !== undefined) {
          // Try to update classes through the API
          if (typeof MLOverlayModule.updateMLOverlayClasses === 'function') {
            const classes = { 
              ...window.mlOverlaySettings?.classes || {},
              showSegmentation
            };
            MLOverlayModule.updateMLOverlayClasses(classes);
          }
          
          // Also update DOM elements directly
          const masks = document.querySelectorAll('.segmentation-mask');
          masks.forEach(mask => {
            mask.style.display = showSegmentation ? 'block' : 'none';
          });
        }
      } else if (showOverlay === true) {
        // If overlay should be visible but doesn't exist yet, try to create it
        console.log('MLOverlayInitializer: Overlay should be visible but does not exist, attempting to create');
        
        // Only proceed if we have detection data and map
        if ((window.mlDetectionData || window.detectionData) && 
            typeof MLOverlayModule.renderMLOverlay === 'function') {
          
          // Find map instance
          const mapInstance = window.map || window.googleMapsInstance || window._googleMap;
          
          if (mapInstance) {
            console.log('MLOverlayInitializer: Creating new overlay with existing data');
            
            // Create overlay with current settings
            MLOverlayModule.renderMLOverlay(
              mapInstance,
              window.mlDetectionData || window.detectionData,
              {
                opacity: opacity !== undefined ? opacity : 0.7,
                showSegmentation: showSegmentation !== undefined ? showSegmentation : true,
                forceRenderBoxes: true
              }
            );
          } else {
            console.warn('MLOverlayInitializer: Cannot create overlay - no map instance found');
          }
        } else if (!window.mlDetectionData && !window.detectionData) {
          console.log('MLOverlayInitializer: No detection data available to show overlay');
        }
      }
      
      // Update global settings
      window.mlOverlaySettings = {
        ...(window.mlOverlaySettings || {}),
        showOverlay: showOverlay !== undefined ? showOverlay : window.mlOverlaySettings?.showOverlay,
        showSegmentation: showSegmentation !== undefined ? showSegmentation : window.mlOverlaySettings?.showSegmentation,
        opacity: opacity !== undefined ? opacity : window.mlOverlaySettings?.opacity
      };
    };
    
    // Store handler in ref for reference stability
    handlersRef.current.handleOverlaySettingsChange = handleOverlaySettingsChange;
    
    // Initialize overlay with any existing data and get cleanup function
    const cleanupInitialization = initializeWithExistingData();
    handlersRef.current.cleanupInitialization = cleanupInitialization;
    
    // Event listener for visibility toggle
    window.addEventListener('mlOverlaySettingsChanged', handleOverlaySettingsChange);
    
    // Listen for new detection data
    const handleDetectionDataUpdate = (event) => {
      // If we received new detection data, update the overlay
      if (event.detail && typeof MLOverlayModule.renderMLOverlay === 'function') {
        // Check if this detection was triggered by button click (from buttonTriggered flag)
        const isButtonTriggered = event.buttonTriggered === true || 
                                  (event.detail && event.detail.buttonTriggered === true) ||
                                  window.mlOverlaySettings?.showOverlay === true;
        
        console.log('MLOverlayInitializer: Received detection data update, buttonTriggered=', isButtonTriggered);
        
        const mapInstance = window.map || window.googleMapsInstance || window._googleMap;
        
        if (mapInstance) {
          const settings = window.mlOverlaySettings || {};
          // Only show overlay if explicitly requested or button triggered
          const showOverlay = isButtonTriggered || settings.showOverlay === true || 
                             (window.detectionShowOverlay === true);
          
          // Only render if overlay should be visible
          if (showOverlay) {
            console.log('MLOverlayInitializer: Rendering overlay with detection data');
            
            // IMPORTANT: Merge with existing data instead of replacing it
            // This ensures we don't lose existing markers when adding new ones
            if (window.mlDetectionData && window.mlDetectionData.trees && event.detail.trees) {
              console.log('MLOverlayInitializer: Merging new detection data with existing data');
              // Create new trees array with both existing and new trees
              const mergedTrees = [...window.mlDetectionData.trees, ...event.detail.trees];
              // Update the event data with merged trees
              event.detail.trees = mergedTrees;
            }
            
            // Store data globally for access by other components
            window.mlDetectionData = event.detail;
            
            // Ensure overlay is visible
            MLOverlayModule.renderMLOverlay(
              mapInstance,
              event.detail,
              {
                opacity: settings.opacity !== undefined ? settings.opacity : 0.7,
                showSegmentation: settings.showSegmentation !== undefined ? settings.showSegmentation : true,
                forceRenderBoxes: true,
                appendMode: true  // Use append mode to add to existing overlay
              }
            );
            
            // Broadcast event that overlay is showing
            window.dispatchEvent(new CustomEvent('mlOverlayShowing', {
              detail: { 
                source: 'detection_data_update',
                data: event.detail
              }
            }));
          } else {
            console.log('MLOverlayInitializer: Not showing overlay automatically - waiting for Detect button');
          }
        }
        
        // Always show the detection preview with the new data, regardless of overlay visibility
        // This ensures the user can see the detection results even if the overlay is hidden
        console.log('MLOverlayInitializer: Showing detection preview with new data');
        try {
          if (typeof window.showDetectionPreview === 'function') {
            window.showDetectionPreview(event.detail);
          } else {
            console.warn('MLOverlayInitializer: window.showDetectionPreview not available, trying after delay');
            // Try again after a short delay in case it's not defined yet
            setTimeout(() => {
              if (typeof window.showDetectionPreview === 'function') {
                window.showDetectionPreview(event.detail);
              }
            }, 500);
          }
        } catch (err) {
          console.error('MLOverlayInitializer: Error showing detection preview:', err);
        }
      }
    };
    
    // Store handler in ref for reference stability
    handlersRef.current.handleDetectionDataUpdate = handleDetectionDataUpdate;
    
    window.addEventListener('detectionDataLoaded', handleDetectionDataUpdate);
    
    return () => {
      window.removeEventListener('mlOverlaySettingsChanged', handlersRef.current.handleOverlaySettingsChange);
      window.removeEventListener('detectionDataLoaded', handlersRef.current.handleDetectionDataUpdate);
      
      // Call the cleanup function from initializeWithExistingData
      if (typeof handlersRef.current.cleanupInitialization === 'function') {
        handlersRef.current.cleanupInitialization();
      }
    };
  }, [initialized]);
  
  // Performance monitoring
  useEffect(() => {
    if (!initialized) return;
    
    // Set up performance monitoring
    let frameCount = 0;
    let lastFrameTime = performance.now();
    let fpsInterval;
    
    // Only run in development mode
    if (process.env.NODE_ENV === 'development') {
      fpsInterval = setInterval(() => {
        const now = performance.now();
        const elapsed = now - lastFrameTime;
        
        if (elapsed > 0 && window._mlDetectionOverlay) {
          const fps = frameCount * 1000 / elapsed;
          console.log(`MLOverlay performance: ${fps.toFixed(1)} FPS, ${frameCount} frames in ${(elapsed/1000).toFixed(1)}s`);
          frameCount = 0;
          lastFrameTime = now;
        }
      }, 5000); // Check every 5 seconds
    }
    
    // Count frames when drawing occurs
    const handleDraw = () => {
      frameCount++;
    };
    
    window.addEventListener('mlOverlayDrawComplete', handleDraw);
    
    // Clean up
    return () => {
      window.removeEventListener('mlOverlayDrawComplete', handleDraw);
      
      if (fpsInterval) clearInterval(fpsInterval);
    };
  }, [initialized]);
  
  // Dispatch initialization event when component initializes
  useEffect(() => {
    if (initialized) {
      // Dispatch event to notify other components that MLOverlay is ready
      window.dispatchEvent(new CustomEvent('mlOverlayInitialized', {
        detail: { success: true }
      }));
      
      // Add helper functions to the global scope for easier access from anywhere
      window.toggleMLOverlay = (show) => {
        const current = window.mlOverlaySettings?.showOverlay;
        const newValue = show !== undefined ? show : !current;
        
        window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
          detail: { 
            showOverlay: newValue,
            showSegmentation: window.mlOverlaySettings?.showSegmentation,
            opacity: window.mlOverlaySettings?.opacity 
          }
        }));
        
        return newValue;
      };
      
      window.setMLOverlayOpacity = (opacity) => {
        console.log(`MLOverlayInitializer: setMLOverlayOpacity called with ${opacity}`);
        
        // Update global settings for consistency
        window.mlOverlaySettings = {
          ...(window.mlOverlaySettings || {}),
          opacity: opacity
        };
        
        // Save to localStorage for persistence
        try {
          localStorage.setItem('ml-overlay-opacity', opacity.toString());
        } catch (e) {
          console.error("Error saving opacity to localStorage:", e);
        }
        
        // APPROACH 1: Use the module function if available
        if (typeof MLOverlayModule.updateMLOverlayOpacity === 'function') {
          MLOverlayModule.updateMLOverlayOpacity(opacity);
        }
        
        // APPROACH 2: Use the global function if available
        if (typeof window.updateMLOverlayOpacity === 'function') {
          window.updateMLOverlayOpacity(opacity);
        }
        
        // APPROACH 3: Update the MLDetectionOverlay instance directly
        if (window._mlDetectionOverlay) {
          // Use updateOpacity method if available
          if (typeof window._mlDetectionOverlay.updateOpacity === 'function') {
            window._mlDetectionOverlay.updateOpacity(opacity);
          } 
          // Direct access to div if method not available
          else if (window._mlDetectionOverlay.div) {
            window._mlDetectionOverlay.div.style.opacity = opacity.toString();
          }
        }
        
        // APPROACH 4: Update DOM elements directly as fallback
        const overlayElement = document.getElementById('ml-detection-overlay');
        if (overlayElement) {
          overlayElement.style.opacity = opacity.toString();
          
          // Also update background color for any tint elements
          const tintElements = overlayElement.querySelectorAll('.tint-layer') || [];
          if (tintElements.length > 0) {
            tintElements.forEach(tint => {
              tint.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
            });
          } else {
            // If no tint elements, update the overlay background directly
            overlayElement.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
          }
        }
        
        // APPROACH 5: Update segmentation masks directly
        const masks = document.querySelectorAll('.segmentation-mask');
        if (masks.length > 0) {
          masks.forEach(mask => {
            mask.style.opacity = Math.min(opacity * 1.2, 0.8); // Adjust for better visibility
          });
        }
        
        // Dispatch event for other components
        window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
          detail: { 
            opacity: opacity,
            showOverlay: window.mlOverlaySettings?.showOverlay,
            showSegmentation: window.mlOverlaySettings?.showSegmentation,
            source: 'set_ml_overlay_opacity',
            timestamp: Date.now()
          }
        }));
        
        return true;
      };
      
      // Listen for tree detection events to properly handle button-triggered detection
      const handleOpenTreeDetection = (event) => {
        // Check if this was triggered by button click
        const isButtonTriggered = event.buttonTriggered === true || 
                               (event.detail && event.detail.buttonTriggered === true);
        
        if (isButtonTriggered) {
          console.log('MLOverlayInitializer: Button-triggered detection, enabling overlay');
          
          // Update settings to show overlay
          window.mlOverlaySettings = {
            ...(window.mlOverlaySettings || {}),
            showOverlay: true,
            pendingButtonTrigger: false
          };
          
          // If we have saved data from earlier, restore it
          if (!window.mlDetectionData && window._savedMlDetectionData) {
            window.mlDetectionData = window._savedMlDetectionData;
          }
          
          // Wait for server response instead of polling
          if (event.detail && event.detail.job_id) {
            const clientJobId = event.detail.job_id;
            console.log(`MLOverlayInitializer: Detection requested with initial ID ${clientJobId}, waiting for server response`);
            
            // Register a listener for when the server response is received (via MapControls)
            const handleServerJobIdUpdate = () => {
              // Check if global job ID has been updated by MapControls
              if (window.currentDetectionJobId && window.currentDetectionJobId !== clientJobId) {
                console.log(`MLOverlayInitializer: Using server-returned job ID ${window.currentDetectionJobId} instead of client ID ${clientJobId}`);
                
                // Use detectionService to load data once available
                import('./detectionService').then(module => {
                  if (typeof module.loadDetectionData === 'function') {
                    console.log(`MLOverlayInitializer: Using detectionService to load data for job ${window.currentDetectionJobId}`);
                    module.loadDetectionData(window.currentDetectionJobId);
                  }
                });
                
                // Clean up the interval since we got what we needed
                clearInterval(checkInterval);
              }
            };
            
            // Check every second for a short period if the global job ID has been updated
            let checkCount = 0;
            const maxChecks = 10; // Check for up to 10 seconds
            const checkInterval = setInterval(() => {
              checkCount++;
              handleServerJobIdUpdate();
              if (checkCount >= maxChecks) {
                clearInterval(checkInterval);
                console.log('MLOverlayInitializer: Finished waiting for server job ID');
              }
            }, 1000);
          }
        } else {
          console.log('MLOverlayInitializer: Non-button detection event, waiting for explicit trigger');
        }
      };
      
      window.addEventListener('openTreeDetection', handleOpenTreeDetection);
      
      console.log('MLOverlayInitializer: MLOverlay initialization complete with helper functions');
      
      // Return cleanup function for all event listeners
      return () => {
        window.removeEventListener('openTreeDetection', handleOpenTreeDetection);
      };
    }
  }, [initialized]);

  // This component doesn't render anything visible
  return null;
};

export default MLOverlayInitializer;