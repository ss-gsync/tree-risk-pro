// src/components/visualization/Detection/MLOverlayInitializer.jsx
//
// This component ensures that MLOverlay.js is properly initialized
// and handles the Google Maps API dependency correctly.
//
// Enhanced version with performance optimizations and improved initialization

import React, { useEffect, useState } from 'react';
// Import MLOverlay without destructuring to ensure we get the full module
import MLOverlayModule from './MLOverlay';

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
    // Check if MLOverlay has an ensureInitialized method
    if (typeof MLOverlayModule.ensureInitialized === 'function') {
      console.log('MLOverlayInitializer: Checking MLOverlay initialization...');
      
      // Check initialization status
      const isInitialized = MLOverlayModule.ensureInitialized();
      
      // If already initialized, just log success and update state
      if (isInitialized) {
        console.log('MLOverlayInitializer: MLOverlay already initialized');
        setInitialized(true);
        
        // Ensure we also initialize the overlay with any existing data
        initializeWithExistingData();
        return;
      }
      
      console.log('MLOverlayInitializer: MLOverlay not initialized, setting up event listeners');
      
      // Set up event listeners to initialize MLOverlay when Maps API is ready
      
      // First, try window.google check with interval
      const checkInterval = setInterval(() => {
        // Check if Google Maps is available now
        if (typeof google !== 'undefined' && google.maps && google.maps.OverlayView) {
          console.log('MLOverlayInitializer: Google Maps API detected, initializing MLOverlay');
          clearInterval(checkInterval);
          
          // Try initializing MLOverlay
          const success = MLOverlayModule.ensureInitialized();
          setInitialized(success);
          
          if (success) {
            initializeWithExistingData();
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
          const success = MLOverlayModule.ensureInitialized();
          setInitialized(success);
          
          if (success) {
            initializeWithExistingData();
          }
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
    } else {
      console.error('MLOverlayInitializer: MLOverlay module does not have ensureInitialized method');
    }
  }, []);
  
  // Function to initialize overlay with existing data - now strictly respects user control
  const initializeWithExistingData = () => {
    // IMPORTANT: Don't automatically initialize overlay with existing data - let user control with Detect button
    console.log('MLOverlayInitializer: Initialized, but waiting for Detect button to show overlay');
    
    // Check if we already have a flag set to show the overlay (from previous Detect button click)
    const shouldShowOverlay = window.detectionShowOverlay === true || 
                             window.mlOverlaySettings?.showOverlay === true;
    
    // Only clear detection data if we don't have a request to show the overlay
    if (!shouldShowOverlay && window.mlDetectionData) {
      console.log('MLOverlayInitializer: Clearing existing detection data to prevent auto-rendering');
      // Save a backup copy in case we need to restore it later, but don't render it
      window._savedMlDetectionData = { ...window.mlDetectionData };
      window.mlDetectionData = null;
    }
    
    // Explicitly mark overlay as hidden by default UNLESS we've already clicked the Detect button
    if (!shouldShowOverlay) {
      window.detectionShowOverlay = false;
      window.mlOverlaySettings = {
        ...(window.mlOverlaySettings || {}),
        showOverlay: false,
        pendingButtonTrigger: true  // Indicate we're waiting for explicit button trigger
      };
      
      // Remove any existing overlay to prevent it from showing
      if (window._mlDetectionOverlay && window._mlDetectionOverlay.div) {
        console.log('MLOverlayInitializer: Hiding existing overlay');
        window._mlDetectionOverlay.div.style.display = 'none';
      }
    } else {
      console.log('MLOverlayInitializer: Detect button already clicked, overlay should be visible');
    }
    
    // Listen for openTreeDetection events to handle button triggers
    const handleOpenTreeDetection = (event) => {
      // Check if this was triggered by button click
      const isButtonTriggered = event.buttonTriggered === true || 
                             (event.detail && event.detail.buttonTriggered === true);
      
      if (isButtonTriggered) {
        console.log('MLOverlayInitializer: Detect button clicked, setting showOverlay flag');
        
        // Set the global flag for showing the overlay
        window.detectionShowOverlay = true;
        window.mlOverlaySettings = {
          ...(window.mlOverlaySettings || {}),
          showOverlay: true,
          pendingButtonTrigger: false
        };
        
        // If we have saved data, restore it
        if (!window.mlDetectionData && window._savedMlDetectionData) {
          window.mlDetectionData = window._savedMlDetectionData;
        }
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
  
  // Create mock data for testing if needed
  const createMockDataIfNeeded = () => {
    // Only create mock data if we're in development and user has enabled test mode
    const isTestMode = localStorage.getItem('enable_ml_test_mode') === 'true';
    if (process.env.NODE_ENV !== 'development' && !isTestMode) return;
    
    console.log('MLOverlayInitializer: Creating mock data for testing ML overlay');
    
    // Create basic mock data with random tree positions
    const mockData = {
      job_id: `mock_${Date.now()}`,
      timestamp: new Date().toISOString(),
      trees: []
    };
    
    // Generate random trees of different categories
    const categories = [
      'healthy_tree',
      'hazardous_tree',
      'dead_tree',
      'low_canopy_tree',
      'pest_disease_tree'
    ];
    
    // Generate 15 random trees
    for (let i = 0; i < 15; i++) {
      const category = categories[Math.floor(Math.random() * categories.length)];
      
      mockData.trees.push({
        id: `mock-tree-${i}`,
        category,
        class: category,
        confidence: 0.7 + (Math.random() * 0.2),
        box: {
          x: 0.3 + (Math.random() * 0.5), // Center around the view
          y: 0.3 + (Math.random() * 0.5),
          width: 0.05 + (Math.random() * 0.05),
          height: 0.05 + (Math.random() * 0.05)
        }
      });
    }
    
    // Store in global scope
    window.mlDetectionData = mockData;
    console.log('MLOverlayInitializer: Created mock data with', mockData.trees.length, 'trees');
    
    // Try to render the overlay with mock data
    setTimeout(() => {
      initializeWithExistingData();
    }, 1000);
  };
  
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
        
        // Update opacity if specified
        if (typeof MLOverlayModule.updateMLOverlayOpacity === 'function' && 
            opacity !== undefined) {
          // Apply to the ML overlay
          MLOverlayModule.updateMLOverlayOpacity(opacity);
          
          // Also update the element directly
          const overlay = document.getElementById('ml-detection-overlay');
          if (overlay) {
            overlay.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
          }
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
    
    // Initialize overlay with any existing data and get cleanup function
    const cleanupInitialization = initializeWithExistingData();
    
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
            
            // Store data globally for access by other components
            window.mlDetectionData = event.detail;
            
            // Ensure overlay is visible
            MLOverlayModule.renderMLOverlay(
              mapInstance,
              event.detail,
              {
                opacity: settings.opacity !== undefined ? settings.opacity : 0.7,
                showSegmentation: settings.showSegmentation !== undefined ? settings.showSegmentation : true,
                forceRenderBoxes: true
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
      }
    };
    
    window.addEventListener('detectionDataLoaded', handleDetectionDataUpdate);
    
    return () => {
      window.removeEventListener('detectionDataLoaded', handleDetectionDataUpdate);
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
      window.removeEventListener('mlOverlaySettingsChanged', handleOverlaySettingsChange);
      window.removeEventListener('detectionDataLoaded', handleDetectionDataUpdate);
      window.removeEventListener('mlOverlayDrawComplete', handleDraw);
      
      // Call the cleanup functions for the event handlers
      
      // Call the cleanup function from initializeWithExistingData
      if (typeof cleanupInitialization === 'function') {
        cleanupInitialization();
      }
      
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
        if (typeof MLOverlayModule.updateMLOverlayOpacity === 'function') {
          MLOverlayModule.updateMLOverlayOpacity(opacity);
        }
        
        // Also update global settings
        window.mlOverlaySettings = {
          ...(window.mlOverlaySettings || {}),
          opacity
        };
        
        window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
          detail: { 
            opacity,
            showOverlay: window.mlOverlaySettings?.showOverlay,
            showSegmentation: window.mlOverlaySettings?.showSegmentation 
          }
        }));
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