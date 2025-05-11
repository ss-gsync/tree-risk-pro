// src/components/common/Layout/Sidebar/index.jsx

import React, { useState, useEffect } from 'react';
import { Settings, BarChart, Database, Box, FileText, ChevronLeft, ChevronRight, Eye, ClipboardList } from 'lucide-react';
import MapControls from '../../../visualization/MapView/MapControls';

const SidebarItem = ({ icon: Icon, label, active = false, onClick }) => {
  return (
    <button
      className={`flex items-center w-full p-3 mb-2 rounded-md transition-colors
        ${active ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:bg-gray-100'}`}
      onClick={() => onClick && onClick(label)}
    >
      <Icon className="h-5 w-5 mr-3" />
      <span className="font-medium">{label}</span>
    </button>
  );
};

const Sidebar = ({ onNavigate, mapRef, mapDataRef }) => {
  const [collapsed, setCollapsed] = useState(false);
  const [selectedItem, setSelectedItem] = useState('Map');
  const [dataAnalysisCollapsed, setDataAnalysisCollapsed] = useState(false);
  const [mapControlsCollapsed, setMapControlsCollapsed] = useState(false);
  
  // Get saved map type from localStorage or default to hybrid
  const savedMapType = localStorage.getItem('currentMapType') || 'hybrid';
  const [mapType, setMapType] = useState(savedMapType);
  
  /**
   * Toggles the sidebar between collapsed and expanded states
   * 
   * This function dispatches a custom event first so other components can
   * react to the sidebar state change before updating the local state.
   * Many components need to adjust their position based on sidebar width.
   */
  const toggleSidebar = () => {
    // Dispatch event first so components can react to this change
    window.dispatchEvent(new CustomEvent('leftSidebarToggle', {
      detail: { 
        collapsed: !collapsed 
      }
    }));
    
    // Then update local state
    setCollapsed(!collapsed);
  };
  
  /**
   * Handler for forced sidebar collapse events from other components
   * 
   * This allows other components (like the Settings panel) to collapse
   * the sidebar when needed without direct access to the sidebar state.
   */
  useEffect(() => {
    const handleForceSidebarCollapse = (event) => {
      // Only handle this event if we're not already collapsed
      if (!collapsed) {
        toggleSidebar();
      }
    };
    
    window.addEventListener('forceSidebarCollapse', handleForceSidebarCollapse);
    
    return () => {
      window.removeEventListener('forceSidebarCollapse', handleForceSidebarCollapse);
    };
  }, [collapsed]);
  
  // Toggle the Data Analysis section
  const toggleDataAnalysis = () => {
    setDataAnalysisCollapsed(!dataAnalysisCollapsed);
  };
  
  // Toggle the Map Controls section
  const toggleMapControls = () => {
    setMapControlsCollapsed(!mapControlsCollapsed);
    
    // If expanding map controls, ensure we're in Map view
    if (mapControlsCollapsed) {
      setSelectedItem('Map');
      
      // Dispatch navigate event to show Map view
      window.dispatchEvent(new CustomEvent('navigateTo', {
        detail: { 
          view: 'Map',
          preserve3DState: true
        }
      }));
    }
  };
  
  // Track header collapsed state for badge positioning
  const [isHeaderCollapsed, setIsHeaderCollapsed] = useState(false);
  
  // Helper function to ensure Map view is active before performing map operations
  const ensureMapViewActive = (callback) => {
    if (selectedItem !== 'Map') {
      console.log("Switching to Map view from:", selectedItem);
      
      // First switch to Map view
      setSelectedItem('Map');
      
      // Dispatch navigate event
      window.dispatchEvent(new CustomEvent('navigateTo', {
        detail: { 
          view: 'Map',
          preserve3DState: true
        }
      }));
      
      // Wait for view to change before executing callback
      setTimeout(() => {
        if (callback) callback();
      }, 100);
    } else {
      // Already in Map view, execute callback immediately
      if (callback) callback();
    }
  };
  
  // Handler for header state changes from other components
  useEffect(() => {
    const handleHeaderCollapse = (event) => {
      setIsHeaderCollapsed(event.detail.collapsed);
    };
    
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    return () => {
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
    };
  }, []);
  
  
  // Detect and load initial header state
  useEffect(() => {
    const header = document.querySelector('header');
    if (header) {
      const collapsed = header.classList.contains('collapsed') || header.offsetHeight < 50;
      setIsHeaderCollapsed(collapsed);
    }
  }, []);
  
  // Handle sidebar item click
  const handleItemClick = (label) => {
    setSelectedItem(label);
    
    // We don't need to track 3D mode in the sidebar anymore
    
    // Handle navigation based on the label
    if (typeof onNavigate === 'function') {
      onNavigate(label);
    }
    
    // Dispatch a custom event for other components to listen for
    window.dispatchEvent(new CustomEvent('navigateTo', {
      detail: { 
        view: label,
        preserve3DState: true
      }
    }));
  };
  
  // Initialize map state when component mounts
  useEffect(() => {
    setSelectedItem('Map');
    
    // Initialize Map Controls to collapsed state
    setMapControlsCollapsed(true);
    
    
    // Initialize marker variables for manual placement
    window.manualTreeMarkers = window.manualTreeMarkers || [];
    window.manualTreePlacementActive = false;
    window.markerSize = 'medium';
    window.markerColor = '#2563eb';
    
    // Set up handler for database feature selection with green theme
    const handleDatabaseFeatureSelection = (event) => {
      // React to any feature selection event to ensure detection badges are removed
      console.log("Feature selection event received, source:", event.detail?.source);
      
      // Always remove detection badges and overlays regardless of source
      const removeDetectionElements = () => {
        try {
          // Remove detection badge if present (with multiple selector patterns)
          document.querySelectorAll('#detection-debug, .detection-debug, [id*="detection-debug"]').forEach(badge => {
            if (badge && document.body.contains(badge)) {
              try {
                console.log("Feature selection handler removing badge:", badge.id || badge.className);
                badge.style.display = 'none';
                badge.style.opacity = '0';
                if (badge.parentNode) {
                  badge.parentNode.removeChild(badge);
                }
              } catch (e) {}
            }
          });
          
          // Remove ML overlay if present (with multiple selector patterns)
          document.querySelectorAll('#ml-detection-overlay, .ml-detection-overlay, [id*="ml-detection"]').forEach(overlay => {
            if (overlay && document.body.contains(overlay)) {
              try {
                console.log("Feature selection handler removing overlay:", overlay.id || overlay.className);
                overlay.style.display = 'none';
                overlay.style.opacity = '0';
                if (overlay.parentNode) {
                  overlay.parentNode.removeChild(overlay);
                }
              } catch (e) {}
            }
          });
        } catch (e) {
          console.log("Error in removeDetectionElements:", e.message);
        }
      };
      
      // Run the cleanup now and again with a delay
      removeDetectionElements();
      setTimeout(removeDetectionElements, 100);
      
      // Only continue with styling if this is coming from the database sidebar
      if (event.detail?.source !== 'database_sidebar') {
        return;
      }
      
      // Safely remove DOM elements with error handling
      const safeRemove = (elementId) => {
        try {
          const element = document.getElementById(elementId);
          if (element && element.parentNode) {
            element.parentNode.removeChild(element);
            console.log(`Safely removed ${elementId}`);
          }
        } catch (e) {
          console.log(`Failed to remove ${elementId}: ${e.message}`);
        }
      };
      
      // Ensure that any detection overlays are removed safely
      safeRemove('ml-detection-overlay');
      safeRemove('detection-debug');
      
      // Apply green theme to database sidebar if created
      setTimeout(() => {
        const databaseSidebar = document.querySelector('.database-sidebar, .feature-selection-sidebar');
        if (databaseSidebar) {
          // Apply green theme
          const header = databaseSidebar.querySelector('div[style*="border-bottom"]');
          if (header) {
            header.style.backgroundColor = '#e6f4ea'; // Light green header
            header.style.borderBottom = '1px solid rgba(22, 101, 52, 0.1)';
            
            // Update header text color
            const headerText = header.querySelector('span');
            if (headerText) {
              headerText.style.color = '#166534'; // Dark green
            }
          }
          
          // Style buttons and badges 
          databaseSidebar.querySelectorAll('button').forEach(btn => {
            if (btn.className.includes('bg-blue')) {
              btn.className = btn.className.replace('bg-blue', 'bg-emerald');
              btn.className = btn.className.replace('text-blue', 'text-emerald');
            }
            if (btn.style.backgroundColor && btn.style.backgroundColor.includes('rgb(13, 71, 161)')) {
              btn.style.backgroundColor = '#166534'; // Dark green
            }
          });
        }
      }, 200);
    };
    
    window.addEventListener('openFeatureSelection', handleDatabaseFeatureSelection);
    
    // Add an event listener to handle 'forceCloseObjectDetection' event
    const handleForceCloseObjectDetection = (event) => {
      console.log('Force close object detection event received from:', event.detail?.source);
      
      // Helper function for safely removing elements by selector
      const safeRemoveBySelector = (selector) => {
        try {
          const elements = document.querySelectorAll(selector);
          elements.forEach(el => {
            try {
              // Check that element still exists in DOM
              if (el && document.body.contains(el) && el.parentNode) {
                // First hide it to prevent visual glitches
                el.style.display = 'none';
                el.style.opacity = '0';
                el.style.visibility = 'hidden';
                el.style.pointerEvents = 'none';
                
                // Then remove it
                setTimeout(() => {
                  try {
                    if (el && el.parentNode && document.body.contains(el)) {
                      el.parentNode.removeChild(el);
                      console.log(`Safely removed element with selector: ${selector}`);
                    }
                  } catch (err) {
                    console.log(`Error in delayed removal for ${selector}: ${err.message}`);
                  }
                }, 50);
              }
            } catch (err) {
              console.log(`Error processing element ${selector}: ${err.message}`);
            }
          });
        } catch (e) {
          console.log(`Error in selector ${selector}: ${e.message}`);
        }
      };
      
      // Helper function for safely removing elements by ID
      const safeRemoveById = (id) => {
        try {
          const element = document.getElementById(id);
          if (element && document.body.contains(element) && element.parentNode) {
            // First hide it
            element.style.display = 'none';
            element.style.opacity = '0';
            element.style.visibility = 'hidden';
            element.style.pointerEvents = 'none';
            
            // Then remove it
            setTimeout(() => {
              try {
                if (element.parentNode && document.body.contains(element)) {
                  element.parentNode.removeChild(element);
                  console.log(`Safely removed element with ID: ${id}`);
                }
              } catch (e) {
                console.log(`Error in delayed removal of ${id}: ${e.message}`);
              }
            }, 50);
          }
        } catch (e) {
          console.log(`Error removing element with ID ${id}: ${e.message}`);
        }
      };
      
      // First hide elements before removing them to prevent visual glitches
      document.querySelectorAll('.detection-sidebar, .detection-debug-badge').forEach(el => {
        el.style.display = 'none';
        el.style.opacity = '0';
        el.style.visibility = 'hidden';
      });
      
      // Hide ML overlay
      const overlay = document.getElementById('ml-detection-overlay');
      if (overlay) {
        overlay.style.display = 'none';
        overlay.style.opacity = '0';
        overlay.style.visibility = 'hidden';
      }
      
      // Notify map about right panel closing
      window.dispatchEvent(new CustomEvent('rightPanelToggle', {
        detail: {
          isOpen: false,
          source: 'detection_close',
          panelType: 'detection'
        }
      }));
      
      // Allow a tick for the hiding to take effect
      setTimeout(() => {
        // Now safely remove elements using more thorough selectors
        safeRemoveBySelector('.detection-sidebar');
        safeRemoveBySelector('.detection-debug-badge');
        safeRemoveBySelector('[id*="detection-debug"]');
        safeRemoveBySelector('[class*="detection-debug"]');
        safeRemoveById('ml-detection-overlay');
        safeRemoveById('detection-debug');
        
        // Remove any elements with detection in their ID/class
        safeRemoveBySelector('[id*="detection"]');
        safeRemoveBySelector('[class*="detection"]');
        
        // Clean up any markers that might have been added
        if (window.manualTreeMarkers && event.detail?.clearMarkers) {
          try {
            window.manualTreeMarkers.forEach(marker => {
              if (marker) marker.setMap(null);
            });
            window.manualTreeMarkers = [];
            console.log('Cleared manual tree markers');
          } catch (e) {
            console.log(`Error clearing markers: ${e.message}`);
          }
        }
        
        // Reset active detection flags
        window.isDetectionSidebarActive = false;
        window.isDetectionModeActive = false;
        window.manualTreePlacementActive = false;
        
        // Reset container width
        const mapContainer = document.getElementById('map-container');
        if (mapContainer) {
          mapContainer.style.right = '0';
        }
        
        // Trigger resize event to update layout
        window.dispatchEvent(new Event('resize'));
        
        // Make another cleanup pass after brief delay to catch any elements that might have been missed
        setTimeout(() => {
          safeRemoveBySelector('.detection-sidebar');
          safeRemoveBySelector('.detection-debug-badge');
          safeRemoveById('ml-detection-overlay');
          safeRemoveById('detection-debug');
          window.dispatchEvent(new Event('resize'));
        }, 200);
      }, 50);
    };
    
    window.addEventListener('forceCloseObjectDetection', handleForceCloseObjectDetection);
    
    return () => {
      window.removeEventListener('forceCloseObjectDetection', handleForceCloseObjectDetection);
      window.removeEventListener('openFeatureSelection', handleDatabaseFeatureSelection);
    };
  }, []);
  
  
  useEffect(() => {
    // This section intentionally left empty after moving 3D functionality back to MapControls
    
    // If no map type preference exists yet, set hybrid as default
    if (!localStorage.getItem('currentMapType')) {
      localStorage.setItem('currentMapType', 'hybrid');
    }
    
    // Listen for settings updates
    const handleSettingsUpdated = (event) => {
      try {
        const settings = event.detail?.settings;
        if (settings && settings.defaultMapType) {
          console.log("Sidebar received settings update with default map type:", settings.defaultMapType);
          
          // Do not change the current map type, just update the stored default
          // The map will be updated separately by MapView.jsx
        }
      } catch (e) {
        console.error("Error handling settings update in sidebar:", e);
      }
    };
    
    // Listen for map type changes
    const handleMapTypeChanged = (event) => {
      const { mapTypeId } = event.detail;
      if (mapTypeId) {
        console.log("Sidebar received map type change:", mapTypeId);
        setMapType(mapTypeId);
      }
    };
    
    window.addEventListener('settingsUpdated', handleSettingsUpdated);
    window.addEventListener('mapTypeChanged', handleMapTypeChanged);
    
    return () => {
      window.removeEventListener('settingsUpdated', handleSettingsUpdated);
      window.removeEventListener('mapTypeChanged', handleMapTypeChanged);
    };
  }, []);

  return (
    <div 
      className={`flex flex-col h-full bg-white border-r border-gray-200 shadow-sm transition-all duration-300 ease-in-out ${
        collapsed ? 'w-10 bg-gray-100' : 'w-64'
      }`}
    >

      {/* Sidebar Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Map Controls Section */}
        <div className={`${collapsed ? 'py-4' : 'p-4'} border-b border-gray-200`}>
          <div className="flex justify-between items-center relative">
            <div className={`flex items-center ${collapsed ? 'ml-0 justify-center w-full' : '-ml-4'}`}>
              <button 
                className={`p-1 ${collapsed ? '' : 'mr-2'} rounded transition-colors ${
                  collapsed ? 'bg-white hover:bg-gray-200 shadow-sm' : 'hover:bg-gray-100'
                }`}
                onClick={toggleSidebar}
                aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              >
                {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
              </button>
              {!collapsed && <h3 className="text-sm font-medium text-gray-600 -ml-2 cursor-pointer hover:text-gray-800" onClick={toggleSidebar}>Controls</h3>}
            </div>
            
            {/* 3D button moved to MapControls */}

            {!collapsed && (
              <div className="flex items-center">
                <div className="flex items-center cursor-pointer" onClick={toggleMapControls}>
                  <span className="text-sm font-medium text-gray-600 mr-1">Map</span>
                  <div 
                    className="transition-transform duration-200" 
                    style={{ transform: mapControlsCollapsed ? 'rotate(180deg)' : 'rotate(0deg)' }}
                  >
                    <ChevronLeft size={16} className="rotate-90" />
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {!collapsed && (
            <div className={`space-y-1 transition-all duration-200 overflow-hidden ${mapControlsCollapsed ? 'max-h-0 opacity-0 mt-0 p-0' : 'max-h-96 opacity-100 mt-3'}`}>
              <MapControls 
                mapRef={mapRef}
                mapDataRef={mapDataRef}
                viewSwitchFunction={ensureMapViewActive}
              />
            </div>
          )}
        </div>
        
        {/* Data Analysis Section */}
        <div className={`${collapsed ? 'hidden' : 'p-4'} border-b border-gray-200`}>
          <div className="flex justify-between items-center cursor-pointer" onClick={toggleDataAnalysis}>
            <h3 className="text-sm font-medium text-gray-600">Analysis</h3>
            <div className="transition-transform duration-200" style={{ transform: dataAnalysisCollapsed ? 'rotate(180deg)' : 'rotate(0deg)' }}>
              <ChevronLeft size={16} className="rotate-90" />
            </div>
          </div>
          <div className={`space-y-2 transition-all duration-200 overflow-hidden ${dataAnalysisCollapsed ? 'max-h-0 opacity-0 mb-0 mt-0 p-0' : 'max-h-96 opacity-100 mt-3'}`}>
            {/* First row - Video and Detection buttons side-by-side */}
            <div className="grid grid-cols-2 gap-2">
              {/* Video Button */}
              <button
                onClick={() => {
                  // First close any visible sidebars
                  window.dispatchEvent(new CustomEvent('exitValidationMode', {
                    detail: { 
                      source: 'sidebar', 
                      target: 'imagery',
                      clearExisting: true
                    }
                  }));
                  
                  // Make sure Object Detection sidebar is closed
                  window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                    detail: { source: 'sidebar' }
                  }));
                  
                  // Notify map that right panel is closing
                  window.dispatchEvent(new CustomEvent('rightPanelToggle', {
                    detail: {
                      isOpen: false,
                      source: 'sidebar'
                    }
                  }));
                  
                  // Make sure Tree Database sidebar is closed
                  window.dispatchEvent(new CustomEvent('forceCloseTreeDatabase', {
                    detail: { source: 'sidebar' }
                  }));
                  
                  // Close the Review panel if it's open
                  window.dispatchEvent(new CustomEvent('closeReviewPanel', {
                    detail: { source: 'aerial' }
                  }));
                  
                  // Then make sure Imagery panel is expanded with Review tab active
                  window.dispatchEvent(new CustomEvent('expandImageryPanel', {
                    detail: { 
                      source: 'sidebar',
                      tab: 'imagery' // 'imagery' is the Review tab
                    }
                  }));
                }}
                className="flex items-center justify-center p-2 bg-indigo-50 text-indigo-900 hover:bg-indigo-100 rounded cursor-pointer border border-indigo-300/20"
              >
                <Box className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Aerial</span>}
              </button>
              
              {/* Components Button (renamed from Detection) */}
              <button
                onClick={() => {
                  // Set the selected item to Map and trigger navigation
                  setSelectedItem('Map');
                  
                  // Explicitly navigate to Map view
                  window.dispatchEvent(new CustomEvent('navigateTo', {
                    detail: { 
                      view: 'Map',
                      preserve3DState: true
                    }
                  }));
                  
                  // First ensure we're in Map view before doing anything else
                  ensureMapViewActive(() => {
                    // First, close ALL other panels/sidebars including Analytics panel
                    // Close any imagery panels
                    window.dispatchEvent(new CustomEvent('forceCloseImageryPanel', {
                      detail: { source: 'detection' }
                    }));
                    
                    // Close any database panels
                    window.dispatchEvent(new CustomEvent('forceCloseTreeDatabase', {
                      detail: { source: 'detection' }
                    }));
                    
                    // Close any review panels
                    window.dispatchEvent(new CustomEvent('closeReviewPanel', {
                      detail: { source: 'detection' }
                    }));
                    
                    // Close analytics panel if open
                    try {
                      // Hide the analytics panel directly first for immediate effect
                      const analyticsSidebar = document.getElementById('analytics-sidebar');
                      if (analyticsSidebar) {
                        analyticsSidebar.style.display = 'none';
                        analyticsSidebar.style.visibility = 'hidden';
                        analyticsSidebar.style.opacity = '0';
                      }
                      
                      // Then dispatch the event for proper cleanup
                      window.dispatchEvent(new CustomEvent('closeAnalyticsPanel', {
                        detail: { source: 'detection' }
                      }));
                    } catch (e) {
                      console.error("Error closing analytics panel:", e);
                    }
                    
                    // Creating ML overlay and badge immediately
                    const createOverlayAndBadge = () => {
                      try {
                        // Get the map container
                        const mapContainer = document.getElementById('map-container');
                        if (!mapContainer) {
                          console.error("Map container not found for ML overlay!");
                          return;
                        }
                        
                        // Remove any existing overlay safely
                        const existingOverlay = document.getElementById('ml-detection-overlay');
                        if (existingOverlay) {
                          try {
                            if (existingOverlay.parentNode && document.body.contains(existingOverlay)) {
                              existingOverlay.parentNode.removeChild(existingOverlay);
                            }
                          } catch (err) {
                            console.log("Error removing existing overlay:", err);
                          }
                        }
                        
                        // Create a new overlay with enhanced styling
                        const newOverlay = document.createElement('div');
                        newOverlay.id = 'ml-detection-overlay';
                        newOverlay.style.position = 'absolute';
                        newOverlay.style.top = '0';
                        newOverlay.style.left = '0';
                        newOverlay.style.width = '100%';
                        newOverlay.style.height = '100%';
                        newOverlay.style.pointerEvents = 'none';
                        newOverlay.style.zIndex = '50';
                        newOverlay.style.transition = 'opacity 0.3s ease, background 0.3s ease';
                        
                        // CRITICAL: Always ensure display properties are set before appending
                        newOverlay.style.display = 'block'; 
                        newOverlay.style.opacity = '1';
                        newOverlay.style.visibility = 'visible';
                        
                        // Read stored opacity or use default
                        let opacity = 0.14; // Default
                        try {
                          const savedOpacity = localStorage.getItem('ml-overlay-opacity');
                          if (savedOpacity !== null) {
                            opacity = parseFloat(savedOpacity);
                          } else {
                            // Save default if no value exists
                            localStorage.setItem('ml-overlay-opacity', opacity.toString());
                          }
                        } catch (e) {
                          console.error("Error reading opacity setting:", e);
                        }
                        
                        // Apply gradient style
                        newOverlay.style.background = `linear-gradient(135deg, 
                          rgba(0, 22, 40, ${opacity}) 0%, 
                          rgba(0, 44, 80, ${parseFloat(opacity) * 0.9}) 50%, 
                          rgba(10, 53, 114, ${parseFloat(opacity) * 0.7}) 100%)`;
                          
                        // Add subtle grid pattern for depth perception
                        newOverlay.style.backgroundImage = `
                          linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255, 255, 255, 0.04) 1px, transparent 1px)
                        `;
                        newOverlay.style.backgroundSize = '20px 20px';
                        
                        // Add to the map container
                        mapContainer.appendChild(newOverlay);
                        
                        console.log("ML overlay created/updated successfully");
                        
                        // Now create the detection badge right after creating the overlay
                        // First check if badge exists and remove if needed
                        const existingBadge = document.getElementById('detection-debug');
                        if (existingBadge) {
                          try {
                            if (existingBadge.parentNode && document.body.contains(existingBadge)) {
                              existingBadge.parentNode.removeChild(existingBadge);
                            }
                          } catch (err) {
                            console.log("Error removing existing badge:", err);
                          }
                        }
                        
                        // Create new badge
                        const badge = document.createElement('div');
                        badge.id = 'detection-debug';
                        badge.textContent = 'OBJECT DETECTION';
                        badge.style.display = 'block';
                        badge.style.position = 'absolute';
                        badge.style.top = '0';
                        badge.style.right = '384px'; // Match the sidebar width for proper positioning
                        badge.style.background = 'rgba(13, 71, 161, 0.85)';
                        badge.style.zIndex = '2000'; // Increased z-index for better visibility
                        badge.style.padding = '5px 12px';
                        badge.style.fontSize = '12px';
                        badge.style.color = 'white';
                        badge.style.fontWeight = '500';
                        badge.style.borderBottomLeftRadius = '3px';
                        badge.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                        badge.style.letterSpacing = '0.5px';
                        
                        // Set initial style to invisible for smooth appearance
                        badge.style.opacity = '0';
                        badge.style.transition = 'opacity 0.3s ease-in';
                        
                        // Try to append badge safely but delay its visibility
                        try {
                          mapContainer.appendChild(badge);
                          console.log("Detection badge created successfully (initially hidden)");
                          
                          // Show the badge after a delay to allow sidebar and map to load
                          setTimeout(() => {
                            if (document.body.contains(badge)) {
                              badge.style.opacity = '1';
                              console.log("Detection badge made visible after delay");
                            }
                          }, 1000); // 1 second delay
                        } catch (err) {
                          console.error("Error adding detection badge:", err);
                        }
                      } catch (e) {
                        console.error("Error creating ML overlay:", e);
                      }
                    };
                    
                    // Execute immediately for the overlay
                    createOverlayAndBadge();
                    
                    // Also run again on a short delay to ensure it appears even if there's a race condition
                    // This will create a second badge with its own 1-second fade-in
                    setTimeout(createOverlayAndBadge, 100);
                    
                    // Continue with Detection functionality after ensuring Map view is active
                    
                    // Check if detection sidebar is already visible
                    const existingDetectionSidebar = document.querySelector('.detection-sidebar');
                    const isDetectionActive = existingDetectionSidebar && 
                      window.getComputedStyle(existingDetectionSidebar).display !== 'none' &&
                      window.getComputedStyle(existingDetectionSidebar).opacity !== '0';
                    
                    // If detection is already active, just ensure overlay and badge are visible
                    if (isDetectionActive) {
                      console.log("Detection sidebar already active, ensuring overlay visible");
                      
                      // Make sure overlay is visible
                      const overlay = document.getElementById('ml-detection-overlay');
                      if (!overlay) {
                        // Create it if it doesn't exist
                        const mapContainer = document.getElementById('map-container');
                        if (mapContainer) {
                          const newOverlay = document.createElement('div');
                          newOverlay.id = 'ml-detection-overlay';
                          newOverlay.style.position = 'absolute';
                          newOverlay.style.top = '0';
                          newOverlay.style.left = '0';
                          newOverlay.style.width = '100%';
                          newOverlay.style.height = '100%';
                          
                          // Default to 10% opacity and use stored value if available
                          let opacity = 0.1; // Default to 10%
                          try {
                            const savedOpacity = localStorage.getItem('ml-overlay-opacity');
                            if (savedOpacity !== null) {
                              opacity = parseFloat(savedOpacity);
                            } else {
                              // Save default if no value exists
                              localStorage.setItem('ml-overlay-opacity', opacity.toString());
                            }
                          } catch (e) {
                            console.error("Error reading opacity setting:", e);
                          }
                          
                          // Use gradient for consistent appearance with the slider updates
                          newOverlay.style.background = `linear-gradient(135deg, 
                            rgba(0, 22, 40, ${opacity}) 0%, 
                            rgba(0, 44, 80, ${parseFloat(opacity) * 0.9}) 50%, 
                            rgba(10, 53, 114, ${parseFloat(opacity) * 0.7}) 100%)`;
                          newOverlay.style.pointerEvents = 'none';
                          newOverlay.style.zIndex = '50';
                          newOverlay.style.transition = 'opacity 0.3s ease, background 0.3s ease';
                          mapContainer.appendChild(newOverlay);
                          
                          console.log(`Created new ML overlay with opacity ${opacity}`);
                          
                          // Removed layer indicator graphic
                        }
                      } else {
                        // Ensure overlay is visible with correct styling
                        overlay.style.display = 'block';
                        overlay.style.opacity = '1';
                        
                        // Update to current stored opacity if available
                        try {
                          const savedOpacity = localStorage.getItem('ml-overlay-opacity');
                          if (savedOpacity !== null) {
                            const opacity = parseFloat(savedOpacity);
                            // Use gradient for consistent appearance with the slider updates
                            overlay.style.background = `linear-gradient(135deg, 
                              rgba(0, 22, 40, ${opacity}) 0%, 
                              rgba(0, 44, 80, ${parseFloat(opacity) * 0.9}) 50%, 
                              rgba(10, 53, 114, ${parseFloat(opacity) * 0.7}) 100%)`;
                            console.log(`Updated existing overlay to stored opacity: ${opacity}`);
                          }
                        } catch (e) {
                          console.error("Error applying stored opacity:", e);
                        }
                      }
                      
                      // This section has been moved earlier to ensure the badge is always created/shown when the Components button is clicked
                      
                      // Make sure sidebar content is showing
                      const detectionContent = document.querySelector('.detection-sidebar');
                      if (detectionContent) {
                        // Force show sidebar if somehow hidden
                        detectionContent.style.display = 'block';
                        detectionContent.style.opacity = '1';
                        detectionContent.style.transform = 'translateX(0)';
                        detectionContent.style.visibility = 'visible';
                        
                        // Check if counter exists, update it if it does
                        const counterElement = document.getElementById('detected-objects-count');
                        if (counterElement) {
                          // Update with a random count for demonstration purposes
                          const randomCount = Math.floor(Math.random() * 30) + 5;
                          counterElement.textContent = randomCount.toString();
                        }
                      }
                      
                      return; // Exit early since we've reactivated the existing interface
                    }
                    
                    // Only continue if we need to create the Detection interface
                    console.log("Detection button: Current header state =", isHeaderCollapsed);
                    
                    // Notify map that right panel will be opening for detection mode
                    window.dispatchEvent(new CustomEvent('rightPanelToggle', {
                      detail: {
                        isOpen: true,
                        panelWidth: 384,
                        panelType: 'detection',
                        source: 'sidebar'
                      }
                    }));
                    
                    // Get the map container for other operations (overlay already created above)
                    const mapContainer = document.getElementById('map-container');
                    
                    // Simple approach: Just make sure the map container is adjusted
                    // for the detection sidebar and trigger the original detection mode
                    if (mapContainer) {
                      mapContainer.style.right = '384px';
                    }
                    
                    // Remove any existing object recognition badge to avoid duplicates
                    const oldBadge = document.getElementById('object-recognition-badge');
                    if (oldBadge) {
                      try {
                        if (oldBadge.parentNode && document.body.contains(oldBadge)) {
                          oldBadge.parentNode.removeChild(oldBadge);
                        }
                      } catch (err) {
                        console.log("Error removing existing badge:", err);
                      }
                    }
                    
                    // Make sure we have no database badges visible
                    const databaseBadges = document.querySelectorAll('.database-badge, [id*="database-badge"]');
                    databaseBadges.forEach(badge => {
                      try {
                        if (badge && badge.parentNode && document.body.contains(badge)) {
                          badge.parentNode.removeChild(badge);
                        }
                      } catch (err) {
                        console.log("Error removing database badge:", err);
                      }
                    });
                    
                    // Don't trigger the React component events to avoid conflicts
                    console.log("Creating standalone Object Recognition sidebar");
                    
                    // Skip triggering the enterTreeValidationMode and requestTreeDetection events
                    // since they're causing conflicts with React components
                    
                    // Create the new Object Recognition sidebar with light-blue theme
                    setTimeout(() => {
                      // Remove any existing sidebar with the same class to avoid duplicates
                      const existingSidebar = document.querySelector('.detection-sidebar');
                      if (existingSidebar) {
                        try {
                          if (existingSidebar.parentNode && document.body.contains(existingSidebar)) {
                            existingSidebar.parentNode.removeChild(existingSidebar);
                          }
                        } catch (err) {
                          console.log("Error removing existing sidebar:", err);
                          // If removal fails, try to hide it
                          try {
                            existingSidebar.style.display = 'none';
                            existingSidebar.style.visibility = 'hidden';
                          } catch (e) {}
                        }
                      }
                      
                      // Create the sidebar with refined Zen-like styling and improved accessibility
                      const sidebarElement = document.createElement('div');
                      sidebarElement.className = 'detection-sidebar';
                      sidebarElement.setAttribute('role', 'complementary');
                      sidebarElement.setAttribute('aria-label', 'Object Detection Panel');
                      sidebarElement.style.position = 'fixed';
                      sidebarElement.style.top = isHeaderCollapsed ? '40px' : '64px';
                      sidebarElement.style.right = '0';
                      sidebarElement.style.width = '384px'; // Standard width to match other sidebars
                      sidebarElement.style.bottom = '0';
                      
                      // More subtle, unified color scheme with lighter background
                      sidebarElement.style.backgroundColor = '#fafbfe'; // Even lighter, subtle blue-white
                      sidebarElement.style.color = '#1e293b'; // Darker base text color
                      sidebarElement.style.zIndex = '100';
                      sidebarElement.style.overflow = 'auto';
                      
                      // Refined shadow with subtle inner light
                      sidebarElement.style.boxShadow = '-1px 0 15px rgba(0,0,0,0.08), inset 1px 0 0 rgba(255,255,255,0.6)';
                      
                      // Light border for better definition
                      sidebarElement.style.borderLeft = '1px solid rgba(226, 232, 240, 0.8)';
                      
                      // Add unique ID for debugging and targeting
                      sidebarElement.setAttribute('data-id', 'detection-sidebar-' + new Date().getTime());
                      
                      // Add smooth entrance animation
                      sidebarElement.style.transform = 'translateX(360px)';
                      sidebarElement.style.transition = 'transform 0.3s ease-out';
                      
                      // Force hardware acceleration for smoother animations
                      sidebarElement.style.willChange = 'transform';
                      sidebarElement.style.backfaceVisibility = 'hidden';
                      
                      // Animate entrance after a brief delay
                      setTimeout(() => {
                        sidebarElement.style.transform = 'translateX(0)';
                      }, 50);
                      
                      // Add a close button at the top-right
                      const closeButtonContainer = document.createElement('div');
                      closeButtonContainer.style.position = 'absolute';
                      closeButtonContainer.style.top = '10px';
                      closeButtonContainer.style.right = '10px';
                      closeButtonContainer.style.zIndex = '101';
                      
                      const closeButton = document.createElement('button');
                      closeButton.innerHTML = '&times;'; // × symbol
                      closeButton.style.width = '24px';
                      closeButton.style.height = '24px';
                      closeButton.style.borderRadius = '50%';
                      closeButton.style.background = 'rgba(13, 71, 161, 0.1)';
                      closeButton.style.color = '#0d47a1';
                      closeButton.style.border = 'none';
                      closeButton.style.fontSize = '16px';
                      closeButton.style.display = 'flex';
                      closeButton.style.alignItems = 'center';
                      closeButton.style.justifyContent = 'center';
                      closeButton.style.cursor = 'pointer';
                      closeButton.style.transition = 'all 0.2s ease';
                      
                      // Add hover effects
                      closeButton.onmouseover = () => {
                        closeButton.style.background = 'rgba(13, 71, 161, 0.2)';
                      };
                      closeButton.onmouseout = () => {
                        closeButton.style.background = 'rgba(13, 71, 161, 0.1)';
                      };
                      
                      // Add click handler to close detection mode
                      closeButton.onclick = () => {
                        window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                          detail: { 
                            source: 'close_button',
                            clearMarkers: true,
                            forceRemove: true
                          }
                        }));
                      };
                      
                      closeButtonContainer.appendChild(closeButton);
                      sidebarElement.appendChild(closeButtonContainer);
                      
                      // Create a content container
                      const content = document.createElement('div');
                      content.style.padding = '12px';
                      
                      // Add header with title
                      const headerSection = document.createElement('div');
                      headerSection.style.padding = '12px 12px 10px';
                      headerSection.style.borderBottom = '1px solid rgba(226, 232, 240, 0.5)';
                      headerSection.style.marginBottom = '12px';
                      headerSection.style.display = 'flex';
                      headerSection.style.alignItems = 'center';
                      headerSection.style.justifyContent = 'space-between';
                      
                      const headerTitle = document.createElement('h2');
                      headerTitle.textContent = 'Component Detection';
                      headerTitle.style.margin = '0';
                      headerTitle.style.fontSize = '16px';
                      headerTitle.style.fontWeight = '500';
                      headerTitle.style.color = '#1e293b';
                      headerTitle.style.letterSpacing = '0.01em';
                      
                      // Add AI/ML icon for visual appeal
                      const headerIcon = document.createElement('div');
                      headerIcon.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v6"></path><path d="M12 22v-6"></path><path d="M4.93 4.93l4.24 4.24"></path><path d="M14.83 14.83l4.24 4.24"></path><path d="M2 12h6"></path><path d="M22 12h-6"></path><path d="M4.93 19.07l4.24-4.24"></path><path d="M14.83 9.17l4.24-4.24"></path></svg>`;
                      headerIcon.style.color = '#475569';
                      headerIcon.style.marginRight = '8px';
                      
                      const headerTitleWrapper = document.createElement('div');
                      headerTitleWrapper.style.display = 'flex';
                      headerTitleWrapper.style.alignItems = 'center';
                      headerTitleWrapper.appendChild(headerIcon);
                      headerTitleWrapper.appendChild(headerTitle);
                      
                      headerSection.appendChild(headerTitleWrapper);
                      content.appendChild(headerSection);
                      
                      // Object Counter Section with refined styling
                      const counterSection = document.createElement('div');
                      counterSection.style.margin = '0 12px 16px';
                      counterSection.style.borderRadius = '3px';
                      counterSection.style.padding = '8px 12px';
                      counterSection.style.backgroundColor = 'rgba(241, 245, 249, 0.6)';
                      counterSection.style.border = '1px solid rgba(226, 232, 240, 0.7)';
                      counterSection.style.display = 'flex';
                      counterSection.style.alignItems = 'center';
                      counterSection.style.justifyContent = 'space-between';
                      counterSection.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.03)';
                      
                      // Left side - label with icon
                      const counterLabel = document.createElement('div');
                      counterLabel.style.display = 'flex';
                      counterLabel.style.alignItems = 'center';
                      
                      // Add ML detection icon
                      const detectionIcon = document.createElement('span');
                      detectionIcon.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2" ry="2"></rect><line x1="9" y1="2" x2="9" y2="22"></line><line x1="15" y1="2" x2="15" y2="22"></line><line x1="2" y1="9" x2="22" y2="9"></line><line x1="2" y1="15" x2="22" y2="15"></line></svg>`;
                      detectionIcon.style.color = '#475569';
                      detectionIcon.style.marginRight = '8px';
                      
                      const labelText = document.createElement('span');
                      labelText.textContent = 'Detected Components';
                      labelText.style.fontWeight = '500';
                      labelText.style.fontSize = '13px';
                      labelText.style.color = '#475569';
                      
                      counterLabel.appendChild(detectionIcon);
                      counterLabel.appendChild(labelText);
                      
                      // Right side - count with refined styling
                      const counterValue = document.createElement('div');
                      counterValue.id = 'detected-objects-count';
                      counterValue.textContent = '0';
                      counterValue.style.fontWeight = '600';
                      counterValue.style.fontSize = '16px';
                      counterValue.style.color = '#475569';
                      counterValue.style.padding = '5px 10px';
                      counterValue.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
                      counterValue.style.borderRadius = '6px';
                      counterValue.style.minWidth = '36px';
                      counterValue.style.textAlign = 'center';
                      counterValue.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.06)';
                      counterValue.style.border = '1px solid rgba(226, 232, 240, 0.6)';
                      
                      // Add counter parts to the section
                      counterSection.appendChild(counterLabel);
                      counterSection.appendChild(counterValue);
                      
                      // Add counter section to content
                      content.appendChild(counterSection);
                      
                      // Settings section with refined styling for opacity control
                      const settingsSection = document.createElement('div');
                      settingsSection.style.margin = '0 12px 16px';
                      settingsSection.style.borderRadius = '3px';
                      settingsSection.style.padding = '10px 12px';
                      settingsSection.style.backgroundColor = '#e7f1fd'; // Light blue tint
                      settingsSection.style.border = '1px solid rgba(186, 213, 241, 0.7)';
                      settingsSection.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.03)';
                      
                      // Settings section header with icon
                      const settingsHeaderWrapper = document.createElement('div');
                      settingsHeaderWrapper.style.display = 'flex';
                      settingsHeaderWrapper.style.alignItems = 'center';
                      settingsHeaderWrapper.style.marginBottom = '12px';
                      
                      // Settings icon
                      const settingsIcon = document.createElement('span');
                      settingsIcon.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v18"></path><rect x="4" y="4" width="16" height="16" rx="2"></rect><path d="M9 10a2 2 0 0 1-2-2V3h10v5a2 2 0 0 1-2 2H9z"></path></svg>`;
                      settingsIcon.style.color = '#475569';
                      settingsIcon.style.marginRight = '8px';
                      
                      // Settings title
                      const settingsTitle = document.createElement('div');
                      settingsTitle.textContent = 'ML Overlay Settings';
                      settingsTitle.style.fontWeight = '500';
                      settingsTitle.style.fontSize = '13px';
                      settingsTitle.style.color = '#475569';
                      
                      // Build header
                      settingsHeaderWrapper.appendChild(settingsIcon);
                      settingsHeaderWrapper.appendChild(settingsTitle);
                      settingsSection.appendChild(settingsHeaderWrapper);
                      
                      // Opacity slider control
                      const sliderContainer = document.createElement('div');
                      sliderContainer.style.marginBottom = '8px';
                      
                      // Slider label with info icon
                      const sliderLabelWrapper = document.createElement('div');
                      sliderLabelWrapper.style.display = 'flex';
                      sliderLabelWrapper.style.alignItems = 'center';
                      sliderLabelWrapper.style.justifyContent = 'space-between';
                      sliderLabelWrapper.style.marginBottom = '8px';
                      
                      const sliderLabelLeft = document.createElement('div');
                      sliderLabelLeft.style.display = 'flex';
                      sliderLabelLeft.style.alignItems = 'center';
                      
                      const sliderLabel = document.createElement('span');
                      sliderLabel.textContent = 'Overlay Density';
                      sliderLabel.style.fontSize = '12px';
                      sliderLabel.style.fontWeight = '500';
                      sliderLabel.style.color = '#475569';
                      
                      const infoIcon = document.createElement('span');
                      infoIcon.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>`;
                      infoIcon.style.color = '#94a3b8';
                      infoIcon.style.marginLeft = '6px';
                      infoIcon.style.cursor = 'help';
                      infoIcon.title = 'Adjust the intensity of the ML detection overlay layer';
                      
                      sliderLabelLeft.appendChild(sliderLabel);
                      sliderLabelLeft.appendChild(infoIcon);
                      
                      sliderLabelWrapper.appendChild(sliderLabelLeft);
                      sliderContainer.appendChild(sliderLabelWrapper);
                      
                      // Slider and value display wrapper with refined styling
                      const sliderWrap = document.createElement('div');
                      sliderWrap.style.display = 'flex';
                      sliderWrap.style.alignItems = 'center';
                      sliderWrap.style.gap = '12px';
                      sliderWrap.style.position = 'relative';
                      
                      // Create enhanced slider 
                      const sliderContainer2 = document.createElement('div');
                      sliderContainer2.style.position = 'relative';
                      sliderContainer2.style.flex = '1';
                      sliderContainer2.style.height = '28px';
                      sliderContainer2.style.display = 'flex';
                      sliderContainer2.style.alignItems = 'center';
                      
                      // Track background
                      const sliderTrack = document.createElement('div');
                      sliderTrack.style.position = 'absolute';
                      sliderTrack.style.width = '100%';
                      sliderTrack.style.height = '4px';
                      sliderTrack.style.backgroundColor = '#e2e8f0';
                      sliderTrack.style.borderRadius = '2px';
                      
                      // The actual slider input
                      const slider = document.createElement('input');
                      slider.type = 'range';
                      slider.min = '0';
                      slider.max = '100';
                      slider.style.position = 'relative';
                      slider.style.width = '100%';
                      slider.style.zIndex = '2';
                      slider.style.appearance = 'none';
                      slider.style.backgroundColor = 'transparent';
                      slider.style.outline = 'none';
                      slider.style.cursor = 'pointer';
                      
                      // Custom slider styling
                      slider.style.accentColor = '#1e40af';
                      
                      // Add track and slider to container
                      sliderContainer2.appendChild(sliderTrack);
                      sliderContainer2.appendChild(slider);
                      sliderWrap.appendChild(sliderContainer2);
                      
                      // Value display with improved styling
                      const valueDisplay = document.createElement('div');
                      valueDisplay.style.fontSize = '12px';
                      valueDisplay.style.fontWeight = '500';
                      valueDisplay.style.color = '#475569';
                      valueDisplay.style.width = '40px';
                      valueDisplay.style.textAlign = 'center';
                      valueDisplay.style.padding = '4px 6px';
                      valueDisplay.style.borderRadius = '4px';
                      valueDisplay.style.backgroundColor = 'white';
                      valueDisplay.style.border = '1px solid #e2e8f0';
                      
                      // Update opacity when slider changes
                      slider.oninput = () => {
                        const opacity = slider.value / 100;
                        valueDisplay.textContent = slider.value + '%';
                        
                        // Update the ML overlay if it exists
                        const overlay = document.getElementById('ml-detection-overlay');
                        if (overlay) {
                          // Update the gradient with the new opacity
                          overlay.style.background = `linear-gradient(135deg, 
                            rgba(0, 22, 40, ${opacity}) 0%, 
                            rgba(0, 44, 80, ${parseFloat(opacity) * 0.9}) 50%, 
                            rgba(10, 53, 114, ${parseFloat(opacity) * 0.7}) 100%)`;
                            
                          console.log(`Updated overlay opacity to ${opacity}`);
                        }
                        
                        // Dispatch event for React component to update
                        window.dispatchEvent(new CustomEvent('mlOverlayOpacityChange', {
                          detail: { opacity }
                        }));
                        
                        // Store in localStorage for persistence
                        try {
                          localStorage.setItem('ml-overlay-opacity', opacity);
                          console.log(`Saved opacity value: ${opacity}`);
                        } catch (e) {
                          console.error("Error saving opacity setting:", e);
                        }
                      };
                      
                      // Set default opacity to 10% and try to use saved value if available
                      let defaultOpacity = 10; // 10%
                      try {
                        const savedOpacity = localStorage.getItem('ml-overlay-opacity');
                        if (savedOpacity !== null) {
                          const opacityValue = Math.round(parseFloat(savedOpacity) * 100);
                          slider.value = opacityValue;
                          valueDisplay.textContent = opacityValue + '%';
                        } else {
                          // If no saved value, use default 10%
                          slider.value = defaultOpacity;
                          valueDisplay.textContent = defaultOpacity + '%';
                          localStorage.setItem('ml-overlay-opacity', (defaultOpacity/100).toString());
                        }
                        
                        // Apply immediately to any existing overlay
                        const overlay = document.getElementById('ml-detection-overlay');
                        if (overlay) {
                          // Use the value from the slider (either from storage or default)
                          const opacity = parseInt(slider.value) / 100;
                          
                          // Apply gradient with proper opacity
                          overlay.style.background = `linear-gradient(135deg, 
                            rgba(0, 22, 40, ${opacity}) 0%, 
                            rgba(0, 44, 80, ${parseFloat(opacity) * 0.9}) 50%, 
                            rgba(10, 53, 114, ${parseFloat(opacity) * 0.7}) 100%)`;
                            
                          console.log(`Applied initial opacity: ${opacity}`);
                        }
                      } catch (e) {
                        console.error("Error loading opacity setting:", e);
                      }
                      
                      sliderWrap.appendChild(slider);
                      sliderWrap.appendChild(valueDisplay);
                      sliderContainer.appendChild(sliderWrap);
                      settingsSection.appendChild(sliderContainer);
                      
                      content.appendChild(settingsSection);
                      
                      // Manual Marker Placement Section with Zen-like styling
                      const markerSection = document.createElement('div');
                      markerSection.style.margin = '0 12px 16px';
                      markerSection.style.borderRadius = '3px';
                      markerSection.style.padding = '10px 12px';
                      markerSection.style.backgroundColor = '#ecfdf5'; // Light green tint
                      markerSection.style.border = '1px solid rgba(167, 243, 208, 0.5)';
                      markerSection.style.boxShadow = '0 1px 2px rgba(0, 0, 0, 0.03)';
                      
                      // Section title with icon
                      const markerHeaderWrapper = document.createElement('div');
                      markerHeaderWrapper.style.display = 'flex';
                      markerHeaderWrapper.style.alignItems = 'center';
                      markerHeaderWrapper.style.marginBottom = '12px';
                      
                      // Marker icon
                      const markerIcon = document.createElement('span');
                      markerIcon.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>`;
                      markerIcon.style.color = '#475569';
                      markerIcon.style.marginRight = '8px';
                      
                      // Marker title
                      const markerTitle = document.createElement('div');
                      markerTitle.textContent = 'Custom Marker Placement';
                      markerTitle.style.fontWeight = '500';
                      markerTitle.style.fontSize = '13px';
                      markerTitle.style.color = '#475569';
                      
                      // Build header
                      markerHeaderWrapper.appendChild(markerIcon);
                      markerHeaderWrapper.appendChild(markerTitle);
                      markerSection.appendChild(markerHeaderWrapper);
                      
                      // Marker controls container
                      const markerControls = document.createElement('div');
                      markerControls.style.display = 'flex';
                      markerControls.style.flexDirection = 'column';
                      markerControls.style.gap = '8px';
                      
                      // Toggle button for manual placement mode
                      const markerToggleBtn = document.createElement('button');
                      markerToggleBtn.textContent = 'Enable Manual Placement';
                      markerToggleBtn.style.backgroundColor = '#eff6ff';
                      markerToggleBtn.style.color = '#0d47a1';
                      markerToggleBtn.style.border = '1px solid rgba(13, 71, 161, 0.15)';
                      markerToggleBtn.style.borderRadius = '4px';
                      markerToggleBtn.style.padding = '6px 10px';
                      markerToggleBtn.style.fontSize = '12px';
                      markerToggleBtn.style.fontWeight = '500';
                      markerToggleBtn.style.cursor = 'pointer';
                      markerToggleBtn.style.textAlign = 'center';
                      markerToggleBtn.style.transition = 'all 0.2s ease';
                      
                      // Toggle state
                      let manualPlacementActive = false;
                      markerToggleBtn.onclick = () => {
                        manualPlacementActive = !manualPlacementActive;
                        if (manualPlacementActive) {
                          markerToggleBtn.textContent = 'Disable Manual Placement';
                          markerToggleBtn.style.backgroundColor = '#dbeafe';
                          markerToggleBtn.style.fontWeight = '600';
                          // Set global flag for other components to detect
                          window.manualTreePlacementActive = true;
                          // Dispatch event for other components
                          window.dispatchEvent(new CustomEvent('manualMarkerModeChange', {
                            detail: { active: true }
                          }));
                          // Show instructions
                          markerInstructions.style.display = 'block';
                        } else {
                          markerToggleBtn.textContent = 'Enable Manual Placement';
                          markerToggleBtn.style.backgroundColor = '#eff6ff';
                          markerToggleBtn.style.fontWeight = '500';
                          // Reset global flag
                          window.manualTreePlacementActive = false;
                          // Dispatch event for other components
                          window.dispatchEvent(new CustomEvent('manualMarkerModeChange', {
                            detail: { active: false }
                          }));
                          // Hide instructions
                          markerInstructions.style.display = 'none';
                        }
                      };
                      markerControls.appendChild(markerToggleBtn);
                      
                      // Marker adjustment controls
                      const markerAdjust = document.createElement('div');
                      markerAdjust.style.display = 'flex';
                      markerAdjust.style.gap = '8px';
                      
                      // Marker size control
                      const sizeControl = document.createElement('div');
                      sizeControl.style.flex = '1';
                      
                      const sizeLabel = document.createElement('div');
                      sizeLabel.textContent = 'Marker Size';
                      sizeLabel.style.fontSize = '11px';
                      sizeLabel.style.color = '#64748b';
                      sizeLabel.style.marginBottom = '4px';
                      sizeControl.appendChild(sizeLabel);
                      
                      const sizeOptions = document.createElement('div');
                      sizeOptions.style.display = 'flex';
                      sizeOptions.style.border = '1px solid rgba(13, 71, 161, 0.1)';
                      sizeOptions.style.borderRadius = '4px';
                      sizeOptions.style.overflow = 'hidden';
                      
                      const sizes = ['S', 'M', 'L'];
                      sizes.forEach((size, index) => {
                        const option = document.createElement('button');
                        option.textContent = size;
                        option.style.flex = '1';
                        option.style.backgroundColor = index === 1 ? '#dbeafe' : '#f8fafc';
                        option.style.color = index === 1 ? '#0d47a1' : '#64748b';
                        option.style.border = 'none';
                        option.style.padding = '4px 0';
                        option.style.fontSize = '11px';
                        option.style.fontWeight = index === 1 ? '600' : '400';
                        option.style.cursor = 'pointer';
                        option.onclick = () => {
                          // Update UI
                          sizeOptions.querySelectorAll('button').forEach((btn, idx) => {
                            btn.style.backgroundColor = idx === index ? '#dbeafe' : '#f8fafc';
                            btn.style.color = idx === index ? '#0d47a1' : '#64748b';
                            btn.style.fontWeight = idx === index ? '600' : '400';
                          });
                          
                          // Set size globally for marker placement
                          window.markerSize = ['small', 'medium', 'large'][index];
                          
                          // Dispatch event for other components
                          window.dispatchEvent(new CustomEvent('markerSizeChange', {
                            detail: { size: window.markerSize }
                          }));
                        };
                        sizeOptions.appendChild(option);
                      });
                      sizeControl.appendChild(sizeOptions);
                      markerAdjust.appendChild(sizeControl);
                      
                      // Marker color control
                      const colorControl = document.createElement('div');
                      colorControl.style.flex = '1';
                      
                      const colorLabel = document.createElement('div');
                      colorLabel.textContent = 'Marker Color';
                      colorLabel.style.fontSize = '11px';
                      colorLabel.style.color = '#64748b';
                      colorLabel.style.marginBottom = '4px';
                      colorControl.appendChild(colorLabel);
                      
                      const colorOptions = document.createElement('div');
                      colorOptions.style.display = 'flex';
                      colorOptions.style.border = '1px solid rgba(13, 71, 161, 0.1)';
                      colorOptions.style.borderRadius = '4px';
                      colorOptions.style.overflow = 'hidden';
                      
                      const colors = [
                        { name: 'Blue', value: '#2563eb' },
                        { name: 'Green', value: '#16a34a' },
                        { name: 'Red', value: '#dc2626' }
                      ];
                      
                      colors.forEach((color, index) => {
                        const option = document.createElement('button');
                        option.style.flex = '1';
                        option.style.backgroundColor = index === 0 ? '#dbeafe' : '#f8fafc';
                        option.style.color = index === 0 ? '#0d47a1' : '#64748b';
                        option.style.border = 'none';
                        option.style.padding = '4px 0';
                        option.style.fontSize = '11px';
                        option.style.fontWeight = index === 0 ? '600' : '400';
                        option.style.cursor = 'pointer';
                        
                        const colorDot = document.createElement('div');
                        colorDot.style.width = '8px';
                        colorDot.style.height = '8px';
                        colorDot.style.borderRadius = '50%';
                        colorDot.style.backgroundColor = color.value;
                        colorDot.style.display = 'inline-block';
                        colorDot.style.marginRight = '4px';
                        
                        option.appendChild(colorDot);
                        option.appendChild(document.createTextNode(color.name));
                        
                        option.onclick = () => {
                          // Update UI
                          colorOptions.querySelectorAll('button').forEach((btn, idx) => {
                            btn.style.backgroundColor = idx === index ? '#dbeafe' : '#f8fafc';
                            btn.style.color = idx === index ? '#0d47a1' : '#64748b';
                            btn.style.fontWeight = idx === index ? '600' : '400';
                          });
                          
                          // Set color globally for marker placement
                          window.markerColor = color.value;
                          
                          // Dispatch event for other components
                          window.dispatchEvent(new CustomEvent('markerColorChange', {
                            detail: { color: window.markerColor }
                          }));
                        };
                        colorOptions.appendChild(option);
                      });
                      
                      colorControl.appendChild(colorOptions);
                      markerAdjust.appendChild(colorControl);
                      markerControls.appendChild(markerAdjust);
                      
                      // Clear button for markers
                      const clearMarkersBtn = document.createElement('button');
                      clearMarkersBtn.textContent = 'Clear All Markers';
                      clearMarkersBtn.style.backgroundColor = '#fef2f2';
                      clearMarkersBtn.style.color = '#991b1b';
                      clearMarkersBtn.style.border = '1px solid #fecaca';
                      clearMarkersBtn.style.borderRadius = '2px';
                      clearMarkersBtn.style.padding = '6px 10px';
                      clearMarkersBtn.style.fontSize = '12px';
                      clearMarkersBtn.style.fontWeight = '500';
                      clearMarkersBtn.style.cursor = 'pointer';
                      clearMarkersBtn.style.marginTop = '8px';
                      
                      clearMarkersBtn.onclick = () => {
                        // Clear markers from map
                        if (window.manualTreeMarkers && window.manualTreeMarkers.length > 0) {
                          window.manualTreeMarkers.forEach(marker => {
                            if (marker) marker.setMap(null);
                          });
                          window.manualTreeMarkers = [];
                          
                          // Dispatch event for other components
                          window.dispatchEvent(new CustomEvent('markersCleared', {
                            detail: { source: 'sidebar' }
                          }));
                          
                          // Update UI feedback
                          clearMarkersBtn.textContent = 'Markers Cleared';
                          clearMarkersBtn.style.backgroundColor = '#f1f5f9';
                          clearMarkersBtn.style.color = '#64748b';
                          
                          // Reset button after timeout
                          setTimeout(() => {
                            clearMarkersBtn.textContent = 'Clear All Markers';
                            clearMarkersBtn.style.backgroundColor = '#fef2f2';
                            clearMarkersBtn.style.color = '#991b1b';
                            clearMarkersBtn.style.border = '1px solid #fecaca';
                          }, 2000);
                        }
                      };
                      markerControls.appendChild(clearMarkersBtn);
                      
                      // Instructions for marker placement
                      const markerInstructions = document.createElement('div');
                      markerInstructions.style.fontSize = '11px';
                      markerInstructions.style.color = '#64748b';
                      markerInstructions.style.padding = '8px';
                      markerInstructions.style.backgroundColor = 'rgba(15, 23, 42, 0.05)';
                      markerInstructions.style.borderRadius = '2px';
                      markerInstructions.style.border = '1px solid rgba(226, 232, 240, 0.8)';
                      markerInstructions.style.marginTop = '8px';
                      markerInstructions.style.display = 'none'; // Hidden by default
                      markerInstructions.innerHTML = 'Click on the map to place markers<br>• Right-click a marker to remove it<br>• Hold Shift and click to move a marker';
                      markerControls.appendChild(markerInstructions);
                      
                      markerSection.appendChild(markerControls);
                      content.appendChild(markerSection);
                      
                      // Add button container for side-by-side buttons
                      const buttonContainer = document.createElement('div');
                      buttonContainer.style.display = 'grid';
                      buttonContainer.style.gridTemplateColumns = '1fr 1fr';
                      buttonContainer.style.gap = '8px';
                      buttonContainer.style.marginTop = '14px';
                      
                      // ML Detection button (simplified)
                      const startBtn = document.createElement('button');
                      startBtn.textContent = 'ML Detection';
                      
                      // Style the button to match Components button
                      startBtn.style.display = 'block';
                      startBtn.style.backgroundColor = '#0a3572'; // Match Components button
                      startBtn.style.color = 'white';
                      startBtn.style.fontWeight = '500';
                      startBtn.style.padding = '8px 12px';
                      startBtn.style.borderRadius = '4px';
                      startBtn.style.border = '1px solid #072448';
                      startBtn.style.cursor = 'pointer';
                      startBtn.style.boxShadow = '0 1px 2px rgba(0,0,0,0.1)';
                      startBtn.style.fontSize = '13px';
                      
                      // Parameters button with inverted colors
                      const editBtn = document.createElement('button');
                      editBtn.textContent = 'Parameters';
                      
                      // Style with inverted colors (dark blue font and border)
                      editBtn.style.display = 'block';
                      editBtn.style.backgroundColor = 'white';
                      editBtn.style.color = '#0a3572'; // Dark blue text
                      editBtn.style.fontWeight = '500';
                      editBtn.style.padding = '8px 12px';
                      editBtn.style.borderRadius = '4px';
                      editBtn.style.border = '1px solid #0a3572'; // Dark blue border
                      editBtn.style.cursor = 'pointer';
                      editBtn.style.boxShadow = '0 1px 2px rgba(0,0,0,0.05)';
                      editBtn.style.fontSize = '13px';
                      
                      // Add click handler for Parameters button
                      editBtn.onclick = () => {
                        // Show simple alert for now since we're not implementing full feature
                        alert('ML parameter configuration will be available in the next release.');
                      };
                      
                      // Add buttons to container
                      buttonContainer.appendChild(startBtn);
                      buttonContainer.appendChild(editBtn);
                      
                      // Simple hover states for ML Detection button
                      startBtn.onmouseover = () => {
                        startBtn.style.backgroundColor = '#082a5b'; // Slightly darker
                      };
                      startBtn.onmouseout = () => {
                        startBtn.style.backgroundColor = '#0a3572';
                      };
                      startBtn.onmousedown = () => {
                        startBtn.style.backgroundColor = '#072448'; // Even darker when clicked
                      };
                      startBtn.onmouseup = () => {
                        startBtn.style.backgroundColor = '#082a5b';
                      };
                      
                      // Simple hover states for Edit Components button
                      editBtn.onmouseover = () => {
                        editBtn.style.backgroundColor = '#f8fafc';
                      };
                      editBtn.onmouseout = () => {
                        editBtn.style.backgroundColor = 'white';
                      };
                      editBtn.onmousedown = () => {
                        editBtn.style.backgroundColor = '#f1f5f9';
                      };
                      editBtn.onmouseup = () => {
                        editBtn.style.backgroundColor = '#f8fafc';
                      };
                      
                      // Click handler for starting detection
                      startBtn.onclick = () => {
                        // Generate a random number of detected objects for demo purposes
                        const randomCount = Math.floor(Math.random() * 30) + 5;
                        const counter = document.getElementById('detected-objects-count');
                        if (counter) {
                          counter.textContent = randomCount.toString();
                          
                          // Apply animation effect to counter
                          counter.style.transition = 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
                          counter.style.transform = 'scale(1.2)';
                          counter.style.backgroundColor = 'rgba(13, 71, 161, 0.15)';
                          
                          // Reset after animation
                          setTimeout(() => {
                            counter.style.transform = 'scale(1)';
                            counter.style.backgroundColor = 'rgba(13, 71, 161, 0.08)';
                          }, 300);
                        }
                        
                        // Show loading state on button
                        startBtn.disabled = true;
                        startBtn.textContent = 'Processing...';
                        startBtn.style.backgroundColor = '#64748b';
                        
                        // Reset button after delay
                        setTimeout(() => {
                          startBtn.disabled = false;
                          startBtn.textContent = 'ML Detection';
                          startBtn.style.backgroundColor = '#0a3572';
                        }, 1500);
                      };
                      
                      // Create a fixed position container for the start button
                      const fixedButtonContainer = document.createElement('div');
                      fixedButtonContainer.style.position = 'absolute';
                      fixedButtonContainer.style.bottom = '12px';
                      fixedButtonContainer.style.left = '12px';
                      fixedButtonContainer.style.right = '12px';
                      fixedButtonContainer.appendChild(buttonContainer);
                      
                      // Add the content and fixed button to sidebar
                      sidebarElement.appendChild(content);
                      sidebarElement.appendChild(fixedButtonContainer);
                      
                      // Add padding to the bottom of content to create space for fixed button
                      content.style.paddingBottom = '70px';
                      
                      // Add the sidebar to document
                      document.body.appendChild(sidebarElement);
                    }, 300);
                    
                    // Badge creation is now handled in the createOverlayAndBadge function above
                    
                    // Trigger resize to finalize layout
                    setTimeout(() => {
                      window.dispatchEvent(new Event('resize'));
                    }, 300);
                  });
                }}
                className="flex items-center justify-center p-2 bg-blue-50 text-blue-900 hover:bg-blue-100 rounded cursor-pointer border border-blue-300/20"
              >
                <Eye className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Components</span>}
              </button>
            </div>
            
            {/* Second row - Validate and Database buttons */}
            <div className="grid grid-cols-2 gap-2">
              {/* Validate Button (renamed from Validation) */}
              <button
                onClick={() => {
                  // First, close imagery panel if it's open
                  const closeImageryEvent = new CustomEvent('forceCloseImageryPanel', {
                    detail: { source: 'sidebar' }
                  });
                  window.dispatchEvent(closeImageryEvent);
                  
                  // Then exit any active validation mode
                  window.dispatchEvent(new CustomEvent('exitValidationMode', {
                    detail: { 
                      source: 'sidebar', 
                      target: 'review',
                      clearExisting: true
                    }
                  }));
                  
                  // Make sure Object Detection sidebar is closed
                  window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                    detail: { source: 'sidebar' }
                  }));
                  
                  // Notify map that right panel is closing
                  window.dispatchEvent(new CustomEvent('rightPanelToggle', {
                    detail: {
                      isOpen: false,
                      source: 'sidebar'
                    }
                  }));
                  
                  // Open review panel
                  window.dispatchEvent(new CustomEvent('openReviewPanel', {
                    detail: { 
                      source: 'sidebar'
                    }
                  }));
                }}
                className="flex items-center justify-center p-2 bg-orange-50 text-orange-800 hover:bg-orange-100 rounded cursor-pointer border border-orange-300/20"
              >
                <ClipboardList className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Validate</span>}
              </button>
              
              {/* Database Button (moved up next to Validate) */}
              <button
                onClick={() => {
                  console.log("Database button clicked - improved disconnect approach");
                  
                  // Check for existing database sidebar to prevent double-clicking issues
                  const dbSidebar = document.querySelector('.feature-selection-sidebar, .tree-inventory-sidebar');
                  if (dbSidebar) {
                    console.log("Database sidebar already open - ignoring click");
                    return;
                  }
                  
                  // STEP 1: Create a hidden storage div to move DOM elements to instead of removing them
                  // This avoids DOM manipulation errors entirely
                  let detachedStorage = document.getElementById('detached-element-storage');
                  if (!detachedStorage) {
                    detachedStorage = document.createElement('div');
                    detachedStorage.id = 'detached-element-storage';
                    detachedStorage.style.display = 'none';
                    detachedStorage.style.visibility = 'hidden';
                    detachedStorage.style.position = 'absolute';
                    detachedStorage.style.left = '-9999px';
                    detachedStorage.style.width = '0';
                    detachedStorage.style.height = '0';
                    detachedStorage.style.overflow = 'hidden';
                    detachedStorage.style.pointerEvents = 'none';
                    document.body.appendChild(detachedStorage);
                  }
                  
                  // STEP 2: Immediately force reset of all detection and validation state
                  window.isValidationMode = false;
                  window.isDetectionSidebarActive = false;
                  window.isDetectionModeActive = false;
                  window.manualTreePlacementActive = false;
                  
                  // STEP 3: Explicitly clean up ML Overlay first - direct approach
                  // First close object detection mode to clean up React components
                  window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                    detail: { 
                      source: 'database_button',
                      clearExisting: true,
                      clearMarkers: true
                    }
                  }));
                  
                  // Then remove any outstanding ML overlays directly
                  const removeMLOverlays = () => {
                    // Find all ML overlay elements by ID and class
                    const mlOverlays = document.querySelectorAll('#ml-detection-overlay, .ml-detection-overlay');
                    
                    console.log(`Found ${mlOverlays.length} ML overlay elements to remove`);
                    
                    // Remove each one individually to prevent errors
                    mlOverlays.forEach((overlay, index) => {
                      try {
                        if (overlay && document.body.contains(overlay) && overlay.parentNode) {
                          // First hide it with CSS to avoid visual glitches
                          overlay.style.display = 'none';
                          overlay.style.opacity = '0';
                          overlay.style.visibility = 'hidden';
                          overlay.style.pointerEvents = 'none';
                          
                          // Then remove it
                          setTimeout(() => {
                            try {
                              if (overlay.parentNode && document.body.contains(overlay)) {
                                overlay.parentNode.removeChild(overlay);
                                console.log(`Successfully removed ML overlay ${index+1}`);
                              }
                            } catch (e) {
                              console.error(`Error removing ML overlay ${index+1}:`, e);
                            }
                          }, 0);
                        }
                      } catch (e) {
                        console.error(`Error processing ML overlay ${index+1}:`, e);
                      }
                    });
                    
                    // Also check for detection badges
                    const badges = document.querySelectorAll('#detection-debug, .detection-debug');
                    console.log(`Found ${badges.length} detection badges to remove`);
                    
                    badges.forEach((badge, index) => {
                      try {
                        if (badge && document.body.contains(badge) && badge.parentNode) {
                          // Add fade-out transition if not already set
                          if (!badge.style.transition) {
                            badge.style.transition = 'opacity 0.3s ease-out';
                          }
                          
                          // First fade out
                          badge.style.opacity = '0';
                          
                          // Then remove after transition completes
                          setTimeout(() => {
                            try {
                              if (badge && document.body.contains(badge) && badge.parentNode) {
                                badge.style.display = 'none';
                                badge.parentNode.removeChild(badge);
                                console.log(`Successfully removed detection badge ${index+1}`);
                              }
                            } catch (e) {
                              console.error(`Error removing detection badge ${index+1} after fade:`, e);
                            }
                          }, 300);
                        }
                      } catch (e) {
                        console.error(`Error removing detection badge ${index+1}:`, e);
                      }
                    });
                  };
                  
                  // Run removal immediately
                  removeMLOverlays();
                  
                  // STEP 4: Send exitValidationMode event to React components
                  window.dispatchEvent(new CustomEvent('exitValidationMode', {
                    detail: { 
                      source: 'database_button',
                      target: 'database',
                      clearExisting: true,
                      forceRemove: true
                    }
                  }));
                  
                  // STEP 5: Prepare map container dimensions for database sidebar
                  const mapContainer = document.getElementById('map-container');
                  if (mapContainer) {
                    // Set right to database sidebar width (384px) to ensure proper sizing
                    mapContainer.style.right = '384px';
                    mapContainer.style.width = 'auto'; // Let it calculate based on left and right
                  }
                  
                  // Close all other panels
                  window.dispatchEvent(new CustomEvent('forceCloseImageryPanel', { detail: { source: 'database' } }));
                  window.dispatchEvent(new CustomEvent('closeAnalyticsPanel', { detail: { source: 'database' } }));
                  window.dispatchEvent(new CustomEvent('closeReviewPanel', { detail: { source: 'database' } }));
                  
                  // Force resize to ensure proper layout
                  window.dispatchEvent(new Event('resize'));
                  
                  // STEP 6: Get accurate header state
                  const headerElement = document.querySelector('header');
                  let isHeaderCollapsed = true;
                  
                  if (headerElement) {
                    const headerHeight = headerElement.getBoundingClientRect().height;
                    const hasCollapsedClass = headerElement.classList.contains('collapsed');
                    isHeaderCollapsed = hasCollapsedClass || headerHeight < 50;
                    console.log("Database button: Header state =", isHeaderCollapsed);
                  }
                  
                  // STEP 7: Initialize styling function for database sidebar
                  window.applyDatabaseSidebarStyling = () => {
                    setTimeout(() => {
                      try {
                        const databaseSidebar = document.querySelector('.feature-selection-sidebar, .tree-inventory-sidebar');
                        if (databaseSidebar) {
                          console.log("Applying green styling to database sidebar");
                          
                          // Style the header
                          const header = databaseSidebar.querySelector('div[style*="border-bottom"]');
                          if (header) {
                            header.style.backgroundColor = '#e6f4ea'; // Light green background
                            header.style.color = '#166534'; // Dark green text
                            header.style.borderBottom = '1px solid rgba(22, 101, 52, 0.1)';
                            
                            const headerTitle = header.querySelector('span');
                            if (headerTitle) {
                              headerTitle.style.color = '#166534'; // Dark green text
                            }
                          }
                          
                          // Style any buttons
                          databaseSidebar.querySelectorAll('button').forEach(btn => {
                            if (btn.className.includes('bg-blue')) {
                              btn.className = btn.className.replace('bg-blue', 'bg-emerald');
                            }
                            if (btn.style.backgroundColor === 'rgb(13, 71, 161)') {
                              btn.style.backgroundColor = '#166534'; // Dark green
                            }
                          });
                        }
                      } catch (e) {
                        console.error("Error applying database styling:", e);
                      }
                    }, 100);
                  };
                  
                  // STEP 8: Open database sidebar with a delay to ensure cleanup is complete
                  setTimeout(() => {
                    // Run ML overlay cleanup one more time to ensure no interference
                    removeMLOverlays();
                    
                    // Wait for the fade-out transition to complete before continuing
                    setTimeout(() => {
                    
                    // Notify the map that a right panel will open
                    window.dispatchEvent(new CustomEvent('rightPanelToggle', {
                      detail: {
                        isOpen: true,
                        panelWidth: 384, // Standard sidebar width
                        panelType: 'database'
                      }
                    }));
                
                    // Then open feature selection with DATABASE theme and explicit suppressOverlays setting
                    console.log("Opening feature selection with DATABASE theme");
                    window.dispatchEvent(new CustomEvent('openFeatureSelection', {
                      detail: { 
                        mode: 'tree_inventory', 
                        clearExisting: true,
                        tab: 'trees',
                        headerCollapsed: isHeaderCollapsed,
                        source: 'database_sidebar',
                        suppressOverlays: true,
                        suppressBadges: true,
                        styling: {
                          theme: 'green',
                          headerColor: '#e6f4ea',
                          textColor: '#166534',
                          borderColor: 'rgba(22, 101, 52, 0.1)'
                        }
                      }
                    }));
                    
                    // Final layout adjustments and styling
                    setTimeout(() => {
                      window.dispatchEvent(new Event('resize'));
                      if (typeof window.applyDatabaseSidebarStyling === 'function') {
                        window.applyDatabaseSidebarStyling();
                      }
                    }, 200);
                    }, 300); // Wait for badge fade-out to complete
                  }, 150);
                }}
                className="flex items-center justify-center p-2 bg-emerald-50 text-emerald-800 hover:bg-emerald-100 rounded cursor-pointer border border-emerald-300/20"
              >
                <Database className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs font-medium">Database</span>}
              </button>
            </div>
            
            {/* Configure and Analytics buttons */}
            <div className="mt-3 pt-3 border-t border-gray-200">
              <div className="grid grid-cols-2 gap-2 mb-2">
                {/* Configure Button */}
                <button
                  onClick={() => {
                    // Dispatch settings panel event
                    window.dispatchEvent(new CustomEvent('openSettingsPanel', {
                      detail: { 
                        source: 'sidebar'
                      }
                    }));
                  }}
                  className="flex items-center justify-center p-2 bg-gray-50 text-gray-600 hover:bg-gray-100 rounded border border-gray-300/20"
                >
                  <Settings className="h-4 w-4 mr-1" />
                  <span className="text-xs font-medium">Configure</span>
                </button>
                
                {/* Analytics Button (moved down next to Configure) */}
                <button
                  onClick={() => {
                    // First, make sure other panels are closed
                    const closeImageryEvent = new CustomEvent('forceCloseImageryPanel', {
                      detail: { source: 'analytics' }
                    });
                    window.dispatchEvent(closeImageryEvent);
                    
                    // Exit any active validation mode
                    window.dispatchEvent(new CustomEvent('exitValidationMode', {
                      detail: { 
                        source: 'analytics', 
                        clearExisting: true
                      }
                    }));
                    
                    // Make sure Object Detection sidebar is closed
                    window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                      detail: { source: 'analytics' }
                    }));
                    
                    // Make sure Tree Database sidebar is closed
                    window.dispatchEvent(new CustomEvent('forceCloseTreeDatabase', {
                      detail: { source: 'analytics' }
                    }));
                    
                    // Close the Review panel if it's open
                    window.dispatchEvent(new CustomEvent('closeReviewPanel', {
                      detail: { source: 'analytics' }
                    }));
                    
                    // Then open the analytics panel with Reports tab active
                    window.dispatchEvent(new CustomEvent('openAnalyticsPanel', {
                      detail: { 
                        source: 'sidebar',
                        tab: 'reports' // 'reports' is the Analytics tab
                      }
                    }));
                  }}
                  className="flex items-center justify-center p-2 bg-purple-50 text-purple-900 hover:bg-purple-100 rounded cursor-pointer border border-purple-300/20"
                >
                  <BarChart className={`h-4 w-4 mr-1`} />
                  <span className="text-xs">Analytics</span>
                </button>
              </div>
              
              {/* Reports Button (moved below Configure and Analytics) */}
              <div className="mt-2">
                <button
                  onClick={() => {
                    // First ensure Map Controls are collapsed
                    setMapControlsCollapsed(true);
                    
                    // Set the active navigation item to Reports
                    setSelectedItem('Reports');
                    
                    // Close any active panels first
                    window.dispatchEvent(new CustomEvent('exitValidationMode', {
                      detail: { 
                        source: 'sidebar_reports', 
                        clearExisting: true
                      }
                    }));
                    
                    window.dispatchEvent(new CustomEvent('closeAnalyticsPanel', {
                      detail: { source: 'reports' }
                    }));
                    
                    // Make sure Object Detection sidebar is closed
                    window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                      detail: { source: 'reports' }
                    }));
                    
                    // Make sure Tree Database sidebar is closed
                    window.dispatchEvent(new CustomEvent('forceCloseTreeDatabase', {
                      detail: { source: 'reports' }
                    }));
                    
                    // Open Analytics panel with Reports Overview
                    window.dispatchEvent(new CustomEvent('openAnalyticsPanel', {
                      detail: { 
                        source: 'sidebar_reports',
                        tab: 'reports'
                      }
                    }));
                    
                    // Navigate to Reports view
                    window.dispatchEvent(new CustomEvent('navigateTo', { 
                      detail: { 
                        view: 'Reports',
                        source: 'sidebar'
                      }
                    }));
                  }}
                  className="flex items-center justify-center p-2 w-full bg-amber-50 text-amber-900 hover:bg-amber-100 rounded cursor-pointer border border-gray-500/30"
                >
                  <FileText className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                  {!collapsed && <span className="text-xs">Reports</span>}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;