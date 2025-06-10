// src/components/common/Layout/Sidebar/index.jsx

import React, { useState, useEffect } from 'react';
import { Settings, BarChart, Database, Box, FileText, ChevronLeft, ChevronRight, Eye, ClipboardList, Plane, Sparkles, Layers } from 'lucide-react';
import MapControls from '../../../visualization/MapView/MapControls';
import { DetectionSidebarBridge } from '../../../visualization/Detection';

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
  
  /**
   * Toggles the sidebar between collapsed and expanded states
   * 
   * This function dispatches a custom event first so other components can
   * react to the sidebar state change before updating the local state.
   * Many components need to adjust their position based on sidebar width.
   */
  const toggleSidebar = () => {
    // Update collapsed state
    const newState = !collapsed;
    setCollapsed(newState);
    
    // Dispatch event for other components
    window.dispatchEvent(new CustomEvent('leftSidebarToggle', {
      detail: { 
        collapsed: newState 
      }
    }));
    
    // Force map to resize 
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
      
      // Second resize after a longer delay to handle transitions
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 300);
    }, 100);
  };
  
  /**
   * Handler for forced sidebar collapse events from other components
   * 
   * This allows other components (like the Settings panel) to collapse
   * the sidebar when needed without direct access to the sidebar state.
   */
  useEffect(() => {
    const handleForceSidebarCollapse = () => {
      if (!collapsed) {
        // Set local state
        setCollapsed(true);
        
        // Dispatch event for other components
        window.dispatchEvent(new CustomEvent('leftSidebarToggle', {
          detail: { collapsed: true }
        }));
        
        // Force resize events with delays to handle transitions
        setTimeout(() => {
          window.dispatchEvent(new Event('resize'));
          
          // Second resize after transition completes
          setTimeout(() => {
            window.dispatchEvent(new Event('resize'));
          }, 300);
        }, 100);
      }
    };
    
    window.addEventListener('forceSidebarCollapse', handleForceSidebarCollapse);
    
    return () => {
      window.removeEventListener('forceSidebarCollapse', handleForceSidebarCollapse);
    };
  }, [collapsed]);
  
  /**
   * Toggles the Data Analysis section dropdown in the sidebar
   * 
   * This controls the visibility of the data analysis tools like
   * Detection, Database, and Analytics.
   */
  const toggleDataAnalysis = () => {
    setDataAnalysisCollapsed(!dataAnalysisCollapsed);
  };

  /**
   * Handles navigation when a sidebar menu item is clicked
   * 
   * This function:
   * 1. Updates the selected item state
   * 2. Notifies the parent component via a callback
   * 3. Preserves the current 3D state during navigation
   * 4. Dispatches a custom navigateTo event for other components
   * 
   * @param {string} label - The label of the clicked menu item
   */
  const handleItemClick = (label) => {
    setSelectedItem(label);
    
    // Let parent component know about navigation
    if (onNavigate) {
      onNavigate(label);
    }
    
    // Check if current map is in 3D mode
    const is3DMode = typeof window.is3DModeActive !== 'undefined' ? window.is3DModeActive : false;
    
    // Dispatch a custom event for other components to listen for
    window.dispatchEvent(new CustomEvent('navigateTo', {
      detail: { 
        view: label,
        preserve3DState: true,
        is3DMode: is3DMode
      }
    }));
  };
  
  // Initialize map state when component mounts
  useEffect(() => {
    setSelectedItem('Map');
    
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
                el.parentNode.removeChild(el);
                console.log(`Safely removed element with selector: ${selector}`);
              }
            } catch (err) {
              console.log(`Error removing element ${selector}: ${err.message}`);
            }
          });
        } catch (e) {
          console.log(`Error in selector ${selector}: ${e.message}`);
        }
      };
      
      // Helper function for safely hiding elements by ID
      const safeHideById = (id) => {
        try {
          const element = document.getElementById(id);
          if (element && document.body.contains(element)) {
            element.style.display = 'none';
            element.style.opacity = '0';
            console.log(`Safely hid element with ID: ${id}`);
          }
        } catch (e) {
          console.log(`Error hiding element with ID ${id}: ${e.message}`);
        }
      };
      
      // First hide elements before processing
      document.querySelectorAll('.detection-sidebar').forEach(el => {
        el.style.display = 'none';
        el.style.opacity = '0';
      });
      
      // Hide overlay and badge but don't remove them
      safeHideById('ml-detection-overlay');
      safeHideById('detection-debug');
      
      // Allow a tick for the hiding to take effect
      setTimeout(() => {
        // Remove only the sidebar elements
        safeRemoveBySelector('.detection-sidebar');
        
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
        
        // Store state indicating detection was recently closed
        window.detectionRecentlyClosed = true;
        
        // Clear this flag after a delay to allow reopening to work correctly
        setTimeout(() => {
          window.detectionRecentlyClosed = false;
        }, 500);
        
        // Trigger resize event
        window.dispatchEvent(new Event('resize'));
      }, 10);
    };
    
    window.addEventListener('forceCloseObjectDetection', handleForceCloseObjectDetection);
    
    return () => {
      window.removeEventListener('forceCloseObjectDetection', handleForceCloseObjectDetection);
      window.removeEventListener('openFeatureSelection', handleDatabaseFeatureSelection);
    };
  }, []);
  
  // Track 3D mode state
  const [is3DMode, setIs3DMode] = useState(false);
  // Track map type state (roadmap or satellite)
  const [mapType, setMapType] = useState('satellite');
  
  useEffect(() => {
    // Initialize from global state if available
    if (typeof window.is3DModeActive !== 'undefined') {
      setIs3DMode(window.is3DModeActive);
    }
    
    // Listen for map mode changed events
    const handleMapModeChanged = (event) => {
      const { mode } = event.detail;
      setIs3DMode(mode === '3D');
    };
    
    // Let the map initialization handle the map type
    // We don't need to force satellite view anymore
    
    // Keep the state in sync with whatever the map is using
    const savedMapType = localStorage.getItem('currentMapType') || 'satellite';
    setMapType(savedMapType);
    console.log("Sidebar using stored map type:", savedMapType);
    
    // Let the app handle map container positioning via CSS and event listeners
    
    // Listen for map type changes
    const handleMapTypeChanged = (event) => {
      const { mapTypeId } = event.detail;
      if (mapTypeId) {
        console.log("Sidebar received map type change:", mapTypeId);
        setMapType(mapTypeId);
      }
    };
    
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
    
    window.addEventListener('mapModeChanged', handleMapModeChanged);
    window.addEventListener('mapTypeChanged', handleMapTypeChanged);
    window.addEventListener('settingsUpdated', handleSettingsUpdated);
    
    return () => {
      window.removeEventListener('mapModeChanged', handleMapModeChanged);
      window.removeEventListener('mapTypeChanged', handleMapTypeChanged);
      window.removeEventListener('settingsUpdated', handleSettingsUpdated);
    };
  }, [collapsed]);

  return (
    <>
      {/* Include the DetectionSidebarBridge for React integration */}
      <DetectionSidebarBridge />
      
      <div 
        className={`flex flex-col h-full bg-white border-r border-gray-200 shadow-sm transition-all duration-300 ease-in-out ${
          collapsed ? 'w-10 bg-gray-100' : 'w-64'
        }`}
        id="left-sidebar"
      >

      {/* Sidebar Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Map Controls Section */}
        <div className={`${collapsed ? 'py-4' : 'p-4'} border-b border-gray-200`}>
          <div className="flex justify-between items-center mb-3">
            <div className={`flex items-center justify-between w-full ${collapsed ? 'justify-center' : ''}`}>
              <div className={`flex items-center ${collapsed ? 'ml-0 justify-center w-full' : ''}`}>
                <button 
                  className={`p-2 ${collapsed ? '' : 'mr-2'} rounded transition-colors ${
                    collapsed ? 'bg-white hover:bg-gray-200 shadow-sm' : 'hover:bg-gray-100'
                  }`}
                  onClick={toggleSidebar}
                  aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                >
                  {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
                </button>
                {!collapsed && <h3 className="text-sm font-semibold text-gray-700 cursor-pointer hover:text-gray-900" onClick={toggleSidebar}>Controls</h3>}
              </div>
            </div>
          </div>
          {!collapsed && (
            <div className="space-y-2">
              {selectedItem === 'Map' ? (
                <MapControls mapRef={mapRef} mapDataRef={mapDataRef} />
              ) : (
                <div className="text-sm text-gray-500">Map controls available when map is active</div>
              )}
              
              {/* Reports & Review buttons */}
              <div className="mt-4">
                <div className="border-t border-gray-200 mb-4"></div>
                <div className="grid grid-cols-2 gap-2">
                  {/* Reports Button */}
                  <button
                    onClick={() => {
                      // Close ALL sidebars
                      window.dispatchEvent(new CustomEvent('forceCloseImageryPanel', {
                        detail: { source: 'sidebar' }
                      }));
                      window.dispatchEvent(new CustomEvent('closeAnalyticsPanel', {
                        detail: { source: 'sidebar' }
                      }));
                      window.dispatchEvent(new CustomEvent('closeReviewPanel', {
                        detail: { source: 'sidebar' }
                      }));
                      window.dispatchEvent(new CustomEvent('forceCloseTreeDatabase', {
                        detail: { source: 'sidebar' }
                      }));
                      window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                        detail: { source: 'sidebar' }
                      }));
                      
                      // Reset container widths to full size
                      const mapContainer = document.getElementById('map-container');
                      const reportContainer = document.getElementById('report-container');
                      
                      if (mapContainer) mapContainer.style.right = '0px';
                      if (reportContainer) reportContainer.style.right = '0px';
                      
                      // Navigate to Reports view
                      setTimeout(() => {
                        onNavigate('Reports');
                        
                        // Force resize event after navigation
                        window.dispatchEvent(new Event('resize'));
                      }, 100);
                    }}
                    className="flex items-center justify-center p-2 rounded-md border border-gray-300 shadow-sm"
                    style={{ backgroundColor: '#f5f5f5', color: '#4b5563' }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#e5e5e5'}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#f5f5f5'}
                  >
                    <FileText className="h-4 w-4 mr-2" />
                    <span className="text-xs font-medium">Reports</span>
                  </button>
                  
                  {/* Review Button */}
                  <button
                    onClick={() => {
                      // First, close the Imagery sidebar if it's open
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
                      
                      // Open review panel
                      window.dispatchEvent(new CustomEvent('openReviewPanel', {
                        detail: { 
                          source: 'sidebar'
                        }
                      }));
                    }}
                    className="flex items-center justify-center p-2 rounded-md border border-gray-300 shadow-sm"
                    style={{ backgroundColor: '#f5f5f5', color: '#4b5563' }}
                    onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#e5e5e5'}
                    onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#f5f5f5'}
                  >
                    <ClipboardList className="h-4 w-4 mr-2" />
                    <span className="text-xs font-medium">Validate</span>
                  </button>
                </div>
              </div>
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
                className="flex items-center justify-center p-2 bg-indigo-50 text-indigo-900 hover:bg-indigo-100 rounded-md cursor-pointer border border-indigo-200"
              >
                <Layers className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Aerial</span>}
              </button>
              
              {/* Detection Button */}
              <button
                onClick={() => {
                  // Check if detection sidebar is already visible
                  const existingDetectionSidebar = document.querySelector('.detection-sidebar');
                  const isDetectionActive = existingDetectionSidebar && 
                    window.getComputedStyle(existingDetectionSidebar).display !== 'none' &&
                    window.getComputedStyle(existingDetectionSidebar).opacity !== '0';
                  
                  // If detection is already active, just ensure overlay and badge are visible
                  if (isDetectionActive) {
                    console.log("Detection sidebar already active, ensuring overlay visible");
                    
                    // CRITICAL FIX: Dispatch openTreeDetection event to ensure ML overlay is visible
                    // with the 'sidebar_button' source to activate the proper code path
                    // Using buttonTriggered: true to simulate a detect button click to force overlay visibility
                    window.dispatchEvent(new CustomEvent('openTreeDetection', {
                      detail: {
                        sidebarInitialization: true,
                        initialVisibility: true,
                        buttonTriggered: true, // Simulate detection button to ensure overlay appears on first click
                        source: 'sidebar_button',
                        jobId: window.currentDetectionJobId || window._lastDetectionJobId
                      }
                    }));
                    
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
                        
                        newOverlay.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
                        newOverlay.style.pointerEvents = 'none';
                        newOverlay.style.zIndex = '50';
                        newOverlay.style.transition = 'opacity 0.3s ease, background-color 0.3s ease';
                        mapContainer.appendChild(newOverlay);
                        
                        console.log(`Created new ML overlay with opacity ${opacity}`);
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
                          overlay.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
                          console.log(`Updated existing overlay to stored opacity: ${opacity}`);
                        }
                      } catch (e) {
                        console.error("Error applying stored opacity:", e);
                      }
                    }
                    
                    // Make sure badge is visible and positioned correctly
                    const badge = document.getElementById('detection-debug');
                    if (badge) {
                      badge.style.display = 'block';
                      badge.style.opacity = '1';
                      // Position badge correctly to the left of the sidebar
                      badge.style.right = '384px'; // Same as sidebar width
                    } else {
                      // Create badge if it doesn't exist, positioned to the left of sidebar
                      const mapContainer = document.getElementById('map-container');
                      if (mapContainer) {
                        const newBadge = document.createElement('div');
                        newBadge.id = 'detection-debug';
                        newBadge.textContent = 'DETECTION';
                        newBadge.style.display = 'block';
                        newBadge.style.position = 'absolute';
                        newBadge.style.top = '0';
                        newBadge.style.right = '384px'; // Position to the left of sidebar
                        newBadge.style.background = 'rgba(13, 71, 161, 0.85)';
                        newBadge.style.zIndex = '200';
                        newBadge.style.padding = '5px 12px';
                        newBadge.style.fontSize = '12px';
                        newBadge.style.color = 'white';
                        newBadge.style.fontWeight = '500';
                        newBadge.style.borderBottomLeftRadius = '3px';
                        newBadge.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                        newBadge.style.letterSpacing = '0.5px';
                        mapContainer.appendChild(newBadge);
                      }
                    }
                    
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
                        
                        // Visual feedback animation
                        counterElement.style.transform = 'scale(1.2)';
                        counterElement.style.transition = 'transform 0.2s';
                        setTimeout(() => {
                          counterElement.style.transform = 'scale(1)';
                        }, 200);
                      }
                    }
                    
                    return; // Exit early, no need to re-create the sidebar
                  }
                  
                  // Close all other sidebars if creating a new detection sidebar
                  window.dispatchEvent(new CustomEvent('forceCloseImageryPanel', { detail: { source: 'sidebar' } }));
                  window.dispatchEvent(new CustomEvent('forceCloseTreeDatabase', { detail: { source: 'sidebar' } }));
                  window.dispatchEvent(new CustomEvent('closeAnalyticsPanel', { detail: { source: 'detection' } }));
                  window.dispatchEvent(new CustomEvent('closeReviewPanel', { detail: { source: 'detection' } }));
                  window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', { detail: { source: 'sidebar' } }));
                  window.dispatchEvent(new CustomEvent('exitValidationMode', { 
                    detail: { source: 'sidebar', target: 'object_detection', clearExisting: true }
                  }));
                  
                  // If we're in the Reports view, switch to Map view first
                  const reportContainer = document.getElementById('report-container');
                  if (reportContainer && window.getComputedStyle(reportContainer).display !== 'none') {
                    // Trigger navigation to Map view
                    window.dispatchEvent(new CustomEvent('navigateTo', {
                      detail: { view: 'Map', preserve3DState: true }
                    }));
                    
                    // Short delay to allow view to change before proceeding
                    setTimeout(() => continueDetectionProcess(), 200);
                  } else {
                    // We're already in Map view, continue immediately
                    continueDetectionProcess();
                  }
                  
                  // Function to continue the detection process after ensuring we're in Map view
                  function continueDetectionProcess() {
                    // 1. Get current header state
                    const headerElement = document.querySelector('header');
                    const isHeaderCollapsed = headerElement ? 
                      headerElement.classList.contains('collapsed') || 
                      headerElement.offsetHeight < 50 : true;
                    
                    console.log("Detection button: Current header state =", isHeaderCollapsed);
                    
                    // 2. Send the header state event to ensure all components are in sync
                    window.dispatchEvent(new CustomEvent('headerCollapse', {
                      detail: { collapsed: isHeaderCollapsed }
                    }));
                    
                    // 3. Adjust the map container right away
                    const mapContainer = document.getElementById('map-container');
                    if (mapContainer) {
                      mapContainer.style.right = '384px';
                    }
                    
                    // 4. Switch to satellite view for the map
                    window.dispatchEvent(new CustomEvent('setMapTypeId', {
                      detail: { mapTypeId: 'satellite', source: 'sidebar' }
                    }));
                    
                    // 5. Reset validation mode to ensure clean state
                    window.dispatchEvent(new CustomEvent('exitValidationMode', {
                      detail: { source: 'sidebar', clearExisting: true }
                    }));
                    
                    // 6. Add the ML overlay to the map - we need this for the blue tint
                    let overlay = document.getElementById('ml-detection-overlay');
                    
                    // If no overlay exists, create it and ensure it's visible
                    if (!overlay && mapContainer) {
                      console.log("CRITICAL: Creating new ML overlay on first detection click");
                      overlay = document.createElement('div');
                      overlay.id = 'ml-detection-overlay';
                      overlay.style.position = 'absolute';
                      overlay.style.top = '0';
                      overlay.style.left = '0';
                      overlay.style.width = '100%';
                      overlay.style.height = '100%';
                      
                      // Try to use stored opacity
                      let opacity = 0.3; // Default
                      try {
                        const savedOpacity = localStorage.getItem('ml-overlay-opacity');
                        if (savedOpacity !== null) {
                          opacity = parseFloat(savedOpacity);
                        }
                      } catch (e) {
                        console.error("Error reading opacity setting:", e);
                      }
                      
                      // Set global settings and ensure visibility
                      window.mlOverlaySettings = {
                        ...(window.mlOverlaySettings || {}),
                        opacity: opacity,
                        showOverlay: true
                      };
                      window.detectionShowOverlay = true;
                      
                      // Style the overlay to make it visible with the right opacity
                      overlay.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
                      overlay.style.pointerEvents = 'none';
                      overlay.style.zIndex = '50';
                      overlay.style.transition = 'background-color 0.3s ease';
                      overlay.style.display = 'block';
                      mapContainer.appendChild(overlay);
                      
                      console.log(`Created new ML overlay with opacity ${opacity} and forced visibility`);
                      
                      // CRITICAL: Trigger the openTreeDetection event to ensure proper initialization
                      // This is typically only triggered on second click, but we need it on first click too
                      window.dispatchEvent(new CustomEvent('openTreeDetection', {
                        detail: {
                          sidebarInitialization: true,
                          initialVisibility: true,
                          buttonTriggered: true, // Simulate detection button to ensure overlay appears
                          source: 'sidebar_first_click',
                          forceVisibility: true // Special flag for first click
                        }
                      }));
                    } else if (overlay) {
                      // Get stored overlay visibility setting
                      const shouldDisplayOverlay = window.mlOverlaySettings?.showOverlay !== false;
                      
                      // Update visibility based on stored setting
                      overlay.style.display = shouldDisplayOverlay ? 'block' : 'none';
                      
                      // Update to current stored opacity if available
                      try {
                        // First try from global settings
                        let opacity = window.mlOverlaySettings?.opacity;
                        
                        // If not in global settings, try localStorage
                        if (opacity === undefined) {
                          const savedOpacity = localStorage.getItem('ml-overlay-opacity');
                          if (savedOpacity !== null) {
                            opacity = parseFloat(savedOpacity);
                            
                            // Update the global settings for consistency
                            window.mlOverlaySettings = {
                              ...(window.mlOverlaySettings || {}),
                              opacity: opacity
                            };
                          }
                        }
                        
                        // Apply the opacity if we found a value
                        if (opacity !== undefined) {
                          overlay.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
                          console.log(`Updated existing overlay with opacity ${opacity}`);
                        }
                      } catch (e) {
                        console.error("Error applying stored opacity:", e);
                      }
                    }
                    
                    // 7. Trigger ML detection integration event to notify React components
                    window.dispatchEvent(new CustomEvent('mlDetectionActive', {
                      detail: { 
                        active: true, 
                        source: 'sidebar',
                        headerCollapsed: isHeaderCollapsed
                      }
                    }));
                    
                    // Simple approach: Just make sure the map container is adjusted
                    // for the detection sidebar and trigger the original detection mode
                    if (mapContainer) {
                      mapContainer.style.right = '384px';
                    }
                    
                    // Remove any existing object recognition badge to avoid duplicates
                    const oldBadge = document.getElementById('object-recognition-badge');
                    if (oldBadge && oldBadge.parentNode) {
                      oldBadge.parentNode.removeChild(oldBadge);
                    }
                    
                    // Don't trigger the React component events to avoid conflicts
                    console.log("Creating standalone Object Recognition sidebar");
                    
                    // Skip triggering the enterTreeValidationMode and requestTreeDetection events
                    // since they're causing conflicts with React components
                    
                    // Create the new Object Recognition sidebar with light-blue theme
                    setTimeout(() => {
                      // Remove any existing sidebar with the same class to avoid duplicates
                      const existingSidebar = document.querySelector('.detection-sidebar');
                      if (existingSidebar && existingSidebar.parentNode) {
                        existingSidebar.parentNode.removeChild(existingSidebar);
                      }
                      
                      // Create the new sidebar element with light-blue theme
                      // First, ensure any previous detection sidebars are fully removed
                      document.querySelectorAll('.detection-sidebar, .object-recognition-sidebar').forEach(oldSidebar => {
                        if (oldSidebar && oldSidebar.parentNode) {
                          console.log("Removing old detection sidebar before creating new one");
                          oldSidebar.style.display = 'none';
                          oldSidebar.parentNode.removeChild(oldSidebar);
                        }
                      });
                      
                      // This will be the container we'll improve directly
                      const sidebarElement = document.createElement('div');
                      sidebarElement.className = 'detection-sidebar';
                      sidebarElement.setAttribute('data-react-container', 'true'); // Mark as React container
                      sidebarElement.style.position = 'fixed';
                      
                      // Get precise header height to ensure proper positioning
                      const headerEl = document.querySelector('header');
                      const headerHeight = headerEl ? headerEl.offsetHeight : (isHeaderCollapsed ? 40 : 64);
                      
                      sidebarElement.style.top = `${headerHeight}px`; // Position exactly below header
                      sidebarElement.style.right = '0';
                      sidebarElement.style.width = '384px';
                      sidebarElement.style.bottom = '0';
                      sidebarElement.style.height = `calc(100vh - ${headerHeight}px)`; // Correct height calculation
                      sidebarElement.style.backgroundColor = '#f8fafc'; // Very subtle, almost white background
                      sidebarElement.style.color = '#0d47a1'; // Dark blue text
                      sidebarElement.style.zIndex = '100';
                      sidebarElement.style.overflow = 'hidden'; // Changed from auto to hidden to prevent scrollbars
                      sidebarElement.style.boxShadow = '-1px 0 8px rgba(0,0,0,0.08)';
                      sidebarElement.style.transition = 'all 0.3s ease';
                      sidebarElement.style.display = 'block'; // Force block display
                      
                      // Create header for this sidebar
                      // Don't create header here - the React component will handle it
                      console.log("Skipping creation of DOM-based header - using React component header instead");
                      
                      // Create main content area for DetectionSidebar to render in
                      const contentContainer = document.createElement('div');
                      contentContainer.id = 'detection-content-container';
                      contentContainer.style.width = '100%';
                      contentContainer.style.height = '100%'; // Full height since React handles the header
                      contentContainer.style.overflow = 'auto';
                      contentContainer.style.position = 'relative';
                      sidebarElement.appendChild(contentContainer);
                      
                      // React will handle all content padding and layout
                      
                      // Add the sidebar to document
                      document.body.appendChild(sidebarElement);
                      
                      // Store reference to sidebar globally so it doesn't get garbage collected
                      window.currentDetectionSidebar = sidebarElement;
                      
                      // Set global flag to prevent sidebar from being closed immediately
                      window.isDetectionSidebarActive = true;
                      
                      // Notify React components that detection is active
                      window.dispatchEvent(new CustomEvent('mlDetectionActive', {
                        detail: { 
                          active: true, 
                          source: 'object_recognition_sidebar',
                          headerCollapsed: isHeaderCollapsed,
                          sidebarElement
                        }
                      }));
                    }, 300);
                    
                    // Create the DETECTION badge for the sidebar
                    window.dispatchEvent(new CustomEvent('createDetectionBadge', {
                      detail: {
                        width: '384px', // Sidebar width
                        headerCollapsed: isHeaderCollapsed
                      }
                    }));
                    
                    // Add a small delay to ensure badge is created
                    setTimeout(() => {
                      const badge = document.getElementById('detection-debug');
                      if (!badge) {
                        // If event handler didn't create it, create it manually
                        const detectionDebug = document.createElement('div');
                        detectionDebug.id = 'detection-debug';
                        detectionDebug.textContent = 'DETECTION';
                        detectionDebug.style.display = 'block';
                        detectionDebug.style.position = 'absolute';
                        detectionDebug.style.top = '0';
                        detectionDebug.style.right = '384px'; // Position to the left of sidebar
                        detectionDebug.style.background = 'rgba(13, 71, 161, 0.85)'; // Dark blue to match theme
                        detectionDebug.style.zIndex = '200';
                        detectionDebug.style.padding = '5px 12px';
                        detectionDebug.style.fontSize = '12px';
                        detectionDebug.style.color = 'white';
                        detectionDebug.style.fontWeight = '500';
                        detectionDebug.style.borderBottomLeftRadius = '3px';
                        detectionDebug.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                        detectionDebug.style.letterSpacing = '0.5px';
                        
                        if (mapContainer) {
                          mapContainer.appendChild(detectionDebug);
                        }
                      }
                    }, 200);
                    
                    // Trigger resize to finalize layout
                    setTimeout(() => {
                      window.dispatchEvent(new Event('resize'));
                    }, 300);
                  }
                }}
                className="flex items-center justify-center p-2 bg-blue-50 text-blue-900 hover:bg-blue-100 rounded-md cursor-pointer border border-blue-200"
              >
                <Box className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Detection</span>}
              </button>
            </div>
            
            {/* Second row - Database and Response buttons side-by-side */}
            <div className="grid grid-cols-2 gap-2">
              {/* Database Button */}
              <button
                onClick={() => {
                  console.log("Database button clicked - FORCEFULLY DISCONNECTING Detection mode");
                  
                  // IMMEDIATE STEP: Prevent rapid double-clicking
                  // Check if database sidebar is already open
                  const featureSelectionSidebar = document.querySelector('.feature-selection-sidebar, .tree-inventory-sidebar');
                  if (featureSelectionSidebar) {
                    console.log("Database sidebar is already open - ignoring click");
                    return; // Exit early if sidebar is already open
                  }
                  
                  // VERY IMPORTANT: Use a comprehensive approach to completely disconnect Detection mode
                  // before launching Database mode to prevent any interaction between them
                  
                  try {
                    // PHASE 1: Emergency state reset - force React component unmounting
                    
                    // Force the React component to unmount by setting application state variables
                    window.isValidationMode = false;
                    window.isDetectionSidebarActive = false;
                    window.isDetectionModeActive = false;
                    window.manualTreePlacementActive = false;
                    
                    // Dispatch reset events to ensure state is updated throughout the app
                    window.dispatchEvent(new CustomEvent('validationModeForceClosed', {
                      detail: { source: 'database_button', emergency: true }
                    }));
                    
                    window.dispatchEvent(new CustomEvent('exitValidationMode', {
                      detail: { 
                        source: 'database_button',
                        emergency: true,
                        target: 'feature_selection',
                        clearExisting: true,
                        forceRemove: true
                      }
                    }));
                    
                    // PHASE 2: Complete DOM cleansing - remove any DOM elements directly
                    
                    // 1. First hide all elements (safer than removing them immediately)
                    const elementsToHide = document.querySelectorAll(
                      '#detection-debug, .detection-debug, ' +
                      '#ml-detection-overlay, .ml-detection-overlay, ' +
                      '[id*="detection"], [class*="detection"], ' +
                      '[id*="ml-overlay"], [class*="ml-overlay"]'
                    );
                    
                    console.log(`Found ${elementsToHide.length} detection elements to hide`);
                    
                    // Hide all elements first
                    elementsToHide.forEach(el => {
                      try {
                        if (el && document.body.contains(el)) {
                          el.style.display = 'none';
                          el.style.visibility = 'hidden';
                          el.style.opacity = '0';
                          el.style.pointerEvents = 'none';
                          el.style.position = 'absolute';
                          el.style.left = '-9999px';
                          el.style.top = '-9999px';
                          el.style.zIndex = '-9999';
                          el.style.height = '0';
                          el.style.width = '0';
                          el.style.overflow = 'hidden';
                          console.log(`Hidden element: ${el.id || el.className}`);
                        }
                      } catch (e) {}
                    });
                    
                    // 2. Move all detection mode content out of the DOM entirely
                    const rootDetectionElements = document.querySelectorAll(
                      '.detection-sidebar, #detection-sidebar, .detection-sidebar-container, ' +
                      '#detection-mode-container, .detection-mode-container'
                    );
                    
                    // Create a hidden storage div to hold removed elements
                    // This prevents DOM errors by keeping them in the document but invisible
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
                      detachedStorage.style.zIndex = '-9999';
                      document.body.appendChild(detachedStorage);
                    }
                    
                    // Move elements to storage instead of removing them
                    console.log(`Found ${rootDetectionElements.length} root detection elements to detach`);
                    rootDetectionElements.forEach(el => {
                      try {
                        if (el && document.body.contains(el)) {
                          // Apply deep hiding styles first
                          el.style.display = 'none';
                          el.style.visibility = 'hidden';
                          
                          // Instead of removing, move to invisible storage div
                          detachedStorage.appendChild(el);
                          console.log(`Safely detached element: ${el.id || el.className}`);
                        }
                      } catch (e) {
                        console.error(`Error detaching element: ${e.message}`);
                      }
                    });
                    
                    // 3. Reset map container dimensions
                    let mapContainerEl = document.getElementById('map-container');
                    if (mapContainerEl) {
                      mapContainerEl.style.right = '0px';
                      mapContainerEl.style.width = '100%';
                      mapContainerEl.style.left = '0';
                    }
                    
                    // 4. Force resize event to update layout
                    window.dispatchEvent(new Event('resize'));
                    
                    // PHASE 3: Use our safe cleanup function for anything we missed
                    const safeCleanup = (selector) => {
                      try {
                        document.querySelectorAll(selector).forEach(el => {
                          if (el && document.body.contains(el) && el.parentNode) {
                            // First hide
                            try {
                              el.style.display = 'none';
                              el.style.opacity = '0';
                              el.style.visibility = 'hidden';
                            } catch (e) {}
                            
                            // Then move to storage instead of removing
                            try {
                              if (detachedStorage) {
                                detachedStorage.appendChild(el);
                              }
                            } catch (e) {}
                          }
                        });
                      } catch (e) {
                        console.error(`Error in selector ${selector}: ${e.message}`);
                      }
                    };
                    
                    // Run cleanup on all possible detection-related elements
                    safeCleanup('#ml-detection-overlay');
                    safeCleanup('.ml-detection-overlay');
                    safeCleanup('[id*="ml-detection"]');
                    safeCleanup('[class*="ml-detection"]');
                    safeCleanup('#detection-debug');
                    safeCleanup('.detection-debug');
                    safeCleanup('[id*="detection-debug"]');
                    safeCleanup('.detection-sidebar');
                    safeCleanup('#detection-sidebar');
                  } catch (e) {
                    console.error("Error in database button click handler:", e);
                  }
                  
                  // Step 2: Close all other sidebar panels
                  window.dispatchEvent(new CustomEvent('forceCloseImageryPanel', { detail: { source: 'database' } }));
                  window.dispatchEvent(new CustomEvent('closeAnalyticsPanel', { detail: { source: 'database' } }));
                  window.dispatchEvent(new CustomEvent('closeReviewPanel', { detail: { source: 'database' } }));
                  
                  // Step 3: ONLY NOW open the database sidebar with a slight delay to ensure complete cleanup
                  setTimeout(() => {
                    // Get accurate header state for positioning
                    const headerElement = document.querySelector('header');
                    let isHeaderCollapsed = true;
                    
                    if (headerElement) {
                      const headerHeight = headerElement.getBoundingClientRect().height;
                      const hasCollapsedClass = headerElement.classList.contains('collapsed');
                      isHeaderCollapsed = hasCollapsedClass || headerHeight < 50;
                      console.log("Database button: Header state =", isHeaderCollapsed);
                    }
                    
                    // Broadcast header state
                    window.dispatchEvent(new CustomEvent('headerCollapse', {
                      detail: { collapsed: isHeaderCollapsed }
                    }));
                    
                    // Force resize event
                    window.dispatchEvent(new Event('resize'));
                    
                    // Create the database styling function
                    window.applyDatabaseSidebarStyling = () => {
                      setTimeout(() => {
                        try {
                          // Find the database sidebar that was just created
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
                    
                    // Finally, open the feature selection mode with Database theme
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
                  }, 100);
                }
              }
              className="flex items-center justify-center p-2 bg-emerald-700 text-emerald-50 hover:bg-emerald-600 rounded-md cursor-pointer border border-emerald-500/30"
              >
                <Database className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs font-light">Database</span>}
              </button>
              
              {/* Analytics Button */}
              <button
                onClick={() => {
                  // First, close ALL sidebars
                  window.dispatchEvent(new CustomEvent('forceCloseImageryPanel', {
                    detail: { source: 'analytics' }
                  }));
                  
                  window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                    detail: { source: 'analytics' }
                  }));
                  
                  window.dispatchEvent(new CustomEvent('forceCloseTreeDatabase', {
                    detail: { source: 'analytics' }
                  }));
                  
                  window.dispatchEvent(new CustomEvent('closeReviewPanel', {
                    detail: { source: 'analytics' }
                  }));
                  
                  window.dispatchEvent(new CustomEvent('exitValidationMode', {
                    detail: { 
                      source: 'analytics', 
                      clearExisting: true
                    }
                  }));
                  
                  // Reset map container to full width
                  const mapContainer = document.getElementById('map-container');
                  if (mapContainer) {
                    mapContainer.style.right = '0';
                  }
                  
                  // Force resize event after closing sidebars
                  setTimeout(() => {
                    window.dispatchEvent(new Event('resize'));
                  }, 100);
                  
                  // Finally, open the analytics panel
                  setTimeout(() => {
                    window.dispatchEvent(new CustomEvent('openAnalyticsPanel', {
                      detail: {
                        source: 'sidebar'
                      }
                    }));
                  }, 150);
                }}
                className="flex items-center justify-center p-2 bg-emerald-50 text-emerald-800 hover:bg-emerald-100 rounded-md cursor-pointer border border-emerald-300/20"
              >
                <Sparkles className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Analytics</span>}
              </button>
            </div>
            
            {/* Reports button was moved to Map Controls section */}
          </div>
        </div>
        
        {/* Bottom border/divider - empty div to maintain spacing */}
        <div className="mt-auto border-t border-gray-200"></div>
      </div>
    </div>
    </>
  );
};

export default Sidebar;