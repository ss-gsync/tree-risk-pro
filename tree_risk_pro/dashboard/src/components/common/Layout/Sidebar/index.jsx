// src/components/common/Layout/Sidebar/index.jsx

import React, { useState, useEffect } from 'react';
import { Settings, BarChart, Database, Box, FileText, ChevronLeft, ChevronRight, Eye } from 'lucide-react';
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
    const handleForceSidebarCollapse = () => {
      if (!collapsed) {
        // Only collapse if not already collapsed
        window.dispatchEvent(new CustomEvent('leftSidebarToggle', {
          detail: { collapsed: true }
        }));
        setCollapsed(true);
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
   * Detection, Database, Analytics, and Reports.
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
  }, []);
  
  // Track 3D mode state
  const [is3DMode, setIs3DMode] = useState(false);
  
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
    
    window.addEventListener('mapModeChanged', handleMapModeChanged);
    
    return () => {
      window.removeEventListener('mapModeChanged', handleMapModeChanged);
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
          <div className="flex justify-between items-center mb-3">
            <div className="flex items-center">
              <button 
                className={`p-1 ${collapsed ? 'mx-auto' : 'mr-2'} rounded transition-colors ${
                  collapsed ? 'bg-white hover:bg-gray-200 shadow-sm' : 'hover:bg-gray-100'
                }`}
                onClick={toggleSidebar}
                aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              >
                {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
              </button>
              {!collapsed && <h3 className="text-sm font-medium text-gray-600">Map Controls</h3>}
            </div>
            {!collapsed && (
              <button 
                onClick={() => {
                  // Dispatch the 3D toggle event
                  window.dispatchEvent(new CustomEvent('requestToggle3DViewType', {
                    detail: { 
                      show3D: !is3DMode 
                    }
                  }));
                }}
                className={`flex items-center justify-center px-3 py-1 text-xs rounded-md border transition-colors ${
                  is3DMode 
                    ? 'bg-blue-50 text-blue-600 hover:bg-blue-100 border-blue-200' 
                    : 'bg-white text-gray-700 hover:bg-gray-50 border-gray-200'
                }`}
              >
                {is3DMode ? '2D View' : '3D View'}
              </button>
            )}
          </div>
          {!collapsed && (
            <div className="space-y-2">
              {selectedItem === 'Map' ? (
                <MapControls mapRef={mapRef} mapDataRef={mapDataRef} />
              ) : (
                <div className="text-sm text-gray-500">Map controls available when map is active</div>
              )}
            </div>
          )}
        </div>
        
        {/* Data Analysis Section */}
        <div className={`${collapsed ? 'hidden' : 'p-4'} border-b border-gray-200`}>
          <div className="flex justify-between items-center cursor-pointer" onClick={toggleDataAnalysis}>
            <h3 className="text-sm font-medium text-gray-600">Data Analysis</h3>
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
                  
                  // Then make sure Imagery panel is expanded with Review tab active
                  window.dispatchEvent(new CustomEvent('expandImageryPanel', {
                    detail: { 
                      source: 'sidebar',
                      tab: 'imagery' // 'imagery' is the Review tab
                    }
                  }));
                }}
                className="flex items-center justify-center p-2 bg-indigo-50 text-indigo-600 hover:bg-indigo-100 rounded-md cursor-pointer"
              >
                <Box className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Review</span>}
              </button>
              
              {/* Detection Button */}
              <button
                onClick={() => {
                  // First, close the Imagery sidebar if it's open
                  const closeImageryEvent = new CustomEvent('forceCloseImageryPanel', {
                    detail: { source: 'sidebar' }
                  });
                  window.dispatchEvent(closeImageryEvent);
                  
                  // Next, close any active validation mode with full reset
                  window.dispatchEvent(new CustomEvent('exitValidationMode', {
                    detail: { 
                      source: 'sidebar', 
                      target: 'object_detection',
                      clearExisting: true  // Changed to true to fully reset
                    }
                  }));
                  
                  // Force close Tree Database sidebar if open
                  window.dispatchEvent(new CustomEvent('forceCloseTreeDatabase', {
                    detail: { source: 'sidebar' }
                  }));
                  
                  // Adjust the map container right away to prevent grey space
                  const mapContainer = document.getElementById('map-container');
                  if (mapContainer) {
                    mapContainer.style.right = '384px';
                  }
                  
                  // Short delay to ensure other sidebars have time to close
                  setTimeout(() => {
                    // Force map resize to adjust to the new container size
                    window.dispatchEvent(new Event('resize'));
                    
                    // Enable satellite view mode for the map
                    window.dispatchEvent(new CustomEvent('setMapTypeId', {
                      detail: { 
                        mapTypeId: 'satellite',
                        source: 'sidebar' 
                      }
                    }));
                    
                    // Only open the detection sidebar, but don't trigger detection yet
                    // Create a proper event with the required details
                    const validationEvent = new CustomEvent('enterTreeValidationMode', {
                      detail: {
                        mode: 'detection', 
                        source: 'sidebar',
                        clearExisting: true,
                        useSatelliteImagery: true,
                        initialDetectionOnly: true  // Flag to indicate we only want to open the sidebar
                      }
                    });
                    window.dispatchEvent(validationEvent);
                    
                    // One more resize after a bit longer delay to ensure everything has loaded
                    setTimeout(() => {
                      window.dispatchEvent(new Event('resize'));
                    }, 300);
                  }, 100); // Short delay
                }}
                className="flex items-center justify-center p-2 bg-blue-50 text-blue-600 hover:bg-blue-100 rounded-md cursor-pointer"
              >
                <Eye className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Detection</span>}
              </button>
            </div>
            
            {/* Second row - Database and Response buttons side-by-side */}
            <div className="grid grid-cols-2 gap-2">
              {/* Database Button */}
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
                      target: 'tree_inventory',
                      clearExisting: true  // Changed to true to fully reset
                    }
                  }));
                  
                  // Make sure Object Detection sidebar is closed
                  window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                    detail: { source: 'sidebar' }
                  }));
                  
                  // Short delay to ensure other sidebars close
                  setTimeout(() => {
                    // Then open feature selection with Index tab active
                    window.dispatchEvent(new CustomEvent('openFeatureSelection', {
                      detail: { 
                        mode: 'tree_inventory', 
                        clearExisting: true,
                        tab: 'trees' // Specify Index tab
                      }
                    }));
                  }, 100); // Short delay
                }}
                className="flex items-center justify-center p-2 bg-emerald-50 text-emerald-600 hover:bg-emerald-100 rounded-md cursor-pointer"
              >
                <Database className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Database</span>}
              </button>
              
              {/* Response Viewer Button */}
              <button
                onClick={() => {
                  // Show a prompt to get the job ID
                  const jobId = prompt("Enter the detection job ID to view Gemini response:", "");
                  if (!jobId) return;
                  
                  // Create a response directory path
                  const responsePath = `/ttt/data/temp/${jobId}/gemini_response`;
                  
                  // Execute commands to view the response
                  alert(`Viewing response files for job ${jobId}.\nLook in ${responsePath}`);
                  
                  // Create a custom event to open a response viewer
                  const viewResponseEvent = new CustomEvent('viewGeminiResponse', {
                    detail: {
                      jobId: jobId,
                      responsePath: responsePath
                    }
                  });
                  window.dispatchEvent(viewResponseEvent);
                }}
                className="flex items-center justify-center p-2 bg-purple-50 text-purple-600 hover:bg-purple-100 rounded-md cursor-pointer"
              >
                <BarChart className={`h-4 w-4 ${!collapsed && 'mr-1'}`} />
                {!collapsed && <span className="text-xs">Analytics</span>}
              </button>
            </div>
            
            {/* Third row - Reports Summary button (full width) */}
            <div className="mt-2">
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
                      target: 'tree_inventory',
                      clearExisting: true
                    }
                  }));
                  
                  // Make sure Object Detection sidebar is closed
                  window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                    detail: { source: 'sidebar' }
                  }));
                  
                  // Short delay to ensure other sidebars close
                  setTimeout(() => {
                    // Open feature selection with reports tab active
                    window.dispatchEvent(new CustomEvent('openFeatureSelection', {
                      detail: { 
                        mode: 'tree_inventory', 
                        clearExisting: true,
                        tab: 'reports' // Specify reports tab
                      }
                    }));
                  }, 100); // Short delay
                }}
                className="flex items-center justify-center w-full p-2 bg-gray-50 text-gray-600 hover:bg-gray-100 rounded-md"
              >
                <FileText className={`h-4 w-4 ${!collapsed && 'mr-2'}`} />
                {!collapsed && <span className="text-xs font-medium">Reports</span>}
              </button>
            </div>
          </div>
        </div>
        
        {/* Bottom border/divider - empty div to maintain spacing */}
        <div className="mt-auto border-t border-gray-200"></div>
      </div>
    </div>
  );
};

export default Sidebar;