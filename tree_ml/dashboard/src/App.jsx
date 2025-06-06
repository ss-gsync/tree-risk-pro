// src/App.jsx
import React, { Suspense, useRef, useState, useEffect, createContext, useContext, useMemo, Component } from 'react';
import { Provider } from 'react-redux';
import { store } from './store';
import Layout from './components/common/Layout';
import LoadingSpinner from './components/common/Loading';
import MapView from './components/visualization/MapView/MapView.jsx';
import ValidationQueue from './components/assessment/Validation/ValidationQueue';
import ValidationSystem from './components/assessment/Validation/ValidationSystem';
import MapAssessmentPanel from './components/assessment/MapAssessmentPanel';
import AnalyticsPanel from './components/analytics/AnalyticsPanel';
import Login from './components/auth/Login';
import { AuthProvider, useAuth } from './components/auth/AuthContext';
import { 
  Layers, 
  Globe, 
  ChevronRight,
  ChevronLeft,
  X,
  Box, 
  BarChart, 
  Map, 
  FileText, 
  Database,
  CheckCircle,
  ClipboardList,
  Eye,
  Search
} from 'lucide-react';

// Import the settings panel
import SettingsPanel from './components/settings/SettingsPanel';

// Create a context for the global validation system
const ValidationContext = createContext();

// ValidationSystemProvider component
const ValidationSystemProvider = ({ children }) => {
  const [validationData, setValidationData] = useState(null);
  
  // Listen for showValidationSystem events
  useEffect(() => {
    const handleShowValidation = (event) => {
      console.log('Validation system triggered with data:', event.detail);
      setValidationData(event.detail);
    };
    
    window.addEventListener('showValidationSystem', handleShowValidation);
    
    return () => {
      window.removeEventListener('showValidationSystem', handleShowValidation);
    };
  }, []);
  
  const closeValidation = () => {
    console.log('Closing validation system');
    setValidationData(null);
    window.validationSystemActive = false;
  };
  
  return (
    <ValidationContext.Provider value={{ validationData, setValidationData, closeValidation }}>
      {children}
      {validationData && (
        <ValidationSystem 
          selectedTree={validationData.tree} 
          onClose={closeValidation} 
        />
      )}
    </ValidationContext.Provider>
  );
};

// Hook to use validation context
const useValidationSystem = () => useContext(ValidationContext);

// Base sidebar component for reuse
const ResizableSidebar = ({
  id,
  title,
  icon: Icon,
  color = "blue",
  children,
  openEventName,
  closeEventName,
  otherEvents = [],
  defaultOpen = false // New parameter to control initial state
}) => {
  const [collapsed, setCollapsed] = useState(!defaultOpen);
  const [width, setWidth] = useState(384); // Default width 384px
  const [isDragging, setIsDragging] = useState(false);
  const [headerCollapsed, setHeaderCollapsed] = useState(false); // Track header state - default expanded
  
  // Handle mousedown on the resize handle
  const handleMouseDown = (e) => {
    setIsDragging(true);
    e.preventDefault();
  };
  
  // Listen for header collapse/expand events and directly update DOM
  React.useEffect(() => {
    const handleHeaderCollapse = (event) => {
      const isCollapsed = event.detail.collapsed;
      setHeaderCollapsed(isCollapsed);
      
      // Also update all sidebar positions immediately to be responsive to header changes
      // This ensures sidebars don't overlap the header when it expands or collapses
      const sidebarElements = document.querySelectorAll('[id$="-sidebar"]');
      sidebarElements.forEach(sidebar => {
        if (sidebar) {
          try {
            sidebar.style.top = isCollapsed ? '40px' : '64px';
          } catch (e) {
            console.warn("Could not update sidebar position:", e);
          }
        }
      });
    };
    
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    return () => {
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
    };
  }, []);
  
  // Handle mouse movement for resize
  React.useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging || collapsed) return;
      
      // Calculate new width
      const newWidth = window.innerWidth - e.clientX;
      
      // Set minimum and maximum width
      const clampedWidth = Math.min(Math.max(newWidth, 300), 600);
      
      // Update DOM immediately for smooth drag
      const sidebar = document.getElementById(id);
      const mapContainer = document.getElementById('map-container');
      const reportContainer = document.getElementById('report-container');
      
      if (sidebar) {
        // Apply new width directly to the sidebar element
        sidebar.style.width = `${clampedWidth}px`;
      }
      
      // Always adjust relevant containers regardless of which sidebar is active
      if (mapContainer && document.body.contains(mapContainer)) {
        mapContainer.style.right = `${clampedWidth}px`;
      }
      
      if (reportContainer && document.body.contains(reportContainer)) {
        reportContainer.style.right = `${clampedWidth}px`;
      }
      
      // Update state after DOM manipulation for smoother experience
      setWidth(clampedWidth);
      
      // Mark this sidebar as open for tracking
      sidebar.classList.add('sidebar-open');
    };
    
    const handleMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
        
        // Final update of sidebar width
        const sidebar = document.getElementById(id);
        const mapContainer = document.getElementById('map-container');
        const reportContainer = document.getElementById('report-container');
        
        if (sidebar) {
          // Update sidebar width
          sidebar.style.width = `${width}px`;
          
          // Always update both containers if they exist
          if (mapContainer && document.body.contains(mapContainer)) {
            mapContainer.style.right = `${width}px`;
          }
          
          if (reportContainer && document.body.contains(reportContainer)) {
            reportContainer.style.right = `${width}px`;
          }
        }
        
        // Trigger window resize to update all components that depend on window size
        window.dispatchEvent(new Event('resize'));
      }
    };
    
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, collapsed, width, id]);
  
  // Close all other sidebars and open this one
  const openSidebar = () => {
    // Close all other sidebars
    const allSidebarEvents = [
      'closeReviewPanel',
      'forceCloseImageryPanel',
      'closeAnalyticsPanel',
      'forceCloseTreeDatabase',
      'forceCloseObjectDetection'
    ];
    
    // Dispatch all close events except our own
    allSidebarEvents.forEach(event => {
      if (event !== closeEventName) {
        window.dispatchEvent(new CustomEvent(event, {
          detail: { source: id }
        }));
      }
    });
    
    // Open this sidebar
    setCollapsed(false);
    
    // Update DOM
    const sidebar = document.getElementById(id);
    const mapContainer = document.getElementById('map-container');
    const reportContainer = document.getElementById('report-container');
    
    if (sidebar) {
      sidebar.style.width = `${width}px`;
      
      // Ensure all content elements also get proper width
      const contentElements = sidebar.querySelectorAll('.sidebar-content');
      if (contentElements.length > 0) {
        contentElements.forEach(el => {
          el.style.width = `${width}px`;
        });
      }
    }
    
    // Always update both containers if they exist
    if (mapContainer && document.body.contains(mapContainer)) {
      mapContainer.style.right = `${width}px`;
    }
    
    if (reportContainer && document.body.contains(reportContainer)) {
      reportContainer.style.right = `${width}px`;
    }
    
    // Mark this sidebar as open
    if (sidebar) {
      sidebar.classList.add('sidebar-open');
    }
    
    // Trigger resize event
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
      // Force a second resize after elements have fully rendered
      setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
    }, 50);
  };
  
  // Close this sidebar
  const closeSidebar = () => {
    setCollapsed(true);
    
    // Update DOM
    const sidebar = document.getElementById(id);
    const mapContainer = document.getElementById('map-container');
    const reportContainer = document.getElementById('report-container');
    
    if (sidebar) {
      sidebar.style.width = '0px';
      
      // Also set content elements width to 0
      const contentElements = sidebar.querySelectorAll('.sidebar-content');
      if (contentElements.length > 0) {
        contentElements.forEach(el => {
          el.style.width = '0px';
        });
      }
      
      // If this is the database sidebar, ensure it's completely hidden
      if (id === 'database-sidebar') {
        sidebar.style.display = 'none';
        setTimeout(() => {
          sidebar.style.display = 'block';
          sidebar.style.width = '0px';
        }, 100);
      }
    }
    
    // Check if any sidebars are still open
    setTimeout(() => {
      // Check by examining the rendered DOM directly
      const visibleSidebars = Array.from(document.querySelectorAll('[id$="-sidebar"]'))
        .filter(el => el.offsetWidth > 0 && el.style.width !== '0px');
      
      // If no sidebars are visible, reset the container widths
      if (visibleSidebars.length === 0) {
        if (mapContainer) {
          mapContainer.style.right = '0px';
          console.log('Resetting mapContainer right margin to 0px');
        }
        if (reportContainer) {
          reportContainer.style.right = '0px';
          console.log('Resetting reportContainer right margin to 0px');
        }
      }
      
      // Force resize events with delays to ensure proper layout updates
      window.dispatchEvent(new Event('resize'));
      setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
      setTimeout(() => window.dispatchEvent(new Event('resize')), 250);
    }, 50);
  };
  
  // Listen for events
  React.useEffect(() => {
    // Event handler for opening
    const handleOpen = (event) => {
      if (collapsed) {
        openSidebar();
      }
    };
    
    // Event handler for closing
    const handleClose = (event) => {
      if (!collapsed) {
        closeSidebar();
      }
    };
    
    // Handle other sidebars opening
    const handleOtherOpen = (event) => {
      if (!collapsed && event.detail.source !== id) {
        closeSidebar();
      }
    };
    
    // Add event listeners
    window.addEventListener(openEventName, handleOpen);
    window.addEventListener(closeEventName, handleClose);
    
    // Add listeners for other sidebar events
    const openEvents = [
      'openReviewPanel',
      'expandImageryPanel',
      'openAnalyticsPanel',
      'openFeatureSelection',
      'enterTreeValidationMode'
    ];
    
    openEvents.forEach(event => {
      if (event !== openEventName) {
        window.addEventListener(event, handleOtherOpen);
      }
    });
    
    // Add other custom events
    otherEvents.forEach(event => {
      window.addEventListener(event, handleOtherOpen);
    });
    
    // Cleanup
    return () => {
      window.removeEventListener(openEventName, handleOpen);
      window.removeEventListener(closeEventName, handleClose);
      
      openEvents.forEach(event => {
        if (event !== openEventName) {
          window.removeEventListener(event, handleOtherOpen);
        }
      });
      
      otherEvents.forEach(event => {
        window.removeEventListener(event, handleOtherOpen);
      });
    };
  }, [collapsed, width, id, openEventName, closeEventName, otherEvents]);
  
  // Initialize sidebar position based on header state (this runs once immediately)
  React.useEffect(() => {
    if (!collapsed) {
      // If sidebar is open on initial render, make sure it's positioned correctly
      const header = document.querySelector('header');
      const headerHeight = header ? header.offsetHeight : 64;
      const sidebar = document.getElementById(id);
      
      if (sidebar) {
        try {
          sidebar.style.top = `${headerHeight}px`;
        } catch (e) {
          console.warn("Could not set initial sidebar position:", e);
        }
      }
      
      // If initially open, make sure to update all containers
      const mapContainer = document.getElementById('map-container');
      const reportContainer = document.getElementById('report-container');
      
      if (mapContainer && document.body.contains(mapContainer)) {
        mapContainer.style.right = `${width}px`;
      }
      
      if (reportContainer && document.body.contains(reportContainer)) {
        reportContainer.style.right = `${width}px`;
      }
      
      // Trigger resize events to ensure proper layout
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 50);
    }
  }, []);
  
  // Only render content if not collapsed
  if (collapsed) return null;
  
  // Get the appropriate background color based on the color prop
  const getColorClasses = () => {
    switch (color) {
      case 'blue':
        return 'bg-blue-600 text-white hover:bg-blue-500';
      case 'indigo':
        return 'bg-indigo-600 text-white hover:bg-indigo-500';
      case 'purple':
        return 'bg-purple-600 text-white hover:bg-purple-500';
      case 'dark-purple':
        return 'bg-gray-100 text-purple-900 border-b border-purple-200/50 shadow-sm hover:bg-gray-200';
      case 'gray':
        return 'bg-white text-gray-800 border-b border-gray-300 shadow-sm hover:bg-gray-50';
      case 'emerald':
        return 'bg-emerald-600 text-white hover:bg-emerald-500';
      case 'white':
        return 'bg-white text-blue-600 border-b border-gray-200 hover:bg-gray-50';
      case 'dark-indigo':
        return 'bg-indigo-50 text-indigo-900 border-b border-indigo-200 shadow-sm hover:bg-indigo-100';
      case 'dark-green':
        return 'bg-green-50 text-green-900 border-b border-green-200 shadow-sm hover:bg-green-100';
      default:
        return 'bg-blue-600 text-white hover:bg-blue-500';
    }
  };
  
  return (
    <div 
      id={id} 
      className={`fixed right-0 bottom-0 bg-white shadow-md transition-all duration-300 ease-in-out z-20 overflow-hidden`}
      style={{ 
        width: `${width}px`, 
        maxWidth: "600px",
        top: headerCollapsed ? '40px' : '64px' // Use pixel values for more reliable positioning
      }}
    >
      {/* Resize handle - standardized design across all sidebars */}
      <div 
        className="absolute top-0 bottom-0 left-0 w-4 cursor-col-resize z-50 hover:bg-gray-100/30"
        onMouseDown={handleMouseDown}
        style={{
          background: isDragging ? 'rgba(59, 130, 246, 0.1)' : 'transparent'
        }}
      >
        <div className="absolute top-0 bottom-0 left-0 w-0.5 bg-gray-300"></div>
      </div>
      
      <div className="h-full flex flex-col w-full">
        <div className={`py-1.5 px-3 border-b ${getColorClasses()}`}>
          <div className="flex justify-between items-center">
            <div className={`font-medium text-sm flex items-center ${
              color === 'white' || color === 'gray' ? 'text-gray-800' : 
              color === 'dark-purple' ? 'text-purple-900' :
              color === 'dark-indigo' ? 'text-indigo-900' :
              color === 'dark-green' ? 'text-green-900' :
              'text-white'
            }`}>
              {Icon && <Icon className={`h-4 w-4 mr-1.5 ${
                color === 'dark-purple' ? 'text-purple-900' :
                color === 'dark-indigo' ? 'text-indigo-900' :
                color === 'dark-green' ? 'text-green-900' :
                color === 'white' || color === 'gray' ? 'text-gray-800 opacity-90' : 
                'text-white opacity-90'
              }`} />}
              {title}
            </div>
            <button
              onClick={closeSidebar}
              className={`p-1 rounded-md ${
                color === 'white' || color === 'gray' ? 'hover:bg-gray-200 text-gray-700' :
                color === 'dark-purple' ? 'hover:bg-gray-200 text-purple-900' :
                color === 'dark-indigo' ? 'hover:bg-indigo-100 text-indigo-900' :
                color === 'dark-green' ? 'hover:bg-green-100 text-green-900' :
                'hover:bg-white/20 text-white'
              }`}
              aria-label="Close sidebar"
              title="Close"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        
        <div className="flex-1 overflow-auto" style={{ width: '100%', boxSizing: 'border-box' }}>
          {children}
        </div>
      </div>
    </div>
  );
};

// Sidebar Components

// Detection sidebar component (placeholder)
// IMPORTANT: The actual component is managed by DetectionMode in MapView.jsx
// This placeholder exists only for API compatibility but should never render
const DetectionSidebar = () => {
  return null; // Always return null - never render this placeholder
};

// Aerial sidebar component
const AerialImagery = () => {
  // Separate component to ensure proper rendering
  const AerialContent = () => (
    <div className="bg-white overflow-auto h-full">
      <div className="p-4">
        <div className="mb-4 bg-slate-50 p-3 rounded-md border border-slate-200">
          <h3 className="text-sm font-medium text-slate-700 mb-1">Raw Data Input</h3>
          <p className="text-xs text-slate-500">Upload satellite or lidar imagery for analysis</p>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="aspect-video bg-slate-100 rounded-md flex items-center justify-center text-slate-400 border border-slate-200">
            <p className="text-xs">Aerial Image 1</p>
          </div>
          <div className="aspect-video bg-slate-100 rounded-md flex items-center justify-center text-slate-400 border border-slate-200">
            <p className="text-xs">Lidar View 1</p>
          </div>
          <div className="aspect-video bg-slate-100 rounded-md flex items-center justify-center text-slate-400 border border-slate-200">
            <p className="text-xs">Aerial Image 2</p>
          </div>
          <div className="aspect-video bg-slate-100 rounded-md flex items-center justify-center text-slate-400 border border-slate-200">
            <p className="text-xs">Lidar View 2</p>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <ResizableSidebar 
      id="imagery-sidebar"
      title="Aerial Imagery"
      icon={Box}
      color="dark-indigo" // Using custom color for Aerial sidebar
      openEventName="expandImageryPanel"
      closeEventName="forceCloseImageryPanel"
      defaultOpen={true} // Make this sidebar open by default
    >
      <AerialContent />
    </ResizableSidebar>
  );
};

// Analytics sidebar component
const AnalyticsSidebar = () => {
  // Use the existing AnalyticsPanel component for consistent behavior
  return (
    <ResizableSidebar
      id="analytics-sidebar"
      title="Analytics"
      icon={BarChart}
      color="dark-purple"
      openEventName="openAnalyticsPanel"
      closeEventName="closeAnalyticsPanel"
    >
      <AnalyticsPanel />
    </ResizableSidebar>
  );
};

// Database sidebar component
const DatabaseSidebar = () => {
  // Database content with object details view
  const DatabaseContent = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [activeFilter, setActiveFilter] = useState('all');
    const [region, setRegion] = useState('all');
    const [selectedTab, setSelectedTab] = useState('trees');
    const [selectedObject, setSelectedObject] = useState(null);
    
    // Mock data for tree objects
    const mockTreeObjects = useMemo(() => {
      return Array(5).fill(null).map((_, i) => ({
        id: `tree_${i}_${Math.random().toString(36).substring(7)}`,
        type: 'tree',
        name: `Oak Tree ${i + 1}`,
        species: ['Red Oak', 'White Oak', 'Maple', 'Pine', 'Birch'][i % 5],
        region: ['North', 'Central', 'East', 'West', 'South'][i % 5],
        regionCode: ['S2-N1', 'S2-C3', 'S2-E2', 'S2-W4', 'S2-S5'][i % 5],
        height: 20 + Math.floor(Math.random() * 40),
        diameter: 10 + Math.floor(Math.random() * 20),
        riskLevel: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
        validated: i % 3 === 0,
        lastUpdated: new Date(2025, 4, 1 + i).toISOString().split('T')[0],
        metadata: {
          longitude: -96.8 - (Math.random() * 0.2),
          latitude: 32.8 + (Math.random() * 0.2),
          path: `/data/zarr/trees/${i}`,
          reportIds: [`rep-00${i + 1}`, `rep-00${i + 2}`],
          detections: Math.floor(Math.random() * 5) + 1
        }
      }));
    }, []);
    
    // Mock data for report objects
    const mockReportObjects = useMemo(() => {
      return Array(3).fill(null).map((_, i) => ({
        id: `rep-00${i + 1}`,
        type: 'report',
        title: [`North Sector Assessment`, `Central District Tree Health`, `South Region Analysis`][i],
        region: ['North', 'Central', 'South'][i],
        regionCode: ['S2-N1', 'S2-C3', 'S2-S5'][i],
        date: new Date(2025, 4, 1 + i).toISOString().split('T')[0],
        status: ['complete', 'in_progress', 'complete'][i],
        trees: [87, 143, 56][i],
        high_risk: [12, 23, 4][i],
        validated: [true, false, true][i],
        path: `/data/zarr/reports/${i}`,
      }));
    }, []);
    
    // Combined data based on active tab
    const activeObjects = useMemo(() => {
      let objects = selectedTab === 'trees' ? mockTreeObjects : 
                   selectedTab === 'reports' ? mockReportObjects : [];
                   
      // Apply filters
      if (activeFilter === 'validated') {
        objects = objects.filter(obj => obj.validated);
      } else if (activeFilter === 'high-risk') {
        objects = objects.filter(obj => obj.riskLevel === 'high' || 
                                      (obj.high_risk && obj.high_risk > 10));
      } else if (activeFilter === 'recent') {
        // Sort by date (if reports) or lastUpdated (if trees)
        objects.sort((a, b) => {
          const dateA = a.date || a.lastUpdated;
          const dateB = b.date || b.lastUpdated;
          return new Date(dateB) - new Date(dateA);
        });
      }
      
      // Apply region filter
      if (region !== 'all') {
        const regionPrefix = region.charAt(0).toUpperCase() + region.slice(1);
        objects = objects.filter(obj => 
          obj.region === regionPrefix || 
          (obj.regionCode && obj.regionCode.includes(regionPrefix.substring(0, 1).toUpperCase()))
        );
      }
      
      // Apply search filter
      if (searchQuery.trim()) {
        const query = searchQuery.toLowerCase();
        objects = objects.filter(obj => 
          obj.name?.toLowerCase().includes(query) || 
          obj.title?.toLowerCase().includes(query) || 
          obj.species?.toLowerCase().includes(query) ||
          obj.region?.toLowerCase().includes(query) ||
          obj.regionCode?.toLowerCase().includes(query) ||
          obj.id?.toLowerCase().includes(query)
        );
      }
      
      return objects;
    }, [selectedTab, mockTreeObjects, mockReportObjects, activeFilter, region, searchQuery]);
    
    // Function to view object details in the main view
    const viewObjectDetails = (object) => {
      // First show the object in the sidebar
      setSelectedObject(object);
      
      // Then trigger an event to show the object in the main view
      window.dispatchEvent(new CustomEvent('showObjectReport', {
        detail: { 
          object,
          source: 'database'
        }
      }));
    };
    
    // Function to return to object list view
    const backToList = () => {
      setSelectedObject(null);
    };
    
    return (
      <div className="bg-white overflow-auto h-full">
        {selectedObject ? (
          // Object details view
          <div className="p-4">
            <button 
              onClick={backToList}
              className="flex items-center text-green-700 mb-4 hover:text-green-800"
            >
              <ChevronLeft className="h-4 w-4 mr-1" /> Back to list
            </button>
            
            <div className="bg-green-50 p-3 rounded-lg border border-green-200 mb-4">
              <div className="flex justify-between items-start">
                <h3 className="text-base font-medium text-green-900">
                  {selectedObject.name || selectedObject.title}
                </h3>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  selectedObject.status === 'complete' || selectedObject.validated ? 
                    'bg-green-100 text-green-800 border border-green-200' : 
                  selectedObject.riskLevel === 'high' ?
                    'bg-red-100 text-red-800 border border-red-200' :
                    'bg-gray-100 text-gray-600 border border-gray-200'
                }`}>
                  {selectedObject.status === 'complete' || selectedObject.validated ? 'Validated' :
                   selectedObject.status === 'in_progress' ? 'In Progress' :
                   selectedObject.riskLevel === 'high' ? 'High Risk' : 'Pending'}
                </span>
              </div>
              <div className="text-sm text-green-800 mt-1">ID: {selectedObject.id}</div>
            </div>
            
            <div className="space-y-4">
              {/* Object metadata */}
              <div className="bg-white p-3 rounded-lg border border-gray-200">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Details</h4>
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  {selectedObject.type === 'tree' && (
                    <>
                      <div className="text-xs">
                        <span className="text-gray-500">Species:</span>{' '}
                        <span className="font-medium">{selectedObject.species}</span>
                      </div>
                      <div className="text-xs">
                        <span className="text-gray-500">Height:</span>{' '}
                        <span className="font-medium">{selectedObject.height} ft</span>
                      </div>
                      <div className="text-xs">
                        <span className="text-gray-500">Diameter:</span>{' '}
                        <span className="font-medium">{selectedObject.diameter} inches</span>
                      </div>
                      <div className="text-xs">
                        <span className="text-gray-500">Risk Level:</span>{' '}
                        <span className={`font-medium ${
                          selectedObject.riskLevel === 'high' ? 'text-red-600' :
                          selectedObject.riskLevel === 'medium' ? 'text-yellow-600' :
                          'text-green-600'
                        }`}>
                          {selectedObject.riskLevel.charAt(0).toUpperCase() + selectedObject.riskLevel.slice(1)}
                        </span>
                      </div>
                    </>
                  )}
                  
                  {selectedObject.type === 'report' && (
                    <>
                      <div className="text-xs">
                        <span className="text-gray-500">Trees:</span>{' '}
                        <span className="font-medium">{selectedObject.trees}</span>
                      </div>
                      <div className="text-xs">
                        <span className="text-gray-500">High Risk:</span>{' '}
                        <span className="font-medium text-red-600">{selectedObject.high_risk}</span>
                      </div>
                      <div className="text-xs">
                        <span className="text-gray-500">Health Rate:</span>{' '}
                        <span className="font-medium text-green-600">
                          {Math.round((selectedObject.trees - selectedObject.high_risk) / selectedObject.trees * 100)}%
                        </span>
                      </div>
                      <div className="text-xs">
                        <span className="text-gray-500">Status:</span>{' '}
                        <span className={`font-medium ${
                          selectedObject.status === 'complete' ? 'text-green-600' :
                          selectedObject.status === 'in_progress' ? 'text-yellow-600' :
                          'text-gray-600'
                        }`}>
                          {selectedObject.status === 'complete' ? 'Complete' :
                           selectedObject.status === 'in_progress' ? 'In Progress' : 'Pending'}
                        </span>
                      </div>
                    </>
                  )}
                  
                  <div className="text-xs">
                    <span className="text-gray-500">Region:</span>{' '}
                    <span className="font-medium">{selectedObject.region} ({selectedObject.regionCode})</span>
                  </div>
                  <div className="text-xs">
                    <span className="text-gray-500">Last Updated:</span>{' '}
                    <span className="font-medium">{selectedObject.lastUpdated || selectedObject.date}</span>
                  </div>
                </div>
              </div>
              
              {/* Storage path */}
              <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                <h4 className="text-xs font-medium text-gray-700 mb-1">Storage Path</h4>
                <div className="font-mono text-xs text-gray-600 bg-white p-1.5 rounded border border-gray-200">
                  {selectedObject.path || selectedObject.metadata?.path || `/data/zarr/${selectedObject.type}s/${selectedObject.id}`}
                </div>
              </div>
              
              {/* Actions */}
              <div className="flex gap-2 pt-2">
                <button 
                  onClick={() => viewObjectDetails(selectedObject)}
                  className="flex-1 bg-green-50 hover:bg-green-100 text-green-900 border border-green-200 rounded-md py-2 text-sm font-medium"
                >
                  View Report
                </button>
                <button 
                  onClick={() => window.dispatchEvent(new CustomEvent('showOnMap', {
                    detail: { 
                      object: selectedObject,
                      source: 'database'
                    }
                  }))}
                  className="flex-1 bg-gray-50 hover:bg-gray-100 text-gray-800 border border-gray-200 rounded-md py-2 text-sm font-medium"
                >
                  Show on Map
                </button>
              </div>
            </div>
          </div>
        ) : (
          // Database list view
          <div className="p-4">
            {/* Search bar */}
            <div className="mb-4 relative">
              <input
                type="text"
                placeholder="Search database..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full p-2 pl-8 pr-4 border border-gray-200 bg-gray-50 rounded-md shadow-sm focus:ring-green-500 focus:border-green-500 focus:bg-white text-sm"
              />
              <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-400" />
            </div>
            
            {/* Tab selector */}
            <div className="mb-4 border-b border-gray-200">
              <nav className="-mb-px flex space-x-4">
                <button
                  onClick={() => setSelectedTab('trees')}
                  className={`whitespace-nowrap pb-2 px-1 border-b-2 font-medium text-sm ${
                    selectedTab === 'trees' 
                      ? 'border-green-500 text-green-600' 
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Trees
                </button>
                <button
                  onClick={() => setSelectedTab('reports')}
                  className={`whitespace-nowrap pb-2 px-1 border-b-2 font-medium text-sm ${
                    selectedTab === 'reports' 
                      ? 'border-green-500 text-green-600' 
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Reports
                </button>
                <button
                  onClick={() => setSelectedTab('analytics')}
                  className={`whitespace-nowrap pb-2 px-1 border-b-2 font-medium text-sm ${
                    selectedTab === 'analytics' 
                      ? 'border-green-500 text-green-600' 
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Analytics
                </button>
              </nav>
            </div>
            
            {/* Filter bar */}
            <div className="mb-4 flex flex-wrap gap-2">
              <button 
                onClick={() => setActiveFilter('all')}
                className={`px-2 py-1 text-xs rounded-md ${
                  activeFilter === 'all' 
                    ? 'bg-green-100 text-green-800 border border-green-300' 
                    : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                }`}
              >
                All
              </button>
              <button 
                onClick={() => setActiveFilter('validated')}
                className={`px-2 py-1 text-xs rounded-md ${
                  activeFilter === 'validated' 
                    ? 'bg-green-100 text-green-800 border border-green-300' 
                    : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                }`}
              >
                Validated
              </button>
              <button 
                onClick={() => setActiveFilter('high-risk')}
                className={`px-2 py-1 text-xs rounded-md ${
                  activeFilter === 'high-risk' 
                    ? 'bg-red-100 text-red-800 border border-red-300' 
                    : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                }`}
              >
                High Risk
              </button>
              <button 
                onClick={() => setActiveFilter('recent')}
                className={`px-2 py-1 text-xs rounded-md ${
                  activeFilter === 'recent' 
                    ? 'bg-blue-100 text-blue-800 border border-blue-300' 
                    : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                }`}
              >
                Recent
              </button>
            </div>
            
            {/* Region filter dropdown */}
            <div className="mb-4">
              <label className="block text-xs font-medium text-gray-700 mb-1">S2 Region</label>
              <select
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                className="w-full rounded-md border border-gray-200 bg-gray-50 shadow-sm focus:ring-green-500 focus:border-green-500 focus:bg-white text-sm p-2"
              >
                <option value="all">All Regions</option>
                <option value="north">North Sector (S2-N1)</option>
                <option value="central">Central Sector (S2-C3)</option>
                <option value="east">East Sector (S2-E2)</option>
                <option value="west">West Sector (S2-W4)</option>
                <option value="south">South Sector (S2-S5)</option>
              </select>
            </div>
            
            {/* Database content */}
            <div className="mb-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">Data Storage</h3>
              <div className="grid grid-cols-2 gap-2 mb-3">
                <div className="bg-gray-50 p-2 rounded-md border border-gray-200">
                  <h4 className="text-xs font-medium text-gray-600">Zarr Store</h4>
                  <p className="text-xs text-gray-500">/data/zarr/{selectedTab}/</p>
                </div>
                <div className="bg-gray-50 p-2 rounded-md border border-gray-200">
                  <h4 className="text-xs font-medium text-gray-600">Cache</h4>
                  <p className="text-xs text-gray-500">/data/ml/</p>
                </div>
              </div>
            </div>
            
            {/* Object list based on selected tab */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <h3 className="text-sm font-medium text-gray-700">
                  {selectedTab === 'trees' ? 'Tree Index' : 
                   selectedTab === 'reports' ? 'Reports' : 'Analytics'}
                </h3>
                <span className="text-xs bg-gray-100 px-2 py-0.5 rounded-full text-gray-600">
                  {activeObjects.length} entries
                </span>
              </div>
              
              <div className="space-y-2 max-h-96 overflow-auto">
                {activeObjects.length === 0 ? (
                  <div className="p-4 text-center text-gray-500 text-sm">
                    No items found for the current filter.
                  </div>
                ) : (
                  activeObjects.map((obj) => (
                    <div 
                      key={obj.id} 
                      onClick={() => viewObjectDetails(obj)}
                      className="p-2 bg-white border border-gray-200 rounded-md hover:bg-gray-50 cursor-pointer"
                    >
                      <div className="flex justify-between">
                        <h4 className="text-xs font-medium text-gray-800">
                          {obj.name || obj.title}
                        </h4>
                        {obj.validated || obj.status === 'complete' ? (
                          <span className="text-xs bg-green-100 text-green-700 px-1.5 py-0.5 rounded">Validated</span>
                        ) : obj.riskLevel === 'high' || (obj.high_risk && obj.high_risk > obj.trees * 0.1) ? (
                          <span className="text-xs bg-red-100 text-red-700 px-1.5 py-0.5 rounded">High Risk</span>
                        ) : obj.status === 'in_progress' ? (
                          <span className="text-xs bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded">In Progress</span>
                        ) : (
                          <span className="text-xs bg-gray-100 text-gray-700 px-1.5 py-0.5 rounded">Pending</span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">ID: {obj.id}</div>
                      <div className="text-xs text-gray-500">
                        Region: {obj.region} {obj.regionCode ? `(${obj.regionCode})` : ''}
                      </div>
                      <div className="text-xs text-gray-500">
                        {obj.type === 'tree' ? (
                          `${obj.species || 'Unknown'}, ${obj.height || '0'} ft`
                        ) : obj.type === 'report' ? (
                          `${obj.trees} trees, ${obj.high_risk} high risk`
                        ) : (
                          `Last updated: ${obj.date || obj.lastUpdated}`
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <ResizableSidebar 
      id="database-sidebar"
      title="Database"
      icon={Database}
      color="dark-green" // Using custom color for Database sidebar
      openEventName="openFeatureSelection"
      closeEventName="forceCloseTreeDatabase"
    >
      <DatabaseContent />
    </ResizableSidebar>
  );
};

// Validation sidebar component
const ValidationSidebar = () => {
  // We use standalone=false to ensure ValidationQueue knows it's embedded
  // in the ResizableSidebar component and doesn't try to manage sidebar state itself
  return (
    <ResizableSidebar
      id="validation-sidebar"
      title="Validation Queue"
      icon={CheckCircle}
      color="gray" // Using gray for the header background
      openEventName="openReviewPanel"
      closeEventName="closeReviewPanel"
    >
      <ValidationQueue standalone={false} />
    </ResizableSidebar>
  );
};

// Object Report component for showing details of a selected database object 
const ObjectReport = ({ object, onBackToReports }) => {
  const [linkedReports, setLinkedReports] = useState([]);
  const [isLoadingLinked, setIsLoadingLinked] = useState(false);
  const [validationReports, setValidationReports] = useState([]);
  const [isLoadingValidation, setIsLoadingValidation] = useState(false);
  
  // Fetch linked reports for area reports using S2 index
  useEffect(() => {
    if (object?.type === 'report') {
      const fetchLinkedReports = async () => {
        try {
          setIsLoadingLinked(true);
          
          // For demonstration purposes, generate mock linked reports
          // In production, this would be fetched from API using reportApi.getLinkedValidationReports
          const mockLinkedReports = [
            {
              id: `validation-${object.id}-1`,
              title: `Validation Report for ${object.title} (South)`,
              region: object.region,
              subregion: 'South',
              date: new Date(new Date(object.date).getTime() + 86400000 * 2).toISOString().split('T')[0], // 2 days after
              status: 'complete',
              trees_validated: Math.floor(object.trees * 0.3),
              high_risk_confirmed: Math.floor(object.high_risk * 0.35)
            },
            {
              id: `validation-${object.id}-2`,
              title: `Validation Report for ${object.title} (North)`,
              region: object.region,
              subregion: 'North',
              date: new Date(new Date(object.date).getTime() + 86400000 * 3).toISOString().split('T')[0], // 3 days after
              status: 'in_progress',
              trees_validated: Math.floor(object.trees * 0.2),
              high_risk_confirmed: Math.floor(object.high_risk * 0.22)
            }
          ];
          
          setLinkedReports(mockLinkedReports);
        } catch (error) {
          console.error('Error fetching linked reports:', error);
        } finally {
          setIsLoadingLinked(false);
        }
      };
      
      // Fetch S2 cell validation reports if this is an area report
      const fetchValidationReportsByS2 = async () => {
        try {
          setIsLoadingValidation(true);
          if (!object.regionCode) return;
          
          // Extract the S2 cell token from the region code
          // In a real implementation, this would be a call to reportApi.getValidationReportsByS2Cell
          const s2CellToken = object.regionCode.replace('S2-', '');
          
          // Mock validation reports by S2 cell
          const mockValidationReports = [
            {
              id: `s2-validation-${s2CellToken}-1`,
              title: `Validation in ${s2CellToken} Cell - East`,
              type: 'validation',
              trees: Math.floor(object.trees * 0.15),
              date: new Date(new Date(object.date).getTime() - 86400000 * 5).toISOString().split('T')[0], // 5 days before
              status: 'complete',
              cell_token: s2CellToken,
              sub_region: 'East',
              linked: false
            },
            {
              id: `s2-validation-${s2CellToken}-2`,
              title: `Validation in ${s2CellToken} Cell - West`,
              type: 'validation',
              trees: Math.floor(object.trees * 0.25),
              date: new Date(new Date(object.date).getTime() - 86400000 * 2).toISOString().split('T')[0], // 2 days before
              status: 'complete',
              cell_token: s2CellToken,
              sub_region: 'West',
              linked: false
            }
          ];
          
          setValidationReports(mockValidationReports);
        } catch (error) {
          console.error('Error fetching validation reports by S2 cell:', error);
        } finally {
          setIsLoadingValidation(false);
        }
      };
      
      fetchLinkedReports();
      fetchValidationReportsByS2();
    }
  }, [object]);
  
  // Handle linking a validation report to an area report
  const handleLinkReport = (validationReportId) => {
    // In a real implementation, this would dispatch to the API
    // For demo purposes, update the local state
    setValidationReports(prev => 
      prev.map(report => 
        report.id === validationReportId 
          ? { ...report, linked: true } 
          : report
      )
    );
    
    // Add to linked reports
    const reportToLink = validationReports.find(r => r.id === validationReportId);
    if (reportToLink) {
      setLinkedReports(prev => [
        ...prev, 
        { 
          ...reportToLink, 
          status: 'complete',
          trees_validated: reportToLink.trees,
          high_risk_confirmed: Math.floor(reportToLink.trees * 0.1)
        }
      ]);
    }
    
    // Show success notification
    alert(`Linked validation report ${validationReportId} to area report ${object.id}`);
  };
  
  if (!object) return <ReportsOverview onNavigate={() => {}} />;
  
  return (
    <div className="h-full bg-gray-50 p-6 overflow-auto">
      <div className="max-w-5xl mx-auto">
        {/* Header with back button only */}
        <div className="flex items-center mb-6">
          <button 
            onClick={() => {
              // Use the onBackToReports callback that maintains sidebar state
              // This ensures proper container width is preserved when returning to Reports view
              onBackToReports();
            }}
            className="flex items-center text-gray-700 hover:text-blue-700 mr-4 font-medium"
          >
            <ChevronLeft className="h-5 w-5 mr-1" /> Back to Reports
          </button>
        </div>
        
        {/* Object banner */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 mb-6 overflow-hidden">
          <div className="border-b border-gray-200 bg-blue-50 px-6 py-4">
            <div className="flex justify-between items-start">
              <div>
                <h2 className="text-xl font-medium text-blue-900">
                  {object.name || object.title}
                </h2>
                <p className="text-sm text-blue-800 mt-1">ID: {object.id}</p>
              </div>
              <div className="relative">
                <select 
                  className="appearance-none pl-3 pr-8 py-1 rounded-md bg-white border border-blue-200 text-sm font-medium text-blue-800 cursor-pointer hover:border-blue-300 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  defaultValue={object.status || "pending"}
                  onChange={(e) => {
                    // This would update the status in a real implementation
                    console.log(`Status changed to: ${e.target.value}`);
                    // Update global state or dispatch an event
                    window.dispatchEvent(new CustomEvent('updateObjectStatus', {
                      detail: {
                        objectId: object.id,
                        newStatus: e.target.value,
                        source: 'assessment_header'
                      }
                    }));
                  }}
                >
                  <option value="pending">Pending</option>
                  <option value="in_progress">In Progress</option>
                  <option value="complete">Complete</option>
                  <option value="validated">Validated</option>
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-blue-800">
                  <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                    <path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" fillRule="evenodd"></path>
                  </svg>
                </div>
              </div>
            </div>
          </div>
          
          <div className="p-6">
            {/* Basic info */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Location</h3>
                <p className="text-sm text-gray-600">Region: {object.region} ({object.regionCode})</p>
                {object.metadata && (
                  <p className="text-sm text-gray-600 mt-1">
                    Coordinates: {object.metadata.latitude?.toFixed(6)}, {object.metadata.longitude?.toFixed(6)}
                  </p>
                )}
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Status</h3>
                <div className="text-sm text-gray-600 flex items-center justify-between">
                  <span>Updated: {object.lastUpdated || object.date}</span>
                  <span className={`ml-2 px-2 py-0.5 rounded-full text-xs ${
                    object.type === 'tree' ? (
                      object.riskLevel === 'high' ? 'bg-red-50/70 text-red-700' :
                      object.riskLevel === 'medium' ? 'bg-yellow-50/70 text-yellow-700' :
                      'bg-green-50/70 text-green-700'
                    ) : (
                      object.status === 'complete' ? 'bg-green-50/70 text-green-700' :
                      object.status === 'in_progress' ? 'bg-yellow-50/70 text-yellow-700' :
                      'bg-blue-50/70 text-blue-700'
                    )
                  }`}>
                    {object.type === 'tree' ? (
                      object.riskLevel?.charAt(0).toUpperCase() + object.riskLevel?.slice(1)
                    ) : (
                      object.status === 'complete' ? 'Complete' :
                      object.status === 'in_progress' ? 'In Progress' : 'Pending'
                    )}
                  </span>
                </div>
              </div>
              
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <h3 className="text-sm font-medium text-gray-700 mb-2">Storage</h3>
                <p className="text-sm text-gray-600 font-mono text-xs">
                  {object.path || object.metadata?.path || `/data/zarr/${object.type}s/${object.id}`}
                </p>
                {object.regionCode && (
                  <p className="text-xs text-blue-600 mt-2 font-medium">
                    S2 Cell: {object.regionCode}
                  </p>
                )}
              </div>
            </div>
            
            {/* Specific details based on type */}
            {object.type === 'tree' ? (
              <div className="mb-8">
                <h3 className="text-base font-medium text-gray-800 mb-4">Tree Details</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <div className="mb-3">
                      <h4 className="text-sm font-medium text-gray-700 mb-1">Species</h4>
                      <p className="text-sm">{object.species || 'Unknown'}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-1">Height</h4>
                        <p className="text-sm">{object.height} ft</p>
                      </div>
                      <div>
                        <h4 className="text-sm font-medium text-gray-700 mb-1">Diameter</h4>
                        <p className="text-sm">{object.diameter} inches</p>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Detection History</h4>
                    <p className="text-sm text-gray-600">
                      Detected {object.metadata?.detections || 1} time(s)
                    </p>
                    <div className="mt-2">
                      <h5 className="text-xs font-medium text-gray-600 mb-1">Related Reports</h5>
                      <div className="space-y-1">
                        {object.metadata?.reportIds?.map(reportId => (
                          <div key={reportId} className="text-xs bg-gray-50 p-1.5 rounded border border-gray-200">
                            {reportId}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : object.type === 'report' ? (
              <div className="mb-8">
                <h3 className="text-base font-medium text-gray-800 mb-4">Report Details</h3>
                <div className="bg-white p-4 rounded-lg border border-gray-200 mb-4">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-sm font-medium text-gray-700">Tree Statistics</h4>
                    <span className="text-xs text-gray-500">Data from {object.date}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div className="bg-gray-50 p-3 rounded-lg text-center">
                      <div className="text-2xl font-bold text-gray-900">{object.trees}</div>
                      <div className="text-xs text-gray-500">Total Trees</div>
                    </div>
                    <div className="bg-red-50 p-3 rounded-lg text-center">
                      <div className="text-2xl font-bold text-red-600">{object.high_risk}</div>
                      <div className="text-xs text-gray-500">High Risk</div>
                    </div>
                    <div className="bg-green-50 p-3 rounded-lg text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {Math.round((object.trees - object.high_risk) / object.trees * 100)}%
                      </div>
                      <div className="text-xs text-gray-500">Healthy Rate</div>
                    </div>
                  </div>
                  
                  {/* Progress bar */}
                  <div className="mb-2">
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                      <span>Risk Distribution</span>
                      <span>{Math.round((object.high_risk / object.trees) * 100)}% at risk</span>
                    </div>
                    <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-red-500 rounded-full"
                        style={{ width: `${Math.round((object.high_risk / object.trees) * 100)}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
                
                {/* Linked validation reports section */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-base font-medium text-gray-800">Linked Validation Reports</h3>
                    <span className="bg-blue-50 text-blue-600 text-xs px-2 py-1 rounded-md">
                      S2 Cell: {object.regionCode}
                    </span>
                  </div>
                  
                  {isLoadingLinked ? (
                    <div className="p-4 flex justify-center">
                      <div className="animate-spin h-5 w-5 border-2 border-blue-500 rounded-full border-t-transparent"></div>
                    </div>
                  ) : linkedReports.length > 0 ? (
                    <div className="space-y-3">
                      {linkedReports.map(report => (
                        <div key={report.id} className="bg-white p-3 border border-blue-100 rounded-md shadow-sm hover:shadow-md transition-shadow">
                          <div className="flex justify-between items-start">
                            <div>
                              <h4 className="text-sm font-medium text-gray-800">{report.title}</h4>
                              <p className="text-xs text-gray-500 mt-1">
                                {report.subregion} • {new Date(report.date).toLocaleDateString()}
                              </p>
                            </div>
                            <span className={`text-xs px-2 py-0.5 rounded-full ${
                              report.status === 'complete' ? 'bg-green-100 text-green-800' :
                              'bg-yellow-100 text-yellow-800'
                            }`}>
                              {report.status === 'complete' ? 'Complete' : 'In Progress'}
                            </span>
                          </div>
                          
                          <div className="mt-3 grid grid-cols-2 gap-2">
                            <div className="bg-gray-50 p-2 rounded text-center text-xs">
                              <span className="block text-blue-600 font-medium">{report.trees_validated}</span>
                              <span className="text-gray-500">Trees Validated</span>
                            </div>
                            <div className="bg-gray-50 p-2 rounded text-center text-xs">
                              <span className="block text-red-600 font-medium">{report.high_risk_confirmed}</span>
                              <span className="text-gray-500">High Risk Confirmed</span>
                            </div>
                          </div>
                          
                          <div className="mt-3 flex justify-end">
                            <button 
                              className="text-xs text-blue-600 hover:text-blue-800"
                              onClick={() => {
                                // Navigate to validation report details
                                window.dispatchEvent(new CustomEvent('showObjectReport', {
                                  detail: { 
                                    object: report,
                                    source: 'report'
                                  }
                                }));
                              }}
                            >
                              View details →
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="bg-gray-50 border border-gray-200 rounded-md p-4 text-center">
                      <p className="text-sm text-gray-600">No validation reports linked to this area report yet.</p>
                    </div>
                  )}
                </div>
                
                {/* S2 Cell Validation Reports */}
                {validationReports.length > 0 && (
                  <div className="mb-6">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-base font-medium text-gray-800">Available Validation Reports in S2 Cell</h3>
                    </div>
                    
                    {isLoadingValidation ? (
                      <div className="p-4 flex justify-center">
                        <div className="animate-spin h-5 w-5 border-2 border-green-500 rounded-full border-t-transparent"></div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {validationReports.filter(r => !r.linked).map(report => (
                          <div key={report.id} className="bg-white p-3 border border-green-100 rounded-md shadow-sm hover:shadow-md transition-shadow">
                            <div className="flex justify-between items-start">
                              <div>
                                <h4 className="text-sm font-medium text-gray-800">{report.title}</h4>
                                <p className="text-xs text-gray-500 mt-1">
                                  {report.sub_region} • {new Date(report.date).toLocaleDateString()}
                                </p>
                              </div>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-800">
                                {report.trees} Trees
                              </span>
                            </div>
                            
                            <div className="mt-3 flex justify-end space-x-2">
                              <button 
                                className="text-xs text-green-600 hover:text-green-800 px-2 py-1 border border-green-200 rounded-md"
                                onClick={() => handleLinkReport(report.id)}
                              >
                                Link to Area Report
                              </button>
                              <button 
                                className="text-xs text-blue-600 hover:text-blue-800 px-2 py-1"
                                onClick={() => {
                                  // Navigate to validation report details
                                  window.dispatchEvent(new CustomEvent('showObjectReport', {
                                    detail: { 
                                      object: report,
                                      source: 'report'
                                    }
                                  }));
                                }}
                              >
                                View details →
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : null}
            
            {/* Actions row */}
            <div className="flex gap-4 mt-6">
              <button 
                onClick={() => window.dispatchEvent(new CustomEvent('showObjectReport', {
                  detail: { 
                    object,
                    source: 'report'
                  }
                }))}
                className="flex-1 bg-purple-50/70 hover:bg-purple-50 text-purple-800 border border-purple-200/60 rounded-md py-2 font-medium"
              >
                Examine Data
              </button>
              <button 
                onClick={() => window.dispatchEvent(new CustomEvent('showOnMap', {
                  detail: { 
                    object,
                    source: 'report'
                  }
                }))}
                className="flex-1 bg-gray-50 hover:bg-gray-100 text-gray-800 border border-gray-200 rounded-md py-2 font-medium"
              >
                View on Map
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Reports Overview component to show as default view instead of map
const ReportsOverview = ({ onNavigate }) => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('recent');
  
  // Fetch reports data from mock/database
  useEffect(() => {
    const fetchReports = async () => {
      setLoading(true);
      try {
        // Mock data for now - would connect to actual zarr storage
        const mockReports = [
          { 
            id: 'rep-001', 
            title: 'North Sector Assessment', 
            region: 'S2-N1', 
            date: '2025-05-01', 
            status: 'complete',
            trees: 87,
            high_risk: 12,
            validated: true 
          },
          { 
            id: 'rep-002', 
            title: 'Central District Tree Health', 
            region: 'S2-C3',
            date: '2025-04-28', 
            status: 'in_progress',
            trees: 143,
            high_risk: 23,
            validated: false 
          },
          { 
            id: 'rep-003', 
            title: 'South Region Analysis', 
            region: 'S2-S5',
            date: '2025-04-22', 
            status: 'complete',
            trees: 56,
            high_risk: 4,
            validated: true 
          },
          { 
            id: 'rep-004', 
            title: 'West Sector Assessment', 
            region: 'S2-W4',
            date: '2025-05-03', 
            status: 'pending',
            trees: 112,
            high_risk: 18,
            validated: false 
          },
          { 
            id: 'rep-005', 
            title: 'East District Analysis', 
            region: 'S2-E2',
            date: '2025-04-25', 
            status: 'complete',
            trees: 94,
            high_risk: 9,
            validated: true 
          }
        ];
        
        // Sort by date (recent first)
        mockReports.sort((a, b) => new Date(b.date) - new Date(a.date));
        
        setReports(mockReports);
      } catch (error) {
        console.error("Error fetching reports:", error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchReports();
  }, []);
  
  // Filter the reports based on selection
  const filteredReports = useMemo(() => {
    if (filter === 'recent') {
      return [...reports].sort((a, b) => new Date(b.date) - new Date(a.date));
    } else if (filter === 'high-risk') {
      return [...reports].sort((a, b) => b.high_risk - a.high_risk);
    } else if (filter === 'validated') {
      return reports.filter(r => r.validated);
    } else if (filter === 'in-progress') {
      return reports.filter(r => r.status === 'in_progress');
    }
    return reports;
  }, [reports, filter]);
  
  return (
    <div className="h-full bg-gray-50 p-6 overflow-auto">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Reports Overview</h1>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => {
                // Navigate to Map view
                onNavigate('Map');
                
                // Force satellite view with labels
                setTimeout(() => {
                  window.dispatchEvent(new CustomEvent('setMapTypeId', {
                    detail: { 
                      mapTypeId: 'satellite',
                      showLabels: true,
                      forceApply: true,
                      source: 'reports_to_map' 
                    }
                  }));
                }, 100);
              }}
              className="bg-white px-4 py-2 text-sm font-medium text-gray-700 border border-gray-300 rounded-md shadow-sm hover:bg-gray-50"
            >
              View Map
            </button>
            <button
              onClick={() => {
                // First close any other sidebars
                window.dispatchEvent(new CustomEvent('forceCloseImageryPanel', {
                  detail: { source: 'reports' }
                }));
                window.dispatchEvent(new CustomEvent('closeAnalyticsPanel', {
                  detail: { source: 'reports' }
                }));
                window.dispatchEvent(new CustomEvent('closeReviewPanel', {
                  detail: { source: 'reports' }
                }));
                
                // Then open database sidebar
                setTimeout(() => {
                  window.dispatchEvent(new CustomEvent('openFeatureSelection', {
                    detail: { 
                      mode: 'tree_inventory', 
                      clearExisting: true,
                      source: 'reports'
                    }
                  }));
                }, 50);
              }}
              className="bg-green-50 px-4 py-2 text-sm font-medium text-green-900 border border-green-200 rounded-md shadow-sm hover:bg-green-100"
            >
              Open Database
            </button>
          </div>
        </div>
        
        {/* Filter tabs */}
        <div className="mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-6">
              <button
                onClick={() => setFilter('recent')}
                className={`whitespace-nowrap pb-3 px-1 border-b-2 font-medium text-sm ${
                  filter === 'recent' 
                    ? 'border-blue-500 text-blue-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Recent
              </button>
              <button
                onClick={() => setFilter('high-risk')}
                className={`whitespace-nowrap pb-3 px-1 border-b-2 font-medium text-sm ${
                  filter === 'high-risk' 
                    ? 'border-red-500 text-red-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                High Risk
              </button>
              <button
                onClick={() => setFilter('validated')}
                className={`whitespace-nowrap pb-3 px-1 border-b-2 font-medium text-sm ${
                  filter === 'validated' 
                    ? 'border-emerald-700 text-emerald-800' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Validated
              </button>
              <button
                onClick={() => setFilter('in-progress')}
                className={`whitespace-nowrap pb-3 px-1 border-b-2 font-medium text-sm ${
                  filter === 'in-progress' 
                    ? 'border-yellow-500 text-yellow-600' 
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                In Progress
              </button>
            </nav>
          </div>
        </div>
        
        {/* Reports grid */}
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-green-500"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredReports.map(report => (
              <div key={report.id} className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden hover:shadow-md transition-shadow">
                <div className="border-b border-gray-200 bg-gray-50 px-4 py-3">
                  <div className="flex justify-between items-start">
                    <h2 className="text-lg font-medium text-gray-900 truncate" title={report.title}>
                      {report.title}
                    </h2>
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      report.status === 'complete' ? 'bg-green-100 text-green-800' :
                      report.status === 'in_progress' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {report.status === 'complete' ? 'Complete' : 
                       report.status === 'in_progress' ? 'In Progress' : 'Pending'}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">Region: {report.region}</p>
                </div>
                
                <div className="p-4">
                  <div className="flex justify-between mb-4">
                    <div className="text-center bg-gray-50/70 px-3 py-2 rounded">
                      <div className="text-2xl font-bold text-gray-900/80">{report.trees}</div>
                      <div className="text-xs text-gray-500">Trees</div>
                    </div>
                    <div className="text-center bg-red-50/70 px-3 py-2 rounded">
                      <div className="text-2xl font-bold text-red-600/80">{report.high_risk}</div>
                      <div className="text-xs text-gray-500">High Risk</div>
                    </div>
                    <div className="text-center bg-emerald-50/70 px-3 py-2 rounded">
                      <div className="text-2xl font-bold text-emerald-800/80">
                        {Math.round((report.trees - report.high_risk) / report.trees * 100)}%
                      </div>
                      <div className="text-xs text-gray-500">Health</div>
                    </div>
                  </div>
                  
                  <div className="border-t border-gray-200 pt-4 flex justify-between">
                    <div className="text-xs text-gray-500">
                      Created: {new Date(report.date).toLocaleDateString()}
                    </div>
                    <button 
                      className="text-sm text-gray-700 hover:text-gray-800 font-medium"
                      onClick={() => {
                        // First close any other sidebars
                        window.dispatchEvent(new CustomEvent('forceCloseImageryPanel', {
                          detail: { source: 'reports' }
                        }));
                        window.dispatchEvent(new CustomEvent('closeAnalyticsPanel', {
                          detail: { source: 'reports' }
                        }));
                        window.dispatchEvent(new CustomEvent('closeReviewPanel', {
                          detail: { source: 'reports' }
                        }));
                        
                        // Then open database sidebar with this report's filter
                        setTimeout(() => {
                          window.dispatchEvent(new CustomEvent('openFeatureSelection', {
                            detail: { 
                              mode: 'tree_inventory', 
                              region: report.region,
                              reportId: report.id,
                              source: 'reports'
                            }
                          }));
                          
                          // Create mock report object for viewing
                          const reportObject = {
                            id: report.id,
                            type: 'report',
                            title: report.title,
                            region: report.region,
                            regionCode: report.region.split(' ')[0],
                            date: report.date,
                            status: report.status,
                            trees: report.trees,
                            high_risk: report.high_risk,
                            validated: report.validated,
                            path: `/data/zarr/reports/${report.id}`
                          };
                          
                          // Show report details in main view
                          window.dispatchEvent(new CustomEvent('showObjectReport', {
                            detail: { 
                              object: reportObject,
                              source: 'reports'
                            }
                          }));
                        }, 50);
                      }}
                    >
                      Details →
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const DashboardContent = () => {
  const mapRef = useRef(null);
  const cesiumRef = useRef(null);
  const mapDataRef = useRef({
    trees: [],
    properties: []
  });
  const [showNewInterface, setShowNewInterface] = useState(false); // Default to classic view
  const [currentView, setCurrentView] = useState('Reports'); // Default to Reports view instead of Map
  const [selectedObject, setSelectedObject] = useState(null); // For object reports from database
  const [headerCollapsed, setHeaderCollapsed] = useState(false); // Track header collapse state
  
  // Header collapse state tracking
  React.useEffect(() => {
    const handleHeaderCollapse = (event) => {
      setHeaderCollapsed(event.detail.collapsed);
    };
    
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    return () => {
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
    };
  }, []);
  
  // Add initialization effect to set default map type to hybrid
  React.useEffect(() => {
    console.log("App initialization - setting default map type to hybrid");
    
    // Initialize map after component has mounted
    const initializeMapView = () => {
      // Get user's saved preference or default to hybrid
      const savedMapType = localStorage.getItem('currentMapType') || 'hybrid';
      
      // Force the saved map type or hybrid by default
      window.dispatchEvent(new CustomEvent('setMapTypeId', {
        detail: { 
          mapTypeId: savedMapType,
          showLabels: true,
          forceApply: true,
          source: 'app_init_preference' 
        }
      }));
      
      // Ensure localStorage is updated
      localStorage.setItem('currentMapType', savedMapType);
      
      // Note: We no longer need to dispatch an event to open the Aerial sidebar
      // since it's now open by default through the defaultOpen prop
    };
    
    // Wait until everything is loaded - just need this once for map type
    setTimeout(initializeMapView, 1000);
    
    // No cleanup needed
  }, []);

  // Add effect to handle both map filtering events and 3D view toggle
  React.useEffect(() => {
    // Handler for 3D view toggle request from MapControls
    const handleToggle3DViewRequest = (event) => {
      const { show3D } = event.detail;
      setShowNewInterface(false); // Always stay in Classic View
      
      // Set global flag for 3D mode
      window.is3DModeActive = show3D;
    };
    
    // Handler for navigation events from components
    const handleNavigationEvent = (event) => {
      const { view } = event.detail;
      if (view) {
        console.log("Navigation event received:", view);
        handleNavigation(view);
      }
    };
    
    // Handler for sidebar resize events
    const handleSidebarResize = (event) => {
      const { id, width } = event.detail;
      const mapContainer = document.getElementById('map-container');
      const reportContainer = document.getElementById('report-container');
      
      // Apply resizing to the active container based on current view
      if (currentView === 'Map' && mapContainer) {
        mapContainer.style.right = `${width}px`;
      } else if ((currentView === 'Reports' || currentView === 'ObjectReport') && reportContainer) {
        reportContainer.style.right = `${width}px`;
      }
    };
    
    // Handler for showing TreeDetection component (Object Recognition)
    const handleShowTreeDetection = (event) => {
      console.log("Received request to show TreeDetection/Object Recognition");
      
      // Create a container for the TreeDetection component if it doesn't exist
      let recognitionContainer = document.getElementById('object-recognition-container');
      if (!recognitionContainer) {
        recognitionContainer = document.createElement('div');
        recognitionContainer.id = 'object-recognition-container';
        recognitionContainer.style.position = 'fixed';
        recognitionContainer.style.top = '0';
        recognitionContainer.style.left = '0';
        recognitionContainer.style.width = '100%';
        recognitionContainer.style.height = '100%';
        recognitionContainer.style.zIndex = '9999';
        recognitionContainer.style.display = 'flex';
        recognitionContainer.style.alignItems = 'center';
        recognitionContainer.style.justifyContent = 'center';
        recognitionContainer.style.background = 'rgba(0, 0, 0, 0.25)';
        
        // Create modal container with subtle styling
        const modalDiv = document.createElement('div');
        modalDiv.style.background = 'white';
        modalDiv.style.borderRadius = '4px';
        modalDiv.style.width = '500px';
        modalDiv.style.maxWidth = '90%';
        modalDiv.style.boxShadow = '0 1px 10px rgba(0, 0, 0, 0.08)';
        modalDiv.style.overflow = 'hidden';
        modalDiv.style.border = '1px solid rgba(0, 0, 0, 0.04)';
        
        // Create header that matches sidebar
        const headerDiv = document.createElement('div');
        headerDiv.style.background = '#edf6ff';
        headerDiv.style.padding = '10px 16px';
        headerDiv.style.display = 'flex';
        headerDiv.style.justifyContent = 'space-between';
        headerDiv.style.alignItems = 'center';
        headerDiv.style.borderBottom = '1px solid rgba(0, 0, 0, 0.05)';
        
        // Header title
        const headerTitle = document.createElement('span');
        headerTitle.textContent = 'Object Recognition';
        headerTitle.style.color = '#0d47a1';
        headerTitle.style.fontWeight = '500';
        headerTitle.style.fontSize = '15px';
        headerDiv.appendChild(headerTitle);
        
        // Close button
        const closeButton = document.createElement('button');
        closeButton.innerHTML = '×';
        closeButton.style.background = 'none';
        closeButton.style.border = 'none';
        closeButton.style.color = '#0d47a1';
        closeButton.style.fontSize = '22px';
        closeButton.style.cursor = 'pointer';
        closeButton.style.padding = '0 4px';
        closeButton.style.lineHeight = '1';
        closeButton.onclick = () => {
          document.body.removeChild(recognitionContainer);
        };
        headerDiv.appendChild(closeButton);
        
        // Create content
        const contentDiv = document.createElement('div');
        contentDiv.style.padding = '20px';
        
        // Form content
        contentDiv.innerHTML = `
          <div style="margin-bottom: 16px;">
            <h2 style="margin: 0 0 8px; font-size: 18px; font-weight: 600;">Start Tree Detection</h2>
            <p style="margin: 0; color: #555; font-size: 14px;">
              Detect trees in the current map view using AI-powered object recognition.
            </p>
          </div>
          
          <div style="margin: 20px 0;">
            <label style="display: block; margin-bottom: 8px; font-weight: 500; font-size: 14px;">
              Detection Area ID
            </label>
            <input id="area-id-input" type="text" value="area_${Date.now()}" 
              style="width: 100%; padding: 8px; border: 1px solid rgba(0,0,0,0.12); border-radius: 3px; font-size: 14px; background-color: white;">
          </div>
          
          <div style="margin: 16px 0; display: flex; align-items: center;">
            <input id="use-current-view" type="checkbox" checked style="margin-right: 8px;">
            <label for="use-current-view" style="font-size: 14px;">Use current map view</label>
          </div>
          
          <div style="margin-top: 24px; display: flex; gap: 12px; justify-content: flex-end;">
            <button id="cancel-detection" style="padding: 8px 16px; background: none; border: 1px solid #ddd; border-radius: 4px; cursor: pointer;">
              Cancel
            </button>
            <button id="start-detection" style="padding: 8px 16px; background: #0d47a1; color: white; border: none; border-radius: 3px; font-weight: 500; cursor: pointer;">
              Start Detection
            </button>
          </div>
        `;
        
        // Add components to DOM
        modalDiv.appendChild(headerDiv);
        modalDiv.appendChild(contentDiv);
        recognitionContainer.appendChild(modalDiv);
        document.body.appendChild(recognitionContainer);
        
        // Add event listeners
        setTimeout(() => {
          const cancelBtn = document.getElementById('cancel-detection');
          const startBtn = document.getElementById('start-detection');
          
          if (cancelBtn) {
            cancelBtn.addEventListener('click', () => {
              document.body.removeChild(recognitionContainer);
            });
          }
          
          if (startBtn) {
            startBtn.addEventListener('click', () => {
              const areaId = document.getElementById('area-id-input')?.value || `area_${Date.now()}`;
              const useCurrentView = document.getElementById('use-current-view')?.checked || true;
              
              alert(`Starting object recognition with areaId: ${areaId}`);
              // Here you would trigger your actual detection process
              
              document.body.removeChild(recognitionContainer);
            });
          }
        }, 100);
      } else {
        // Make sure it's visible if it already exists
        recognitionContainer.style.display = 'flex';
      }
    };
    
    // Handler for closing Object Detection sidebar
    const handleCloseObjectDetection = (event) => {
      console.log("Closing Object Recognition sidebar");
      
      // 1. Clean up the Object Recognition sidebar (light blue theme)
      const detectionSidebar = document.querySelector('.detection-sidebar');
      if (detectionSidebar) {
        console.log("Found Object Recognition sidebar, hiding it");
        // Just hide the sidebar to avoid DOM parent/child errors
        detectionSidebar.style.transform = 'translateX(385px)';
        detectionSidebar.style.opacity = '0';
        
        // Set to display none after animation
        setTimeout(() => {
          detectionSidebar.style.display = 'none';
        }, 300);
      }
      
      // 2. Hide the detection debug badge (OBJECT DETECTION text)
      const detectionDebug = document.getElementById('detection-debug');
      if (detectionDebug) {
        // Just hide the element to avoid DOM errors
        detectionDebug.style.opacity = '0';
        setTimeout(() => {
          detectionDebug.style.display = 'none';
        }, 200);
      }
      
      // 3. Close the ML overlay (blue tint)
      const overlay = document.getElementById('ml-detection-overlay');
      if (overlay) {
        // Just hide the overlay to avoid DOM errors
        overlay.style.opacity = '0';
        setTimeout(() => {
          overlay.style.display = 'none';
        }, 200);
      }
      
      // 4. Ensure any manual tree placement mode is disabled
      window.dispatchEvent(new CustomEvent('disableManualTreePlacement', {
        detail: { source: 'app', forced: true }
      }));
      
      // 5. Reset map container width
      const mapContainer = document.getElementById('map-container');
      if (mapContainer) {
        // Animate the transition back
        mapContainer.style.transition = 'right 0.3s ease';
        mapContainer.style.right = '0px';
      }
      
      // 6. Also dispatch the exitValidationMode event
      window.dispatchEvent(new CustomEvent('exitValidationMode', {
        detail: { 
          source: 'app', 
          clearExisting: true,
          forceRemove: true
        }
      }));
      
      // 7. Remove any manual placement notification that might still be visible
      const manualPlacementNotification = document.querySelector('div[style*="Manual tree placement"]');
      if (manualPlacementNotification && manualPlacementNotification.parentNode) {
        manualPlacementNotification.parentNode.removeChild(manualPlacementNotification);
      }
      
      // 8. Force resize event to ensure layout updates properly
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 300);
    };

    // Add event listeners
    window.addEventListener('requestToggle3DViewType', handleToggle3DViewRequest);
    window.addEventListener('navigateTo', handleNavigationEvent);
    window.addEventListener('sidebarResize', handleSidebarResize);
    window.addEventListener('sidebarResizeEnd', handleSidebarResize);
    window.addEventListener('showTreeDetection', handleShowTreeDetection);
    window.addEventListener('forceCloseObjectDetection', handleCloseObjectDetection);
    
    // Handle manual tree placement events
    const handleEnableManualTreePlacement = (event) => {
      console.log("Enabling manual tree placement mode");
      // Set a global flag to inform map click handlers
      window.manualTreePlacementActive = true;
      
      // Update map cursor if possible
      const mapElement = document.querySelector('#map-container');
      if (mapElement) {
        mapElement.style.cursor = 'crosshair';
      }
      
      // Add map click event listener for placing trees
      if (window.google && window.google.maps && mapRef.current) {
        try {
          // Store the click listener to remove it later
          window.manualTreeClickListener = window.google.maps.event.addListener(
            mapRef.current, 
            'click', 
            (clickEvent) => {
              if (window.manualTreePlacementActive) {
                console.log("Manual tree placement at:", clickEvent.latLng.lat(), clickEvent.latLng.lng());
                
                // Create a marker at the click location
                const marker = new window.google.maps.Marker({
                  position: clickEvent.latLng,
                  map: mapRef.current,
                  icon: {
                    path: window.google.maps.SymbolPath.CIRCLE,
                    scale: 8,
                    fillColor: '#2ecc71',
                    fillOpacity: 0.6,
                    strokeColor: '#ffffff',
                    strokeWeight: 1,
                    strokeOpacity: 0.7
                  },
                  zIndex: 1000
                });
                
                // Store marker for later removal if needed
                if (!window.manualTreeMarkers) {
                  window.manualTreeMarkers = [];
                }
                window.manualTreeMarkers.push(marker);
                
                // Add right-click to exit placement mode
                window.google.maps.event.addListenerOnce(mapRef.current, 'rightclick', () => {
                  window.dispatchEvent(new CustomEvent('disableManualTreePlacement', {
                    detail: { source: 'map_right_click' }
                  }));
                });
              }
            }
          );
        } catch (error) {
          console.error("Error setting up manual tree placement:", error);
        }
      }
    };
    
    const handleDisableManualTreePlacement = (event) => {
      console.log("Disabling manual tree placement mode");
      // Clear the global flag
      window.manualTreePlacementActive = false;
      
      // Reset cursor to default
      const mapElement = document.querySelector('#map-container');
      if (mapElement) {
        mapElement.style.cursor = 'auto';
      }
      
      // Remove the click listener if it exists
      if (window.google && window.google.maps && window.manualTreeClickListener) {
        window.google.maps.event.removeListener(window.manualTreeClickListener);
        window.manualTreeClickListener = null;
      }
      
      // Keep markers visible unless forced cleanup is requested
      if (event?.detail?.forced && window.manualTreeMarkers) {
        window.manualTreeMarkers.forEach(marker => {
          if (marker) marker.setMap(null);
        });
        window.manualTreeMarkers = [];
      }
    };
    
    // Add manual tree placement event listeners
    window.addEventListener('enableManualTreePlacement', handleEnableManualTreePlacement);
    window.addEventListener('disableManualTreePlacement', handleDisableManualTreePlacement);
    
    // Handler for the new applyMapFilters event that supports all risk levels
    const handleMapFilters = (event) => {
      console.log("Applying map filters:", event.detail.filters);
      const { riskLevel } = event.detail.filters;
      
      // Set global variables
      window.currentRiskFilter = riskLevel;
      window.highRiskFilterActive = riskLevel === 'high';
      window.showOnlyHighRiskTrees = riskLevel !== 'all';
      
      console.log("Filter state set: currentRiskFilter =", window.currentRiskFilter,
                  "highRiskFilterActive =", window.highRiskFilterActive,
                  "showOnlyHighRiskTrees =", window.showOnlyHighRiskTrees);
      
      if (mapRef.current && mapDataRef.current) {
        const googleMap = mapRef.current.getMap();
        if (!googleMap) return;
        
        // Clear existing markers
        const markers = mapRef.current.getMarkers();
        if (markers) {
          markers.forEach(marker => {
            if (marker.setMap) {
              marker.setMap(null);
            } else if (marker.map) {
              marker.map = null;
            }
          });
          
          // Clear the markers array
          mapRef.current.getMarkers().length = 0;
        }
        
        // Refresh markers with the selected filter
        const refreshEvent = new CustomEvent('refreshMapMarkers', { 
          detail: { 
            riskFilter: riskLevel
          } 
        });
        window.dispatchEvent(refreshEvent);
        
        // Update the validation queue filter
        if (window.setValidationRiskFilter) {
          window.setValidationRiskFilter(riskLevel);
        }
      }
    };
    
    // Legacy handler for high risk trees only (for backward compatibility)
    const showHighRiskTrees = () => {
      console.log("Showing high risk trees");
      // Set global variables to indicate high risk filter is active
      window.currentRiskFilter = 'high';
      window.highRiskFilterActive = true;
      window.showOnlyHighRiskTrees = true;
      
      if (mapRef.current && mapDataRef.current) {
        const googleMap = mapRef.current.getMap();
        if (!googleMap) return;
        
        // Dispatch events to update filters
        const filterEvent = new CustomEvent('filterHighRiskOnly', { 
          detail: { active: true } 
        });
        window.dispatchEvent(filterEvent);
        
        // Refresh map markers to show only high risk trees
        const refreshEvent = new CustomEvent('refreshMapMarkers', { 
          detail: { 
            riskFilter: 'high'
          } 
        });
        window.dispatchEvent(refreshEvent);
        
        // Update the validation queue filter
        if (window.setValidationRiskFilter) {
          window.setValidationRiskFilter('high');
        }
      }
    };
    
    // Function to reset filters
    const resetFilters = () => {
      console.log("Resetting filters");
      // Set global flags
      window.currentRiskFilter = 'all';
      window.highRiskFilterActive = false;
      window.showOnlyHighRiskTrees = false;
      
      // Dispatch events to reset filters
      const filterEvent = new CustomEvent('filterHighRiskOnly', { 
        detail: { active: false } 
      });
      window.dispatchEvent(filterEvent);
      
      // Refresh map markers to show all trees
      const refreshEvent = new CustomEvent('refreshMapMarkers', { 
        detail: { 
          riskFilter: 'all'
        } 
      });
      window.dispatchEvent(refreshEvent);
      
      // Update the validation queue filter
      if (window.setValidationRiskFilter) {
        window.setValidationRiskFilter('all');
      }
    };
    
    // Add event listeners
    window.addEventListener('applyMapFilters', handleMapFilters);
    window.addEventListener('showHighRiskTrees', showHighRiskTrees);
    window.addEventListener('resetFilters', resetFilters);
    
    // Cleanup
    return () => {
      window.removeEventListener('applyMapFilters', handleMapFilters);
      window.removeEventListener('showHighRiskTrees', showHighRiskTrees);
      window.removeEventListener('resetFilters', resetFilters);
      window.removeEventListener('requestToggle3DViewType', handleToggle3DViewRequest);
      window.removeEventListener('navigateTo', handleNavigationEvent);
      window.removeEventListener('sidebarResize', handleSidebarResize);
      window.removeEventListener('sidebarResizeEnd', handleSidebarResize);
      window.removeEventListener('showTreeDetection', handleShowTreeDetection);
      window.removeEventListener('forceCloseObjectDetection', handleCloseObjectDetection);
      window.removeEventListener('enableManualTreePlacement', handleEnableManualTreePlacement);
      window.removeEventListener('disableManualTreePlacement', handleDisableManualTreePlacement);
    };
  }, []);
  
  // CRITICAL: Unified layout management system for all center content views
  useEffect(() => {
    // Unified function to handle main content positioning regardless of view type
    const updateContentLayout = () => {
      console.log(`Updating layout for current view: ${currentView}`);
      
      // Get the active container based on current view
      const activeContainer = currentView === 'Map' 
        ? document.getElementById('map-container')
        : document.getElementById('report-container');
        
      // Track header collapsed state
      const headerCollapsed = document.querySelector('header')?.classList.contains('collapsed') || false;
      
      if (!activeContainer) {
        console.log(`No container found for view: ${currentView}`);
        return;
      }
      
      // Calculate left margin (main sidebar)
      let leftMargin = 0;
      const leftSidebar = document.querySelector('.flex.flex-col.h-full.bg-white.border-r');
      
      if (leftSidebar) {
        const sidebarRect = leftSidebar.getBoundingClientRect();
        const sidebarWidth = sidebarRect.width;
        const sidebarVisible = sidebarWidth > 0 && 
                              window.getComputedStyle(leftSidebar).display !== 'none' &&
                              window.getComputedStyle(leftSidebar).visibility !== 'hidden';
                              
        if (sidebarVisible) {
          const isSidebarCollapsed = leftSidebar.classList.contains('collapsed') || sidebarWidth < 100;
          leftMargin = isSidebarCollapsed ? 40 : 256; // 40px collapsed, 256px expanded
          console.log(`Left sidebar: ${isSidebarCollapsed ? 'collapsed' : 'expanded'}, width: ${leftMargin}px`);
        }
      }
      
      // Check for any active right sidebar
      let rightMargin = 0;
      const rightSidebars = [
        document.getElementById('imagery-sidebar'),
        document.getElementById('analytics-sidebar'),
        document.getElementById('validation-sidebar'),
        document.getElementById('database-sidebar'),
        document.getElementById('review-panel')
      ].filter(Boolean);
      
      // Count visible sidebars and find the widest one
      let visibleSidebarCount = 0;
      rightSidebars.forEach(sidebar => {
        if (sidebar && 
            window.getComputedStyle(sidebar).display !== 'none' && 
            window.getComputedStyle(sidebar).visibility !== 'hidden') {
          
          const rect = sidebar.getBoundingClientRect();
          // Only count sidebars with meaningful width
          if (rect.width > 50) {
            visibleSidebarCount++;
            if (rect.width > rightMargin) {
              rightMargin = rect.width;
              console.log(`Right sidebar detected: ${sidebar.id}, width: ${rightMargin}px`);
            }
          }
        }
      });
      
      // Log when all right sidebars are closed
      if (visibleSidebarCount === 0) {
        console.log("No right sidebars are currently visible - resetting right margin to 0");
        rightMargin = 0;
      }
      
      // Apply positioning to the active container - use fixed positioning for Reports
      if (currentView === 'Reports' || currentView === 'ObjectReport') {
        // For Reports view - use fixed positioning which worked correctly
        activeContainer.style.position = 'fixed';
        activeContainer.style.left = `${leftMargin}px`;
        activeContainer.style.right = `${rightMargin}px`;
        activeContainer.style.top = headerCollapsed ? '40px' : '64px'; // Account for header height
        activeContainer.style.bottom = '0';
        activeContainer.style.width = 'auto'; // Auto width to fill the space
        activeContainer.style.margin = '0';
        
        // Log the exact positioning being applied
        console.log(`Report container fixed position: left=${leftMargin}px, right=${rightMargin}px`);
      } else {
        // For Map view, use the same fixed positioning approach for consistency
        activeContainer.style.position = 'fixed';
        activeContainer.style.left = `${leftMargin}px`;
        activeContainer.style.right = `${rightMargin}px`;
        activeContainer.style.top = headerCollapsed ? '40px' : '64px'; // Account for header height
        activeContainer.style.bottom = '0';
        activeContainer.style.width = 'auto'; // Auto width to fill the space
        activeContainer.style.margin = '0';
        
        // Log the exact positioning being applied
        console.log(`Map container fixed position: left=${leftMargin}px, right=${rightMargin}px`);
      }
      
      // Store dimensions for debugging
      activeContainer.dataset.leftMargin = leftMargin;
      activeContainer.dataset.rightMargin = rightMargin;
      activeContainer.dataset.totalWidth = `calc(100% - ${leftMargin + rightMargin}px)`;
      
      // Make container visible
      activeContainer.style.opacity = '1';
      
      // Special handling for Google Maps when in Map view
      if (currentView === 'Map' && window.google && window.google.maps) {
        // Add a small delay to let the DOM update first
        setTimeout(() => {
          try {
            if (mapRef.current && typeof mapRef.current.getMap === 'function') {
              const mapInstance = mapRef.current.getMap();
              if (mapInstance) {
                google.maps.event.trigger(mapInstance, 'resize');
                console.log("Google Maps resize triggered");
              }
            }
          } catch (e) {
            console.warn("Error resizing Google Maps:", e);
          }
        }, 100);
      }
    };
    
    // Set up event listeners for content layout changes
    const setupLayoutListeners = () => {
      // Initial layout update
      updateContentLayout();
      
      // Update layout when sidebars open/close
      window.addEventListener('expandImageryPanel', updateContentLayout);
      window.addEventListener('forceCloseImageryPanel', updateContentLayout);
      window.addEventListener('openAnalyticsPanel', updateContentLayout);
      window.addEventListener('closeAnalyticsPanel', updateContentLayout);
      window.addEventListener('openReviewPanel', updateContentLayout); 
      window.addEventListener('closeReviewPanel', updateContentLayout);
      window.addEventListener('openFeatureSelection', updateContentLayout);
      window.addEventListener('closeFeatureSelection', updateContentLayout);
      window.addEventListener('sidebarResize', updateContentLayout);
      window.addEventListener('sidebarResizeEnd', updateContentLayout);
      window.addEventListener('resize', updateContentLayout);
      
      // Update when header collapses/expands
      window.addEventListener('headerCollapse', updateContentLayout);
      
      // Update when current view changes
      window.addEventListener('navigateTo', (event) => {
        // Schedule layout updates (with delay to let view changes settle)
        setTimeout(updateContentLayout, 50);
        setTimeout(updateContentLayout, 300);
      });
    };
    
    // Set up the event listeners
    setupLayoutListeners();
    
    // Clean up all event listeners on unmount
    return () => {
      window.removeEventListener('expandImageryPanel', updateContentLayout);
      window.removeEventListener('forceCloseImageryPanel', updateContentLayout);
      window.removeEventListener('openAnalyticsPanel', updateContentLayout);
      window.removeEventListener('closeAnalyticsPanel', updateContentLayout);
      window.removeEventListener('openReviewPanel', updateContentLayout);
      window.removeEventListener('closeReviewPanel', updateContentLayout);
      window.removeEventListener('openFeatureSelection', updateContentLayout);
      window.removeEventListener('closeFeatureSelection', updateContentLayout);
      window.removeEventListener('sidebarResize', updateContentLayout);
      window.removeEventListener('sidebarResizeEnd', updateContentLayout);
      window.removeEventListener('resize', updateContentLayout);
      window.removeEventListener('headerCollapse', updateContentLayout);
    };
  }, [currentView]);
  
  // Add an effect for Cesium 3D view handling
  useEffect(() => {
    // Only handle Cesium resizing if we're in a 3D view
    if (currentView !== 'Map' || !window.is3DModeActive) return;
    
    // Function to resize Cesium viewer when needed
    const resizeCesium = () => {
      try {
        const cesiumViewer = document.querySelector('.cesium-viewer');
        if (cesiumViewer) {
          console.log("Attempting to resize Cesium viewer");
          // For Cesium, we need to trigger a resize on the window
          if (window.cesiumResizeEvent) {
            window.cesiumResizeEvent();
            console.log("Triggered Cesium resize event");
          } else {
            // Fallback to window resize which Cesium listens for
            window.dispatchEvent(new Event('resize'));
            console.log("Triggered window resize for Cesium");
          }
        }
      } catch (e) {
        console.warn("Error resizing Cesium:", e);
      }
    };
    
    // Set up listeners for Cesium resizing
    window.addEventListener('resize', resizeCesium);
    window.addEventListener('sidebarResize', resizeCesium);
    window.addEventListener('cesiumNeedsResize', resizeCesium);
    
    // Resize Cesium after a delay to ensure the container has settled
    setTimeout(resizeCesium, 200);
    
    // Clean up
    return () => {
      window.removeEventListener('resize', resizeCesium);
      window.removeEventListener('sidebarResize', resizeCesium);
      window.removeEventListener('cesiumNeedsResize', resizeCesium);
    };
  }, [currentView]);
  
  // Create a custom event to force layout updates when needed from anywhere in the app
  useEffect(() => {
    // Custom force layout update handler
    const handleForceLayoutUpdate = () => {
      // Dispatch a resize event which will trigger our layout handlers
      window.dispatchEvent(new Event('resize'));
    };
    
    // Listen for force layout update events
    window.addEventListener('forceLayoutUpdate', handleForceLayoutUpdate);
    
    // Also trigger a resize when view changes
    if (currentView) {
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 100);
    }
    
    return () => {
      window.removeEventListener('forceLayoutUpdate', handleForceLayoutUpdate);
    };
  }, [currentView]);

  // Listen for database object selections
  useEffect(() => {
    const handleShowObjectReport = (event) => {
      const { object, source } = event.detail;
      console.log(`Showing object report from ${source}:`, object);
      
      // Set the selected object
      setSelectedObject(object);
      
      // Also ensure we're in the Reports view
      setCurrentView('ObjectReport');
    };
    
    const handleBackToReports = () => {
      // Maintain the sidebar state when going back to Reports
      // Don't reset container widths
      
      // Check which sidebars are currently visible
      const visibleSidebars = Array.from(document.querySelectorAll('[id$="-sidebar"]'))
        .filter(el => el.offsetWidth > 0 && el.style.width !== '0px');
      
      // Get the current width to maintain for report container
      const sidebarWidth = visibleSidebars.length > 0 ? 
        parseInt(visibleSidebars[0].style.width || '0', 10) : 0;
      
      // Navigate back to Reports
      setSelectedObject(null);
      setCurrentView('Reports');
      
      // If a sidebar is open, ensure the report container keeps the right width
      setTimeout(() => {
        const reportContainer = document.getElementById('report-container');
        if (reportContainer && sidebarWidth > 0) {
          reportContainer.style.right = `${sidebarWidth}px`;
        }
        
        // Force resize event to ensure proper layout
        window.dispatchEvent(new Event('resize'));
      }, 50);
    };
    
    // Add event listeners
    window.addEventListener('showObjectReport', handleShowObjectReport);
    window.addEventListener('backToReports', handleBackToReports);
    
    // Cleanup
    return () => {
      window.removeEventListener('showObjectReport', handleShowObjectReport);
      window.removeEventListener('backToReports', handleBackToReports);
    };
  }, []);

  // Handler for navigation from sidebar
  const handleNavigation = (view) => {
    console.log("Navigating to view:", view);
    
    // Before changing views, capture current sidebar state
    const visibleSidebars = Array.from(document.querySelectorAll('[id$="-sidebar"]'))
      .filter(el => el.offsetWidth > 0 && el.style.width !== '0px');
    
    // Get the current width if a sidebar is open
    const sidebarWidth = visibleSidebars.length > 0 ? 
      parseInt(visibleSidebars[0].style.width || '0', 10) : 0;
    
    // Reset selected object when navigating away from ObjectReport
    if (view !== 'ObjectReport') {
      setSelectedObject(null);
    }
    
    // Update the current view
    setCurrentView(view);
    
    // CRITICAL: Apply the correct container width after the view changes
    // This ensures going back to Reports maintains proper sidebar spacing
    setTimeout(() => {
      if (view === 'Reports' && sidebarWidth > 0) {
        const reportContainer = document.getElementById('report-container');
        if (reportContainer) {
          reportContainer.style.right = `${sidebarWidth}px`;
        }
      }
      
      // When navigating back to Map view, ensure the UI is reset properly
      if (view === "Map") {
        setShowNewInterface(false);
        
        // Make sure aerial sidebar is expanded when switching to Map view
        // Find the aerial sidebar and manually open it if it's not already open
        const aerialSidebar = document.getElementById('imagery-sidebar');
        if (aerialSidebar && (!aerialSidebar.offsetWidth || aerialSidebar.offsetWidth === 0)) {
          console.log("Opening Aerial sidebar on navigation to Map view");
          // Dispatch event to open the aerial sidebar
          window.dispatchEvent(new CustomEvent('expandImageryPanel', {
            detail: { source: 'navigation_to_map' }
          }));
        }
        
        // Preserve the 3D mode state if it's set globally
        if (typeof window.is3DModeActive !== 'undefined') {
          console.log("Preserving 3D mode state:", window.is3DModeActive);
          // Dispatch an event to ensure 3D mode is maintained if needed
          if (window.is3DModeActive) {
            // Wait a small delay to ensure the map is ready before applying 3D mode
            setTimeout(() => {
              window.dispatchEvent(new CustomEvent('mapModeChanged', { 
                detail: { mode: '3D', tilt: 45 } 
              }));
            }, 300);
          }
        }
      }
      
      // Force a resize event to ensure proper layout
      window.dispatchEvent(new Event('resize'));
    }, 50);
  };

  return (
    <Layout mapRef={mapRef} mapDataRef={mapDataRef} onNavigate={handleNavigation}>
      <Suspense fallback={<LoadingSpinner />}>
        {currentView === 'Settings' ? (
          <SettingsPanel />
        ) : currentView === 'Reports' ? (
          // Default view - Reports Overview with sidebars
          <div className="h-full w-full relative">
            <div className="w-full h-full overflow-auto" id="report-container" style={{position: 'relative'}}>
              <ReportsOverview onNavigate={handleNavigation} />
            </div>
            {/* All sidebars - positioned absolutely through their own internal styling */}
            {/* DetectionSidebar is now managed by MapView directly */}
            <AerialImagery />
            <ValidationSidebar />
            <AnalyticsSidebar />
            <DatabaseSidebar />
          </div>
        ) : currentView === 'ObjectReport' ? (
          // Object Report view with sidebars - simplified layout
          <div className="h-full w-full relative">
            <div className="w-full h-full overflow-auto" id="report-container" style={{position: 'relative'}}>
              <ObjectReport 
                object={selectedObject} 
                onBackToReports={() => {
                  // Use handleNavigation to ensure proper navigation back to Reports
                  handleNavigation('Reports');
                }} 
              />
            </div>
            {/* All sidebars */}
            {/* DetectionSidebar is now managed by MapView directly */}
            <AerialImagery />
            <ValidationSidebar />
            <AnalyticsSidebar />
            <DatabaseSidebar />
          </div>
        ) : (
          // Map view with sidebars when selected
          showNewInterface ? (
            <div className="h-full">
              <MapAssessmentPanel />
            </div>
          ) : (
            <div className="h-full w-full relative">
              <div className="w-full h-full overflow-auto" id="map-container" style={{position: 'relative'}}>
                <MapView 
                  ref={mapRef} 
                  onDataLoaded={(data) => {
                    mapDataRef.current = data;
                  }}
                  headerState={headerCollapsed}
                />
                {/* MapControls moved to sidebar */}
              </div>
              {/* All sidebars */}
              {/* DetectionSidebar is now managed by MapView directly */}
              <AerialImagery />
              <ValidationSidebar />
              <AnalyticsSidebar />
              <DatabaseSidebar />
            </div>
          )
        )}
      </Suspense>
    </Layout>
  );
};

/**
 * Main application content with authentication flow
 * Renders login screen, loading state, or dashboard based on auth status
 */
const AppContent = () => {
  const { isAuthenticated, isValidating, login } = useAuth();

  // Display loading spinner during authentication check
  if (isValidating) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-200">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-800 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // Security: Verify authentication status before showing content
  // This prevents content from being shown briefly before auth check completes
  if (!isAuthenticated) {
    // Add a unique key to force complete remount after logout
    return <Login onLogin={login} key={`login-${Date.now()}`} />;
  }

  // Render dashboard for authenticated users
  return <DashboardContent />;
};

/**
 * Root application component
 * Sets up Redux store and authentication provider
 */
const App = () => {
  return (
    <Provider store={store}>
      <AuthProvider>
        <ValidationSystemProvider>
          <AppContent />
        </ValidationSystemProvider>
      </AuthProvider>
    </Provider>
  );
};

// Error Boundary component to catch errors in child components
export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // You can also log the error to an error reporting service
    console.error("Error caught by boundary:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      // If component is the DetectionMode or related to it, render an empty div
      if (this.props.componentName === 'DetectionMode') {
        console.log("DetectionMode error suppressed by boundary");
        return null;
      }
      
      // For other components, you can render a fallback UI
      return (
        <div style={{ padding: '20px', color: '#666' }}>
          <h3>Something went wrong in {this.props.componentName || "a component"}</h3>
          <p>This error has been contained and the rest of the app should function normally.</p>
        </div>
      );
    }

    return this.props.children;
  }
}

export default App;