// src/components/assessment/Validation/ValidationQueue.jsx

import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, XCircle, Clock, Filter, Eye, MapPin, ChevronLeft, ChevronRight, Box, Search } from 'lucide-react';
import { useValidation } from '../../../hooks/useReportValidation';
import { PropertyService } from '../../../services/api/apiService';
import TreeAnalysis from '../TreeAnalysis/TreeAnalysis';
import ValidationSystem from './ValidationSystem';

const ValidationQueue = () => {
  const { validationItems, validateItem, isProcessing, refreshValidationItems } = useValidation();
  const [statusFilter, setStatusFilter] = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');
  const [selectedItemId, setSelectedItemId] = useState(null);
  const [showValidation, setShowValidation] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [propertyData, setPropertyData] = useState({});
  const [collapsed, setCollapsed] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('imagery'); // 'imagery' or 'images'
  
  // Handle address search
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    
    if (!searchQuery.trim()) return;
    
    console.log('Searching for address:', searchQuery);
    
    // In a real app, this would use the Google Maps Geocoding API
    // For now, just display an alert
    alert(`Search functionality coming soon.`);
  };
  
  // Properly implement sidebar toggle
  const toggleCollapse = () => {
    const newCollapsedState = !collapsed;
    setCollapsed(newCollapsedState);
    
    // Update sidebar width
    const sidebar = document.getElementById('imagery-sidebar');
    if (sidebar) {
      sidebar.style.width = newCollapsedState ? '0px' : '384px';
      
      // Also adjust the map container width to prevent grey areas
      const mapContainer = document.getElementById('map-container');
      if (mapContainer) {
        mapContainer.style.right = newCollapsedState ? '0px' : '384px';
      }
    }
    
    // Notify other components about the state change
    window.dispatchEvent(new CustomEvent('validationQueueToggle', {
      detail: { 
        collapsed: newCollapsedState,
        source: 'validationQueue'
      }
    }));
    
    console.log("Video sidebar toggled:", newCollapsedState ? "collapsed" : "expanded");
  };
  
  // Expose the setRiskFilter function globally so it can be called directly
  useEffect(() => {
    // New function to handle any risk level
    window.setValidationRiskFilter = (riskLevel) => {
      console.log("Setting validation risk filter:", riskLevel);
      setRiskFilter(riskLevel);
    };
    
    // Keep the old function for backward compatibility
    window.setValidationHighRiskFilter = (showHighRiskOnly) => {
      console.log("Setting validation high risk filter:", showHighRiskOnly);
      setRiskFilter(showHighRiskOnly ? 'high' : 'all');
    };
    
    return () => {
      delete window.setValidationRiskFilter;
      delete window.setValidationHighRiskFilter;
    };
  }, []);

  // Fetch property data when validation items change
  useEffect(() => {
    const fetchPropertyData = async () => {
      const properties = {};
      
      // Create a unique set of property IDs
      const propertyIds = [...new Set(validationItems.map(item => item.property_id))];
      
      // Fetch each property's data
      for (const propertyId of propertyIds) {
        try {
          const propertyInfo = await PropertyService.getProperty(propertyId);
          properties[propertyId] = propertyInfo;
        } catch (error) {
          console.error(`Error fetching property ${propertyId}:`, error);
        }
      }
      
      setPropertyData(properties);
    };
    
    if (validationItems.length > 0 && !collapsed) {
      fetchPropertyData();
    }
  }, [validationItems, collapsed]);
  
  // Handle validation queue opening and zarr database updates
  useEffect(() => {
    const handleOpenValidationQueue = (event) => {
      // Expand the validation queue when explicitly requested
      console.log("Opening ValidationQueue as requested");
      if (collapsed) {
        toggleCollapse();
      }
    };
    
    // Handler for trees saved to zarr database
    const handleTreesSavedToZarr = (event) => {
      if (event.detail && event.detail.success) {
        console.log("Trees were saved to zarr database. Refreshing validation queue.");
        
        // Refresh validation items from zarr source
        refreshValidationItems();
        
        // Expand the sidebar if it's collapsed
        if (collapsed) {
          toggleCollapse();
        }
      }
    };
    
    // Handle events when trees are saved to zarr database
    const handleZarrSaveEvent = (event) => {
      if (event.detail && event.detail.target === 'zarr') {
        console.log("Received zarr save event:", event.detail);
        
        // This would trigger a backend refresh in a real implementation
        // Then we'll refresh our local validation items
        setTimeout(() => {
          refreshValidationItems();
          console.log("Refreshed validation items from zarr");
          
          // Expand the sidebar if it's collapsed
          if (collapsed) {
            toggleCollapse();
          }
        }, 500); // Short delay to simulate backend processing
      }
    };
    
    // Handle forced close from other sidebar buttons
    const handleForceCloseImageryPanel = (event) => {
      if (!collapsed) {
        console.log("Force closing Imagery sidebar from:", event.detail.source);
        setCollapsed(true);
        
        // Update the parent container's width
        const sidebar = document.getElementById('imagery-sidebar');
        if (sidebar) {
          sidebar.style.width = '0px';
          
          // Also adjust the map container width to prevent grey areas
          const mapContainer = document.getElementById('map-container');
          if (mapContainer) {
            mapContainer.style.right = '0px';
          }
        }
        
        // Reset tab to default when sidebar is collapsed for politeness
        setActiveTab('imagery');
      }
    };
    
    // Handle expand request (ensure Imagery is always opened, not toggled)
    const handleExpandImageryPanel = (event) => {
      // Check if we should switch tabs even when already open
      if (!collapsed && event.detail && event.detail.source === 'sidebar') {
        console.log("Sidebar already open, switching to Review tab");
        // Just switch to the Review tab without toggling the sidebar
        setActiveTab('imagery');
        return;
      }
      
      if (collapsed) {
        console.log("Expanding Imagery sidebar from:", event.detail.source);
        setCollapsed(false);
        
        // Update the parent container's width
        const sidebar = document.getElementById('imagery-sidebar');
        if (sidebar) {
          sidebar.style.width = '384px'; // w-96 (384px) to match other sidebars
          
          // Also adjust the map container width to prevent grey areas
          const mapContainer = document.querySelector('.flex-1.absolute.inset-0');
          if (mapContainer) {
            mapContainer.style.right = '384px';
          }
        }
        
        // Set active tab based on the source of the event
        if (event.detail && event.detail.source === 'sidebar') {
          // If expanding from sidebar button, show Review tab
          setActiveTab('imagery');
        } else if (event.detail && event.detail.source === 'caretButton') {
          // If expanding from caret button, show Images tab
          setActiveTab('images');
        } else if (event.detail && event.detail.tab) {
          // If a specific tab is requested, use that
          setActiveTab(event.detail.tab);
        }
        
        // Notify other components
        window.dispatchEvent(new CustomEvent('validationQueueToggle', {
          detail: { 
            collapsed: false,
            source: 'validationQueue'
          }
        }));
      }
    };
    
    // Add event listeners
    window.addEventListener('saveTreesToZarr', handleZarrSaveEvent);
    window.addEventListener('treesZarrSaveComplete', handleTreesSavedToZarr);
    window.addEventListener('openValidationQueue', handleOpenValidationQueue);
    window.addEventListener('forceCloseImageryPanel', handleForceCloseImageryPanel);
    window.addEventListener('expandImageryPanel', handleExpandImageryPanel);
    window.addEventListener('toggleImageryPanel', toggleCollapse);  // Preserve original toggle functionality
    
    return () => {
      window.removeEventListener('saveTreesToZarr', handleZarrSaveEvent);
      window.removeEventListener('treesZarrSaveComplete', handleTreesSavedToZarr);
      window.removeEventListener('openValidationQueue', handleOpenValidationQueue);
      window.removeEventListener('forceCloseImageryPanel', handleForceCloseImageryPanel);
      window.removeEventListener('expandImageryPanel', handleExpandImageryPanel);
      window.removeEventListener('toggleImageryPanel', toggleCollapse);
    };
  }, [collapsed, refreshValidationItems, toggleCollapse]);

  // Add effect to listen for high risk filter events
  useEffect(() => {
    const handleHighRiskFilter = (event) => {
      if (event.detail.active) {
        setRiskFilter('high');
      } else {
        setRiskFilter('all');
      }
    };
    
    window.addEventListener('filterHighRiskOnly', handleHighRiskFilter);
    
    return () => {
      window.removeEventListener('filterHighRiskOnly', handleHighRiskFilter);
    };
  }, []);
  
  // Add effect to listen for map filter events
  useEffect(() => {
    const handleMapFilters = (event) => {
      if (event.detail?.filters) {
        const { riskLevel, treeSpecies } = event.detail.filters;
        
        // Update risk filter if provided
        if (riskLevel) {
          setRiskFilter(riskLevel);
        }
        
        // Store the species filter globally (we'll use it in the filter function)
        window.currentSpeciesFilter = treeSpecies || '';
      }
    };
    
    // Listen for resetFilters event to clear all filters
    const handleReset = () => {
      window.currentSpeciesFilter = '';
    };
    
    // Listen for refreshMapMarkers to reload validation items
    const handleMapRefresh = () => {
      refreshValidationItems();
    };
    
    window.addEventListener('applyMapFilters', handleMapFilters);
    window.addEventListener('resetFilters', handleReset);
    window.addEventListener('refreshMapMarkers', handleMapRefresh);
    
    return () => {
      window.removeEventListener('applyMapFilters', handleMapFilters);
      window.removeEventListener('resetFilters', handleReset);
      window.removeEventListener('refreshMapMarkers', handleMapRefresh);
    };
  }, [refreshValidationItems]);

  // Apply filters to validation items
  const filteredItems = validationItems.filter(item => {
    // Status filter
    if (statusFilter !== 'all' && item.status !== statusFilter) {
      return false;
    }
    
    // Risk filter
    if (riskFilter !== 'all') {
      const hasHighRisk = item.riskFactors.some(factor => factor.level === 'high');
      const hasMediumRisk = item.riskFactors.some(factor => factor.level === 'medium');
      const hasLowRisk = item.riskFactors.some(factor => factor.level === 'low');
      
      if (riskFilter === 'high') {
        if (!hasHighRisk) return false; // Show only high risk items
      } else if (riskFilter === 'high_medium') {
        if (!hasHighRisk && !hasMediumRisk) return false; // Show both high and medium risk items
      } else if (riskFilter === 'medium') {
        if (!hasMediumRisk || hasHighRisk) return false; // Show only medium risk items (not high)
      } else if (riskFilter === 'low') {
        if (!hasLowRisk || hasHighRisk || hasMediumRisk) return false; // Show only low risk items (not high or medium)
      }
    }
    
    // Species filter - use the global variable set by the map filter event
    if (window.currentSpeciesFilter) {
      // Look up the tree to check its species
      if (item.tree_species !== window.currentSpeciesFilter) {
        return false;
      }
    }
    
    return true;
  });

  const handleValidate = (itemId, status) => {
    validateItem(itemId, status);
  };

  const handleViewValidation = (itemId) => {
    setSelectedItemId(itemId);
    setShowValidation(true);
  };

  const handleViewAnalysis = (itemId) => {
    setSelectedItemId(itemId);
    setShowAnalysis(true);
  };

  const closeModals = () => {
    setShowValidation(false);
    setShowAnalysis(false);
    setSelectedItemId(null);
  };

  // Get selected item data
  const selectedItem = validationItems.find(item => item.id === selectedItemId);
  
  // toggleCollapse function is defined at the top of the component
  
  // Set initial collapsed state on mount and handle resize correctly
  useEffect(() => {
    // Update the parent container's width based on initial state
    const sidebar = document.getElementById('imagery-sidebar');
    if (sidebar) {
      sidebar.style.width = collapsed ? '0px' : '384px';
      console.log("Initializing Video sidebar width:", collapsed ? "0px (hidden)" : "384px (visible)");
      
      // Also adjust the map container width to prevent grey areas
      const mapContainer = document.querySelector('.flex-1.absolute.inset-0');
      if (mapContainer) {
        // Only adjust map if sidebar is visible
        if (!collapsed) {
          mapContainer.style.right = '384px';
        }
      }
      
      // Dispatch event to notify other components about initial state
      window.dispatchEvent(new CustomEvent('validationQueueToggle', {
        detail: { 
          collapsed: collapsed,
          source: 'validationQueue',
          initializing: true
        }
      }));
    }
    
    // When component unmounts, reset map container
    return () => {
      const mapContainer = document.querySelector('.flex-1.absolute.inset-0');
      if (mapContainer) {
        mapContainer.style.right = '0px';
      }
    };
  }, [collapsed]);

  // Listen for TreeValidationMode toggle events and the toggleImageryPanel event
  useEffect(() => {
    // Handle tree validation mode toggle
    const handleTreeValidationModeToggle = (event) => {
      // Only respond to events from TreeValidationMode that aren't initialization events
      if (event.detail.source === 'treeValidationMode' && !event.detail.initializing) {
        console.log("ValidationQueue received event from TreeValidationMode:", event.detail);
        
        // If TreeValidationMode is being expanded, collapse this sidebar
        if (!event.detail.collapsed) {
          // Only collapse if currently expanded
          if (!collapsed) {
            // Use setTimeout to avoid race conditions
            setTimeout(() => {
              // Set collapsed state
              setCollapsed(true);
              
              // Update parent container width
              const sidebar = document.getElementById('imagery-sidebar');
              if (sidebar) {
                sidebar.style.width = '0px';
                
                // Also adjust the map container width to prevent grey areas
                const mapContainer = document.querySelector('.flex-1.absolute.inset-0');
                if (mapContainer) {
                  mapContainer.style.right = '384px'; // Set to Object Detection sidebar width
                }
              }
              
              console.log("Video sidebar collapsed due to TreeValidationMode expansion");
            }, 50);
          }
        }
      }
    };
    
    // Handle toggle imagery panel event from the sidebar button
    const handleToggleImageryPanel = () => {
      // Always expand when this event is triggered
      if (collapsed) {
        toggleCollapse();
        console.log("Video sidebar expanded via toggleImageryPanel event");
      }
    };
    
    // Listen for tree detection events
    const handleTreeDetectionResult = (event) => {
      const { jobId, status, trees } = event.detail;
      
      console.log(`Received tree detection result for job ${jobId}`);
      
      // Expand the imagery panel if it's collapsed
      if (collapsed) {
        toggleCollapse();
      }
      
      // Additional logic for processing tree detection results could go here
    };
    
    window.addEventListener('validationSidebarToggle', handleTreeValidationModeToggle);
    window.addEventListener('toggleImageryPanel', handleToggleImageryPanel);
    window.addEventListener('treeDetectionResult', handleTreeDetectionResult);
    window.addEventListener('enterTreeValidationMode', handleTreeDetectionResult);
    
    return () => {
      window.removeEventListener('validationSidebarToggle', handleTreeValidationModeToggle);
      window.removeEventListener('toggleImageryPanel', handleToggleImageryPanel);
      window.removeEventListener('treeDetectionResult', handleTreeDetectionResult);
      window.removeEventListener('enterTreeValidationMode', handleTreeDetectionResult);
    };
  }, []);

  // Add state to track header collapsed state
  const [headerCollapsed, setHeaderCollapsed] = useState(true);
  
  // Listen for header collapse events
  useEffect(() => {
    const handleHeaderCollapse = (event) => {
      setHeaderCollapsed(event.detail.collapsed);
    };
    
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    return () => {
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
    };
  }, []);

  return (
    <>
      {/* Expand button - visible when sidebar is collapsed */}
      <button 
        className={`fixed ${headerCollapsed ? 'top-32' : 'top-40'} right-0 z-10 h-10 px-2 bg-indigo-600 text-white transition-all duration-300 flex items-center ${collapsed ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        onClick={() => {
          // Instead of using toggleCollapse directly, we'll dispatch an event with a source
          if (collapsed) {
            window.dispatchEvent(new CustomEvent('expandImageryPanel', {
              detail: { source: 'caretButton' }
            }));
          } else {
            setCollapsed(true);
          }
        }}
        aria-label="Expand sidebar"
      >
        <ChevronLeft size={16} />
      </button>
      
      {/* Main content container - position depends on header state */}
      <div className={`fixed ${headerCollapsed ? 'top-10' : 'top-16'} right-0 bottom-0 w-96 flex flex-col transition-opacity ease-in-out duration-200 ${collapsed ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
      <div className={`py-2 px-4 border-b bg-indigo-600 h-10 transition-all duration-300 text-white`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="h-6 w-6 rounded-md bg-indigo-700 flex items-center justify-center mr-2">
              <Box size={14} className="text-white" />
            </div>
            <button 
              onClick={toggleCollapse}
              className="p-0.5 rounded-md hover:bg-indigo-500 transition-colors"
              aria-label="Collapse sidebar"
            >
              <ChevronRight size={16} className="text-white" />
            </button>
          </div>
          <div className="flex items-center space-x-2">
              <button
                className="p-0.5 rounded-md hover:bg-indigo-500 transition-colors"
                title="Search trees"
              >
                <Search className="h-3.5 w-3.5 text-white" />
              </button>
            
              <div className="flex space-x-2">
              <select
                className="text-xs border border-indigo-300 rounded px-2 py-1 text-white bg-indigo-500 focus:bg-indigo-400 focus:text-white focus:outline-none focus:ring-1 focus:ring-white focus:border-white"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="all" className="text-gray-600 bg-white">S2 Areas</option>
                <option value="pending" className="text-gray-600 bg-white">Current Area</option>
                <option value="approved" className="text-gray-600 bg-white">Processed</option>
                <option value="rejected" className="text-gray-600 bg-white">Queued</option>
              </select>
              <select
                className="text-xs border border-indigo-300 rounded px-2 py-1 text-white bg-indigo-500 focus:bg-indigo-400 focus:text-white focus:outline-none focus:ring-1 focus:ring-white focus:border-white"
                value={riskFilter}
                onChange={(e) => {
                  setRiskFilter(e.target.value);
                  // Set the selected risk filter
                  const riskLevel = e.target.value;
                  
                  // Set global flags
                  window.currentRiskFilter = riskLevel;
                  window.highRiskFilterActive = riskLevel === 'high';
                  window.showOnlyHighRiskTrees = riskLevel !== 'all';
                  
                  // Create a filter event with all details
                  const filterEvent = new CustomEvent('applyMapFilters', {
                    detail: {
                      filters: {
                        riskLevel: riskLevel
                      }
                    }
                  });
                  window.dispatchEvent(filterEvent);
                  
                  // For backward compatibility
                  if (riskLevel === 'all') {
                    // If changing to all, reset the filters
                    const resetEvent = new CustomEvent('resetFilters');
                    window.dispatchEvent(resetEvent);
                  }
                }}
              >
                <option value="all" className="text-gray-600 bg-white">Tree Groups</option>
                <option value="high" className="text-gray-600 bg-white">Latest Group</option>
                <option value="high_medium" className="text-gray-600 bg-white">By Location</option>
                <option value="medium" className="text-gray-600 bg-white">By Timestamp</option>
                <option value="low" className="text-gray-600 bg-white">By Area ID</option>
              </select>
              </div>
            </div>
          </div>
      </div>
      
      {/* Tab Navigation */}
      {!collapsed && (
        <div className="flex border-b">
          <button
            className={`flex-1 py-3 font-medium text-sm ${activeTab === 'imagery' ? 'border-b-2 border-indigo-500 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
            onClick={() => setActiveTab('imagery')}
          >
            Review
          </button>
          <button
            className={`flex-1 py-3 font-medium text-sm ${activeTab === 'images' ? 'border-b-2 border-indigo-500 text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
            onClick={() => setActiveTab('images')}
          >
            Images
          </button>
        </div>
      )}
      
      {/* Validation Steps - Progress bar and steps, only in Validation tab */}
      {!collapsed && activeTab === 'imagery' && (
        <div className="px-4 py-3 border-b bg-white">
          <div className="mb-3">
            <div className="w-full bg-gray-100 rounded-full h-2">
              <div 
                className="bg-indigo-500 h-2 rounded-full" 
                style={{ width: filteredItems.length > 0 ? '33%' : '0%' }}
              ></div>
            </div>
          </div>
          <div className="flex justify-between text-xs text-gray-500">
            <div className="flex items-center">
              <div className="w-5 h-5 rounded-full bg-indigo-500 text-white flex items-center justify-center mr-1">1</div>
              <span>Review</span>
            </div>
            <div className="flex items-center text-gray-400">
              <div className="w-5 h-5 rounded-full bg-gray-300 text-gray-500 flex items-center justify-center mr-1">2</div>
              <span>Validate</span>
            </div>
            <div className="flex items-center text-gray-400">
              <div className="w-5 h-5 rounded-full bg-gray-300 text-gray-500 flex items-center justify-center mr-1">3</div>
              <span>Export</span>
            </div>
          </div>
        </div>
      )}
      
      {!collapsed && (
        <div className="flex-1 overflow-auto p-0 bg-white">
          {/* Imagery Tab Content */}
          {activeTab === 'imagery' && filteredItems.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-gray-500">
                <Filter className="h-8 w-8 mx-auto mb-2" />
                <p>No items match your filters</p>
              </div>
            </div>
          ) : activeTab === 'imagery' ? (
          <div className="divide-y">
            {filteredItems.map((item) => (
              <div key={item.id} className="p-3 hover:bg-gray-50">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h3 className="font-medium">
                      {item.tree_species} - {item.tree_height}ft 
                      {item.tree_diameter && ` - Diameter: ${item.tree_diameter}in`}
                      {item.tree_condition && ` - ${item.tree_condition}`}
                    </h3>
                    <div className="text-sm text-gray-500 mb-1">
                      <div className="flex items-start">
                        <MapPin className="h-3 w-3 mr-1 mt-0.5" />
                        <span>
                          {(() => {
                            // Extract location from description (after species)
                            const parts = item.location.description?.split(' - ');
                            let locationDesc = '';
                            
                            if (parts && parts.length > 1) {
                              // If there are multiple parts, the last part is typically the location
                              locationDesc = parts[parts.length - 1];
                            } else if (item.location.description && !item.location.description.includes(item.tree_species)) {
                              // If description doesn't include species, use whole description
                              locationDesc = item.location.description;
                            }
                            
                            // Format for display
                            if (!locationDesc) {
                              return propertyData[item.property_id]?.address || 'Unknown address';
                            }
                            
                            // Remove "of property" if it exists to prevent "of property of address"
                            if (locationDesc.toLowerCase().endsWith('of property')) {
                              locationDesc = locationDesc.substring(0, locationDesc.length - 11);
                            }
                            
                            return propertyData[item.property_id]?.address || locationDesc || 'Unknown address';
                          })()}
                        </span>
                      </div>
                    </div>
                    <div className="flex space-x-2 mt-1">
                      {item.status === 'pending' && (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-yellow-100 text-yellow-800">
                          <Clock className="h-3 w-3 mr-1" />
                          Pending
                        </span>
                      )}
                      {item.status === 'approved' && (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                          <CheckCircle className="h-3 w-3 mr-1" />
                          Approved
                        </span>
                      )}
                      {item.status === 'rejected' && (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-red-100 text-red-800">
                          <XCircle className="h-3 w-3 mr-1" />
                          Rejected
                        </span>
                      )}
                      
                      {/* Show high risk badge if any high risk factors */}
                      {item.riskFactors.some(factor => factor.level === 'high') && (
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-red-100 text-red-800">
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          High Risk
                        </span>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex space-x-1">
                    <button
                      onClick={() => handleViewAnalysis(item.id)}
                      className="p-1 rounded hover:bg-gray-200"
                      title="View Analysis"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                    {item.status === 'pending' && (
                      <>
                        <button
                          onClick={() => handleValidate(item.id, 'approved')}
                          className="p-1 rounded bg-green-100 hover:bg-green-200"
                          disabled={isProcessing}
                          title="Approve"
                        >
                          <CheckCircle className="h-4 w-4 text-green-600" />
                        </button>
                        <button
                          onClick={() => handleValidate(item.id, 'rejected')}
                          className="p-1 rounded bg-red-100 hover:bg-red-200"
                          disabled={isProcessing}
                          title="Reject"
                        >
                          <XCircle className="h-4 w-4 text-red-600" />
                        </button>
                      </>
                    )}
                  </div>
                </div>
                
                <div className="mt-2">
                  <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                    <div>
                      <h4 className="text-xs text-gray-500 mb-1">Tree Attributes:</h4>
                      <ul className="text-sm space-y-1">
                        <li className="flex items-start text-gray-700">
                          <span className="inline-block w-24 text-xs font-medium">Size Category:</span>
                          <span>
                            {item.tree_height <= 30 ? 'Short (<30 ft)' : 
                             item.tree_height <= 50 ? 'Medium (30-50 ft)' : 
                             'Tall (>50 ft)'}
                          </span>
                        </li>
                        {item.tree_diameter && (
                          <li className="flex items-start text-gray-700">
                            <span className="inline-block w-24 text-xs font-medium">DBH:</span>
                            <span>{item.tree_diameter} inches</span>
                          </li>
                        )}
                        {item.canopy_density && (
                          <li className="flex items-start text-gray-700">
                            <span className="inline-block w-24 text-xs font-medium">Canopy:</span>
                            <span>{item.canopy_density}</span>
                          </li>
                        )}
                        {item.leaning_angle && (
                          <li className="flex items-start text-gray-700">
                            <span className="inline-block w-24 text-xs font-medium">Leaning:</span>
                            <span>{item.leaning_angle}</span>
                          </li>
                        )}
                      </ul>
                    </div>
                    
                    <div>
                      <h4 className="text-xs text-gray-500 mb-1">Location Context:</h4>
                      <ul className="text-sm space-y-1">
                        <li className="flex items-start text-gray-700">
                          <span className="inline-block w-24 text-xs font-medium">Property:</span>
                          <span>{item.property_type || 'Residential'}</span>
                        </li>
                        <li className="flex items-start text-gray-700">
                          <span className="inline-block w-24 text-xs font-medium">Proximity:</span>
                          <span>{item.proximity || 'Near structure'}</span>
                        </li>
                        {item.lidar_height && (
                          <li className="flex items-start text-gray-700">
                            <span className="inline-block w-24 text-xs font-medium">LiDAR Height:</span>
                            <span>{item.lidar_height} ft</span>
                          </li>
                        )}
                      </ul>
                    </div>
                  </div>

                  <h4 className="text-xs text-gray-500 my-2">ML Detection Risk Factors:</h4>
                  <ul className="text-sm space-y-1">
                    {item.riskFactors.map((factor, index) => (
                      <li 
                        key={index} 
                        className={`flex items-start ${
                          factor.level === 'high' 
                            ? 'text-red-600' 
                            : factor.level === 'medium' 
                              ? 'text-orange-600' 
                              : 'text-gray-600'
                        }`}
                      >
                        <AlertTriangle className="h-4 w-4 mr-1 flex-shrink-0 mt-0.5" />
                        {factor.description}
                      </li>
                    ))}
                  </ul>
                </div>
                
                <div className="mt-3 flex space-x-2">
                  <button
                    onClick={() => handleViewValidation(item.id)}
                    className="text-xs px-3 py-1 bg-indigo-50 text-indigo-600 rounded hover:bg-indigo-100"
                  >
                    Start Validation
                  </button>
                </div>
              </div>
            ))}
          </div>
          ) : activeTab === 'images' ? (
            <div className="grid grid-cols-2 gap-2 p-4">
              <div className="aspect-video bg-gray-100 rounded-md flex items-center justify-center text-gray-400">
                <p className="text-xs">Aerial Image 1</p>
              </div>
              <div className="aspect-video bg-gray-100 rounded-md flex items-center justify-center text-gray-400">
                <p className="text-xs">Lidar View 1</p>
              </div>
              <div className="aspect-video bg-gray-100 rounded-md flex items-center justify-center text-gray-400">
                <p className="text-xs">Aerial Image 2</p>
              </div>
              <div className="aspect-video bg-gray-100 rounded-md flex items-center justify-center text-gray-400">
                <p className="text-xs">Lidar View 2</p>
              </div>
              <div className="aspect-video bg-gray-100 rounded-md flex items-center justify-center text-gray-400">
                <p className="text-xs">Aerial Image 3</p>
              </div>
              <div className="aspect-video bg-gray-100 rounded-md flex items-center justify-center text-gray-400">
                <p className="text-xs">Lidar View 3</p>
              </div>
            </div>
          ) : null}
        </div>
      )}
      
      {/* Modals */}
      {showAnalysis && <TreeAnalysis selectedTree={selectedItem} onClose={closeModals} />}
      {showValidation && <ValidationSystem selectedTree={selectedItem} onClose={closeModals} />}
      </div>
    </>
  );
};

export default ValidationQueue;