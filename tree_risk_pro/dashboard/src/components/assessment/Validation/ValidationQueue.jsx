// src/components/assessment/Validation/ValidationQueue.jsx

import React, { useState, useEffect } from 'react';
import { AlertTriangle, CheckCircle, XCircle, Clock, Filter, Eye, MapPin, ChevronLeft, ChevronRight, ClipboardList } from 'lucide-react';
import { useValidation } from '../../../hooks/useReportValidation';
import { PropertyService } from '../../../services/api/apiService';
import TreeAnalysis from '../TreeAnalysis/TreeAnalysis';
import ValidationSystem from './ValidationSystem';

// PropTypes:
// showHeader: boolean - Whether to show the header with close button (false when embedded)
// standalone: boolean - Whether this component manages its own sidebar state

const ValidationQueue = ({ showHeader = true, standalone = false }) => {
  const { validationItems, validateItem, isProcessing, refreshValidationItems } = useValidation();
  const [statusFilter, setStatusFilter] = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');
  const [selectedItemId, setSelectedItemId] = useState(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [propertyData, setPropertyData] = useState({});
  const [headerCollapsed, setHeaderCollapsed] = useState(true); // Track header state
  
  // Load data and setup on component mount
  useEffect(() => {
    // Force validation items to load immediately
    if (validationItems.length === 0) {
      refreshValidationItems();
    }
    
    // Set header collapsed state based on current app state
    const handleHeaderCollapse = (event) => {
      setHeaderCollapsed(event.detail.collapsed);
    };
    
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    return () => {
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
    };
  }, [validationItems.length, refreshValidationItems]);
  
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
    
    if (validationItems.length > 0) {
      fetchPropertyData();
    }
  }, [validationItems]);

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
    
    window.addEventListener('applyMapFilters', handleMapFilters);
    window.addEventListener('resetFilters', handleReset);
    
    return () => {
      window.removeEventListener('applyMapFilters', handleMapFilters);
      window.removeEventListener('resetFilters', handleReset);
    };
  }, []);

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

  // Start tree validation mode
  const handleViewValidation = (itemId) => {
    if (!itemId) return;
    
    // Store selected item ID for reference
    setSelectedItemId(itemId);
    
    // Find the selected item with error handling
    const selectedItem = validationItems.find(item => item.id === itemId);
    if (!selectedItem) {
      console.error('Selected validation item not found:', itemId);
      return;
    }
    
    console.log('Starting validation for item:', selectedItem);
    
    // Enhanced tree data object for validation
    // First, make a deep copy of the selected item
    const treeToValidate = JSON.parse(JSON.stringify(selectedItem));
    
    // Ensure consistent data format - depending on the data source, coordinates might be in different formats
    let coordinates;
    if (treeToValidate.location) {
      // Handle case where location is an object with coordinates array
      if (Array.isArray(treeToValidate.location.coordinates)) {
        coordinates = treeToValidate.location.coordinates;
      } 
      // Handle case where location itself is the coordinates array
      else if (Array.isArray(treeToValidate.location)) {
        coordinates = treeToValidate.location;
      }
    } 
    // Fallback to coordinates property if it exists
    else if (treeToValidate.coordinates && Array.isArray(treeToValidate.coordinates)) {
      coordinates = treeToValidate.coordinates;
    }
    
    // Ensure coordinates are present and valid
    if (!coordinates || !Array.isArray(coordinates) || coordinates.length < 2) {
      console.error('Cannot start validation: Invalid coordinates format for item', itemId, coordinates);
      
      // Create fallback coordinates for testing
      coordinates = [-96.75, 32.85]; // Default coordinates (Dallas area)
      console.log('Using fallback coordinates for testing:', coordinates);
    }
    
    // Ensure consistent format for coordinates in the tree object
    treeToValidate.coordinates = coordinates;
    
    // Ensure location object is properly formatted
    if (!treeToValidate.location || !Array.isArray(treeToValidate.location)) {
      treeToValidate.location = coordinates;
    }
    
    // Add risk_level field if not present (for compatibility with detection mode)
    if (!treeToValidate.risk_level) {
      let highestRisk = 'low';
      if (treeToValidate.riskFactors && treeToValidate.riskFactors.length > 0) {
        // Find the highest risk level from risk factors
        treeToValidate.riskFactors.forEach(factor => {
          if (factor.level === 'high') highestRisk = 'high';
          else if (factor.level === 'medium' && highestRisk !== 'high') highestRisk = 'medium';
        });
      }
      treeToValidate.risk_level = highestRisk;
    }
    
    // Calculate a simple bounding box around the tree location
    const margin = 0.001; // About 100 meters depending on latitude
    const bounds = [
      [coordinates[0] - margin, coordinates[1] - margin], 
      [coordinates[0] + margin, coordinates[1] + margin]
    ];
    
    // Create a custom event to show the full screen validation system
    const validationModalEvent = new CustomEvent('showValidationSystem', {
      detail: {
        tree: treeToValidate,
        itemId: itemId
      }
    });
    
    // Dispatch the event to show the validation system
    window.dispatchEvent(validationModalEvent);
    
    // No need to close the sidebar because we want it to remain visible
    // The ValidationSystem will be shown in front of everything, including the map and sidebar
    
    // Set a global flag to indicate that the validation system is shown
    window.validationSystemActive = true;
  };

  const handleViewAnalysis = (itemId) => {
    setSelectedItemId(itemId);
    setShowAnalysis(true);
  };

  const closeModals = () => {
    setShowAnalysis(false);
    setSelectedItemId(null);
    
    // Exit validation mode if necessary
    window.dispatchEvent(new CustomEvent('exitValidationMode', {
      detail: { 
        source: 'validationQueue', 
        clearExisting: true
      }
    }));
  };

  // Get selected item data
  const selectedItem = validationItems.find(item => item.id === selectedItemId);

  // When used in embedded mode (ResizableSidebar), make the content only
  if (!standalone) {
    return (
      <div className="h-full flex flex-col overflow-auto" style={{ width: '100%', boxSizing: 'border-box' }}>
        {/* Filter row with dropdown menus - positioned on the left */}
        <div className="py-2 px-3 flex justify-start space-x-2 bg-gray-100 border-b w-full flex-shrink-0">
          <select
            className="text-xs border rounded px-2 py-1 text-gray-700 bg-white focus:bg-white focus:text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
          >
            <option value="all" className="text-gray-700 bg-white">All Statuses</option>
            <option value="pending" className="text-gray-700 bg-white">Pending</option>
            <option value="approved" className="text-gray-700 bg-white">Approved</option>
            <option value="rejected" className="text-gray-700 bg-white">Rejected</option>
          </select>
          <select
            className="text-xs border rounded px-2 py-1 text-gray-700 bg-white focus:bg-white focus:text-gray-700 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
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
            <option value="all" className="text-gray-400 bg-white">All Risk Levels</option>
            <option value="high" className="text-gray-600 bg-white">High Risk Only</option>
            <option value="medium" className="text-gray-600 bg-white">Medium Risk Only</option>
            <option value="low" className="text-gray-600 bg-white">Low Risk Only</option>
          </select>
        </div>
        
        {/* Main content area */}
        <div className="flex-1 overflow-auto p-0 bg-white w-full">
          {filteredItems.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-gray-500">
                <Filter className="h-8 w-8 mx-auto mb-2" />
                <p>No items match your filters</p>
              </div>
            </div>
          ) : (
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
                              const parts = item.location?.description?.split(' - ');
                              let locationDesc = '';
                              
                              if (parts && parts.length > 1) {
                                // If there are multiple parts, the last part is typically the location
                                locationDesc = parts[parts.length - 1];
                              } else if (item.location?.description && !item.location.description.includes(item.tree_species)) {
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
                          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-amber-100 text-amber-800">
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
                      className="text-xs px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
                    >
                      Start Validation
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* Modals */}
        {showAnalysis && <TreeAnalysis selectedTree={selectedItem} onClose={closeModals} />}
        {/* ValidationSystem is now rendered globally through the ValidationSystemProvider */}
      </div>
    );
  }
  
  // This was the old standalone mode, but we're not using it anymore
  // The component is now managed by the ResizableSidebar in App.jsx
  return null;
};

export default ValidationQueue;