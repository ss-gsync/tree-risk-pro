// src/components/visualization/MapView/MapControls.jsx

import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Search, Home, Crosshair, Filter, MapPin, Download, Target, Box } from 'lucide-react';
import { setMapView } from '../../../features/map/mapSlice';
import { DetectionService } from '../../../services/api/apiService';
import { TreeService } from '../../../services/api/apiService';
import Map3DToggle from './Map3DToggle';

const MapControls = ({ mapRef, mapDataRef }) => {
  const dispatch = useDispatch();
  const { center, zoom } = useSelector((state) => state.map);
  const [searchQuery, setSearchQuery] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    riskLevel: 'all',
    treeHeight: 'all',
    treeSpecies: ''
  });
  const [mapLoaded, setMapLoaded] = useState(false);
  const [treeSpeciesList, setTreeSpeciesList] = useState([]);
  const [is3DMode, setIs3DMode] = useState(true); // Start in 3D mode by default
  const [is3DSupported, setIs3DSupported] = useState(false);
  const [exportStatus, setExportStatus] = useState(null);
  const [isExporting, setIsExporting] = useState(false);
  
  useEffect(() => {
    // Function to check if map is available
    const checkMapReady = () => {
      if (mapRef && mapRef.current && mapRef.current.getMap()) {
        setMapLoaded(true);
        console.log('Map is loaded and ready');
        
        // Check if 3D mode is supported and get current state
        try {
          const mapModeInfo = mapRef.current.getMapMode();
          setIs3DMode(mapModeInfo.is3DMode);
          setIs3DSupported(mapModeInfo.is3DSupported);
          console.log('3D mode support detected:', mapModeInfo.is3DSupported);
        } catch (error) {
          console.warn('Could not get 3D mode info:', error);
          setIs3DSupported(false);
        }
        
        return true;
      }
      return false;
    };

    // Initial check
    if (checkMapReady()) {
      return;
    }

    // If map is not ready, set up an interval to check
    const mapReadyInterval = setInterval(() => {
      if (checkMapReady()) {
        clearInterval(mapReadyInterval);
      }
    }, 500);

    // Clean up interval
    return () => {
      clearInterval(mapReadyInterval);
    };
  }, [mapRef]);
  
  // Listen for map mode changes
  useEffect(() => {
    const handleMapModeChange = (event) => {
      const { mode } = event.detail;
      setIs3DMode(mode === '3D');
    };
    
    window.addEventListener('mapModeChanged', handleMapModeChange);
    
    return () => {
      window.removeEventListener('mapModeChanged', handleMapModeChange);
    };
  }, []);
  
  // Monitor filters state changes and apply them
  useEffect(() => {
    if (mapLoaded) {
      console.log('Filters state changed:', filters);
    }
  }, [filters, mapLoaded]);
  
  // Initialize global state variables on component mount
  useEffect(() => {
    // Set default values for global state variables if they don't exist
    if (window.currentRiskFilter === undefined) {
      window.currentRiskFilter = 'all';
    }
    if (window.highRiskFilterActive === undefined) {
      window.highRiskFilterActive = false;
    }
    if (window.showOnlyHighRiskTrees === undefined) {
      window.showOnlyHighRiskTrees = false;
    }
    
    console.log('Initialized global state variables:', {
      currentRiskFilter: window.currentRiskFilter,
      highRiskFilterActive: window.highRiskFilterActive,
      showOnlyHighRiskTrees: window.showOnlyHighRiskTrees
    });
  }, []);
  
  // Fetch tree species list
  useEffect(() => {
    const fetchTreeSpecies = async () => {
      try {
        // Get species list directly from the API
        const speciesList = await TreeService.getTreeSpecies();
        setTreeSpeciesList(speciesList);
        
        console.log('Loaded tree species list:', speciesList);
      } catch (error) {
        console.error('Error fetching tree species:', error);
        // Fallback to hardcoded species list for demo
        const fallbackSpecies = [
          'Live Oak', 'Post Oak', 'Chinese Pistache', 'Bald Cypress', 
          'Red Oak', 'Southern Magnolia', 'Cedar Elm', 'Pecan', 'Shumard Oak'
        ];
        setTreeSpeciesList(fallbackSpecies);
      }
    };
    
    fetchTreeSpecies();
  }, []);

  // Reset view to default location and clear all filters
  const handleResetView = () => {
    console.log("Reset button clicked");
    
    // Reset map view
    dispatch(setMapView({ 
      center: [-96.7800, 32.8600], // North Dallas area coordinates
      zoom: 10
    }));
    
    // Reset filter values in state
    setFilters({
      riskLevel: 'all',
      treeHeight: 'all',
      treeSpecies: ''
    });
    
    // Set global flags
    window.currentRiskFilter = 'all';
    window.currentSpeciesFilter = '';
    window.currentHeightFilter = 'all';
    window.highRiskFilterActive = false;
    window.showOnlyHighRiskTrees = false;
    
    // Clear existing markers first
    if (mapRef && mapRef.current) {
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
    }
    
    // Create and dispatch event with all filters reset
    const filterEvent = new CustomEvent('applyMapFilters', {
      detail: {
        filters: {
          riskLevel: 'all',
          treeHeight: 'all',
          treeSpecies: ''
        }
      }
    });
    window.dispatchEvent(filterEvent);
    
    // Also dispatch the reset event for backward compatibility
    const resetEvent = new CustomEvent('resetFilters');
    window.dispatchEvent(resetEvent);
    
    // Update the validation queue filter directly
    if (window.setValidationRiskFilter) {
      window.setValidationRiskFilter('all');
    }
  };

  // Get user's current location and place marker
  const handleGetCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { longitude, latitude } = position.coords;
          
          // Update the center position
          dispatch(setMapView({
            center: [longitude, latitude],
            zoom: 15
          }));
          
          // Create and dispatch an event to add a crosshair marker
          const markerEvent = new CustomEvent('addCenterMarker', {
            detail: {
              position: [longitude, latitude]
            }
          });
          window.dispatchEvent(markerEvent);
          
          console.log('Pin placed at:', longitude, latitude);
        },
        (error) => {
          console.error('Error getting current location:', error);
          alert('Unable to retrieve your location. Please check your browser permissions.');
        }
      );
    } else {
      alert('Geolocation is not supported by your browser.');
    }
  };

  // Handle address search
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    
    if (!searchQuery.trim()) return;
    
    // In a real app, this would use the Google Maps Geocoding API
    console.log('Searching for address:', searchQuery);
    alert(`Search functionality coming soon.`);
    
    // Simulated search behavior - move slightly from current position
    const newLng = center[0] + (Math.random() * 0.02 - 0.01);
    const newLat = center[1] + (Math.random() * 0.02 - 0.01);
    
    dispatch(setMapView({
      center: [newLng, newLat],
      zoom: 16
    }));
  };

  // Handle filter changes
  const handleFilterChange = (name, value) => {
    setFilters(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Listen for tree detection requests from sidebar
  useEffect(() => {
    const handleOpenTreeDetection = (event) => {
      console.log("Tree detection requested from sidebar", event.detail);
      
      // Switch to satellite view mode if available
      if (mapRef && mapRef.current && mapRef.current.getMap) {
        try {
          const map = mapRef.current.getMap();
          if (map && map.setMapTypeId) {
            console.log("Setting map to satellite mode for tree detection");
            map.setMapTypeId(window.google.maps.MapTypeId.SATELLITE);
          }
        } catch (error) {
          console.error("Error setting map to satellite mode:", error);
        }
      }
      
      // Get all parameters from the event
      const useSatelliteImagery = event.detail?.useSatelliteImagery !== false;
      const useRealGemini = event.detail?.useRealGemini === true;
      const saveToResponseJson = event.detail?.saveToResponseJson === true;
      const geminiParams = event.detail?.geminiParams || {};
      
      console.log("Tree detection parameters:", {
        useSatelliteImagery,
        useRealGemini,
        saveToResponseJson,
        geminiParamsKeys: Object.keys(geminiParams)
      });
      
      // Start detection with all parameters
      handleExportData(useSatelliteImagery, useRealGemini, saveToResponseJson, geminiParams);
    };
    
    window.addEventListener('openTreeDetection', handleOpenTreeDetection);
    
    return () => {
      window.removeEventListener('openTreeDetection', handleOpenTreeDetection);
    };
  }, [mapRef]);

  // Apply filters
  const handleApplyFilters = (specificRiskLevel = null) => {
    // Use specific risk level if provided, otherwise use state
    const riskLevel = specificRiskLevel || filters.riskLevel;
    
    console.log('Applying filters:', { ...filters, riskLevel });
    
    // Set global flags for all filters
    window.currentRiskFilter = riskLevel;
    window.currentSpeciesFilter = filters.treeSpecies || '';
    window.currentHeightFilter = filters.treeHeight || 'all';
    window.highRiskFilterActive = riskLevel === 'high';
    window.showOnlyHighRiskTrees = riskLevel !== 'all';
    
    // Create and dispatch custom event with filter details
    const filterEvent = new CustomEvent('applyMapFilters', {
      detail: {
        filters: {
          ...filters,
          riskLevel: riskLevel
        }
      }
    });
    window.dispatchEvent(filterEvent);
    
    // Also dispatch refresh event explicitly
    const refreshEvent = new CustomEvent('refreshMapMarkers', { 
      detail: { 
        riskFilter: riskLevel,
        filters: {
          ...filters,
          riskLevel: riskLevel
        }
      } 
    });
    window.dispatchEvent(refreshEvent);
    
    // Update the validation queue filter
    if (window.setValidationRiskFilter) {
      window.setValidationRiskFilter(riskLevel);
    }
    
    // Hide filters panel after applying
    setTimeout(() => {
      setShowFilters(false);
    }, 100);
  };
  
  // Refs for the buttons
  const debugHighRiskButtonRef = React.useRef(null);
  const debugViewAllButtonRef = React.useRef(null);
  const applyFiltersButtonRef = React.useRef(null);

  // Detect trees in the current map view with user validation
  const handleExportData = async (useSatelliteImagery = true, useRealGemini = false, saveToResponseJson = false, geminiParams = {}) => {
    if (!mapRef || !mapRef.current || !mapRef.current.getMap()) {
      alert('Map is not ready. Please try again in a moment.');
      return;
    }
    
    if (isExporting) {
      return; // Already in progress
    }
    
    console.log("Detecting trees with options:", {
      useSatelliteImagery,
      useRealGemini,
      saveToResponseJson,
      geminiParamsKeys: Object.keys(geminiParams)
    });
    
    // Get the current view state from the map
    const map = mapRef.current.getMap();
    const center = [map.getCenter().lng(), map.getCenter().lat()];
    const zoom = map.getZoom();
    const bounds = map.getBounds();
    
    // Check if the zoom level is appropriate for tree detection
    if (zoom < 17) {
      alert('Please zoom in closer to detect trees. We recommend a zoom level of at least 17 for accurate detection.');
      
      // Notify that detection was cancelled due to zoom level
      window.dispatchEvent(new CustomEvent('treeDetectionError', {
        detail: {
          error: 'zoom_level_too_low',
          message: 'Zoom level is too low for accurate detection'
        }
      }));
      
      return;
    }
    
    // Convert bounds to array format
    const boundsArray = bounds ? [
      [bounds.getSouthWest().lng(), bounds.getSouthWest().lat()],
      [bounds.getNorthEast().lng(), bounds.getNorthEast().lat()]
    ] : [
      [center[0] - 0.01, center[1] - 0.01],
      [center[0] + 0.01, center[1] + 0.01]
    ];
    
    // Calculate area size in square degrees - rough estimate
    const areaSize = 
      (boundsArray[1][0] - boundsArray[0][0]) * 
      (boundsArray[1][1] - boundsArray[0][1]);
    
    // Check if area is too large (might be inefficient or slow)
    if (areaSize > 0.01) {
      const continueAnyway = window.confirm(
        'The selected area is quite large. Processing might take longer and could be less accurate. We recommend zooming in closer for better results. Continue anyway?'
      );
      
      if (!continueAnyway) {
        // Notify that detection was cancelled by user
        window.dispatchEvent(new CustomEvent('treeDetectionError', {
          detail: {
            error: 'cancelled_by_user',
            message: 'Detection cancelled by user'
          }
        }));
        return;
      }
    }
    
    // No confirmation needed, just proceed with detection
    
    try {
      setIsExporting(true);
      setExportStatus("Initializing tree detection...");
      
      // We no longer need area_id since all data is stored in TEMP_DIR
      
      // Capture map view parameters for server-side satellite imagery
      setExportStatus("Capturing view parameters...");
      
      let mapViewInfo = null;
      
      try {
        if (!mapRef || !mapRef.current) {
          setExportStatus("Map not initialized");
          console.error("Map reference not available");
          return;
        }
        
        // For 3D view, use Cesium's view parameters
        if (is3DMode) {
          setExportStatus("Capturing 3D view parameters...");
          
          try {
            // Get view parameters directly from Cesium
            const viewData = await mapRef.current.captureCurrentView();
            mapViewInfo = viewData;
            
            setExportStatus("3D view parameters captured!");
            console.log("Captured 3D view parameters for satellite imagery");
          } catch (error) {
            console.log("Error capturing 3D view:", error);
            
            // Fallback to basic coordinates if Cesium capture fails
            mapViewInfo = {
              viewData: {
                bounds: boundsArray,
                center: center,
                zoom: zoom,
                mapType: 'satellite' // Force satellite view
              }
            };
          }
        } else {
          // For 2D view, use Google Maps parameters
          setExportStatus("Capturing 2D view parameters...");
          
          try {
            // Get basic map data directly from Google Maps
            const map = mapRef.current.getMap();
            if (map) {
              const mapBounds = map.getBounds();
              const mapCenter = map.getCenter();
              const viewData = {
                bounds: [
                  [mapBounds.getSouthWest().lng(), mapBounds.getSouthWest().lat()],
                  [mapBounds.getNorthEast().lng(), mapBounds.getNorthEast().lat()]
                ],
                center: [mapCenter.lng(), mapCenter.lat()],
                zoom: map.getZoom(),
                heading: map.getHeading ? map.getHeading() : 0,
                tilt: map.getTilt ? map.getTilt() : 0
              };
              
              mapViewInfo = { viewData };
              setExportStatus("Map parameters captured!");
            } else {
              // Fallback to basic coordinates
              mapViewInfo = {
                viewData: {
                  bounds: boundsArray,
                  center: center,
                  zoom: zoom,
                  mapType: 'satellite' // Force satellite view
                }
              };
            }
          } catch (error) {
            console.error("Error capturing view parameters:", error);
            
            // Last resort - use basic coordinates
            mapViewInfo = {
              viewData: {
                bounds: boundsArray,
                center: center,
                zoom: zoom,
                mapType: 'satellite' // Force satellite view
              }
            };
          }
        }
      } catch (outerError) {
        console.error("Critical error capturing view parameters:", outerError);
        
        // Recover with basic data
        mapViewInfo = {
          viewData: {
            bounds: boundsArray,
            center: center,
            zoom: zoom,
            mapType: 'satellite' // Force satellite view
          }
        };
      }
      
      // Prepare request data with the map view data
      const requestData = {
        debug: true,
        include_bounding_boxes: true,
        assistance_mode: true,
        use_satellite_imagery: useSatelliteImagery,
        use_real_gemini: useRealGemini,
        save_to_response_json: saveToResponseJson
      };
      
      // Add Gemini parameters if provided
      if (Object.keys(geminiParams).length > 0) {
        requestData.gemini_params = geminiParams;
      }
      
      // Include view parameters for server-side satellite imagery
      console.log("Preparing request data with view parameters:", { hasViewInfo: !!mapViewInfo });
      
      // Add view information for server-side satellite image retrieval
      if (mapViewInfo) {
        console.log("Including map view parameters in request");
        requestData.map_view_info = mapViewInfo;
        
        // Include coordinates directly if available
        if (mapViewInfo.viewData) {
          requestData.coordinates = mapViewInfo.viewData;
        } else {
          // Fallback to basic coordinates
          requestData.coordinates = {
            center: center,
            zoom: zoom,
            bounds: boundsArray
          };
        }
      } else {
        // Last resort - just use the current map state
        console.log("No view parameters available, using basic coordinates");
        requestData.coordinates = {
          center: center,
          zoom: zoom,
          bounds: boundsArray
        };
      }
      
      // Start detection process
      setExportStatus("Running ML detection, please wait...");
      const result = await DetectionService.detectTrees(requestData);
      
      // Check if the initial detection result has an error
      if (result.status === 'error') {
        setIsExporting(false);
        console.error("Detection error:", result.error || "Error in tree detection. Please check the logs.");
        // Don't display alert here to avoid duplicate error messages
        return;
      }
      
      // Store job ID for polling
      const jobId = result.job_id;
      setExportStatus("Processing Job...");
      
      // Get detection results right away
      try {
        const statusResult = await DetectionService.getDetectionStatus(jobId);
        setExportStatus(statusResult.message || "Detection complete");
        setIsExporting(false);
        
        // Get tree count
        const treeCount = statusResult.tree_count || 0;
        
        // Check for errors first, then tree count
        if (statusResult.status === 'error') {
          console.error("Detection status error:", statusResult.error || "Error in tree detection. Please check the logs.");
          // No alert to avoid duplicate messages
          return;
        } else if (treeCount === 0) {
          alert('No trees were detected.');
          return;
        }
        
        // Create and dispatch event to enter tree validation mode with the detection results
        // This will notify the MapView component to enter validation mode
        const validationEvent = new CustomEvent('enterTreeValidationMode', {
          detail: {
            detectionJobId: jobId,
            treeCount: treeCount,
            bounds: boundsArray,
            trees: statusResult.trees || [] // Include detected trees directly
          }
        });
        window.dispatchEvent(validationEvent);
        
        // Set status to show user next steps
        setExportStatus("Validation mode: Review detected trees");
      } catch (error) {
        console.error('Error getting detection results:', error);
        setExportStatus("Error getting detection results");
        setIsExporting(false);
      }
      
    } catch (error) {
      console.error('Error detecting trees:', error);
      setIsExporting(false);
      setExportStatus("Error during detection");
      alert('Failed to detect trees: ' + error.message);
    }
  };
  
  // Modified to toggle between Google Maps and 3D Tiles
  const toggle3DMode = () => {
    // First, toggle local state for UI indication
    setIs3DMode(!is3DMode);
    
    // Check for the 3D API type from settings
    const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
    let map3DApi = 'cesium'; // Default to Cesium
    
    if (savedSettings) {
      try {
        const settings = JSON.parse(savedSettings);
        map3DApi = settings.map3DApi || 'cesium';
      } catch (e) {
        console.error('Error parsing map 3D API settings:', e);
      }
    }
    
    // Dispatch an event that App.jsx will listen for
    const toggleEvent = new CustomEvent('requestToggle3DViewType', {
      detail: {
        show3D: !is3DMode,
        map3DApi: map3DApi // Include the 3D API type from settings
      }
    });
    window.dispatchEvent(toggleEvent);
    
    console.log(`Toggling 3D view with API: ${map3DApi}`);
  };

  return (
    <div className="w-full">
      
      {/* Quick actions */}
      <div className="flex space-x-2 mb-3">
        <button
          onClick={handleResetView}
          className="flex items-center justify-center p-2 bg-gray-100 rounded-md hover:bg-gray-200 flex-1"
          title="Reset view"
        >
          <Home className="h-4 w-4 mr-1" />
          <span className="text-xs">Reset</span>
        </button>
        
        <button
          onClick={handleGetCurrentLocation}
          className="flex items-center justify-center p-2 bg-gray-100 rounded-md hover:bg-gray-200 flex-1"
          title="Go to location"
        >
          <Crosshair className="h-4 w-4 mr-1" />
          <span className="text-xs">Pin</span>
        </button>
        
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`flex items-center justify-center p-2 rounded-md flex-1 ${
            showFilters ? 'bg-emerald-100 text-emerald-600' : 'bg-gray-100 hover:bg-gray-200'
          }`}
          title="Filter data"
        >
          <Filter className="h-4 w-4 mr-1" />
          <span className="text-xs">Filter</span>
        </button>
      </div>
      
      {/* Filters panel */}
      {showFilters && (
        <div className="p-2 bg-gray-50 rounded-md mb-3">
          <h3 className="font-medium text-xs mb-2">Filters</h3>
          
          <div className="mb-2">
            <label className="block text-xs text-gray-600 mb-1">Risk Level</label>
            <select
              className="w-full p-1 text-xs border rounded text-gray-400 bg-gray-50 focus:bg-white focus:text-gray-600 focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
              value={filters.riskLevel}
              onChange={(e) => handleFilterChange('riskLevel', e.target.value)}
            >
              <option value="all" className="text-gray-400 bg-white">Low Risk</option>
              <option value="high" className="text-gray-600 bg-white">High Risk Only</option>
              <option value="medium" className="text-gray-600 bg-white">Medium Risk Only</option>
              <option value="low" className="text-gray-600 bg-white">Low Risk Only</option>
            </select>
          </div>
          
          <div className="mb-2">
            <label className="block text-xs text-gray-600 mb-1">Tree Height</label>
            <select
              className="w-full p-1 text-xs border rounded text-gray-400 bg-gray-50 focus:bg-white focus:text-gray-600 focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
              value={filters.treeHeight}
              onChange={(e) => handleFilterChange('treeHeight', e.target.value)}
            >
              <option value="all" className="text-gray-400 bg-white">All Heights</option>
              <option value="tall" className="text-gray-600 bg-white">Tall ({'>'}50 ft)</option>
              <option value="medium" className="text-gray-600 bg-white">Medium (30-50 ft)</option>
              <option value="short" className="text-gray-600 bg-white">Short ({'<'}30 ft)</option>
            </select>
          </div>
          
          <div className="mb-3">
            <label className="block text-xs text-gray-600 mb-1">Tree Species</label>
            <select
              className="w-full p-1 text-xs border rounded text-gray-400 bg-gray-50 focus:bg-white focus:text-gray-600 focus:outline-none focus:ring-1 focus:ring-emerald-500 focus:border-emerald-500"
              value={filters.treeSpecies}
              onChange={(e) => handleFilterChange('treeSpecies', e.target.value)}
            >
              <option value="" className="text-gray-400 bg-white">All Species</option>
              {treeSpeciesList.map((species, index) => (
                <option key={index} value={species} className="text-gray-600 bg-white">
                  {species}
                </option>
              ))}
            </select>
          </div>
          
          <button
            ref={applyFiltersButtonRef}
            onClick={() => handleApplyFilters()}
            className="w-full bg-emerald-500 hover:bg-emerald-600 text-white p-1.5 rounded text-xs transition-colors"
          >
            Apply Filters
          </button>
        </div>
      )}
      
      {/* Map info */}
      <div className="text-xs text-gray-500 mb-3">
        <div className="flex justify-between">
          <span>Center:</span>
          <span className="font-mono">
            {center[1].toFixed(4)}, {center[0].toFixed(4)}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Zoom:</span>
          <span className="font-mono">{zoom.toFixed(1)}</span>
        </div>
      </div>
      
      {/* 3D View Toggle removed */}
      
      {/* Actions */}
      <div className="space-y-2">
        <div className="flex space-x-2">
          <button
            id="toggleHighRiskButton"
            onClick={() => {
              console.log('Toggle High Risk button clicked');
              
              // Get the current risk filter state
              let currentFilter = window.currentRiskFilter || 'all';
              let newFilter;
              
              // Implement toggle functionality: high -> high+medium -> all -> high
              if (currentFilter === 'high') {
                // First click: already showing high risk, add medium risk
                console.log('Adding medium risk trees');
                newFilter = 'high_medium';
                
                // Set filter values in state
                setFilters({
                  riskLevel: 'high_medium',
                  treeHeight: 'all',
                  treeSpecies: filters.treeSpecies || ''
                });
              } else if (currentFilter === 'high_medium') {
                // Second click: already showing high and medium risk, show all risks
                console.log('Showing all risk levels');
                newFilter = 'all';
                
                // Set filter values in state
                setFilters({
                  riskLevel: 'all',
                  treeHeight: 'all',
                  treeSpecies: filters.treeSpecies || ''
                });
              } else {
                // Third click or first time: only show high risk
                console.log('Showing only high risk trees');
                newFilter = 'high';
                
                // Set filter values in state
                setFilters({
                  riskLevel: 'high',
                  treeHeight: 'all',
                  treeSpecies: filters.treeSpecies || ''
                });
              }
              
              // The more explicit applyMapFilters event
              const filterEvent = new CustomEvent('applyMapFilters', {
                detail: {
                  filters: {
                    riskLevel: newFilter,
                    treeHeight: 'all',
                    treeSpecies: filters.treeSpecies || ''
                  }
                }
              });
              window.dispatchEvent(filterEvent);
              
              // Set global flags (fallback)
              window.currentRiskFilter = newFilter;
              window.highRiskFilterActive = newFilter === 'high' || newFilter === 'high_medium';
              window.showOnlyHighRiskTrees = newFilter !== 'all';
              
              // Directly refresh markers
              const refreshEvent = new CustomEvent('refreshMapMarkers', {
                detail: {
                  riskFilter: newFilter
                }
              });
              window.dispatchEvent(refreshEvent);
              
              // Direct update of validation queue
              if (window.setValidationRiskFilter) {
                window.setValidationRiskFilter(newFilter);
              }
            }}
            disabled={!mapLoaded}
            className="flex items-center justify-center p-2 bg-red-100 text-red-700 rounded-md hover:bg-red-200 flex-1 cursor-pointer"
          >
            <Target className="h-4 w-4 mr-1" />
            <span className="text-xs">Risk</span>
          </button>
          
          <button
            id="mediumRiskButton"
            onClick={() => {
              console.log('Medium Risk button clicked');
              
              // Keep existing species filter but set risk level to medium
              const currentSpecies = filters.treeSpecies || '';
              setFilters({
                riskLevel: 'medium',
                treeHeight: 'all',
                treeSpecies: currentSpecies
              });
              
              // More explicit filter event
              const filterEvent = new CustomEvent('applyMapFilters', {
                detail: {
                  filters: {
                    riskLevel: 'medium',
                    treeHeight: 'all',
                    treeSpecies: currentSpecies
                  }
                }
              });
              window.dispatchEvent(filterEvent);
              
              // Set global flags
              window.currentRiskFilter = 'medium';
              window.highRiskFilterActive = false;
              window.showOnlyHighRiskTrees = false;
              
              // Direct marker refresh
              const refreshEvent = new CustomEvent('refreshMapMarkers', {
                detail: {
                  riskFilter: 'medium'
                }
              });
              window.dispatchEvent(refreshEvent);
              
              // Direct update of validation queue
              if (window.setValidationRiskFilter) {
                window.setValidationRiskFilter('medium');
              }
            }}
            disabled={!mapLoaded}
            className="flex items-center justify-center p-2 bg-orange-100 text-orange-700 rounded-md hover:bg-orange-200 flex-1 cursor-pointer"
          >
            <Target className="h-4 w-4 mr-1" />
            <span className="text-xs">Med</span>
          </button>
          
          <button
            id="lowRiskButton"
            onClick={() => {
              console.log('Low Risk button clicked');
              
              // Keep existing species filter but set risk level to low
              const currentSpecies = filters.treeSpecies || '';
              setFilters({
                riskLevel: 'low',
                treeHeight: 'all',
                treeSpecies: currentSpecies
              });
              
              // More explicit filter event
              const filterEvent = new CustomEvent('applyMapFilters', {
                detail: {
                  filters: {
                    riskLevel: 'low',
                    treeHeight: 'all',
                    treeSpecies: currentSpecies
                  }
                }
              });
              window.dispatchEvent(filterEvent);
              
              // Set global flags (fallback)
              window.currentRiskFilter = 'low';
              window.highRiskFilterActive = false;
              window.showOnlyHighRiskTrees = false;
              
              // Direct marker refresh
              const refreshEvent = new CustomEvent('refreshMapMarkers', {
                detail: {
                  riskFilter: 'low'
                }
              });
              window.dispatchEvent(refreshEvent);
              
              // Direct update of validation queue
              if (window.setValidationRiskFilter) {
                window.setValidationRiskFilter('low');
              }
            }}
            disabled={!mapLoaded}
            className="flex items-center justify-center p-2 bg-yellow-100 text-yellow-700 rounded-md hover:bg-yellow-200 flex-1 cursor-pointer"
          >
            <Target className="h-4 w-4 mr-1" />
            <span className="text-xs">Low</span>
          </button>
        </div>
        
        {/* Tree Detection button removed from here and moved to Analytics section in sidebar */}
      </div>
    </div>
  );
};

export default MapControls;