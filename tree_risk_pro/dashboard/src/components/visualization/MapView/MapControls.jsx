// src/components/visualization/MapView/MapControls.jsx

import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Search, Home, Crosshair, Filter, MapPin, Download, Target, Box, Map, Image, Layers } from 'lucide-react';
import { setMapView } from '../../../features/map/mapSlice';
import { DetectionService } from '../../../services/api/apiService';
import { TreeService } from '../../../services/api/apiService';
import Map3DToggle from './Map3DToggle';

const MapControls = ({ mapRef, mapDataRef, viewSwitchFunction }) => {
  // Helper function to ensure Map view is active
  const ensureMapView = (callback) => {
    // Check if we're in a different view that needs to be switched
    if (typeof viewSwitchFunction === 'function') {
      viewSwitchFunction(callback);
    } else {
      // No view switching function provided, execute callback directly
      if (callback) callback();
    }
  };
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
  const [is3DMode, setIs3DMode] = useState(false); // Start in 2D mode by default
  const [is3DSupported, setIs3DSupported] = useState(false);
  const [exportStatus, setExportStatus] = useState(null);
  const [isExporting, setIsExporting] = useState(false);
  // Get saved map type from localStorage or default to hybrid
  const savedMapType = localStorage.getItem('currentMapType') || 'hybrid';
  // Set global state for consistency
  window.currentMapType = savedMapType;
  
  // Initialize state with saved preference
  const [mapType, setMapType] = useState(savedMapType);
  
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
          
          // Make sure we update the button text to match the current map type
          setTimeout(() => {
            try {
              const currentMapType = mapRef.current.getMap().getMapTypeId();
              
              // Convert Google Maps type to our internal type
              let mappedType = 'hybrid';
              if (currentMapType === window.google.maps.MapTypeId.SATELLITE) {
                mappedType = 'satellite';
              } else if (currentMapType === window.google.maps.MapTypeId.ROADMAP) {
                mappedType = 'map';
              }
              
              // Update button text if needed
              const btn = document.getElementById('map-type-toggle-button');
              const spanElement = btn ? btn.querySelector('span') : null;
              
              if (spanElement) {
                if (mappedType === 'satellite') {
                  spanElement.textContent = 'Satellite';
                } else if (mappedType === 'map') {
                  spanElement.textContent = 'Map';
                } else {
                  spanElement.textContent = 'Hybrid';
                }
                console.log(`Updated map type toggle button text to: ${spanElement.textContent}`);
              }
            } catch (err) {
              console.warn('Error updating map type toggle button:', err);
            }
          }, 200);
                
          
          // Force map to use stored map type preference on initial load
          try {
            const map = mapRef.current.getMap();
            if (map) {
              const storedMapType = localStorage.getItem('currentMapType') || 'hybrid';
              let googleMapType;
              
              switch (storedMapType) {
                case 'map':
                  googleMapType = window.google.maps.MapTypeId.ROADMAP;
                  break;
                case 'satellite':
                  googleMapType = window.google.maps.MapTypeId.SATELLITE;
                  break;
                case 'hybrid':
                default:
                  googleMapType = window.google.maps.MapTypeId.HYBRID;
                  break;
              }
              
              console.log(`Setting initial map type to: ${storedMapType}`);
              map.setMapTypeId(googleMapType);
              
              // Update UI state to match
              setMapType(storedMapType);
              
              // Also update the button text for map type toggle
              setTimeout(() => {
                const btn = document.getElementById('map-type-toggle-button');
                const spanElement = btn ? btn.querySelector('span') : null;
                
                if (spanElement) {
                  if (storedMapType === 'satellite') {
                    spanElement.textContent = 'Satellite';
                  } else if (storedMapType === 'map') {
                    spanElement.textContent = 'Map';
                  } else {
                    spanElement.textContent = 'Hybrid';
                  }
                }
              }, 100);
            }
          } catch (e) {
            console.warn('Error setting initial map type:', e);
          }
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

  // Get user's current location with geocoding and place marker
  const handleGetCurrentLocation = () => {
    // Show loading toast
    const loadingToast = document.createElement('div');
    loadingToast.className = 'geolocation-toast';
    loadingToast.style.position = 'absolute';
    loadingToast.style.top = '70px';
    loadingToast.style.left = '50%';
    loadingToast.style.transform = 'translateX(-50%)';
    loadingToast.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
    loadingToast.style.color = 'white';
    loadingToast.style.padding = '8px 16px';
    loadingToast.style.borderRadius = '4px';
    loadingToast.style.zIndex = '9999';
    loadingToast.style.fontSize = '14px';
    loadingToast.style.fontWeight = '500';
    loadingToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
    loadingToast.textContent = 'Getting your location...';
    document.body.appendChild(loadingToast);

    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { longitude, latitude } = position.coords;
          
          // Update the center position
          dispatch(setMapView({
            center: [longitude, latitude],
            zoom: 15
          }));
          
          // Use Google Maps Geocoding API to get the address
          if (window.google && window.google.maps && window.google.maps.Geocoder) {
            const geocoder = new window.google.maps.Geocoder();
            geocoder.geocode({ location: { lat: latitude, lng: longitude } }, (results, status) => {
              // Remove loading toast
              if (loadingToast.parentNode) {
                loadingToast.parentNode.removeChild(loadingToast);
              }
              
              if (status === "OK" && results[0]) {
                const address = results[0].formatted_address;
                console.log(`Current location address: ${address}`);
                
                // Create and dispatch an event to add a marker with address
                const markerEvent = new CustomEvent('addLocationMarker', {
                  detail: {
                    position: [longitude, latitude],
                    address: address
                  }
                });
                window.dispatchEvent(markerEvent);
                
                // Show success toast with address
                const successToast = document.createElement('div');
                successToast.className = 'location-toast';
                successToast.style.position = 'absolute';
                successToast.style.top = '70px';
                successToast.style.left = '50%';
                successToast.style.transform = 'translateX(-50%)';
                successToast.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
                successToast.style.color = 'white';
                successToast.style.padding = '8px 16px';
                successToast.style.borderRadius = '4px';
                successToast.style.zIndex = '9999';
                successToast.style.fontSize = '14px';
                successToast.style.fontWeight = '500';
                successToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
                successToast.innerHTML = `
                  <div class="flex flex-col">
                    <span>Location found</span>
                    <span class="text-xs mt-1">${address}</span>
                  </div>
                `;
                document.body.appendChild(successToast);
                
                // Remove success toast after 3 seconds
                setTimeout(() => {
                  if (successToast.parentNode) {
                    successToast.parentNode.removeChild(successToast);
                  }
                }, 3000);
              } else {
                console.warn("Geocoder failed due to: " + status);
                
                // Create and dispatch an event to add a marker without address
                const markerEvent = new CustomEvent('addLocationMarker', {
                  detail: {
                    position: [longitude, latitude],
                    address: null
                  }
                });
                window.dispatchEvent(markerEvent);
                
                // Show simple success toast
                const simpleToast = document.createElement('div');
                simpleToast.className = 'location-toast';
                simpleToast.style.position = 'absolute';
                simpleToast.style.top = '70px';
                simpleToast.style.left = '50%';
                simpleToast.style.transform = 'translateX(-50%)';
                simpleToast.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
                simpleToast.style.color = 'white';
                simpleToast.style.padding = '8px 16px';
                simpleToast.style.borderRadius = '4px';
                simpleToast.style.zIndex = '9999';
                simpleToast.style.fontSize = '14px';
                simpleToast.style.fontWeight = '500';
                simpleToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
                simpleToast.textContent = 'Location found';
                document.body.appendChild(simpleToast);
                
                // Remove toast after 2 seconds
                setTimeout(() => {
                  if (simpleToast.parentNode) {
                    simpleToast.parentNode.removeChild(simpleToast);
                  }
                }, 2000);
              }
            });
          } else {
            // Remove loading toast
            if (loadingToast.parentNode) {
              loadingToast.parentNode.removeChild(loadingToast);
            }
            
            console.warn("Google Maps Geocoder not available");
            
            // Create and dispatch an event to add a crosshair marker
            const markerEvent = new CustomEvent('addCenterMarker', {
              detail: {
                position: [longitude, latitude]
              }
            });
            window.dispatchEvent(markerEvent);
            
            // Show simple success toast
            const simpleToast = document.createElement('div');
            simpleToast.className = 'location-toast';
            simpleToast.style.position = 'absolute';
            simpleToast.style.top = '70px';
            simpleToast.style.left = '50%';
            simpleToast.style.transform = 'translateX(-50%)';
            simpleToast.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
            simpleToast.style.color = 'white';
            simpleToast.style.padding = '8px 16px';
            simpleToast.style.borderRadius = '4px';
            simpleToast.style.zIndex = '9999';
            simpleToast.style.fontSize = '14px';
            simpleToast.style.fontWeight = '500';
            simpleToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
            simpleToast.textContent = 'Location found';
            document.body.appendChild(simpleToast);
            
            // Remove toast after 2 seconds
            setTimeout(() => {
              if (simpleToast.parentNode) {
                simpleToast.parentNode.removeChild(simpleToast);
              }
            }, 2000);
          }
          
          console.log('Pin placed at:', longitude, latitude);
        },
        (error) => {
          // Remove loading toast
          if (loadingToast.parentNode) {
            loadingToast.parentNode.removeChild(loadingToast);
          }
          
          console.error('Error getting current location:', error);
          
          // Show error toast
          const errorToast = document.createElement('div');
          errorToast.className = 'error-toast';
          errorToast.style.position = 'absolute';
          errorToast.style.top = '70px';
          errorToast.style.left = '50%';
          errorToast.style.transform = 'translateX(-50%)';
          errorToast.style.backgroundColor = 'rgba(220, 38, 38, 0.9)';
          errorToast.style.color = 'white';
          errorToast.style.padding = '8px 16px';
          errorToast.style.borderRadius = '4px';
          errorToast.style.zIndex = '9999';
          errorToast.style.fontSize = '14px';
          errorToast.style.fontWeight = '500';
          errorToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
          errorToast.textContent = 'Unable to retrieve your location. Please check your browser permissions.';
          document.body.appendChild(errorToast);
          
          // Remove toast after 3 seconds
          setTimeout(() => {
            if (errorToast.parentNode) {
              errorToast.parentNode.removeChild(errorToast);
            }
          }, 3000);
        }
      );
    } else {
      // Remove loading toast
      if (loadingToast.parentNode) {
        loadingToast.parentNode.removeChild(loadingToast);
      }
      
      // Show error toast
      const errorToast = document.createElement('div');
      errorToast.className = 'error-toast';
      errorToast.style.position = 'absolute';
      errorToast.style.top = '70px';
      errorToast.style.left = '50%';
      errorToast.style.transform = 'translateX(-50%)';
      errorToast.style.backgroundColor = 'rgba(220, 38, 38, 0.9)';
      errorToast.style.color = 'white';
      errorToast.style.padding = '8px 16px';
      errorToast.style.borderRadius = '4px';
      errorToast.style.zIndex = '9999';
      errorToast.style.fontSize = '14px';
      errorToast.style.fontWeight = '500';
      errorToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
      errorToast.textContent = 'Geolocation is not supported by your browser.';
      document.body.appendChild(errorToast);
      
      // Remove toast after 3 seconds
      setTimeout(() => {
        if (errorToast.parentNode) {
          errorToast.parentNode.removeChild(errorToast);
        }
      }, 3000);
    }
  };

  // Handle address search with Geocoding API
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    
    if (!searchQuery.trim()) return;
    
    // Show loading toast
    const loadingToast = document.createElement('div');
    loadingToast.className = 'search-toast';
    loadingToast.style.position = 'absolute';
    loadingToast.style.top = '70px';
    loadingToast.style.left = '50%';
    loadingToast.style.transform = 'translateX(-50%)';
    loadingToast.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
    loadingToast.style.color = 'white';
    loadingToast.style.padding = '8px 16px';
    loadingToast.style.borderRadius = '4px';
    loadingToast.style.zIndex = '9999';
    loadingToast.style.fontSize = '14px';
    loadingToast.style.fontWeight = '500';
    loadingToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
    loadingToast.textContent = `Searching for "${searchQuery}"...`;
    document.body.appendChild(loadingToast);
    
    // Use Google Maps Geocoding API
    if (window.google && window.google.maps && window.google.maps.Geocoder) {
      const geocoder = new window.google.maps.Geocoder();
      
      geocoder.geocode({ address: searchQuery }, (results, status) => {
        // Remove loading toast
        if (loadingToast.parentNode) {
          loadingToast.parentNode.removeChild(loadingToast);
        }
        
        if (status === "OK" && results[0]) {
          console.log('Address search result:', results[0]);
          
          // Get location and formatted address
          const location = results[0].geometry.location;
          const address = results[0].formatted_address;
          
          // Update the map view
          dispatch(setMapView({
            center: [location.lng(), location.lat()],
            zoom: 16
          }));
          
          // Create and dispatch an event to add a marker with address
          const markerEvent = new CustomEvent('addLocationMarker', {
            detail: {
              position: [location.lng(), location.lat()],
              address: address
            }
          });
          window.dispatchEvent(markerEvent);
          
          // Clear search query
          setSearchQuery('');
          
          // Show success toast with address
          const successToast = document.createElement('div');
          successToast.className = 'search-success-toast';
          successToast.style.position = 'absolute';
          successToast.style.top = '70px';
          successToast.style.left = '50%';
          successToast.style.transform = 'translateX(-50%)';
          successToast.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
          successToast.style.color = 'white';
          successToast.style.padding = '8px 16px';
          successToast.style.borderRadius = '4px';
          successToast.style.zIndex = '9999';
          successToast.style.fontSize = '14px';
          successToast.style.fontWeight = '500';
          successToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
          successToast.innerHTML = `
            <div class="flex flex-col">
              <span>Address found</span>
              <span class="text-xs mt-1">${address}</span>
            </div>
          `;
          document.body.appendChild(successToast);
          
          // Remove success toast after 3 seconds
          setTimeout(() => {
            if (successToast.parentNode) {
              successToast.parentNode.removeChild(successToast);
            }
          }, 3000);
        } else {
          console.warn("Geocoding failed due to:", status);
          
          // Show error toast
          const errorToast = document.createElement('div');
          errorToast.className = 'search-error-toast';
          errorToast.style.position = 'absolute';
          errorToast.style.top = '70px';
          errorToast.style.left = '50%';
          errorToast.style.transform = 'translateX(-50%)';
          errorToast.style.backgroundColor = 'rgba(220, 38, 38, 0.9)';
          errorToast.style.color = 'white';
          errorToast.style.padding = '8px 16px';
          errorToast.style.borderRadius = '4px';
          errorToast.style.zIndex = '9999';
          errorToast.style.fontSize = '14px';
          errorToast.style.fontWeight = '500';
          errorToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
          errorToast.textContent = `Could not find location "${searchQuery}"`;
          document.body.appendChild(errorToast);
          
          // Remove error toast after 3 seconds
          setTimeout(() => {
            if (errorToast.parentNode) {
              errorToast.parentNode.removeChild(errorToast);
            }
          }, 3000);
        }
      });
    } else {
      // Remove loading toast
      if (loadingToast.parentNode) {
        loadingToast.parentNode.removeChild(loadingToast);
      }
      
      console.warn("Google Maps Geocoder not available");
      
      // Show error toast
      const errorToast = document.createElement('div');
      errorToast.className = 'geocoder-error-toast';
      errorToast.style.position = 'absolute';
      errorToast.style.top = '70px';
      errorToast.style.left = '50%';
      errorToast.style.transform = 'translateX(-50%)';
      errorToast.style.backgroundColor = 'rgba(220, 38, 38, 0.9)';
      errorToast.style.color = 'white';
      errorToast.style.padding = '8px 16px';
      errorToast.style.borderRadius = '4px';
      errorToast.style.zIndex = '9999';
      errorToast.style.fontSize = '14px';
      errorToast.style.fontWeight = '500';
      errorToast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
      errorToast.textContent = 'Geocoding service is not available';
      document.body.appendChild(errorToast);
      
      // Remove error toast after 3 seconds
      setTimeout(() => {
        if (errorToast.parentNode) {
          errorToast.parentNode.removeChild(errorToast);
        }
      }, 3000);
    }
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
            console.log("Setting map to hybrid mode for tree detection");
            map.setMapTypeId(window.google.maps.MapTypeId.HYBRID);
          }
        } catch (error) {
          console.error("Error setting map to hybrid mode:", error);
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
  
  // Enhanced toggle between 2D and 3D views with improved state management
  const toggle3DMode = () => {
    ensureMapView(() => {
      console.log(`Toggling from ${is3DMode ? '3D' : '2D'} to ${!is3DMode ? '3D' : '2D'} mode`);
      
      // Save current map type for when we return to 2D
      const currentMapTypeBeforeToggle = mapType;
      
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
      
      // If switching TO 3D, remember the current 2D state
      if (!is3DMode) {
        localStorage.setItem('lastMapTypeBefore3D', mapType);
        console.log(`Saving current 2D map type (${mapType}) before switching to 3D`);
      }
      // If switching FROM 3D to 2D, restore the saved state or default to hybrid
      else {
        const savedMapType = localStorage.getItem('lastMapTypeBefore3D') || 'hybrid';
        console.log(`Restoring 2D map type to ${savedMapType} after exiting 3D`);
        
        // Update state and button text for hybrid/satellite toggle
        setMapType(savedMapType);
        
        // After a brief delay to let the map initialize, apply the saved map type
        setTimeout(() => {
          const btn = document.getElementById('hybrid-satellite-button');
          const spanElement = btn ? btn.querySelector('span') : null;
          
          if (spanElement) {
            if (savedMapType === 'satellite') {
              spanElement.textContent = 'Satellite';
            } else if (savedMapType === 'hybrid' || savedMapType === 'map') {
              spanElement.textContent = 'Hybrid';
            }
          }
        }, 100);
      }
      
      // Dispatch an event that App.jsx will listen for
      const toggleEvent = new CustomEvent('requestToggle3DViewType', {
        detail: {
          show3D: !is3DMode,
          map3DApi: map3DApi, // Include the 3D API type from settings
          previousMapType: currentMapTypeBeforeToggle // Pass the previous map type
        }
      });
      window.dispatchEvent(toggleEvent);
      
      console.log(`Toggling 3D view with API: ${map3DApi}`);
    });
  };

  // Enhanced map type handler with improved state management for all 5 possible states
  const handleMapTypeChange = (newType) => {
    console.log(`Changing map type to: ${newType}`);

    // If switching to 3D view
    if (newType === '3d') {
      // Only perform the 3D toggle if not already in 3D mode
      if (!is3DMode) {
        toggle3DMode();
      }
      return;
    }
    
    // For 2D modes (Map/Hybrid/Satellite), ensure we're in 2D mode first
    if (is3DMode) {
      // Store the requested map type for after 3D toggle completes
      localStorage.setItem('pendingMapTypeAfter3D', newType);
      console.log(`3D mode active, storing requested map type (${newType}) for after 3D exit`);
      
      // Turn off 3D mode first
      toggle3DMode();
      
      // Apply the requested map type after a delay to allow 3D toggle to complete
      setTimeout(() => {
        const pendingType = localStorage.getItem('pendingMapTypeAfter3D');
        if (pendingType) {
          console.log(`Applying pending map type (${pendingType}) after 3D exit`);
          handleMapTypeChange(pendingType);
          localStorage.removeItem('pendingMapTypeAfter3D');
        }
      }, 700);
      
      return;
    }
    
    // Update React state immediately for responsive UI
    setMapType(newType);
    
    // Update the button text for the map type toggle
    const btn = document.getElementById('map-type-toggle-button');
    const spanElement = btn ? btn.querySelector('span') : null;
    
    if (spanElement) {
      if (newType === 'satellite') {
        spanElement.textContent = 'Satellite';
      } else if (newType === 'map') {
        spanElement.textContent = 'Map';
      } else {
        spanElement.textContent = 'Hybrid';
      }
    }
    
    // Always store preferences regardless of map initialization state
    try {
      localStorage.setItem('currentMapType', newType);
      window.currentMapType = newType; // Update global state
      console.log(`Stored map type preference: ${newType}`);
    } catch (e) {
      console.error("Error saving map type preference:", e);
    }
    
    // Handle case where map is not yet initialized
    if (!mapRef || !mapRef.current || !mapRef.current.getMap) {
      console.warn('Map reference not available, preference stored. Will apply when map initializes.');
      
      // Dispatch event for other components that might need to know
      try {
        window.dispatchEvent(new CustomEvent('mapTypeChanged', {
          detail: { 
            mapTypeId: newType,
            pending: true
          }
        }));
      } catch (e) {
        console.error("Error dispatching map type change event:", e);
      }
      return;
    }
    
    try {
      // Get the Google Maps instance
      const map = mapRef.current.getMap();
      if (!map) {
        console.error("Map instance not available");
        return;
      }
      
      // Safely access Google Maps API
      if (!window.google || !window.google.maps) {
        console.error("Google Maps API not available");
        return;
      }
      
      // Convert our simplified types to Google Maps types
      let googleMapType;
      switch (newType) {
        case 'satellite':
          // Pure satellite view without labels
          googleMapType = window.google.maps.MapTypeId.SATELLITE;
          console.log("Setting map type to SATELLITE (no labels)");
          break;
        case 'map':
          // Road map view
          googleMapType = window.google.maps.MapTypeId.ROADMAP;
          console.log("Setting map type to ROADMAP");
          break;
        case 'hybrid':
          // Satellite with labels (roads, etc)
          googleMapType = window.google.maps.MapTypeId.HYBRID;
          console.log("Setting map type to HYBRID (satellite with labels)");
          break;
        default:
          // Default to hybrid
          googleMapType = window.google.maps.MapTypeId.HYBRID;
          console.log("Setting map type to default HYBRID view");
      }
      
      // Apply the map type with appropriate error handling
      console.log(`Applying map type: ${googleMapType}`);
      
      try {
        // Set map type with direct API call
        map.setMapTypeId(googleMapType);
        
        // Force the change again after a brief delay to ensure it sticks
        setTimeout(() => {
          try {
            if (map) {
              map.setMapTypeId(googleMapType);
              console.log(`Re-applied ${newType} map type after delay`);
              
              // Trigger a resize to ensure labels render correctly 
              window.google.maps.event.trigger(map, 'resize');
            }
          } catch (e) {
            console.error(`Error in delayed ${newType} map type update:`, e);
          }
        }, 100);
      } catch (e) {
        console.error(`Error setting map type to ${newType}:`, e);
      }
      
      // Store the user's selection in multiple places for consistency
      try {
        localStorage.setItem('currentMapType', newType);
        window.currentMapType = newType; // Explicit global state
        
        // Also store as last used 2D mode for when returning from 3D
        localStorage.setItem('lastMapTypeBefore3D', newType);
        
        // Broadcast the change via custom event
        window.dispatchEvent(new CustomEvent('mapTypeChanged', {
          detail: { 
            mapTypeId: newType,
            googleMapTypeId: googleMapType
          }
        }));
      } catch (e) {
        console.error("Error saving map type preference:", e);
      }
    } catch (e) {
      console.error("Error changing map type:", e);
    }
  };
  
  return (
    <div className="w-full">
            
      {/* Quick actions */}
      <div className="flex space-x-2 mb-3">
        <button
          onClick={() => ensureMapView(handleResetView)}
          className="flex items-center justify-center p-2 bg-gray-100 rounded-md hover:bg-gray-200 flex-1 border border-gray-200/50"
          title="Reset view"
        >
          <Home className="h-4 w-4 mr-1" />
          <span className="text-xs">Reset</span>
        </button>
        
        <button
          onClick={() => ensureMapView(handleGetCurrentLocation)}
          className="flex items-center justify-center p-2 bg-gray-100 rounded-md hover:bg-gray-200 flex-1 border border-gray-200/50"
          title="Go to location"
        >
          <Crosshair className="h-4 w-4 mr-1" />
          <span className="text-xs">Pin</span>
        </button>
        
        <button
          onClick={() => ensureMapView(() => setShowFilters(!showFilters))}
          className={`flex items-center justify-center p-2 rounded-md flex-1 border ${
            showFilters ? 'bg-emerald-100 text-emerald-600 border-emerald-300/30' : 'bg-gray-100 hover:bg-gray-200 border-gray-200/50'
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
            className="w-full bg-emerald-500 hover:bg-emerald-600 text-white p-1.5 rounded text-xs transition-colors border border-emerald-400/50"
          >
            Apply Filters
          </button>
        </div>
      )}
      
      {/* 3D View Toggle buttons will be in map type section */}
      
      {/* Actions */}
      <div className="space-y-2">
        <div className="flex space-x-2">
          <button
            id="toggleHighRiskButton"
            onClick={() => ensureMapView(() => {
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
            })}
            disabled={!mapLoaded}
            className="flex items-center justify-center p-2 bg-red-100 text-red-700 rounded-md hover:bg-red-200 flex-1 cursor-pointer border border-red-300/30"
          >
            <Target className="h-4 w-4 mr-1" />
            <span className="text-xs">Risk</span>
          </button>
          
          <button
            id="mediumRiskButton"
            onClick={() => ensureMapView(() => {
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
            })}
            disabled={!mapLoaded}
            className="flex items-center justify-center p-2 bg-orange-100 text-orange-700 rounded-md hover:bg-orange-200 flex-1 cursor-pointer border border-orange-300/30"
          >
            <Target className="h-4 w-4 mr-1" />
            <span className="text-xs">Med</span>
          </button>
          
          <button
            id="lowRiskButton"
            onClick={() => ensureMapView(() => {
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
            })}
            disabled={!mapLoaded}
            className="flex items-center justify-center p-2 bg-yellow-100 text-yellow-700 rounded-md hover:bg-yellow-200 flex-1 cursor-pointer border border-yellow-300/30"
          >
            <Target className="h-4 w-4 mr-1" />
            <span className="text-xs">Low</span>
          </button>
        </div>
        
        {/* Map type controls - 3D toggle and Hybrid/Satellite/Map button */}
        <div className="flex space-x-2 mb-3 mt-3">
          {/* 3D View Toggle Button */}
          <button
            onClick={() => ensureMapView(() => toggle3DMode())}
            className={`flex-1 py-1.5 px-2 rounded-md text-xs relative ${
              is3DMode
                ? 'text-blue-600 bg-blue-50 border border-blue-300/30'
                : 'text-gray-700 bg-white border border-gray-200/50'
            }`}
            disabled={!is3DSupported}
            title={is3DMode ? "Switch to 2D view" : "Switch to 3D view"}
          >
            <div className="flex items-center justify-center">
              <Box className="h-3 w-3 mr-1" />
              <span>3D View</span>
            </div>
          </button>
          
          {/* Enhanced Map Type Toggle (Hybrid/Satellite/Map) */}
          <button 
            id="map-type-toggle-button"
            className={`map-type-toggle-button flex-1 py-1.5 px-2 rounded-md text-xs relative ${
              !is3DMode 
                ? 'text-blue-600 bg-blue-50 border border-blue-300/30'
                : 'text-gray-700 bg-white border border-gray-200/50 hover:bg-gray-50'
            }`}
            onClick={() => ensureMapView(() => {
              // Only functional in 2D mode
              if (is3DMode) {
                toggle3DMode();
                return;
              }
              
              const btn = document.getElementById('map-type-toggle-button');
              const spanElement = btn ? btn.querySelector('span') : null;
              const currentText = spanElement ? spanElement.textContent : 'Hybrid';
              
              console.log("Map type toggle clicked with current text:", currentText);
              
              // Cycle through: Hybrid -> Satellite -> Map -> Hybrid
              if (currentText === 'Hybrid') {
                // Switch to Satellite
                if (spanElement) spanElement.textContent = 'Satellite';
                handleMapTypeChange('satellite');
              } else if (currentText === 'Satellite') {
                // Switch to Map
                if (spanElement) spanElement.textContent = 'Map';
                handleMapTypeChange('map');
              } else {
                // Switch to Hybrid
                if (spanElement) spanElement.textContent = 'Hybrid';
                handleMapTypeChange('hybrid');
              }
            })}
          >
            <div className="flex items-center justify-center">
              {mapType === 'map' ? 
                <Map className="h-3 w-3 mr-1" /> : 
                <Layers className="h-3 w-3 mr-1" />
              }
              <span>{mapType === 'satellite' ? 'Satellite' : mapType === 'map' ? 'Map' : 'Hybrid'}</span>
            </div>
          </button>
        </div>
        
        
        {/* Tree Detection button removed from here and moved to Analytics section in sidebar */}
      </div>
    </div>
  );
};

export default MapControls;