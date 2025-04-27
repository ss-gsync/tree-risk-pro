// src/components/visualization/MapView/LidarVisualization.jsx

import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { Activity, RefreshCw, AlertTriangle } from 'lucide-react';
import { useLidarData } from '../../../hooks/useLidarData';

const LidarVisualization = ({ propertyId, map }) => {
  const [lidarLayer, setLidarLayer] = useState(null);
  const [treeMarkers, setTreeMarkers] = useState([]);
  const { lidarData, isLoading, error, processingProgress, reloadLidarData } = useLidarData(propertyId);
  const { visibleLayers } = useSelector((state) => state.map);

  // Clear the LiDAR visualization when unmounting
  useEffect(() => {
    return () => {
      clearLidarVisualization();
    };
  }, []);

  // Update visibility when layers change
  useEffect(() => {
    if (!map || !lidarData) return;
    
    const isLidarVisible = visibleLayers.includes('lidar');
    
    if (lidarLayer) {
      lidarLayer.setMap(isLidarVisible ? map : null);
    }
    
    treeMarkers.forEach(marker => {
      marker.setMap(isLidarVisible ? map : null);
    });
    
  }, [visibleLayers, lidarLayer, treeMarkers, map]);

  // Add LiDAR visualization when data is loaded
  useEffect(() => {
    if (!map || !lidarData || !visibleLayers.includes('lidar')) return;
    
    // Clear any existing visualization
    clearLidarVisualization();
    
    try {
      // Create LiDAR data overlay
      createLidarOverlay();
      
      // Add tree markers from LiDAR data
      addTreeMarkers();
    } catch (err) {
      console.error('Error creating LiDAR visualization:', err);
    }
    
    // Clean up when component unmounts or when lidarData changes
    return () => {
      clearLidarVisualization();
    };
  }, [lidarData, map, visibleLayers]);

  // Clear all LiDAR visualization elements
  const clearLidarVisualization = () => {
    try {
      if (lidarLayer) {
        lidarLayer.setMap(null);
        setLidarLayer(null);
      }
      
      treeMarkers.forEach(marker => {
        if (marker && marker.setMap) {
          marker.setMap(null);
        }
      });
      setTreeMarkers([]);
    } catch (error) {
      console.error('Error clearing LiDAR visualization:', error);
    }
  };

  // Create the LiDAR data overlay
  const createLidarOverlay = () => {
    if (!lidarData || !lidarData.trees || lidarData.trees.length === 0 || !map) return;
    
    try {
      // Calculate bounds from tree locations
      const bounds = new window.google.maps.LatLngBounds();
      
      lidarData.trees.forEach(tree => {
        if (tree.position && tree.position.length >= 2) {
          bounds.extend(new window.google.maps.LatLng(tree.position[1], tree.position[0]));
        }
      });
      
      // Expand bounds a bit
      bounds.extend(new window.google.maps.LatLng(bounds.getNorthEast().lat() + 0.001, bounds.getNorthEast().lng() + 0.001));
      bounds.extend(new window.google.maps.LatLng(bounds.getSouthWest().lat() - 0.001, bounds.getSouthWest().lng() - 0.001));
      
      // Create a rectangle overlay for the LiDAR scan area
      const newLidarLayer = new window.google.maps.Rectangle({
        bounds: bounds,
        strokeColor: '#1a73e8',
        strokeOpacity: 0.8,
        strokeWeight: 2,
        fillColor: '#1a73e8',
        fillOpacity: 0.1,
        map: map
      });
      
      setLidarLayer(newLidarLayer);
      
      // Add a label to the center
      try {
        const center = bounds.getCenter();
        const labelMarker = new window.google.maps.Marker({
          position: center,
          map: map,
          icon: {
            path: window.google.maps.SymbolPath.CIRCLE,
            scale: 0,
          },
          label: {
            text: `LiDAR Scan Area (${lidarData.points_processed.toLocaleString()} points)`,
            color: '#1a73e8',
            fontWeight: 'bold',
            fontSize: '12px',
            className: 'lidar-label'
          }
        });
        
        // Add to markers array for cleanup
        setTreeMarkers(prev => [...prev, labelMarker]);
      } catch (labelError) {
        console.error('Error adding LiDAR label:', labelError);
      }
    } catch (error) {
      console.error('Error creating LiDAR overlay:', error);
    }
  };

  // Add tree markers based on LiDAR data
  const addTreeMarkers = () => {
    if (!lidarData || !lidarData.trees || !map) return;
    
    try {
      const newMarkers = lidarData.trees.map(tree => {
        if (!tree.position || tree.position.length < 2) return null;
        
        try {
          // Scale based on height, between 0.5 and 1
          const scale = Math.min(1, Math.max(0.5, tree.height / 60)); 
          
          // Use AdvancedMarkerElement if available (newer versions of Maps API)
          if (window.google.maps.marker && window.google.maps.marker.AdvancedMarkerElement) {
            const element = document.createElement('div');
            element.innerHTML = `
              <div style="width: ${18 * scale}px; height: ${24 * scale}px; 
                          background-color: #43a047; border-radius: 4px 4px 0 0;
                          transform: translateY(-100%); position: relative;">
                <div style="width: 100%; height: 80%; background-color: #2e7d32; 
                           border-radius: 50% 50% 0 0; position: absolute; top: 0;"></div>
                <div style="width: 30%; height: 40%; background-color: #795548; 
                           position: absolute; bottom: 0; left: 35%;"></div>
              </div>
            `;
            
            const markerView = new window.google.maps.marker.AdvancedMarkerElement({
              position: { lat: tree.position[1], lng: tree.position[0] },
              map: map,
              title: `Tree ${tree.id} - Height: ${tree.height}m, Width: ${tree.canopy_width}m`,
              content: element
            });
            
            return markerView;
          } else {
            // Fall back to legacy Marker
            const marker = new window.google.maps.Marker({
              position: { lat: tree.position[1], lng: tree.position[0] },
              map: map,
              title: `Tree ${tree.id} - Height: ${tree.height}m, Width: ${tree.canopy_width}m`,
              icon: {
                path: 'M 0,0 C -2,-20 -10,-22 -10,-30 A 10,10 0 1,1 10,-30 C 10,-22 2,-20 0,0 z',
                fillColor: '#43a047',
                fillOpacity: 0.8,
                strokeWeight: 1,
                strokeColor: '#005a00',
                scale: scale,
                labelOrigin: new window.google.maps.Point(0, -30)
              }
            });
            
            return marker;
          }
        } catch (markerError) {
          console.error('Error creating tree marker:', markerError);
          return null;
        }
      }).filter(Boolean); // Remove any null markers
      
      setTreeMarkers(newMarkers);
    } catch (error) {
      console.error('Error adding tree markers:', error);
    }
  };

  // Loading state - show progress
  if (isLoading) {
    return (
      <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 bg-white rounded-md shadow-md p-3 z-20">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-500"></div>
          <div>
            <p className="text-sm font-medium">Processing LiDAR Data</p>
            <div className="w-48 h-2 bg-gray-200 rounded-full overflow-hidden mt-1">
              <div 
                className="h-full bg-blue-500 rounded-full transition-all duration-300" 
                style={{ width: `${processingProgress}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 bg-white rounded-md shadow-md p-3 z-20">
        <div className="flex items-center space-x-2 text-red-600">
          <AlertTriangle className="h-5 w-5" />
          <p className="text-sm font-medium">Error loading LiDAR data</p>
          <button 
            onClick={reloadLidarData}
            className="ml-2 p-1 rounded hover:bg-gray-100"
            title="Retry"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>
    );
  }

  // No data or property selected
  if (!propertyId || !lidarData) {
    return null;
  }

  return (
    <div className="absolute bottom-16 left-4 bg-white rounded-md shadow-md p-3 z-20">
      <div className="flex items-center space-x-2">
        <Activity className="h-5 w-5 text-blue-500" />
        <div>
          <p className="text-sm font-medium">LiDAR Data Active</p>
          <p className="text-xs text-gray-500">
            {lidarData.trees?.length || 0} trees detected • 
            {lidarData.points_processed?.toLocaleString() || 0} points processed
          </p>
        </div>
        <button 
          onClick={reloadLidarData}
          className="ml-2 p-1 rounded hover:bg-gray-100"
          title="Refresh LiDAR Data"
        >
          <RefreshCw className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
};

export default LidarVisualization;