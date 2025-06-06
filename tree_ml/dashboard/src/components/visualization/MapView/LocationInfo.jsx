// src/components/visualization/MapView/LocationInfo.jsx

import React, { useState, useEffect, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { MapPin, Compass, ZoomIn, RefreshCw } from 'lucide-react';

// Create a global reference to store the latest coordinates as single source of truth
if (!window.mapCoordinates) {
  window.mapCoordinates = {
    center: null,
    zoom: null,
    bounds: null,
    heading: null,
    tilt: null,
    lastUpdated: 0
  };
}

const LocationInfo = () => {
  const dispatch = useDispatch();
  const { center, zoom, heading, tilt } = useSelector((state) => state.map);
  const [isUpdating, setIsUpdating] = useState(false);
  
  // Function to get exact coordinates from map instance
  const updateExactMapCoordinates = useCallback(() => {
    try {
      // Get map instance directly from global reference
      const mapInstance = window.googleMapsInstance;
      if (!mapInstance) {
        return null;
      }
      
      // Get center coordinates directly from map
      const mapCenter = mapInstance.getCenter();
      if (!mapCenter) {
        return null;
      }
      
      // Get exact values with maximum precision
      const lng = mapCenter.lng();
      const lat = mapCenter.lat();
      
      // Get zoom
      const mapZoom = mapInstance.getZoom();
      
      // Get bounds
      const mapBounds = mapInstance.getBounds();
      let bounds = null;
      
      if (mapBounds) {
        const ne = mapBounds.getNorthEast();
        const sw = mapBounds.getSouthWest();
        
        bounds = [
          [sw.lng(), sw.lat()],
          [ne.lng(), ne.lat()]
        ];
      }
      
      // Get heading and tilt if available
      const mapHeading = mapInstance.getHeading ? mapInstance.getHeading() : null;
      const mapTilt = mapInstance.getTilt ? mapInstance.getTilt() : null;
      
      // Update global reference with latest coordinates
      window.mapCoordinates = {
        center: [lng, lat],
        zoom: mapZoom,
        bounds: bounds,
        heading: mapHeading,
        tilt: mapTilt,
        lastUpdated: Date.now()
      };
      
      // Dispatch to Redux so other components can use these coordinates
      dispatch({ 
        type: 'map/setMapView', 
        payload: { 
          center: [lng, lat], 
          zoom: mapZoom,
          heading: mapHeading,
          tilt: mapTilt
        } 
      });
      
      // Briefly show update indicator
      setIsUpdating(true);
      setTimeout(() => setIsUpdating(false), 200);
      
      return true;
    } catch (error) {
      console.error('Error getting direct map coordinates:', error);
      return null;
    }
  }, [dispatch]);
  
  // Listen for the map idle event to update coordinates
  useEffect(() => {
    const handleMapIdle = () => {
      updateExactMapCoordinates();
    };
    
    window.addEventListener('mapIdle', handleMapIdle);
    
    return () => {
      window.removeEventListener('mapIdle', handleMapIdle);
    };
  }, [updateExactMapCoordinates]);
  
  // Format coordinates to show 6 decimal places
  const formatCoordinate = (coord) => {
    return typeof coord === 'number' ? coord.toFixed(6) : 'N/A';
  };

  // Don't render if we don't have coordinate data
  if (!center || center.length !== 2) {
    return null;
  }

  return (
    <div className="p-2 bg-white border border-gray-200/50 rounded-md text-xs text-gray-700">
      <div className="flex items-center">
        <MapPin className="h-3 w-3 mr-1 text-gray-500" />
        <span>{formatCoordinate(center[1])}, {formatCoordinate(center[0])}</span>
        {isUpdating && <RefreshCw className="h-2 w-2 ml-1 text-green-500 animate-spin" />}
      </div>
      
      <div className="mt-1 flex items-center">
        <ZoomIn className="h-3 w-3 mr-1 text-gray-500" />
        <span>Zoom: {zoom || 'N/A'}</span>
      </div>
      
      {(heading !== undefined || tilt !== undefined) && (
        <div className="mt-1 flex items-center">
          <Compass className="h-3 w-3 mr-1 text-gray-500" />
          <span>H: {heading?.toFixed(1) || 'N/A'}°, T: {tilt?.toFixed(1) || 'N/A'}°</span>
        </div>
      )}
    </div>
  );
};

export default LocationInfo;