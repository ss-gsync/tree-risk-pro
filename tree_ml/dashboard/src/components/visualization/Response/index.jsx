// src/components/visualization/Response/index.jsx
//
// Main entry point for the Detection Response module
// Coordinates between ResponseList and ResponseViewer components

import React, { useState, useEffect } from 'react';
import ResponseList from './ResponseList';
import ResponseViewer from './ResponseViewer';

/**
 * Response Module
 * 
 * Main container component that manages the detection response viewing system
 * Coordinates between the response list and response viewer
 */
const ResponseModule = ({ 
  initialDetectionId = null,
  onViewOnMap = null,
  refreshInterval = 30000 
}) => {
  const [selectedDetectionId, setSelectedDetectionId] = useState(initialDetectionId);
  const [viewerVisible, setViewerVisible] = useState(!!initialDetectionId);
  
  // Update selected detection when initialDetectionId prop changes
  useEffect(() => {
    if (initialDetectionId && initialDetectionId !== selectedDetectionId) {
      setSelectedDetectionId(initialDetectionId);
      setViewerVisible(true);
    }
  }, [initialDetectionId, selectedDetectionId]);

  // Handle response selection from the list
  const handleSelectResponse = (detectionId) => {
    setSelectedDetectionId(detectionId);
    setViewerVisible(true);
    
    // Dispatch event for any listeners
    window.dispatchEvent(new CustomEvent('responseSelected', {
      detail: { detectionId }
    }));
  };

  // Handle closing the viewer
  const handleCloseViewer = () => {
    setViewerVisible(false);
  };

  // Handle view on map action
  const handleViewOnMap = (detectionId, metadata) => {
    // Call the provided handler if available
    if (onViewOnMap) {
      onViewOnMap(detectionId, metadata);
    } else {
      // Default implementation - dispatch custom event
      window.dispatchEvent(new CustomEvent('viewDetectionOnMap', {
        detail: { 
          detectionId,
          metadata
        }
      }));
    }
    
    // Close the viewer
    setViewerVisible(false);
  };

  return (
    <div className="grid grid-cols-1 gap-4">
      {/* Show either the response list or the viewer */}
      {viewerVisible ? (
        <ResponseViewer
          detectionId={selectedDetectionId}
          onViewOnMap={handleViewOnMap}
          onClose={handleCloseViewer}
        />
      ) : (
        <ResponseList
          onSelectResponse={handleSelectResponse}
          selectedDetectionId={selectedDetectionId}
          refreshInterval={refreshInterval}
        />
      )}
    </div>
  );
};

export default ResponseModule;