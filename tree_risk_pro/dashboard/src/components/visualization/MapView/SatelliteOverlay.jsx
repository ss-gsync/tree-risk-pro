// src/components/visualization/MapView/SatelliteOverlay.jsx

import React, { useEffect, useState } from 'react';

/**
 * ML Overlay component that renders a semi-transparent overlay for machine learning
 * to highlight detected trees and allow placing markers for validation.
 */
const MLOverlay = ({ mapRef, jobId, opacity = 0.3, visible = true }) => {
  const [overlayElement, setOverlayElement] = useState(null);
  
  // Create and manage the overlay
  useEffect(() => {
    if (!mapRef || !mapRef.current || !visible) return;
    
    // Create overlay div
    const overlayDiv = document.createElement('div');
    overlayDiv.className = 'ml-detection-overlay';
    overlayDiv.style.position = 'absolute';
    overlayDiv.style.top = '0';
    overlayDiv.style.left = '0';
    overlayDiv.style.width = '100%';
    overlayDiv.style.height = '100%';
    overlayDiv.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`; // Dark slate blue for aerial/satellite imagery
    overlayDiv.style.pointerEvents = 'none'; // Allow clicks to pass through
    overlayDiv.style.transition = 'opacity 0.3s ease';
    overlayDiv.style.zIndex = '1'; // Above map but below controls
    
    // Add a visual indicator that this is a satellite overlay
    const indicator = document.createElement('div');
    indicator.style.position = 'absolute';
    indicator.style.top = '10px';
    indicator.style.right = '100px';
    indicator.style.padding = '4px 8px';
    indicator.style.fontSize = '11px';
    indicator.style.backgroundColor = 'rgba(0,0,0,0.6)';
    indicator.style.color = 'white';
    indicator.style.borderRadius = '4px';
    indicator.style.zIndex = '2'; // Just above the overlay
    indicator.style.pointerEvents = 'none'; // Don't intercept clicks
    indicator.textContent = 'Satellite Imagery Layer';
    
    // Add it to the overlay for debugging purposes
    overlayDiv.appendChild(indicator);
    
    // Add to map container
    try {
      const mapContainer = mapRef.current.getDiv?.() || 
                          document.getElementById('map-container');
      
      if (mapContainer) {
        mapContainer.appendChild(overlayDiv);
        setOverlayElement(overlayDiv);
        console.log("ML overlay added to map");
      }
    } catch (error) {
      console.error("Error adding ML overlay to map:", error);
    }
    
    // Cleanup function
    return () => {
      try {
        if (overlayDiv && overlayDiv.parentNode) {
          overlayDiv.parentNode.removeChild(overlayDiv);
          console.log("ML overlay removed from map");
        }
      } catch (error) {
        console.error("Error removing ML overlay:", error);
      }
    };
  }, [mapRef, visible]);
  
  // Update opacity when it changes
  useEffect(() => {
    if (overlayElement) {
      overlayElement.style.opacity = visible ? opacity : 0;
    }
  }, [opacity, overlayElement, visible]);
  
  // The component doesn't render anything directly in the React tree
  return null;
};

export default MLOverlay;