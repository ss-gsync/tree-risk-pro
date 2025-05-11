// src/components/visualization/MapView/MLOverlay.jsx

import React, { useEffect, useState } from 'react';

/**
 * ML Overlay component that renders a semi-transparent overlay for machine learning
 * to highlight detected trees and allow placing markers for validation.
 */
const MLOverlay = ({ mapRef, jobId, opacity = 0.3, visible = true }) => {
  const [overlayElement, setOverlayElement] = useState(null);
  
  // Initialize with stored opacity if available
  useEffect(() => {
    try {
      const savedOpacity = localStorage.getItem('ml-overlay-opacity');
      if (savedOpacity !== null) {
        // This will be used for future overlay initialization
        console.log(`Loaded stored overlay opacity: ${savedOpacity}`);
      }
    } catch (e) {
      console.error("Error reading stored opacity:", e);
    }
  }, []);
  
  // Create and manage the overlay
  useEffect(() => {
    if (!mapRef || !mapRef.current || !visible) return;
    
    // Check for stored opacity preference with default of 0.1 (10%)
    let initialOpacity = 0.1; // Default to 10%
    try {
      const savedOpacity = localStorage.getItem('ml-overlay-opacity');
      if (savedOpacity !== null) {
        initialOpacity = parseFloat(savedOpacity);
      } else {
        // Save the default opacity if no saved value exists
        localStorage.setItem('ml-overlay-opacity', initialOpacity.toString());
      }
    } catch (e) {}
    
    // Create overlay div
    const overlayDiv = document.createElement('div');
    overlayDiv.id = 'ml-detection-overlay'; // Add ID for direct DOM access
    overlayDiv.className = 'ml-detection-overlay';
    overlayDiv.style.position = 'absolute';
    overlayDiv.style.top = '0';
    overlayDiv.style.left = '0';
    overlayDiv.style.width = '100%';
    overlayDiv.style.height = '100%';
    overlayDiv.style.backgroundColor = `rgba(0, 30, 60, ${initialOpacity})`; // Use stored or default opacity
    overlayDiv.style.pointerEvents = 'none'; // CRITICAL: Allow clicks to pass through to the map by default
    overlayDiv.style.transition = 'opacity 0.3s ease, background-color 0.3s ease';
    overlayDiv.style.zIndex = '5'; // Higher than default but below markers
    
    // We're keeping the overlay fully transparent to mouse events at all times
    // This ensures markers are always clickable and draggable without complex detection logic
    // The overlay is purely visual and doesn't interfere with user interaction
    console.log("ML Overlay now permanently click-through - all markers should be directly interactive");
    
    // Log ML pipeline mode
    console.log(`ML Overlay initialized with opacity ${initialOpacity}`);
    
    // We no longer need the visual indicator
    // This is intentionally left empty to remove the indicator
    
    // Listen for opacity change events
    const handleOpacityChange = (event) => {
      const newOpacity = event.detail.opacity;
      if (overlayDiv && newOpacity !== undefined) {
        overlayDiv.style.backgroundColor = `rgba(0, 30, 60, ${newOpacity})`;
        console.log(`ML overlay opacity updated to ${newOpacity}`);
      }
    };
    window.addEventListener('mlOverlayOpacityChange', handleOpacityChange);
    
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
    
    // Listen for validation mode cleanup event
    const handleValidationModeCleanup = (event) => {
      try {
        console.log("ValidationModeCleanup event received in MLOverlay");
        
        // First hide the overlay with CSS
        if (overlayDiv) {
          overlayDiv.style.opacity = '0';
          overlayDiv.style.visibility = 'hidden';
          overlayDiv.style.pointerEvents = 'none';
          
          // Then safely remove after a short delay
          setTimeout(() => {
            try {
              if (overlayDiv && overlayDiv.parentNode && document.body.contains(overlayDiv)) {
                overlayDiv.parentNode.removeChild(overlayDiv);
                console.log("ML overlay removed during cleanup");
              }
            } catch (e) {
              console.error("Error removing ML overlay during cleanup:", e);
            }
          }, 100);
        }
      } catch (e) {
        console.error("Error handling cleanup in MLOverlay:", e);
      }
    };
    
    // Register the cleanup listener
    window.addEventListener('validationModeCleanup', handleValidationModeCleanup);
    
    // Cleanup function
    return () => {
      try {
        window.removeEventListener('mlOverlayOpacityChange', handleOpacityChange);
        window.removeEventListener('validationModeCleanup', handleValidationModeCleanup);
        
        if (overlayDiv && overlayDiv.parentNode && document.body.contains(overlayDiv)) {
          // First hide it
          overlayDiv.style.opacity = '0';
          overlayDiv.style.visibility = 'hidden';
          overlayDiv.style.pointerEvents = 'none';
          
          // Then remove it
          try {
            overlayDiv.parentNode.removeChild(overlayDiv);
            console.log("ML overlay removed from map");
          } catch (err) {
            console.error("Error removing ML overlay:", err);
          }
        }
      } catch (error) {
        console.error("Error cleaning up ML overlay:", error);
      }
    };
  }, [mapRef, visible]);
  
  // Update opacity when it changes via props
  useEffect(() => {
    if (overlayElement) {
      overlayElement.style.opacity = visible ? '1' : '0';
      
      // Only update the color if opacity value changes
      if (visible && opacity !== undefined) {
        overlayElement.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
      }
    }
  }, [opacity, overlayElement, visible]);
  
  // The component doesn't render anything directly in the React tree
  return null;
};

export default MLOverlay;