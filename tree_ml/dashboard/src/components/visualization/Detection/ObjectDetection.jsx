// src/components/visualization/Detection/ObjectDetection.jsx
//
// Enhanced Object Detection component that serves as an entry point to the
// detection system and coordinates with the optimized MLOverlay.js
//
// This component:
// 1. Handles the integration with the sidebar
// 2. Ensures proper initialization of MLOverlay
// 3. Manages display states and UI consistency
// 4. Coordinates between DOM-based sidebar and React components

import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Scan } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import DetectionMode from './DetectionMode';
import MLOverlayInitializer from './MLOverlayInitializer';

/**
 * ObjectDetection - Enhanced wrapper component that properly integrates
 * with the optimized MLOverlay system and the sidebar navigation
 */
const ObjectDetection = (props) => {
  const [isActive, setIsActive] = useState(false);
  const [headerCollapsed, setHeaderCollapsed] = useState(false);
  const sidebarRef = useRef(null);
  
  // Detect when the component is being shown in the sidebar
  useEffect(() => {
    // First check if detection sidebar already exists (created by the sidebar.jsx)
    const existingDetectionSidebar = document.querySelector('.detection-sidebar');
    const isDetectionActive = existingDetectionSidebar && 
      window.getComputedStyle(existingDetectionSidebar).display !== 'none' &&
      window.getComputedStyle(existingDetectionSidebar).opacity !== '0';
    
    if (isDetectionActive) {
      setIsActive(true);
      console.log("ObjectDetection: Detected existing detection sidebar, integrating with it");
      
      // Store reference to the existing sidebar
      sidebarRef.current = existingDetectionSidebar;
      
      // Ensure our MLOverlay is initialized and visible
      ensureMLOverlayVisible();
    }
    
    // Listen for header state changes
    const handleHeaderCollapse = (event) => {
      if (event.detail && typeof event.detail.collapsed !== 'undefined') {
        setHeaderCollapsed(event.detail.collapsed);
      }
    };
    
    // Check for existing header state
    try {
      const headerElement = document.querySelector('header');
      const isHeaderCollapsed = headerElement ? 
        headerElement.classList.contains('collapsed') || 
        headerElement.offsetHeight < 50 : true;
      setHeaderCollapsed(isHeaderCollapsed);
    } catch (e) {
      console.error("Error detecting header state:", e);
    }
    
    // Register event listeners
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    // Clean up on unmount
    return () => {
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
    };
  }, []);
  
  // Ensure MLOverlay is visible when needed
  const ensureMLOverlayVisible = useCallback(() => {
    // Check global settings
    const showOverlay = window.mlOverlaySettings?.showOverlay !== false;
    
    // If overlay exists, update its visibility
    if (window._mlDetectionOverlay && window._mlDetectionOverlay.div) {
      window._mlDetectionOverlay.div.style.display = showOverlay ? 'block' : 'none';
    }
    
    // Update the OBJECT DETECTION badge
    const badge = document.getElementById('detection-debug');
    if (badge) {
      badge.style.display = showOverlay ? 'block' : 'none';
      badge.style.opacity = showOverlay ? '1' : '0';
      
      // Ensure badge is positioned correctly relative to sidebar
      badge.style.right = '384px'; // Position to the left of sidebar
    }
    
    // Trigger an overlay settings change event
    window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
      detail: { 
        showOverlay: showOverlay,
        showSegmentation: window.mlOverlaySettings?.showSegmentation !== false,
        opacity: window.mlOverlaySettings?.opacity || 0.7
      }
    }));
  }, []);
  
  // Listen for detection sidebar close events
  useEffect(() => {
    const handleForceClose = (event) => {
      console.log("ObjectDetection: Received force close event");
      setIsActive(false);
    };
    
    window.addEventListener('forceCloseObjectDetection', handleForceClose);
    
    return () => {
      window.removeEventListener('forceCloseObjectDetection', handleForceClose);
    };
  }, []);
  
  // Sync sidebar visibility with detection mode state
  useEffect(() => {
    // Make sidebar dimensions match map adjustment
    if (isActive && sidebarRef.current) {
      // Ensure map container is adjusted
      const mapContainer = document.getElementById('map-container');
      if (mapContainer) {
        mapContainer.style.right = '384px';
      }
      
      // Update badge if needed
      const badge = document.getElementById('detection-debug');
      if (badge) {
        badge.style.right = '384px';
      }
      
      // Ensure sidebar is fully visible
      sidebarRef.current.style.display = 'flex';
      sidebarRef.current.style.transform = 'translateX(0)';
      sidebarRef.current.style.opacity = '1';
    }
  }, [isActive]);

  // Create a title card with the DetectionMode component inside
  return (
    <>
      <Card className="w-full shadow-sm border-0">
        <CardHeader className="py-0.5 px-2">
          <CardTitle className="text-[10px] flex items-center">
            <Scan className="h-2.5 w-2.5 mr-1" />
            Object Recognition
          </CardTitle>
        </CardHeader>
        
        <CardContent className="p-0">
          {/* Only render DetectionMode if active to save resources */}
          {isActive && (
            <DetectionMode 
              {...props} 
              headerCollapsed={headerCollapsed}
            />
          )}
        </CardContent>
      </Card>
      
      {/* Always include the MLOverlayInitializer to ensure overlay is available */}
      <MLOverlayInitializer />
    </>
  );
};

export default ObjectDetection;