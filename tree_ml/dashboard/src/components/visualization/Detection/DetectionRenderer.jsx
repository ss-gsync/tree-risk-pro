// src/components/visualization/Detection/DetectionRenderer.jsx
//
// This component is responsible for rendering detection results on the map.
// It's split from the larger DetectionMode.jsx component.

import React, { useEffect, useCallback, useState } from 'react';
// Import MLOverlay with named imports
import * as MLOverlayModule from './MLOverlay';
import MLOverlayInitializer from './MLOverlayInitializer';
import DetectionPreview from './DetectionPreview';

// Use functions directly from MLOverlayModule

/**
 * Detection Renderer Component
 * 
 * Handles the rendering of detection results on the map, including:
 * - Managing the ML overlay
 * - Adding markers to the map
 * - Handling marker events
 * - Displaying the detection preview
 */
const DetectionRenderer = ({
  mapRef,
  trees = [],
  selectedTree,
  setSelectedTree,
  setCurrentTreeIndex,
  showOverlay = true,
  overlayOpacity = 0.7,
  jobId,
  setCollapsed,
  width,
  showPreview,
  setShowPreview
}) => {
  // Detection preview state is now passed from parent
  const [detectionFilters, setDetectionFilters] = useState({
    trees: true,
    buildings: true,
    powerLines: true
  });
  const [showLabels, setShowLabels] = useState(true);
  const [localShowOverlay, setShowOverlay] = useState(showOverlay);
  
  // When trees data changes, show the center preview
  useEffect(() => {
    if (trees && trees.length > 0) {
      console.log(`DetectionRenderer: Trees data loaded with ${trees.length} trees, showing center preview`);
      setShowPreview(true);
      
      // Process the data for display
      const processedData = processDetectionData();
      
      // Save the data to the window for access by other components
      window.mlDetectionData = processedData;
      
      // Use the global function to show the detection preview directly at the document level
      if (typeof window.showDetectionPreview === 'function') {
        try {
          window.showDetectionPreview(processedData);
          console.log("DetectionRenderer: Showing detection preview via global function");
        } catch (e) {
          console.error("DetectionRenderer: Error showing detection preview:", e);
        }
      }
    }
    
    // Listen for explicit show preview events
    const handleShowPreview = (event) => {
      if (event.detail && event.detail.data) {
        console.log("DetectionRenderer: Received showCenterPanePreview event", {
          dataPresent: !!event.detail.data,
          trees: event.detail.data?.trees?.length || 0
        });
        
        // Store data globally
        window.mlDetectionData = event.detail.data;
        
        // Show center preview
        setShowPreview(true);
        
        // Force preview to appear by adding class to body
        document.body.classList.add('show-detection-preview');
        
        // Use the global function to show the detection preview directly at the document level
        if (typeof window.showDetectionPreview === 'function') {
          try {
            window.showDetectionPreview(event.detail.data);
            console.log("DetectionRenderer: Using direct programmatic method to show preview");
          } catch (e) {
            console.error("DetectionRenderer: Error showing preview via global function:", e);
          }
        }
      }
    };
    
    // Detection complete handler
    const handleDetectionComplete = () => {
      console.log("DetectionRenderer: Detection process completed, ensuring preview stays visible");
      // Ensure preview is shown after detection completes
      setShowPreview(true);
      
      // Force preview to appear
      document.body.classList.add('show-detection-preview');
      
      // Process detection data if available
      if (window.mlDetectionData) {
        console.log("DetectionRenderer: Using existing detection data after detection completion");
        
        // Use the global function to show the detection preview
        if (typeof window.showDetectionPreview === 'function') {
          try {
            window.showDetectionPreview(window.mlDetectionData);
            console.log("DetectionRenderer: Showing detection preview after completion");
          } catch (e) {
            console.error("DetectionRenderer: Error showing detection preview:", e);
          }
        }
      } else {
        // Try to use trees state as fallback
        if (trees && trees.length > 0) {
          console.log("DetectionRenderer: Creating detection data from trees state");
          const data = processDetectionData();
          window.mlDetectionData = data;
          
          // Use the global function to show the detection preview
          if (typeof window.showDetectionPreview === 'function') {
            try {
              window.showDetectionPreview(data);
              console.log("DetectionRenderer: Showing detection preview with processed tree data");
            } catch (e) {
              console.error("DetectionRenderer: Error showing detection preview:", e);
            }
          }
        }
      }
    };
    
    // Handle incoming inference results IMMEDIATELY - don't wait for detection to complete
    const handleFastInference = (event) => {
      if (event.detail) {
        // Display preview IMMEDIATELY without any delay
        console.log("DetectionRenderer: Fast inference results received - showing preview IMMEDIATELY");
        
        // Performance optimization: Process only what we need for quick display
        // Skip any heavy processing that might delay showing results
        
        // Pass data directly to showDetectionPreview without complex processing
        if (typeof window.showDetectionPreview === 'function') {
          try {
            console.log("DetectionRenderer: Showing preview WITH HIGHEST PRIORITY");
            // Call with priority flag to ensure fastest possible display
            window.showDetectionPreview(event.detail, true);
            return; // Exit early to avoid additional processing
          } catch (e) {
            console.error("Error showing immediate preview:", e);
          }
        }
        
        // Fallback to traditional processing only if direct method fails
        console.log("DetectionRenderer: Falling back to category processing");
        
        // Process trees into categories (even partial data is fine)
        const treesByCategory = {
          trees: [],
          buildings: [],
          power_lines: []
        };
        
        // If there are trees in the event detail, process them
        if (event.detail.trees && Array.isArray(event.detail.trees)) {
          console.log(`Processing ${event.detail.trees.length} trees from fast inference`);
          
          event.detail.trees.forEach(tree => {
            const treeClass = (tree.class || 'tree').toLowerCase();
            
            if (treeClass === 'tree') {
              treesByCategory.trees.push(tree);
            } else if (['building', 'roof', 'solar panel'].includes(treeClass)) {
              treesByCategory.buildings.push(tree);
            } else if (['power line', 'railroad'].includes(treeClass)) {
              treesByCategory.power_lines.push(tree);
            } else {
              // Default unknown objects to trees
              treesByCategory.trees.push(tree);
            }
          });
        }
        
        // Make sure there's at least an empty array for each category
        treesByCategory.trees = treesByCategory.trees || [];
        treesByCategory.buildings = treesByCategory.buildings || [];
        treesByCategory.power_lines = treesByCategory.power_lines || [];
        
        // Add any metadata from the event detail
        const enhancedData = {
          ...treesByCategory,
          // Add job ID, metadata, etc. if available
          job_id: event.detail.job_id,
          metadata: event.detail.metadata || {},
          // Include the raw data for debugging
          raw_data: event.detail
        };
        
        // Store processed data globally
        window.mlDetectionData = enhancedData;
        
        // Show the preview IMMEDIATELY
        setShowPreview(true);
        
        // Force document class to indicate preview should be shown
        document.body.classList.add('show-detection-preview');
        
        // DIRECTLY show the preview - don't wait
        if (typeof window.showDetectionPreview === 'function') {
          try {
            window.showDetectionPreview(enhancedData);
            console.log("DetectionRenderer: Showing preview IMMEDIATELY");
          } catch (e) {
            console.error("Error showing detection preview:", e);
          }
        }
        
        // Log the categories
        console.log("DetectionRenderer: Preview data prepared", {
          treeCount: enhancedData.trees.length,
          buildingCount: enhancedData.buildings.length,
          powerLineCount: enhancedData.power_lines.length
        });
      }
    };
    
    // Register event listeners
    window.addEventListener('showCenterPanePreview', handleShowPreview);
    window.addEventListener('enterDetectionMode', handleDetectionComplete);
    window.addEventListener('fastInferenceResults', handleFastInference);
    // New event handler for data from MLOverlay
    window.addEventListener('mlDataAvailable', (event) => {
      if (event.detail && event.detail.data) {
        console.log("DetectionRenderer: Received mlDataAvailable event with data");
        // Handle the same way as fast inference data
        handleFastInference({ detail: event.detail.data });
      }
    });
    
    return () => {
      window.removeEventListener('showCenterPanePreview', handleShowPreview);
      window.removeEventListener('enterDetectionMode', handleDetectionComplete);
      window.removeEventListener('fastInferenceResults', handleFastInference);
      window.removeEventListener('mlDataAvailable', handleFastInference);
    };
  }, [trees]);
  
  // Helper function to add markers directly to Google Maps with improved category handling
  const addGoogleMapMarkers = useCallback((trees) => {
    if (!mapRef.current || !window.google) return;
    
    try {
      console.log("Adding markers directly to Google Maps with tree categories");
      
      // Get the map instance
      const map = mapRef.current.getMap?.() || mapRef.current;
      if (!map) return;
      
      // Clear existing markers if needed
      if (mapRef.current.getMarkers && mapRef.current.getMarkers().length > 0) {
        mapRef.current.getMarkers().forEach(marker => {
          if (marker.setMap) marker.setMap(null);
        });
        // Reset markers array
        mapRef.current.getMarkers().length = 0;
      }
      
      // Create a markers array if it doesn't exist
      if (!mapRef.current.getMarkers) {
        mapRef.current.markers = [];
        mapRef.current.getMarkers = () => mapRef.current.markers;
      }
      
      // Define tree category colors and shapes
      const categoryColors = {
        healthy_tree: '#16a34a',       // Green
        hazardous_tree: '#8b5cf6',     // Purple
        dead_tree: '#6b7280',          // Gray
        low_canopy_tree: '#0ea5e9',    // Blue
        pest_disease_tree: '#84cc16',  // Light Green
        flood_prone_tree: '#0891b2',   // Cyan
        utility_conflict_tree: '#3b82f6', // Blue
        structural_hazard_tree: '#0d9488', // Teal
        fire_risk_tree: '#4f46e5',     // Indigo
        building: '#3b82f6',           // Blue
        power_line: '#ef4444',         // Red
        generic: '#10b981'             // Default Teal
      };
      
      // Add markers for each tree with improved category detection
      trees.forEach(tree => {
        // Handle both coordinate formats - direct location array or coordinates object
        let lat, lng;
        
        // Check if tree has coordinates in nested object
        if (tree.coordinates && tree.coordinates.lat && tree.coordinates.lng) {
          lat = tree.coordinates.lat;
          lng = tree.coordinates.lng;
        } 
        // Check for location array
        else if (tree.location && tree.location.length >= 2) {
          [lng, lat] = tree.location;
        }
        // No valid coordinates found
        else {
          console.log("Skipping tree without valid location", tree);
          return;
        }
        
        // IMPROVED CATEGORY DETECTION - Check multiple properties and normalize
        // First try to get class and category
        let treeClass = '';
        let treeCategory = '';
        
        // Try to get class from different possible locations in the object
        if (tree.class) {
          treeClass = tree.class.toLowerCase();
        } else if (tree.detection && tree.detection.class) {
          treeClass = tree.detection.class.toLowerCase();
        }
        
        // Try to get category if available
        if (tree.category) {
          treeCategory = tree.category.toLowerCase();
        }
        
        // Normalize spaces to underscores for both
        const normalizedClass = treeClass.replace(/\s+/g, '_');
        const normalizedCategory = treeCategory.replace(/\s+/g, '_');
        
        // Determine which category to use
        let markerCategory = 'generic'; // Default category
        
        // Try to determine the specific tree category
        if (normalizedCategory === 'hazardous_tree' || normalizedClass === 'hazardous_tree' || 
            treeClass.includes('hazardous') || treeCategory.includes('hazardous')) {
          markerCategory = 'hazardous_tree';
        } 
        else if (normalizedCategory === 'dead_tree' || normalizedClass === 'dead_tree' || 
                 treeClass.includes('dead') || treeCategory.includes('dead')) {
          markerCategory = 'dead_tree';
        } 
        else if (normalizedCategory === 'healthy_tree' || normalizedClass === 'healthy_tree' || 
                 treeClass.includes('healthy') || treeCategory.includes('healthy')) {
          markerCategory = 'healthy_tree';
        } 
        else if (normalizedCategory === 'low_canopy_tree' || normalizedClass === 'low_canopy_tree' || 
                 treeClass.includes('low canopy') || treeCategory.includes('low canopy')) {
          markerCategory = 'low_canopy_tree';
        } 
        else if (normalizedCategory === 'pest_disease_tree' || normalizedClass === 'pest_disease_tree' || 
                 treeClass.includes('pest') || treeClass.includes('disease') || 
                 treeCategory.includes('pest') || treeCategory.includes('disease')) {
          markerCategory = 'pest_disease_tree';
        } 
        else if (normalizedCategory === 'flood_prone_tree' || normalizedClass === 'flood_prone_tree' || 
                 treeClass.includes('flood') || treeCategory.includes('flood')) {
          markerCategory = 'flood_prone_tree';
        } 
        else if (normalizedCategory === 'utility_conflict_tree' || normalizedClass === 'utility_conflict_tree' || 
                 treeClass.includes('utility') || treeCategory.includes('utility')) {
          markerCategory = 'utility_conflict_tree';
        } 
        else if (normalizedCategory === 'structural_hazard_tree' || normalizedClass === 'structural_hazard_tree' || 
                 treeClass.includes('structural') || treeCategory.includes('structural')) {
          markerCategory = 'structural_hazard_tree';
        } 
        else if (normalizedCategory === 'fire_risk_tree' || normalizedClass === 'fire_risk_tree' || 
                 treeClass.includes('fire') || treeCategory.includes('fire')) {
          markerCategory = 'fire_risk_tree';
        }
        else if (treeClass.includes('building') || treeClass.includes('roof') || treeClass.includes('solar')) {
          markerCategory = 'building';
        }
        else if (treeClass.includes('power') || treeClass.includes('utility') || treeClass.includes('line')) {
          markerCategory = 'power_line';
        }
        else if (treeClass.includes('tree') || treeCategory.includes('tree')) {
          // Generic tree
          markerCategory = 'generic';
        }
        
        // Now filter based on the category
        if (
          (['healthy_tree', 'hazardous_tree', 'dead_tree', 'low_canopy_tree', 'pest_disease_tree',
            'flood_prone_tree', 'utility_conflict_tree', 'structural_hazard_tree', 'fire_risk_tree', 'generic']
              .includes(markerCategory) && !detectionFilters.trees) ||
          (markerCategory === 'building' && !detectionFilters.buildings) ||
          (markerCategory === 'power_line' && !detectionFilters.powerLines)
        ) {
          return; // Skip if category is filtered out
        }
        
        // Get color for the marker based on category
        const markerColor = categoryColors[markerCategory] || categoryColors.generic;
        
        // Create marker with improved styling
        const marker = new window.google.maps.Marker({
          position: { lat, lng },
          map: map,
          title: tree.species || markerCategory.replace('_', ' ') || 'Object',
          icon: {
            path: window.google.maps.SymbolPath.CIRCLE,
            scale: 8,
            fillColor: markerColor,
            fillOpacity: 0.8,
            strokeWeight: 2,
            strokeColor: '#ffffff'
          }
        });
        
        // Add click handler
        marker.addListener('click', () => {
          // Add category to tree object for reference
          const enrichedTree = { 
            ...tree, 
            _derived_category: markerCategory 
          };
          
          // Create and dispatch custom event
          const event = new CustomEvent('treeMarkerSelected', {
            detail: { tree: enrichedTree }
          });
          window.dispatchEvent(event);
        });
        
        // Store marker reference
        mapRef.current.getMarkers().push(marker);
      });
      
      console.log(`Added ${mapRef.current.getMarkers().length} markers to map`);
    } catch (err) {
      console.error("Error adding markers to map:", err);
    }
  }, [mapRef, detectionFilters]);
  
  // Handle view on map from the detection preview
  const handleViewOnMap = (filters, labels) => {
    console.log("Applying detection filters:", filters, "with labels:", labels);
    
    // Update local state
    setDetectionFilters(filters);
    setShowLabels(labels);
    
    // Store for MLOverlay to use
    window.detectionClasses = filters;
    window.detectionShowLabels = labels;
    
    // CRITICAL: Force overlay to be visible
    window.detectionShowOverlay = true;
    
    // Hide preview
    setShowPreview(false);
    
    // Process and display the visible trees with filters
    // IMPORTANT: Use the global mlDetectionData (from detection preview) instead of local trees
    // This ensures we use the ACTUAL coordinates from the detection
    const detectionData = window.mlDetectionData || {};
    
    // Make sure we have detections array properly structured
    if (!detectionData.detections && detectionData.trees) {
      console.log("Converting trees array to detections array for marker placement");
      // Create detections array from trees
      detectionData.detections = detectionData.trees.map(tree => {
        // If tree has a detection property with coordinates, use those
        if (tree.detection && tree.detection.bbox) {
          return {
            ...tree,
            bbox: tree.detection.bbox,
            class: tree.detection.class || 'tree',
            category: tree.category || 'healthy_tree',
            confidence: tree.detection.confidence
          };
        }
        return tree;
      });
    }
    
    // Get all objects from detection data
    const allObjects = detectionData.detections || [];
    console.log(`Rendering ${allObjects.length} objects to map with filters`);
    
    // Dispatch events to trigger rendering
    window.dispatchEvent(new CustomEvent('detectionDataAvailable', {
      detail: { data: detectionData }
    }));
    
    // Add markers with filters applied
    if (mapRef.current && window.google) {
      // First remove existing markers
      if (mapRef.current.getMarkers) {
        mapRef.current.getMarkers().forEach(marker => {
          if (marker.setMap) marker.setMap(null);
        });
        mapRef.current.getMarkers().length = 0;
      }
      
      // Add new markers with filters - use detections array instead of trees
      addGoogleMapMarkers(allObjects);
      
      // CRITICAL: Force ML overlay to update
      if (typeof window.renderDetectionOverlay === 'function') {
        console.log("Forcing ML overlay update with detection data");
        
        // Before rendering, make sure each detection has proper coordinates
        // This is crucial for the overlay to work correctly
        if (detectionData.detections) {
          detectionData.detections.forEach(detection => {
            // If detection has nested detection property but no bbox
            if (detection.detection && !detection.bbox) {
              // Check for bbox in nested detection
              if (detection.detection.bbox) {
                detection.bbox = detection.detection.bbox;
                console.log(`DetectionRenderer: Using nested detection.bbox for detection`, detection.id || 'unknown');
              }
              // Check for box in nested detection and convert to bbox
              else if (detection.detection.box) {
                detection.bbox = [
                  detection.detection.box.x,
                  detection.detection.box.y,
                  detection.detection.box.x + detection.detection.box.width,
                  detection.detection.box.y + detection.detection.box.height
                ];
                console.log(`DetectionRenderer: Converting nested box to bbox for detection`, detection.id || 'unknown');
              }
            }
          });
        }
        
        // Use the window.showDetectionOverlay function which has better debugging
        if (typeof window.showDetectionOverlay === 'function') {
          window.showDetectionOverlay(detectionData, {
            forceRenderBoxes: true,
            debug: true
          });
        } else {
          window.renderMLOverlay(detectionData, {
            forceRenderBoxes: true,
            debug: true
          });
        }
      }
    }
  };
  
  // Initialize markers when trees or map changes
  useEffect(() => {
    if (!trees || trees.length === 0 || !mapRef.current) {
      console.log('DetectionRenderer: Missing trees or mapRef:', {
        hasTrees: !!trees && trees.length > 0,
        hasMapRef: !!mapRef.current
      });
      return;
    }
    
    const visibleTrees = trees.filter(tree => tree.visible);
    console.log(`Rendering ${visibleTrees.length} objects to map`);
    
    // Process trees to ensure class property is set
    const processedTrees = visibleTrees.map(tree => {
      // If no class is set, default to 'tree'
      if (!tree.class) {
        return { ...tree, class: 'tree' };
      }
      return tree;
    });
    
    // Organize trees by category for ML overlay and visualization
    const treesByCategory = {
      trees: processedTrees.filter(t => (!t.class || t.class.toLowerCase() === 'tree')),
      buildings: processedTrees.filter(t => (t.class && ['building', 'roof', 'solar panel'].includes(t.class.toLowerCase()))),
      power_lines: processedTrees.filter(t => (t.class && ['power line', 'railroad'].includes(t.class.toLowerCase())))
    };
    
    // Store data for MLOverlay to use - this format is expected by MLOverlay.js
    window.mlDetectionData = {
      trees: treesByCategory.trees,
      buildings: treesByCategory.buildings,
      power_lines: treesByCategory.power_lines
    };
    
    // Create a global function to force marker rendering
    window.forceRenderMarkers = () => {
      console.log('Force rendering markers via global function');
      addGoogleMapMarkers(processedTrees);
    };
    
    // Wait for Google Maps API to be fully loaded
    if (!window.google || !window.google.maps) {
      console.error('Google Maps API not available! Waiting for it to load...');
      
      // Set up a wait-and-retry mechanism
      const checkGoogleMaps = setInterval(() => {
        if (window.google && window.google.maps) {
          console.log('Google Maps API now available, rendering markers');
          clearInterval(checkGoogleMaps);
          addGoogleMapMarkers(processedTrees);
        }
      }, 500);
      
      // Clear interval after 10 seconds to prevent memory leaks
      setTimeout(() => clearInterval(checkGoogleMaps), 10000);
      return;
    }
    
    // Add markers with a direct approach - no fallbacks as requested by the user
    addGoogleMapMarkers(processedTrees);
    
    // Dispatch event to notify other components that markers have been rendered
    window.dispatchEvent(new CustomEvent('markersRendered', {
      detail: { trees: processedTrees }
    }));
    
  }, [trees, mapRef, addGoogleMapMarkers]);
  
  // Listen for custom marker placement and clearing ML markers
  useEffect(() => {
    const handleCustomMarkerPlacement = (event) => {
      if (!event.detail || !mapRef.current || !window.google) {
        console.error("DetectionRenderer: Cannot handle custom marker placement - missing dependencies", {
          hasDetail: !!event.detail,
          hasMapRef: !!mapRef.current,
          hasGoogleMaps: !!window.google
        });
        return;
      }
      
      console.log("DetectionRenderer: Handling custom marker placement event", event.detail);
      
      // Extract relevant properties from event.detail
      const { filters, showLabels, method, treeData, categoryData } = event.detail;
      
      // CRITICAL: Force overlay to be visible
      window.detectionShowOverlay = true;
      setShowOverlay(true);
      
      // IMPORTANT: Check if there's tree data in the global mlDetectionData object
      // This ensures we're using the most recent detection data
      const detectionData = window.mlDetectionData || {};
      
      // Process the tree data from the event
      if (Array.isArray(treeData) && treeData.length > 0) {
        console.log(`DetectionRenderer: Found ${treeData.length} trees in the event data`);
        
        // Convert tree data to detections array if needed
        if (!detectionData.detections) {
          detectionData.detections = treeData.map(tree => {
            // If tree has a detection property with bbox, use those
            if (tree.detection && tree.detection.bbox) {
              return {
                ...tree,
                bbox: tree.detection.bbox,
                class: tree.detection.class || 'tree',
                category: tree.category || 'healthy_tree',
                confidence: tree.detection.confidence
              };
            }
            return tree;
          });
        }
      }
      
      // Get the detection data if available
      const detectionJobId = event.detail.jobId || 
                    (event.detail.treeData && event.detail.treeData[0]?.job_id) || 
                    (window.mlDetectionData && window.mlDetectionData.job_id);
      
      // If we have a job ID but don't have the data loaded yet, load it
      if (detectionJobId && !window.mlDetectionData?.job_id) {
        console.log(`DetectionRenderer: Loading data for job ${detectionJobId} from custom marker placement`);
        if (typeof window.loadDetectionDataForJob === 'function') {
          window.loadDetectionDataForJob(detectionJobId);
        }
      }
      
      // CRITICAL: Force ML overlay to update
      if (typeof window.renderDetectionOverlay === 'function') {
        console.log("Forcing ML overlay update with detection data");
        window.renderDetectionOverlay(detectionData, {
          forceRenderBoxes: true,
          debug: true
        });
      }
      
      // Collect all trees to display
      let allTrees = [];
      
      // First check if we have treeData in the event
      if (Array.isArray(treeData) && treeData.length > 0) {
        console.log(`DetectionRenderer: Using ${treeData.length} trees from treeData array`);
        allTrees = treeData;
      } 
      // Next, check if there are trees directly in the event detail
      else if (Array.isArray(event.detail.trees) && event.detail.trees.length > 0) {
        console.log(`DetectionRenderer: Using ${event.detail.trees.length} trees from event.detail.trees`);
        allTrees = event.detail.trees;
      }
      // Check for detections array (new format)
      else if (event.detail.detections && Array.isArray(event.detail.detections) && event.detail.detections.length > 0) {
        console.log(`DetectionRenderer: Using ${event.detail.detections.length} trees from event.detail.detections`);
        allTrees = event.detail.detections;
      }
      // Fallback to global detection data if available
      else if (window.mlDetectionData && window.mlDetectionData.detections && 
               Array.isArray(window.mlDetectionData.detections) && 
               window.mlDetectionData.detections.length > 0) {
        console.log(`DetectionRenderer: Using ${window.mlDetectionData.detections.length} trees from global mlDetectionData`);
        allTrees = window.mlDetectionData.detections;
      }
      // Final fallback to component state trees
      else {
        console.log(`DetectionRenderer: Using ${trees.filter(tree => tree.visible).length} trees from component state`);
        allTrees = trees.filter(tree => tree.visible);
      }
      
      // Ensure all trees have job_id for consistency
      if (detectionJobId) {
        allTrees = allTrees.map(tree => {
          if (!tree.job_id) return { ...tree, job_id: detectionJobId };
          return tree;
        });
      }
      
      console.log(`DetectionRenderer: Processing ${allTrees.length} trees for marker placement`);
      
      // Clear existing markers first
      if (mapRef.current.getMarkers) {
        mapRef.current.getMarkers().forEach(marker => {
          if (marker.setMap) marker.setMap(null);
        });
        mapRef.current.getMarkers().length = 0;
        console.log("DetectionRenderer: Cleared existing markers");
      }
      
      // Add new markers using the enhanced marker function
      if (allTrees.length > 0) {
        console.log(`DetectionRenderer: Adding ${allTrees.length} markers to the map`);
        addGoogleMapMarkers(allTrees);
        
        // If we have a jobId, also ensure MLOverlay is rendering the data
        if (detectionJobId && window.mlDetectionData) {
          console.log(`DetectionRenderer: Ensuring MLOverlay shows data for job ${detectionJobId}`);
          
          // If global function is available, use it directly
          if (typeof window.renderDetectionOverlay === 'function') {
            try {
              // CRITICAL FIX: Ensure that mlDetectionData is properly structured with detections array
              let dataToRender = window.mlDetectionData;
              
              // If we have raw detection data but it's not in the right format, convert it
              if (dataToRender && !dataToRender.detections && (dataToRender.trees || allTrees.length > 0)) {
                console.log("DetectionRenderer: Converting data to proper detections format");
                
                // Create a new object with proper structure
                dataToRender = {
                  ...dataToRender,
                  job_id: detectionJobId,
                  // Convert trees array to detections array
                  detections: allTrees.map(tree => {
                    // Ensure tree has bbox or box property
                    if (!tree.bbox && tree.box) {
                      return {
                        ...tree,
                        bbox: [
                          tree.box.x, 
                          tree.box.y, 
                          tree.box.x + tree.box.width, 
                          tree.box.y + tree.box.height
                        ]
                      };
                    } else if (!tree.bbox && tree.coordinates) {
                      // If we have coordinates but no bbox, create a default bbox
                      return {
                        ...tree,
                        bbox: [0.4, 0.4, 0.6, 0.6] // Default centered box
                      };
                    }
                    return tree;
                  })
                };
                
                // Update the global data for future reference
                window.mlDetectionData = dataToRender;
                console.log("DetectionRenderer: Updated mlDetectionData with proper detections array:", dataToRender.detections.length);
              }
              
              console.log("DetectionRenderer: Rendering overlay with data:", {
                hasDetections: !!dataToRender?.detections,
                detectionCount: dataToRender?.detections?.length || 0,
                jobId: detectionJobId
              });
              
              // Call the renderDetectionOverlay function
              window.renderDetectionOverlay(dataToRender, {
                opacity: 1.0,
                classes: {
                  healthy_tree: true,
                  hazardous_tree: true,
                  dead_tree: true,
                  low_canopy_tree: true,
                  pest_disease_tree: true,
                  flood_prone_tree: true,
                  utility_conflict_tree: true,
                  structural_hazard_tree: true,
                  fire_risk_tree: true
                },
                forceRenderBoxes: true,
                jobId: detectionJobId
              });
              console.log("DetectionRenderer: Direct call to renderDetectionOverlay successful");
            } catch (e) {
              console.error("Error calling renderDetectionOverlay:", e);
            }
          }
        }
      } else {
        console.warn("DetectionRenderer: No trees found to add as markers");
      }
    };
    
    // Handler for clearing ML markers
    const handleClearMLMarkers = (event) => {
      console.log("DetectionRenderer: Clearing ML markers", event?.detail?.source || "unknown source");
      
      if (!mapRef.current) return;
      
      // Clear existing markers
      if (mapRef.current.getMarkers) {
        mapRef.current.getMarkers().forEach(marker => {
          if (marker.setMap) marker.setMap(null);
        });
        mapRef.current.getMarkers().length = 0;
      }
      
      // Remove ML data from window to prevent reuse
      window.mlDetectionData = null;
      
      // Also notify other components that markers have been cleared
      window.dispatchEvent(new CustomEvent('mlMarkersCleared'));
    };
    
    window.addEventListener('useCustomMarkerPlacement', handleCustomMarkerPlacement);
    window.addEventListener('clearMLMarkers', handleClearMLMarkers);
    
    return () => {
      window.removeEventListener('useCustomMarkerPlacement', handleCustomMarkerPlacement);
      window.removeEventListener('clearMLMarkers', handleClearMLMarkers);
    };
  }, [mapRef, trees, addGoogleMapMarkers]);
  
  // Listen for tree selection from markers
  useEffect(() => {
    // Handle marker selection events
    const handleTreeSelected = (event) => {
      if (!event.detail) return;
      
      const { tree } = event.detail;
      console.log("Tree selected from map:", tree);
      
      // Find the tree in our current trees list
      const treeInList = trees.find(t => 
        // Try to match by ID first
        (t.id && t.id === tree.id) || 
        // If no ID match, check location
        (t.location && tree.location && 
          t.location[0] === tree.location[0] && 
          t.location[1] === tree.location[1])
      );
      
      if (treeInList) {
        // Update the selected tree
        setSelectedTree(treeInList);
        
        // Also update the current tree index
        const treeIndex = trees.indexOf(treeInList);
        if (treeIndex !== -1) {
          setCurrentTreeIndex(treeIndex);
        }
      }
    };
    
    // Also listen for treeMarkerSelected events from the markers we added
    const handleTreeMarkerSelected = (event) => {
      handleTreeSelected(event);
    };
    
    // Add event listeners
    window.addEventListener('treeSelected', handleTreeSelected);
    window.addEventListener('treeMarkerSelected', handleTreeMarkerSelected);
    
    return () => {
      window.removeEventListener('treeSelected', handleTreeSelected);
      window.removeEventListener('treeMarkerSelected', handleTreeMarkerSelected);
    };
  }, [trees, setSelectedTree, setCurrentTreeIndex]);
  
  // Handle highlighting selected tree
  useEffect(() => {
    if (!selectedTree || !mapRef.current) return;
    
    // Highlight the selected tree if there are markers
    if (mapRef.current.getMarkers) {
      const markers = mapRef.current.getMarkers();
      
      markers.forEach(marker => {
        if (!marker || !marker.getPosition) return;
        
        const pos = marker.getPosition();
        if (!pos) return;
        
        const lat = pos.lat();
        const lng = pos.lng();
        
        // Check if marker position matches selected tree
        if (selectedTree.location && 
            Math.abs(selectedTree.location[1] - lat) < 0.00001 && 
            Math.abs(selectedTree.location[0] - lng) < 0.00001) {
          
          // This is our selected tree, highlight it
          if (marker.setAnimation) {
            marker.setAnimation(window.google.maps.Animation.BOUNCE);
            
            // Stop animation after a short time
            setTimeout(() => {
              marker.setAnimation(null);
            }, 1500);
          }
          
          // Increase size temporarily
          if (marker.getIcon) {
            const icon = marker.getIcon();
            if (icon) {
              const originalScale = icon.scale;
              
              // Increase scale
              icon.scale = originalScale * 1.5;
              marker.setIcon(icon);
              
              // Reset after a delay
              setTimeout(() => {
                icon.scale = originalScale;
                marker.setIcon(icon);
              }, 1500);
            }
          }
        }
      });
    }
  }, [selectedTree, mapRef]);
  
  // Create a state variable to store the loaded detection data
  const [loadedDetectionData, setLoadedDetectionData] = useState(null);
  
  // This function returns the cached data without loading it again
  const getDetectionData = () => {
    // If we have data in component state, use that first
    if (loadedDetectionData && loadedDetectionData.job_id === jobId) {
      return loadedDetectionData;
    }
    
    // Next check if we have data in the global cache
    if (window.mlDetectionData && window.mlDetectionData.job_id === jobId) {
      // Update component state to match global cache
      if (loadedDetectionData !== window.mlDetectionData) {
        setLoadedDetectionData(window.mlDetectionData);
      }
      return window.mlDetectionData;
    }
    
    // If we don't have data cached, return null (data will be loaded by useEffect)
    return null;
  };
  
  // Separate function to load detection data - only called from useEffect
  const loadDetectionData = async () => {
    // Skip if we already have data for this job ID
    if (loadedDetectionData && loadedDetectionData.job_id === jobId) {
      console.log("DetectionRenderer: Already have detection data for job ID:", jobId);
      return loadedDetectionData;
    }
    
    // Skip if no job ID
    if (!jobId) {
      console.error("DetectionRenderer: No job ID provided, cannot load detection data");
      return null;
    }
    
    console.log(`DetectionRenderer: Loading detection data for job ID: ${jobId}`);
    const baseDir = `/data/ml/${jobId}/ml_response`;
    
    try {
      // Load metadata.json
      console.log(`DetectionRenderer: Fetching metadata from ${baseDir}/metadata.json`);
      const metadataResponse = await fetch(`${baseDir}/metadata.json`);
      if (!metadataResponse.ok) {
        throw new Error(`Error loading metadata: ${metadataResponse.status}`);
      }
      const metadata = await metadataResponse.json();
      
      // Load trees.json
      console.log(`DetectionRenderer: Fetching tree data from ${baseDir}/trees.json`);
      const treesResponse = await fetch(`${baseDir}/trees.json`);
      if (!treesResponse.ok) {
        throw new Error(`Error loading trees.json: ${treesResponse.status}`);
      }
      const treesData = await treesResponse.json();
      
      // Combine metadata with detection data
      const combinedData = {
        ...treesData,
        job_id: metadata.job_id || jobId,
        metadata: metadata
      };
      
      console.log("DetectionRenderer: Detection data loaded successfully:", {
        jobId: metadata.job_id,
        detectionCount: treesData.detections?.length || 0
      });
      
      // Store in component state and globally
      setLoadedDetectionData(combinedData);
      window.mlDetectionData = combinedData;
      
      // Notify other components that data is available
      window.dispatchEvent(new CustomEvent('detectionDataAvailable', {
        detail: { data: combinedData }
      }));
      
      return combinedData;
    } catch (error) {
      console.error("DetectionRenderer: Error loading detection data:", error);
      return null;
    }
  };
  
  // Process detection data to ensure it has the correct structure for MLOverlay
  const processDetectionData = () => {
    const data = getDetectionData();
    
    // If we already have properly structured data, return it
    if (data && data.detections && Array.isArray(data.detections) && data.detections.length > 0) {
      console.log(`DetectionRenderer: processDetectionData - Already have ${data.detections.length} detections`);
      return data;
    }
    
    // If we have data but no detections array, convert from trees array
    if (data && !data.detections && data.trees && Array.isArray(data.trees) && data.trees.length > 0) {
      console.log(`DetectionRenderer: processDetectionData - Converting ${data.trees.length} trees to detections format`);
      
      // Create a new data object with the right structure
      const processedData = {
        ...data,
        detections: data.trees.map(tree => {
          // If the tree already has a bbox, use it
          if (tree.bbox) {
            return tree;
          }
          
          // If it has a box property, convert to bbox
          if (tree.box) {
            return {
              ...tree,
              bbox: [
                tree.box.x, 
                tree.box.y, 
                tree.box.x + tree.box.width, 
                tree.box.y + tree.box.height
              ]
            };
          }
          
          // Default to centered bbox if nothing else available
          return {
            ...tree,
            bbox: [0.3, 0.3, 0.7, 0.7] // Default centered box
          };
        })
      };
      
      // Update the global cache
      window.mlDetectionData = processedData;
      console.log(`DetectionRenderer: Updated global mlDetectionData with ${processedData.detections.length} detections`);
      
      return processedData;
    }
    
    // Return original data as fallback
    return data;
  };
  
  // Make sure React component renders properly without direct DOM manipulation
  useEffect(() => {
    if (trees && trees.length > 0) {
      // Just focus on ensuring the React component has the data it needs
      console.log("DetectionRenderer: Ensuring detection data is ready for preview", {
        treeCount: trees.length,
        processedData: !!processDetectionData()
      });
    }
  }, [trees]);
  
  // Add an effect to handle jobId changes and load detection data
  useEffect(() => {
    if (jobId) {
      console.log(`DetectionRenderer: Job ID changed to ${jobId}, loading detection data`);
      
      // Clear any existing detection data from a different job
      if (window.mlDetectionData && window.mlDetectionData.job_id !== jobId) {
        console.log("DetectionRenderer: Clearing previous detection data");
        window.mlDetectionData = null;
      }
      
      // Load the detection data using the async function
      loadDetectionData().then(data => {
        // If data is loaded successfully, trigger preview update
        if (data && data.detections) {
          console.log(`DetectionRenderer: Successfully loaded ${data.detections.length} detections for job ${jobId}`);
          
          // Show preview if not already visible
          if (!showPreview) {
            setShowPreview(true);
          }
        }
      });
    }
  }, [jobId]);
  
  // Create a directly exposed function to load data for a specific job ID
  // This allows other components to call it directly
  const directLoadDataForJobId = useCallback((detectionJobId) => {
    if (!detectionJobId) {
      console.error("DetectionRenderer: No job ID provided to directLoadDataForJobId");
      return;
    }
    
    console.log(`DetectionRenderer: Direct load requested for job ID: ${detectionJobId}`);
    
    // Clear any existing detection data for a different job
    if (window.mlDetectionData && window.mlDetectionData.job_id !== detectionJobId) {
      console.log(`DetectionRenderer: Clearing previous data before loading job ${detectionJobId}`);
      window.mlDetectionData = null;
    }
    
    // Construct base dir using the job ID
    const baseDir = `/data/ml/${detectionJobId}/ml_response`;
    
    // TEMPORARY DEBUG: Set global flag to indicate loading is in progress
    window.mlDataLoading = true;
    window.mlDataLoadingJobId = detectionJobId;
    
    // Show console note about using network tab to verify request
    console.log(`%cDetectionRenderer: Loading ML data for job ${detectionJobId} - check Network tab for requests to trees.json and metadata.json`, 'color: orange; font-weight: bold');
    
    // Asynchronously load the data
    console.log(`DetectionRenderer: Loading data for job ${detectionJobId} directly`);
    
    fetch(`${baseDir}/metadata.json`)
      .then(response => {
        if (!response.ok) throw new Error(`Error loading metadata: ${response.status}`);
        return response.json();
      })
      .then(metadata => {
        console.log(`DetectionRenderer: Successfully loaded metadata for job ${detectionJobId}:`, metadata);
        return fetch(`${baseDir}/trees.json`)
          .then(response => {
            if (!response.ok) throw new Error(`Error loading trees.json: ${response.status}`);
            return response.json();
          })
          .then(treesData => {
            console.log(`DetectionRenderer: Successfully loaded trees.json with ${treesData.detections?.length || 0} detections`);
            const combinedData = {
              ...treesData,
              job_id: metadata.job_id || detectionJobId,
              metadata: metadata
            };
            
            console.log(`DetectionRenderer: Successfully loaded ${combinedData.detections?.length || 0} detections for job ${detectionJobId}`);
            
            // Store in component state and globally
            setLoadedDetectionData(combinedData);
            window.mlDetectionData = combinedData;
            
            // Clear loading flag
            window.mlDataLoading = false;
            
            // Also dispatch a custom event for any component listening for this data
            window.dispatchEvent(new CustomEvent('mlDetectionDataLoaded', {
              detail: {
                data: combinedData,
                jobId: detectionJobId,
                source: 'directLoad'
              }
            }));
            
            // Show preview if not already visible
            if (!showPreview) {
              setShowPreview(true);
            }
            
            // Process detection data for rendering on the map directly
            if (combinedData.detections && combinedData.detections.length > 0) {
              console.log(`DetectionRenderer: Processing ${combinedData.detections.length} detections for map rendering`);
              
              // Force MLOverlay to render the detection data FIRST
              if (typeof window.renderMLOverlay === 'function') {
                try {
                  console.log("DetectionRenderer: Directly calling renderMLOverlay global function");
                  
                  // Get map container element
                  let targetElement = null;
                  if (window.googleMapsInstance && typeof window.googleMapsInstance.getDiv === 'function') {
                    try {
                      targetElement = window.googleMapsInstance.getDiv();
                    } catch (e) {
                      console.error('Error getting div from googleMapsInstance:', e);
                    }
                  }
                  
                  // CRITICAL FIX: Ensure the data has a proper detections array
                  if (!combinedData.detections && combinedData.trees) {
                    console.log("DetectionRenderer: Adding detections array from trees array");
                    
                    // Convert trees to detections format
                    combinedData.detections = combinedData.trees.map(tree => {
                      // If the tree doesn't have a bbox property, add one
                      if (!tree.bbox) {
                        // If it has a box property, convert it to bbox
                        if (tree.box) {
                          return {
                            ...tree,
                            bbox: [
                              tree.box.x, 
                              tree.box.y, 
                              tree.box.x + tree.box.width, 
                              tree.box.y + tree.box.height
                            ]
                          };
                        }
                        // Otherwise create a default centered bbox
                        else {
                          return {
                            ...tree,
                            bbox: [0.3, 0.3, 0.7, 0.7] // Default centered box
                          };
                        }
                      }
                      return tree;
                    });
                    
                    console.log(`DetectionRenderer: Created detections array with ${combinedData.detections.length} items`);
                  }
                  
                  window.renderMLOverlay(combinedData, {
                    opacity: 1.0,
                    classes: {
                      healthy_tree: true,
                      hazardous_tree: true,
                      dead_tree: true,
                      low_canopy_tree: true,
                      pest_disease_tree: true,
                      flood_prone_tree: true,
                      utility_conflict_tree: true,
                      structural_hazard_tree: true,
                      fire_risk_tree: true
                    },
                    targetElement: targetElement,
                    jobId: detectionJobId,
                    forceRenderBoxes: true
                  });
                  
                  console.log(`DetectionRenderer: Overlay rendered with ${combinedData.detections?.length || 0} detections`);
                } catch (e) {
                  console.error("Error directly calling renderDetectionOverlay:", e);
                }
              }
              
              // Then trigger custom marker placement to render markers
              window.dispatchEvent(new CustomEvent('useCustomMarkerPlacement', {
                detail: {
                  treeData: combinedData.detections,
                  jobId: detectionJobId,
                  filters: {
                    trees: true,
                    buildings: true,
                    powerLines: true
                  },
                  showLabels: true,
                  method: 'direct'
                }
              }));
            }
            
            // Dispatch event to notify other components
            window.dispatchEvent(new CustomEvent('detectionDataAvailable', {
              detail: { data: combinedData }
            }));
            
            // Also dispatch an ML Data Available event for MLOverlay
            window.dispatchEvent(new CustomEvent('mlDataAvailable', {
              detail: { data: combinedData }
            }));
            
            return combinedData;
          });
      })
      .catch(error => {
        console.error(`DetectionRenderer: Error loading data for job ${detectionJobId}:`, error);
      });
  }, [showPreview]);
  
  // Save the function to window for direct access
  useEffect(() => {
    window.loadDetectionDataForJob = directLoadDataForJobId;
    return () => {
      delete window.loadDetectionDataForJob;
    };
  }, [directLoadDataForJobId]);
  
  // Listen for detection complete events from the app
  useEffect(() => {
    const handleEnterDetectionMode = (event) => {
      console.log("DetectionRenderer: Received enterDetectionMode event:", event.detail);
      
      // Extract job ID from event - check multiple possible properties
      const detectionJobId = event.detail?.detectionJobId || 
                             event.detail?.jobId || 
                             (event.detail?.ml_response_dir ? event.detail.ml_response_dir.split('/').pop().replace('/ml_response', '') : null);
      
      if (detectionJobId) {
        console.log(`DetectionRenderer: Using job ID from event: ${detectionJobId}`);
        
        // Use our callback function to load the data
        directLoadDataForJobId(detectionJobId);
      } else {
        console.error("DetectionRenderer: Could not find job ID in event:", event.detail);
      }
    };
    
    // Add event listener
    window.addEventListener('enterDetectionMode', handleEnterDetectionMode);
    
    return () => {
      window.removeEventListener('enterDetectionMode', handleEnterDetectionMode);
    };
  }, [directLoadDataForJobId]);
  
  // Manually trigger the detection preview when the showPreview state changes
  useEffect(() => {
    // Use a microtask to avoid React rendering conflicts
    const handlePreviewChange = () => {
      // Defer any DOM manipulations to after React rendering cycle
      setTimeout(() => {
        if (showPreview) {
          // Check if a recent View on Map operation was performed
          // If so, don't reshow the preview
          if (window._preventDetectionPreviewReopen) {
            console.log("DetectionRenderer: Skipping preview reopen due to _preventDetectionPreviewReopen flag");
            return;
          }
          
          console.log("DetectionRenderer: showPreview is true, manually triggering global preview function");
          
          const data = processDetectionData();
          if (data && typeof window.showDetectionPreview === 'function') {
            try {
              window.showDetectionPreview(data);
            } catch (error) {
              console.warn("Error showing detection preview, will retry:", error);
              
              // Retry once after a short delay
              setTimeout(() => {
                try {
                  window.showDetectionPreview(data);
                } catch (retryError) {
                  console.error("Failed to show detection preview after retry:", retryError);
                }
              }, 100);
            }
          }
        } else {
          // If showPreview is false, schedule the cleanup after current render cycle
          setTimeout(() => {
            if (typeof window.destroyDetectionPreview === 'function') {
              try {
                window.destroyDetectionPreview();
              } catch (error) {
                console.warn("Error closing detection preview:", error);
              }
            }
          }, 0);
        }
      }, 0);
    };
    
    // Call the handler after React has completed its current render cycle
    Promise.resolve().then(handlePreviewChange);
  }, [showPreview]);

  // Get the detection data from our cached state - no loading here
  const detectionData = getDetectionData();
  
  // Add a debug output for jobId to help troubleshoot
  console.log("DetectionRenderer: Rendering with jobId:", jobId, 
    detectionData ? 
      `(${detectionData.detections?.length || 0} detections)` : 
      "(no data loaded yet)"
  );
  
  return (
    <>
      {/* MLOverlayInitializer to ensure proper initialization */}
      <MLOverlayInitializer />
      
      {/* ML Detection Overlay */}
      {showOverlay && detectionData && (
        <div id="ml-overlay-react-container">
          {/* Use direct function call instead of component to avoid issues */}
          {(() => {
            // This IIFE executes immediately once the component renders
            if (typeof MLOverlayModule.renderDetectionOverlay === 'function' && detectionData) {
                console.log('Rendering detection overlay through direct function call');
                
                // Force initialization and ensure multiple attempts to render
                const attemptRender = (attempt = 1) => {
                  console.log(`Attempt ${attempt} to render ML overlay`);
                  
                  // Always try to initialize first
                  if (MLOverlayModule.ensureInitialized) {
                    MLOverlayModule.ensureInitialized();
                  }
                  
                  // Even if initialization appears to fail, try rendering anyway
                  try {
                    MLOverlayModule.renderMLOverlay(detectionData, {
                      opacity: overlayOpacity,
                      classes: detectionFilters,
                      forceRenderBoxes: true,
                      targetElement: document.getElementById('map-container') || 
                                   document.getElementById('map')
                    });
                    console.log(`Successfully rendered ML overlay on attempt ${attempt}`);
                  } catch (e) {
                    console.error(`Error rendering ML overlay on attempt ${attempt}:`, e);
                    
                    // Retry with exponential backoff if within retry limit
                    if (attempt < 3) {
                      setTimeout(() => attemptRender(attempt + 1), 200 * attempt);
                    }
                  }
                };
                
                // Start rendering attempts immediately
                attemptRender();
              }
            return null; // This doesn't render anything, just triggers the function
          })()}
        </div>
      )}
      
      {/* 
        Detection Preview is handled via the global window.showDetectionPreview function
        This approach completely bypasses the React component hierarchy
        to ensure the preview renders at document level
      */}
    </>
  );
};

export default DetectionRenderer;