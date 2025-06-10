// src/components/visualization/Detection/DetectionSidebar.jsx
//
// This component is responsible for the sidebar UI that displays
// detected objects and allows users to interact with them.
// It's split from the larger DetectionMode.jsx component.

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { 
  X, Check, Edit, Plus, Trash, ChevronLeft, ChevronRight, 
  Save, AlertTriangle, FileText, MapPin, Clock, CheckCircle, 
  XCircle, Search, Database, BarChart, Settings, Image, 
  Eye, EyeOff, Sliders, Layers
} from 'lucide-react';

// Import the components for detection categories and preview
import DetectionCategories from './DetectionCategories';
import DetectionPreview from './DetectionPreview';
import MLStatusIndicator from './MLStatusIndicator';

/**
 * Detection Sidebar Component
 * 
 * Handles the UI for the detection sidebar, including:
 * - Tree/object list display
 * - Object selection and navigation
 * - Object details editing
 * - Collapsing/expanding the sidebar
 */
const DetectionSidebar = ({
  trees = [],
  selectedTree,
  setSelectedTree,
  currentTreeIndex,
  setCurrentTreeIndex,
  isEditing,
  setIsEditing,
  formData,
  setFormData,
  removeTree,
  validateTree,
  goToNextTree,
  goToPreviousTree,
  handleInputChange,
  saveTreeEdits,
  collapsed,
  setCollapsed,
  width,
  setWidth,
  manualPlacement,
  setManualPlacement,
  showOverlay,
  setShowOverlay,
  overlayOpacity,
  setOverlayOpacity,
  geminiParams,
  setGeminiParams,
  headerCollapsed
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredTrees, setFilteredTrees] = useState([]);
  const [configVisible, setConfigVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('trees');
  const [showSegmentation, setShowSegmentation] = useState(true);
  
  // Filter trees based on search query
  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredTrees(trees.filter(tree => tree.visible));
      return;
    }
    
    const query = searchQuery.toLowerCase();
    const filtered = trees.filter(tree => {
      if (!tree.visible) return false;
      
      // Search in various tree properties
      return (
        (tree.species && tree.species.toLowerCase().includes(query)) ||
        (tree.risk_level && tree.risk_level.toLowerCase().includes(query)) ||
        (tree.address && tree.address.toLowerCase().includes(query)) ||
        (tree.id && tree.id.toLowerCase().includes(query))
      );
    });
    
    setFilteredTrees(filtered);
  }, [searchQuery, trees]);
  
  /**
   * Toggle sidebar collapse state and properly update the map container size
   */
  const toggleCollapse = () => {
    const newCollapsedState = !collapsed;
    setCollapsed(newCollapsedState);
    
    // Create detailed event payload with all relevant information
    const eventDetail = {
      collapsed: newCollapsedState,
      source: 'tree_detection',
      width: newCollapsedState ? 0 : width,
      headerCollapsed: headerCollapsed, // Include header state
      timestamp: Date.now()
    };
    
    console.log(`DetectionSidebar: Toggling collapse to ${newCollapsedState ? 'collapsed' : 'expanded'} with width ${width}px`);
    
    // Batch these operations for better performance
    requestAnimationFrame(() => {
      // 1. First adjust the map container directly for immediate visual feedback
      const mapContainer = document.querySelector('#map-container');
      if (mapContainer) {
        mapContainer.style.right = newCollapsedState ? '0px' : `${width}px`;
      }
      
      // 2. Then dispatch both events for components to handle
      // Legacy event
      window.dispatchEvent(new CustomEvent('validationSidebarToggle', {
        detail: eventDetail
      }));
      
      // New specific event
      window.dispatchEvent(new CustomEvent('detectionSidebarToggle', {
        detail: eventDetail
      }));
      
      // 3. Force Google Maps to recalculate its size
      const mapInstance = window.googleMapsInstance || window._googleMap;
      if (mapInstance && window.google && window.google.maps) {
        setTimeout(() => {
          window.google.maps.event.trigger(mapInstance, 'resize');
          console.log('DetectionSidebar: Triggered Google Maps resize after toggle');
        }, 100);
      }
      
      // 4. Also dispatch a general resize event
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 150);
    });
  };
  
  // Get visible trees
  const visibleTrees = trees.filter(tree => tree.visible);
  
  // Use a ref to track previous visible trees to prevent infinite updates
  const prevVisibleTreesRef = useRef(visibleTrees);
  
  // Initialize filter at first render if needed
  useEffect(() => {
    // Compare trees to avoid unnecessary state updates
    const prevTrees = prevVisibleTreesRef.current || [];
    const treesChanged = 
      prevTrees.length !== visibleTrees.length ||
      JSON.stringify(prevTrees.map(t => t.id)) !== JSON.stringify(visibleTrees.map(t => t.id));
    
    // Only update filtered trees if visibleTrees has actually changed
    if (treesChanged) {
      console.log('DetectionSidebar: Visible trees changed, updating filtered trees');
      setFilteredTrees(visibleTrees);
      // Update the ref to current value
      prevVisibleTreesRef.current = [...visibleTrees];
      
      // Update the DOM counter directly to ensure it's always in sync
      const counterEl = document.getElementById('detected-objects-count');
      if (counterEl) {
        counterEl.textContent = visibleTrees.length.toString();
      }
    }
    
    // CRITICAL: Check for detection data and update counter when we have mlDetectionData
    if (window.mlDetectionData && window.mlDetectionData.trees && window.mlDetectionData.trees.length > 0) {
      console.log(`DetectionSidebar: Found detection data with ${window.mlDetectionData.trees.length} trees`);
      
      // Only update if trees prop is empty but we have detection data
      if ((!trees || trees.length === 0) && window.mlDetectionData.trees.length > 0) {
        console.log('DetectionSidebar: Updating counter with trees from global mlDetectionData');
        // Update the DOM counter directly
        const counterEl = document.getElementById('detected-objects-count');
        if (counterEl) {
          counterEl.textContent = window.mlDetectionData.trees.length.toString();
        }
      }
    }
  }, [visibleTrees, trees]);
  
  // =====================================================================
  // EMERGENCY FIX: COMPLETE REWRITE OF STATE MANAGEMENT TO PREVENT INFINITE UPDATES
  // =====================================================================
  
  // Use a static class variable to track if we've initialized settings
  // This prevents any React state updates from triggering re-renders
  if (!window._mlSettingsInitialized) {
    console.log('DetectionSidebar: ONE-TIME INITIALIZATION of global settings');
    
    // Initialize global settings with 30% opacity default
    window.mlOverlaySettings = {
      showOverlay: true,
      showSegmentation: true,
      opacity: 0.3
    };
    
    window.detectionShowOverlay = true;
    
    // Override any previously saved value with our new default
    try {
      // First clear any existing saved value
      localStorage.removeItem('ml-overlay-opacity');
      
      // Then save our new default
      localStorage.setItem('ml-overlay-opacity', '0.3');
      
      console.log(`DetectionSidebar: Set default opacity to 0.3`);
    } catch (e) {
      console.error("DetectionSidebar: Error accessing localStorage for opacity setting:", e);
    }
    window._mlSettingsInitialized = true;
    
    // Initialize global settings directly without delay
      
    // Ensure global settings are initialized correctly
    window.mlOverlaySettings = {
      showOverlay: true,
      showSegmentation: true,
      opacity: 0.3
    };
    
    window.detectionShowOverlay = true;
      
      // Make sure we have the correct functions registered globally
      // If MLOverlayInitializer has done its job, these should already be available
      if (typeof window.renderMLOverlay === 'function' && 
          typeof window.updateMLOverlayOpacity === 'function' &&
          typeof window.toggleMLOverlayVisibility === 'function') {
        console.log('DetectionSidebar: Working overlay functions are available');
      } else {
        console.warn('DetectionSidebar: Working overlay functions not found, creating fallbacks');
        
        // Create fallback functions directly from integrated MLOverlay.js module
        // Import the MLOverlay module and use its functions directly
        import('./MLOverlay').then(MLOverlayModule => {
          console.log('DetectionSidebar: Imported MLOverlay module, setting up global functions');
          
          // Set up global functions from the module
          window.renderMLOverlay = MLOverlayModule.renderMLOverlay;
          window.updateMLOverlayOpacity = MLOverlayModule.updateMLOverlayOpacity;
          window.toggleMLOverlayVisibility = (visible) => {
            console.log(`DetectionSidebar: toggleMLOverlayVisibility called with ${visible}`);
            if (window._mlDetectionOverlay && window._mlDetectionOverlay.div) {
              window._mlDetectionOverlay.div.style.display = visible ? 'block' : 'none';
              return true;
            }
            return false;
          };
          
          console.log('DetectionSidebar: Global overlay functions set up from integrated module');
        }).catch(err => {
          console.error('DetectionSidebar: Failed to import MLOverlay module, using basic fallbacks', err);
          
          // Provide basic fallback implementations if module import fails
          window.renderMLOverlay = (map, data, options = {}) => {
            console.log('DetectionSidebar: renderMLOverlay fallback called');
            // Basic implementation to register data
            window.mlDetectionData = data;
            return true;
          };
          
          window.updateMLOverlayOpacity = (opacity) => {
            console.log(`DetectionSidebar: updateMLOverlayOpacity fallback called with ${opacity}`);
            // Basic implementation to update settings
            window.mlOverlaySettings = window.mlOverlaySettings || {};
            window.mlOverlaySettings.opacity = opacity;
            
            // Direct DOM update for simple overlay
            const overlay = document.getElementById('ml-detection-overlay');
            if (overlay) {
              overlay.style.opacity = opacity;
              return true;
            }
            return false;
          };
          
          window.toggleMLOverlayVisibility = (visible) => {
            console.log(`DetectionSidebar: toggleMLOverlayVisibility fallback called with ${visible}`);
            // Basic implementation to update settings
            window.mlOverlaySettings = window.mlOverlaySettings || {};
            window.mlOverlaySettings.showOverlay = visible;
            window.detectionShowOverlay = visible;
            
            // Direct DOM update for simple overlay
            const overlay = document.getElementById('ml-detection-overlay');
            if (overlay) {
              overlay.style.display = visible ? 'block' : 'none';
              return true;
            }
            return false;
          };
        });
      }
      
      // Ensure a simple overlay always exists - this is a failsafe
      const createSimpleOverlay = () => {
        // Check if we already have a simple overlay
        if (!document.getElementById('ml-detection-overlay')) {
          console.log('DetectionSidebar: Creating simple overlay as a fallback');
          const mapContainer = document.getElementById('map-container');
          if (mapContainer) {
            const overlay = document.createElement('div');
            overlay.id = 'ml-detection-overlay';
            overlay.style.position = 'absolute';
            overlay.style.top = '0';
            overlay.style.left = '0';
            overlay.style.width = '100%';
            overlay.style.height = '100%';
            overlay.style.backgroundColor = `rgba(0, 30, 60, ${window.mlOverlaySettings.opacity || 0.3})`;
            overlay.style.pointerEvents = 'none';
            overlay.style.zIndex = '50';
            overlay.style.display = window.mlOverlaySettings.showOverlay ? 'block' : 'none';
            mapContainer.appendChild(overlay);
          }
        }
      };
      
      // Create the simple overlay if needed
      createSimpleOverlay();
      
      // Dispatch an event to notify other components that the overlay is ready
      window.dispatchEvent(new CustomEvent('mlOverlayReady', {
        detail: { 
          source: 'detection_sidebar',
          showOverlay: window.mlOverlaySettings.showOverlay,
          opacity: window.mlOverlaySettings.opacity
        }
      }));
  }
  
  // Track last toggle timestamp to prevent duplicate events
  const lastToggleTimeRef = useRef(0);
  
  /**
   * Toggles the visibility of the ML overlay
   * This is a simplified function that directly modifies the existing overlay
   * without recreating it unnecessarily
   * @param {boolean} newValue - Whether the overlay should be visible
   */
  const handleToggleOverlay = useCallback((newValue) => {
    // Prevent duplicate calls within short time window (debounce)
    const now = Date.now();
    if (now - lastToggleTimeRef.current < 100) {
      console.log('DetectionSidebar: Ignoring rapid toggle to prevent infinite loop');
      return;
    }
    lastToggleTimeRef.current = now;
    
    console.log(`DetectionSidebar: Toggle overlay visibility to ${newValue ? 'visible' : 'hidden'}`);
    
    // Update local React state
    setShowOverlay(newValue);
    
    // Update the global state
    window.mlOverlaySettings = {
      ...(window.mlOverlaySettings || {}),
      showOverlay: newValue
    };
    window.detectionShowOverlay = newValue;
    
    // Save to localStorage for persistence
    try {
      localStorage.setItem('ml-overlay-show', newValue ? 'true' : 'false');
    } catch (e) {
      console.error("Error saving overlay visibility to localStorage:", e);
    }
    
    // APPROACH 1: Use direct access to the MLDetectionOverlay instance (most efficient)
    if (window._mlDetectionOverlay) {
      try {
        if (window._mlDetectionOverlay.div) {
          window._mlDetectionOverlay.div.style.display = newValue ? 'block' : 'none';
          console.log(`Updated MLDetectionOverlay div visibility to ${newValue ? 'visible' : 'hidden'}`);
        }
      } catch (e) {
        console.error("Error updating MLDetectionOverlay directly:", e);
      }
    }
    
    // APPROACH 2: Use the global toggleMLOverlayVisibility function if available
    if (typeof window.toggleMLOverlayVisibility === 'function') {
      try {
        window.toggleMLOverlayVisibility(newValue);
      } catch (e) {
        console.error("Error calling toggleMLOverlayVisibility:", e);
      }
    }
    
    // APPROACH 3: Direct DOM manipulation as fallback
    const overlay = document.getElementById('ml-detection-overlay');
    if (overlay) {
      overlay.style.display = newValue ? 'block' : 'none';
    }
    
    // Also toggle detection badges visibility
    const badges = [
      document.getElementById('detection-debug'),
      document.querySelector('.detection-badge')
    ];
    
    badges.forEach(badge => {
      if (badge) {
        badge.style.display = newValue ? 'block' : 'none';
        badge.style.opacity = newValue ? '1' : '0';
      }
    });
    
    // Notify other components about the change
    window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
      detail: { 
        showOverlay: newValue,
        showSegmentation: window.mlOverlaySettings?.showSegmentation !== false,
        opacity: window.mlOverlaySettings?.opacity || 0.3,
        source: 'detection_sidebar_toggle_function',
        timestamp: now
      }
    }));
  }, []);
  
  // Track last segmentation toggle timestamp to prevent duplicate events
  const lastSegToggleTimeRef = useRef(0);
  
  /**
   * Toggles the visibility of segmentation masks in the ML overlay
   * This is a simplified function that avoids recreating the overlay
   * @param {boolean} newValue - Whether segmentation masks should be visible
   */
  const handleSegmentationToggle = useCallback((newValue) => {
    // Prevent duplicate calls within short time window (debounce)
    const now = Date.now();
    if (now - lastSegToggleTimeRef.current < 100) {
      console.log('DetectionSidebar: Ignoring rapid segmentation toggle to prevent infinite loop');
      return;
    }
    lastSegToggleTimeRef.current = now;
    
    console.log(`DetectionSidebar: Toggle segmentation mask visibility to ${newValue ? 'visible' : 'hidden'}`);
    
    // Update local state
    setShowSegmentation(newValue);
    
    // Update global settings
    window.mlOverlaySettings = {
      ...(window.mlOverlaySettings || {}),
      showSegmentation: newValue
    };
    
    // APPROACH 1: Use the updateMLOverlayClasses function from MLOverlay.js
    if (typeof window.updateMLOverlayClasses === 'function') {
      window.updateMLOverlayClasses({
        showSegmentation: newValue
      });
    }
    
    // APPROACH 2: Direct DOM update as fallback
    const masks = document.querySelectorAll('.segmentation-mask');
    masks.forEach(mask => {
      mask.style.display = newValue ? 'block' : 'none';
    });
    
    // Notify other components about the change
    window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
      detail: { 
        showOverlay: window.mlOverlaySettings?.showOverlay !== false,
        showSegmentation: newValue,
        opacity: window.mlOverlaySettings?.opacity || overlayOpacity,
        source: 'detection_sidebar_segmentation_function',
        timestamp: now
      }
    }));
  }, [overlayOpacity]);
  
  // Track last opacity change timestamp to prevent duplicate events
  const lastOpacityChangeTimeRef = useRef(0);
  
  // Update counter when detectionDataLoaded event is received
  useEffect(() => {
    const handleDetectionDataLoaded = (event) => {
      if (event.detail && (event.detail.trees || event.detail.detections)) {
        const data = event.detail;
        const count = data.trees?.length || data.detections?.length || 0;
        
        console.log(`DetectionSidebar: Detection data loaded with ${count} objects`);
        
        // Update the counter directly
        const counterEl = document.getElementById('detected-objects-count');
        if (counterEl) {
          counterEl.textContent = count.toString();
          console.log(`DetectionSidebar: Updated Objects Found counter to ${count} from event`);
        }
      }
    };
    
    // Listen for detection data events
    window.addEventListener('detectionDataLoaded', handleDetectionDataLoaded);
    window.addEventListener('fastInferenceResults', handleDetectionDataLoaded);
    
    return () => {
      // Clean up event listeners
      window.removeEventListener('detectionDataLoaded', handleDetectionDataLoaded);
      window.removeEventListener('fastInferenceResults', handleDetectionDataLoaded);
    };
  }, []);
  
  /**
   * Updates the opacity of the ML overlay - improved real-time implementation
   * @param {number} newValue - The new opacity value (0-1)
   */
  const handleOpacityChange = useCallback((newValue) => {
    // Prevent duplicate calls within short time window (debounce)
    const now = Date.now();
    if (now - lastOpacityChangeTimeRef.current < 50) { // Reduced to 50ms for better real-time feel
      console.log('DetectionSidebar: Ignoring rapid opacity change to prevent infinite loop');
      return;
    }
    lastOpacityChangeTimeRef.current = now;
    
    console.log(`DetectionSidebar: Updating overlay opacity to ${newValue}`);
    
    // Update local React state
    setOverlayOpacity(newValue);
    
    // APPROACH 1: Use the setMLOverlayOpacity helper function if available
    if (typeof window.setMLOverlayOpacity === 'function') {
      window.setMLOverlayOpacity(newValue);
      return; // This function handles everything we need
    }
    
    // APPROACH 2: Use updateMLOverlayOpacity directly
    if (typeof window.updateMLOverlayOpacity === 'function') {
      window.updateMLOverlayOpacity(newValue);
    }
    
    // APPROACH 3: Update the MLDetectionOverlay instance directly
    if (window._mlDetectionOverlay) {
      if (typeof window._mlDetectionOverlay.updateOpacity === 'function') {
        window._mlDetectionOverlay.updateOpacity(newValue);
      } else if (window._mlDetectionOverlay.div) {
        window._mlDetectionOverlay.div.style.opacity = newValue.toString();
      }
    }
    
    // APPROACH 4: Direct DOM manipulation as fallback
    const overlay = document.getElementById('ml-detection-overlay');
    if (overlay) {
      overlay.style.opacity = newValue.toString();
      
      // Also update any tint layers
      const tintLayers = overlay.querySelectorAll('.tint-layer');
      if (tintLayers.length > 0) {
        tintLayers.forEach(layer => {
          layer.style.opacity = newValue;
        });
      } else {
        // If no tint layers, update background color directly
        overlay.style.backgroundColor = `rgba(0, 30, 60, ${newValue})`;
      }
    }
    
    // APPROACH 5: Update all segmentation masks
    const masks = document.querySelectorAll('.segmentation-mask');
    if (masks.length > 0) {
      masks.forEach(mask => {
        mask.style.opacity = Math.min(newValue * 1.2, 0.8); // Slightly adjust for better visibility
      });
    }
    
    // Update global settings for other components
    window.mlOverlaySettings = {
      ...(window.mlOverlaySettings || {}),
      opacity: newValue
    };
    
    // Save to localStorage for persistence
    try {
      localStorage.setItem('ml-overlay-opacity', newValue.toString());
    } catch (e) {
      console.error("Error saving opacity to localStorage:", e);
    }
    
    // Dispatch event for other components
    window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
      detail: { 
        showOverlay: window.mlOverlaySettings?.showOverlay !== false,
        showSegmentation: window.mlOverlaySettings?.showSegmentation !== false,
        opacity: newValue,
        source: 'detection_sidebar_opacity_slider',
        timestamp: now
      }
    }));
    
    // Also dispatch a specific opacity update event that other components can listen for
    window.dispatchEvent(new CustomEvent('mlOverlayOpacityUpdated', {
      detail: { 
        opacity: newValue,
        source: 'detection_sidebar_opacity_slider',
        timestamp: now
      }
    }));
  }, []);
  
  // Calculate top position based on header state
  const [sidebarTopPosition, setSidebarTopPosition] = useState(headerCollapsed ? '0px' : '60px');
  
  // Update sidebar position when header state changes
  useEffect(() => {
    const header = document.querySelector('header');
    const headerHeight = header ? header.offsetHeight : (headerCollapsed ? 0 : 60);
    
    // Update position based on current header state
    setSidebarTopPosition(headerCollapsed ? '0px' : `${headerHeight}px`);
    
    // Apply position directly to DOM element for immediate effect
    const sidebarElement = document.querySelector('.detection-sidebar');
    if (sidebarElement) {
      sidebarElement.style.top = headerCollapsed ? '0px' : `${headerHeight}px`;
      sidebarElement.style.height = headerCollapsed ? '100vh' : `calc(100vh - ${headerHeight}px)`;
      console.log(`DetectionSidebar: Applied initial header state adjustments: top=${headerCollapsed ? '0px' : `${headerHeight}px`}`);
    }
    
    // Listen for header collapse events
    const handleHeaderCollapse = (event) => {
      const isCollapsed = event.detail.collapsed;
      const header = document.querySelector('header');
      const headerHeight = header ? header.offsetHeight : (isCollapsed ? 0 : 60);
      
      // Update sidebar position
      setSidebarTopPosition(isCollapsed ? '0px' : `${headerHeight}px`);
      
      // Apply position directly to DOM element for immediate effect
      const sidebarElement = document.querySelector('.detection-sidebar');
      if (sidebarElement) {
        sidebarElement.style.top = isCollapsed ? '0px' : `${headerHeight}px`;
        sidebarElement.style.height = isCollapsed ? '100vh' : `calc(100vh - ${headerHeight}px)`;
      }
      
      // Force map container to resize
      const mapContainer = document.getElementById('map-container');
      if (mapContainer) {
        // Trigger resize event on map container
        window.dispatchEvent(new Event('resize'));
        
        // Force Google Maps to recalculate its size
        const mapInstance = window.googleMapsInstance || window._googleMap;
        if (mapInstance && window.google && window.google.maps) {
          window.google.maps.event.trigger(mapInstance, 'resize');
        }
      }
      
      console.log(`DetectionSidebar: Adjusted for header ${isCollapsed ? 'collapse' : 'expand'}, top=${isCollapsed ? '0px' : `${headerHeight}px`}`);
    };
    
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    return () => {
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
    };
  }, [headerCollapsed]);
  
  // Calculate risk level color based on risk level
  const getRiskLevelColor = (riskLevel) => {
    switch (riskLevel) {
      case 'high':
        return 'text-red-500';
      case 'medium':
        return 'text-orange-500';
      case 'low':
        return 'text-green-500';
      // Tree risk category colors
      case 'healthy_tree':
        return 'text-green-500';
      case 'hazardous_tree':
        return 'text-purple-500';
      case 'dead_tree':
        return 'text-gray-600';
      case 'low_canopy_tree':
        return 'text-blue-500';
      case 'pest_disease_tree':
        return 'text-lime-600';
      case 'flood_prone_tree':
        return 'text-cyan-600';
      case 'utility_conflict_tree':
        return 'text-blue-600';
      case 'structural_hazard_tree':
        return 'text-teal-600';
      case 'fire_risk_tree':
        return 'text-indigo-600';
      default:
        return 'text-blue-500';
    }
  };
  
  // Calculate risk level text based on risk level
  const getRiskLevelText = (riskLevel) => {
    switch (riskLevel) {
      case 'high':
        return 'High Risk';
      case 'medium':
        return 'Medium Risk';
      case 'low':
        return 'Low Risk';
      case 'new':
        return 'Unassigned';
      // Map tree risk categories to their display text
      case 'healthy_tree':
        return 'Healthy Tree';
      case 'hazardous_tree':
        return 'Hazardous Tree';
      case 'dead_tree':
        return 'Dead Tree';
      case 'low_canopy_tree':
        return 'Low Canopy Tree';
      case 'pest_disease_tree':
        return 'Pest/Disease Tree';
      case 'flood_prone_tree':
        return 'Flood-Prone Tree';
      case 'utility_conflict_tree':
        return 'Utility Conflict Tree';
      case 'structural_hazard_tree':
        return 'Structural Hazard Tree';
      case 'fire_risk_tree':
        return 'Fire Risk Tree';
      default:
        return 'Unknown';
    }
  };

  // This component is rendered inside a portal to the DOM-created sidebar
  // It should only render content, not create a new sidebar
  return (
    <div 
      className={`detection-content flex-col overflow-auto w-full h-full ${collapsed ? 'hidden' : 'flex'}`}
      style={{
        color: '#0d47a1',
        backgroundColor: '#f8fafc'
      }}
    >
      {/* Streamlined Detection Header with Close Button */}
      <div className="px-3 pt-3 pb-2 mb-1 border-b border-slate-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <MLStatusIndicator />
          </div>
          
          <div className="flex items-center">
            <button 
              className="flex items-center bg-white px-2 py-1 rounded border border-blue-100 hover:bg-blue-50 transition-colors mr-2"
              onClick={() => {
                // When clicking the Objects Found button, show the detection preview
                if (window.mlDetectionData && typeof window.showDetectionPreview === 'function') {
                  console.log('DetectionSidebar: Showing detection preview from Objects Found counter');
                  
                  // CRITICAL FIX: Ensure the ML overlay is visible and has the data
                  // Force visibility with explicit display
                  const overlay = document.getElementById('ml-detection-overlay');
                  if (overlay) {
                    overlay.style.display = 'block';
                    overlay.style.opacity = '1';
                    
                    // Try to use stored opacity from localStorage
                    try {
                      const savedOpacity = localStorage.getItem('ml-overlay-opacity');
                      if (savedOpacity !== null) {
                        const opacity = parseFloat(savedOpacity);
                        overlay.style.backgroundColor = `rgba(0, 30, 60, ${opacity})`;
                        console.log(`DetectionSidebar: Applied stored opacity to overlay: ${opacity}`);
                      }
                    } catch (e) {
                      console.error("DetectionSidebar: Error applying stored opacity:", e);
                    }
                  }
                  
                  // Also ensure the ML overlay has the data by explicitly calling renderMLOverlay
                  const mapInstance = window.map || window.googleMapsInstance || window._googleMap;
                  if (mapInstance && typeof window.renderMLOverlay === 'function') {
                    // This triggers the overlay to render with existing data
                    window.renderMLOverlay(
                      mapInstance,
                      window.mlDetectionData,
                      {
                        opacity: window.mlOverlaySettings?.opacity || 0.3,
                        showSegmentation: window.mlOverlaySettings?.showSegmentation !== false,
                        forceVisible: true // Added flag to force visibility
                      }
                    );
                    
                    console.log('DetectionSidebar: Forced ML overlay visibility with existing data');
                  }
                  
                  // Show the preview with detection data
                  // Add more debugging to identify potential data issues
                  console.log('DetectionSidebar: ML detection data structure:', {
                    jobId: window.mlDetectionData?.job_id,
                    treesLength: window.mlDetectionData?.trees?.length,
                    detectionsLength: window.mlDetectionData?.detections?.length,
                    hasDetections: !!window.mlDetectionData?.detections,
                    hasTrees: !!window.mlDetectionData?.trees
                  });
                  
                  // Ensure we have a properly structured object to pass to preview
                  const enhancedData = {
                    ...window.mlDetectionData,
                    job_id: window.mlDetectionData?.job_id || window.currentDetectionJobId,
                    timestamp: window.mlDetectionData?.timestamp || new Date().toISOString(),
                    // Ensure we have the trees array populated, using detections as fallback
                    trees: window.mlDetectionData?.trees || window.mlDetectionData?.detections || [],
                    // Always include both trees and detections arrays for maximum compatibility
                    detections: window.mlDetectionData?.detections || window.mlDetectionData?.trees || []
                  };
                  
                  // Call the preview with the enhanced data
                  window.showDetectionPreview(enhancedData);
                } else if (typeof window.checkAndShowDetectionPreview === 'function') {
                  console.log('DetectionSidebar: Checking for detection data to show preview');
                  window.checkAndShowDetectionPreview();
                }
              }}
              title="Click to show detection results"
            >
              <span className="text-xs text-blue-700 font-medium mr-2">Objects:</span>
              <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded font-semibold" id="detected-objects-count">
                {window.mlDetectionData?.trees?.length || window.mlDetectionData?.detections?.length || filteredTrees.length || 0}
              </span>
            </button>
            
            {/* Close Button */}
            <button
              className="text-gray-500 hover:text-blue-700 focus:outline-none"
              onClick={() => {
                // Dispatch event to close the detection sidebar
                window.dispatchEvent(new CustomEvent('forceCloseObjectDetection', {
                  detail: { source: 'detection_sidebar_close_button' }
                }));
              }}
              title="Close detection panel"
            >
              <X size={18} />
            </button>
          </div>
        </div>
      </div>
      
      {/* Configuration Panel (conditionally rendered) */}
      {configVisible && (
        <div className="bg-slate-50 px-3 py-2 border-b border-slate-200">
          <h3 className="text-sm font-semibold mb-2">Detection Settings</h3>
          
          <div className="space-y-2">
            {/* Detection threshold and other ML settings */}
            
            <div>
              <label className="text-xs text-gray-500 block">Detection Threshold</label>
              <div className="flex items-center space-x-2">
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.1"
                  value={geminiParams.detectionThreshold}
                  onChange={(e) => setGeminiParams(prev => ({
                    ...prev,
                    detectionThreshold: parseFloat(e.target.value)
                  }))}
                  className="w-full"
                />
                <span className="text-xs">{Math.round(geminiParams.detectionThreshold * 100)}%</span>
              </div>
            </div>
            
            <div>
              <label className="text-xs text-gray-500 block">Max Objects</label>
              <input
                type="number"
                min="5"
                max="100"
                value={geminiParams.maxTrees}
                onChange={(e) => setGeminiParams(prev => ({
                  ...prev,
                  maxTrees: parseInt(e.target.value)
                }))}
                className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
              />
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="includeRiskAnalysis"
                checked={geminiParams.includeRiskAnalysis}
                onChange={(e) => setGeminiParams(prev => ({
                  ...prev,
                  includeRiskAnalysis: e.target.checked
                }))}
                className="mr-2"
              />
              <label htmlFor="includeRiskAnalysis" className="text-xs text-gray-500">
                Include risk analysis
              </label>
            </div>
            
            <div>
              <label className="text-xs text-gray-500 block">Detail Level</label>
              <select
                value={geminiParams.detailLevel}
                onChange={(e) => setGeminiParams(prev => ({
                  ...prev,
                  detailLevel: e.target.value
                }))}
                className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
          </div>
        </div>
      )}
      
      
      {/* Main Content Tabs */}
      <div className="flex border-b border-slate-200">
        <button
          className={`flex-1 py-2 text-sm font-medium ${activeTab === 'trees' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
          onClick={() => setActiveTab('trees')}
        >
          Detected
        </button>
        <button
          className={`flex-1 py-2 text-sm font-medium ${activeTab === 'params' ? 'text-blue-600 border-b-2 border-blue-600' : 'text-gray-500'}`}
          onClick={() => setActiveTab('params')}
        >
          Parameters
        </button>
      </div>
      
      {/* Tree List or Parameters Panel */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'trees' ? (
          /* Tree List */
          <div>
            {/* Tree List items */}
            <div className="px-3 py-2 space-y-1.5">
            {filteredTrees.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <AlertTriangle size={24} className="mx-auto mb-2" />
                <p className="text-sm">No objects found</p>
                <p className="text-xs mt-1">Try detecting objects or adjusting search filters</p>
              </div>
            ) : (
              filteredTrees.map((tree, index) => (
                <div
                  key={tree.id || index}
                  className={`border rounded-md p-1.5 cursor-pointer transition-colors ${selectedTree && selectedTree.id === tree.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:bg-gray-50'}`}
                  onClick={() => {
                    setSelectedTree(tree);
                    setCurrentTreeIndex(index);
                    setFormData({
                      species: tree.species || 'Unknown Species',
                      height: tree.height || 30,
                      diameter: tree.diameter || 12,
                      riskLevel: tree.risk_level || 'medium'
                    });
                  }}
                >
                  <div className="flex justify-between items-center">
                    <div className="truncate pr-2">
                      <h3 className="font-medium text-xs truncate">{tree.species || 'Unknown Species'}</h3>
                      <div className="flex text-xs text-gray-500">
                        {tree.height && <span className="mr-1">{tree.height}m</span>}
                        {tree.diameter && <span className="mr-1">{tree.diameter}cm</span>}
                        {tree.validated && <CheckCircle size={10} className="text-green-600" />}
                      </div>
                    </div>
                    <span className={`text-xs px-1.5 py-0.5 rounded ${getRiskLevelColor(tree.risk_level)}`}>
                      {tree.risk_level === 'high' ? 'High' : tree.risk_level === 'medium' ? 'Med' : 'Low'}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
        ) : (
          /* Parameters Panel */
          <div className="px-3 py-2">
            <Card className="p-3">
              <div className="space-y-4">
                {/* ML Overlay Controls (direct implementation) */}
                <div>
                  <h3 className="text-sm font-semibold mb-2">ML Overlay Settings</h3>
                  <div className="ml-overlay-controls">
                    <div className="control-section">
                      {/* Main overlay toggle */}
                      <div className="mb-3">
                        <div className="flex items-center justify-between">
                          <label htmlFor="overlay-toggle" className="text-sm text-gray-700">
                            Show Overlay
                          </label>
                          <div className="relative inline-block w-10 mr-2 align-middle select-none">
                            <input
                              id="overlay-toggle"
                              type="checkbox"
                              className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"
                              checked={window.mlOverlaySettings?.showOverlay || showOverlay}
                              onChange={() => {
                                // Simply call our reusable toggle function with the inverse of current state
                                handleToggleOverlay(!showOverlay);
                              }}
                              style={{
                                right: showOverlay ? '0' : '4px',
                                transition: 'right 0.2s ease',
                                backgroundColor: showOverlay ? '#10b981' : 'white',
                                borderColor: showOverlay ? '#10b981' : '#d1d5db'
                              }}
                            />
                            <label 
                              htmlFor="overlay-toggle" 
                              className="toggle-label block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer"
                              style={{
                                backgroundColor: showOverlay ? '#d1fae5' : '#e5e7eb'
                              }}
                            />
                          </div>
                        </div>
                      </div>
                      
                      {/* Segmentation mask toggle */}
                      <div className="mb-3">
                        <div className="flex items-center justify-between">
                          <label htmlFor="segmentation-toggle" className="text-sm text-gray-700">
                            Show Segmentation Masks
                          </label>
                          <div className="relative inline-block w-10 mr-2 align-middle select-none">
                            <input
                              id="segmentation-toggle"
                              type="checkbox"
                              className="toggle-checkbox absolute block w-6 h-6 rounded-full bg-white border-4 appearance-none cursor-pointer"
                              checked={window.mlOverlaySettings?.showSegmentation || showSegmentation}
                              onChange={() => {
                                // Simply call our reusable segmentation toggle function with the inverse of current state
                                handleSegmentationToggle(!showSegmentation);
                              }}
                              style={{
                                right: showSegmentation ? '0' : '4px',
                                transition: 'right 0.2s ease',
                                backgroundColor: showSegmentation ? '#10b981' : 'white',
                                borderColor: showSegmentation ? '#10b981' : '#d1d5db'
                              }}
                            />
                            <label 
                              htmlFor="segmentation-toggle" 
                              className="toggle-label block overflow-hidden h-6 rounded-full bg-gray-300 cursor-pointer"
                              style={{
                                backgroundColor: showSegmentation ? '#d1fae5' : '#e5e7eb'
                              }}
                            />
                          </div>
                        </div>
                      </div>
                      
                      {/* Opacity slider */}
                      <div className="mb-3">
                        <label htmlFor="opacity-slider" className="block text-sm text-gray-700 mb-1">
                          Overlay Opacity ({Math.round((window.mlOverlaySettings?.opacity || overlayOpacity) * 100)}%)
                        </label>
                        <input
                          id="opacity-slider"
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={window.mlOverlaySettings?.opacity || overlayOpacity}
                          onChange={(e) => {
                            // Get the opacity value and update
                            const newOpacity = parseFloat(e.target.value);
                            handleOpacityChange(newOpacity);
                          }}
                          className="w-full"
                          // Add a fast initial load animation to draw attention
                          style={{
                            transition: "all 0.2s ease",
                            opacity: 1
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* T4 GPU Integration Status Section */}
                <div className="mt-4 mb-2">
                  <h3 className="text-sm font-semibold mb-2">ML Engine Status</h3>
                  <div className="bg-gray-50 p-3 rounded border border-gray-200">
                    <div>
                      <MLStatusIndicator />
                      
                      <div className="mt-2 text-xs text-gray-600">
                        <p>Tree detection can run on the local CPU or GPU for faster processing.</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Detection Parameters section moved below overlay settings */}
                <div className="mt-6">
                  <h3 className="text-sm font-semibold mb-2">Detection Parameters</h3>
                  
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm text-gray-600 block mb-1">Detection Mode</label>
                      <select
                        className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm bg-blue-50 text-blue-700 font-medium"
                        value={geminiParams.detailLevel}
                        onChange={(e) => setGeminiParams(prev => ({
                          ...prev,
                          detailLevel: e.target.value
                        }))}
                      >
                        <option value="high">Grounded SAM</option>
                        <option value="medium" disabled>Gemini (coming soon)</option>
                      </select>
                    </div>
                    
                    {/* Detection Categories */}
                    <div>
                      <label className="text-sm text-gray-600 block mb-1">Categories</label>
                      <DetectionCategories />
                    </div>
                    
                  </div>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
      
      {/* Tree Detail Section (when a tree is selected) */}
      {selectedTree && (
        <div className="border-t border-slate-200 px-3 py-2">
          <div className="flex justify-between items-center mb-2">
            <h3 className="font-semibold">Selected Object</h3>
            <div className="flex space-x-1">
              <Button
                variant="ghost"
                size="sm"
                className="p-1 h-8 w-8"
                title="Previous object"
                onClick={goToPreviousTree}
                disabled={visibleTrees.length <= 1}
              >
                <ChevronLeft size={16} />
              </Button>
              <span className="text-xs text-gray-500 flex items-center">
                {currentTreeIndex + 1}/{visibleTrees.length}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="p-1 h-8 w-8"
                title="Next object"
                onClick={goToNextTree}
                disabled={visibleTrees.length <= 1}
              >
                <ChevronRight size={16} />
              </Button>
            </div>
          </div>
          
          {isEditing ? (
            /* Edit Mode */
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-500 block">Species</label>
                <input
                  type="text"
                  name="species"
                  value={formData.species}
                  onChange={handleInputChange}
                  className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                />
              </div>
              <div className="flex space-x-2">
                <div className="flex-1">
                  <label className="text-xs text-gray-500 block">Height (m)</label>
                  <input
                    type="number"
                    name="height"
                    value={formData.height}
                    onChange={handleInputChange}
                    className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                  />
                </div>
                <div className="flex-1">
                  <label className="text-xs text-gray-500 block">Diameter (cm)</label>
                  <input
                    type="number"
                    name="diameter"
                    value={formData.diameter}
                    onChange={handleInputChange}
                    className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                  />
                </div>
              </div>
              <div>
                <label className="text-xs text-gray-500 block">Risk Level</label>
                <select
                  name="riskLevel"
                  value={formData.riskLevel}
                  onChange={handleInputChange}
                  className="w-full border border-gray-300 rounded px-2 py-1 text-sm"
                >
                  <option value="high">High Risk</option>
                  <option value="medium">Medium Risk</option>
                  <option value="low">Low Risk</option>
                  <option value="new">Unassigned</option>
                </select>
              </div>
              <div className="flex space-x-2 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={() => setIsEditing(false)}
                >
                  Cancel
                </Button>
                <Button
                  variant="default"
                  size="sm"
                  className="flex-1 bg-blue-600 hover:bg-blue-700"
                  onClick={saveTreeEdits}
                >
                  <Save size={14} className="mr-1" />
                  Save
                </Button>
              </div>
            </div>
          ) : (
            /* View Mode */
            <div>
              <Card className="p-3">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Species:</span>
                    <span className="text-sm">{selectedTree.species || 'Unknown'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Height:</span>
                    <span className="text-sm">{selectedTree.height ? `${selectedTree.height}m` : 'Unknown'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Diameter:</span>
                    <span className="text-sm">{selectedTree.diameter ? `${selectedTree.diameter}cm` : 'Unknown'}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Risk Level:</span>
                    <span className={`text-sm font-medium ${getRiskLevelColor(selectedTree.risk_level)}`}>
                      {getRiskLevelText(selectedTree.risk_level)}
                    </span>
                  </div>
                  {selectedTree.confidence && (
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Confidence:</span>
                      <span className="text-sm">{Math.round(selectedTree.confidence * 100)}%</span>
                    </div>
                  )}
                  {selectedTree.address && (
                    <div className="flex justify-between">
                      <span className="text-sm font-medium">Location:</span>
                      <span className="text-sm truncate max-w-[180px]" title={selectedTree.address}>
                        {selectedTree.address}
                      </span>
                    </div>
                  )}
                </div>
              </Card>
              
              <div className="flex space-x-2 mt-3">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={removeTree}
                >
                  <Trash size={14} className="mr-1" />
                  Remove
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={() => setIsEditing(true)}
                >
                  <Edit size={14} className="mr-1" />
                  Edit
                </Button>
                <Button
                  variant="default"
                  size="sm"
                  className="flex-1 bg-blue-600 hover:bg-blue-700"
                  onClick={validateTree}
                >
                  <Check size={14} className="mr-1" />
                  Approve
                </Button>
              </div>
              
              {selectedTree.validated && (
                <div className="mt-3 text-center">
                  <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full inline-flex items-center">
                    <CheckCircle size={12} className="mr-1" />
                    Object approved and saved
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      )}
      
      {/* Manual Placement Controls - Enhanced with Category Support */}
      <div className="border-t border-slate-200 px-3 py-2">
        <div className="flex space-x-2">
          <div className="flex-1">
            <Button
              variant={manualPlacement ? "default" : "outline"}
              size="sm"
              className={`w-full ${manualPlacement ? 'bg-blue-600 hover:bg-blue-700 ring-1 ring-blue-300 ring-opacity-50' : ''}`}
              onClick={() => {
                const newState = !manualPlacement;
                setManualPlacement(newState);
                
                // Dispatch event to notify map components about placement mode
                window.dispatchEvent(new CustomEvent(newState ? 'enableManualTreePlacement' : 'disableManualTreePlacement', {
                  detail: { 
                    source: 'detection_sidebar', 
                    objectType: window.manualObjectType || 'tree'
                  }
                }));
                
                // Show confirmation toast for better UX
                if (newState) {
                  const toast = document.createElement('div');
                  toast.className = 'fixed top-4 right-4 bg-blue-50 text-blue-800 px-4 py-2 rounded shadow-md z-50 flex items-center';
                  toast.innerHTML = `
                    <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    Click on map to add markers
                  `;
                  document.body.appendChild(toast);
                  
                  // Remove toast after 3 seconds
                  setTimeout(() => {
                    if (document.body.contains(toast)) {
                      document.body.removeChild(toast);
                    }
                  }, 3000);
                }
              }}
              title={manualPlacement ? "Placement mode active - click on map to add markers" : "Enable manual marker placement"}
            >
              <Plus size={14} className="mr-1" />
              {manualPlacement ? 'Adding...' : 'Add Markers'}
            </Button>
            
            {manualPlacement && (
              <div className="space-y-2 mt-2">
                {/* Category selection from DetectionCategories */}
                <select
                  className="w-full border border-gray-300 rounded text-xs py-1 bg-blue-50 text-blue-700"
                  onChange={(e) => {
                    const selectedValue = e.target.value;
                    // Set object type for manual placement
                    window.manualObjectType = selectedValue;
                    
                    // Dispatch event to update placement type
                    window.dispatchEvent(new CustomEvent('updateManualPlacementType', {
                      detail: { 
                        objectType: selectedValue,
                        source: 'detection_sidebar'
                      }
                    }));
                    
                    console.log(`DetectionSidebar: Manual placement type set to ${selectedValue}`);
                  }}
                  defaultValue="healthy_tree"
                >
                  {/* Tree risk categories matching DetectionCategories.jsx */}
                  <option value="healthy_tree">Healthy Tree</option>
                  <option value="hazardous_tree">Hazardous Tree</option>
                  <option value="dead_tree">Dead Tree</option>
                  <option value="low_canopy_tree">Low Canopy Tree</option>
                  <option value="pest_disease_tree">Pest/Disease Tree</option>
                  <option value="flood_prone_tree">Flood-Prone Tree</option>
                  <option value="utility_conflict_tree">Utility Conflict Tree</option>
                  <option value="structural_hazard_tree">Structural Hazard Tree</option>
                  <option value="fire_risk_tree">Fire Risk Tree</option>
                  {/* Other object types */}
                  <option value="building">Building</option>
                  <option value="power_line">Power Line</option>
                </select>
                
                {/* Help text */}
                <p className="text-xs text-gray-500 italic mt-1">
                  Click on the map to place markers at specific locations. Select object category above.
                </p>
              </div>
            )}
          </div>
          
          <Button
            variant="outline"
            size="sm"
            className="flex-none px-4 bg-blue-700 hover:bg-blue-800 border-blue-600 font-medium text-black"
            id="detect-trees-btn"
            onClick={e => {
              console.log("DetectionSidebar: Detect button clicked - starting ML pipeline");
              
              // Show a loading state for the button
              const button = e.currentTarget;
              const originalText = button.innerHTML;
              button.innerHTML = '<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-gray-300" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg><span class="text-gray-300">Detecting...</span>';
              button.disabled = true;
              button.style.cursor = 'not-allowed';
              
              // Create progress indicator
              const progressContainer = document.createElement('div');
              progressContainer.id = 'detection-progress-container';
              progressContainer.style.width = '100%';
              progressContainer.style.marginTop = '8px';
              progressContainer.style.marginBottom = '12px';
              progressContainer.style.padding = '0 12px';
              
              const progressBarOuter = document.createElement('div');
              progressBarOuter.style.width = '100%';
              progressBarOuter.style.height = '6px';
              progressBarOuter.style.backgroundColor = '#e2e8f0';
              progressBarOuter.style.borderRadius = '3px';
              progressBarOuter.style.overflow = 'hidden';
              
              const progressBar = document.createElement('div');
              progressBar.id = 'detection-progress-bar';
              progressBar.style.width = '5%';
              progressBar.style.height = '100%';
              progressBar.style.backgroundColor = '#3b82f6';
              progressBar.style.borderRadius = '3px';
              progressBar.style.transition = 'width 0.3s ease';
              
              const progressText = document.createElement('div');
              progressText.id = 'detection-progress-text';
              progressText.style.fontSize = '11px';
              progressText.style.color = '#64748b';
              progressText.style.marginTop = '4px';
              progressText.textContent = 'Loading model...';
              
              progressBarOuter.appendChild(progressBar);
              progressContainer.appendChild(progressBarOuter);
              progressContainer.appendChild(progressText);
              
              // Add progress bar near the button
              const buttonContainer = button.parentNode;
              if (buttonContainer && buttonContainer.parentNode) {
                buttonContainer.parentNode.insertBefore(progressContainer, buttonContainer.nextSibling);
              }
              
              // Use existing job ID if available, otherwise generate one
              const jobId = window.currentDetectionJobId || `detection_${Date.now()}`;
              // Store it globally so other components can access it
              window.currentDetectionJobId = jobId;
              window._lastDetectionJobId = jobId; // Save a backup copy
              console.log(`DetectionSidebar: Using job ID: ${jobId}`);
              
              // Also dispatch an event with the job ID
              window.dispatchEvent(new CustomEvent('detectionJobIdUpdated', {
                detail: { job_id: jobId }
              }));
              
              // Helper to update progress
              const updateProgress = (percent, message) => {
                const bar = document.getElementById('detection-progress-bar');
                const text = document.getElementById('detection-progress-text');
                
                if (bar) bar.style.width = `${percent}%`;
                if (text) text.textContent = message;
              };
              
              // Helper to cleanup
              const cleanupDetection = () => {
                // Restore button
                button.innerHTML = originalText;
                button.disabled = false;
                button.style.backgroundColor = '';
                button.style.cursor = 'pointer';
                
                // Remove progress bar after delay
                setTimeout(() => {
                  const progressContainer = document.getElementById('detection-progress-container');
                  if (progressContainer && progressContainer.parentNode) {
                    progressContainer.parentNode.removeChild(progressContainer);
                  }
                }, 1500);
              };
              
              // Listen for detection events
              const handleDetectionComplete = () => {
                console.log("ML detection complete");
                updateProgress(100, "Detection complete!");
                cleanupDetection();
                
                // After detection completes, explicitly show the detection preview
                setTimeout(() => {
                  console.log("DetectionSidebar: Showing detection preview with ML detection data");
                  
                  // Get the detection data from the global variable
                  const detectionData = window.mlDetectionData;
                  
                  if (detectionData) {
                    console.log("DetectionSidebar: Found detection data with:", {
                      jobId: detectionData.job_id,
                      treesCount: detectionData.trees?.length || 0,
                      hasDetections: !!detectionData.detections,
                      detectionsCount: detectionData.detections?.length || 0
                    });
                    
                    // CRITICAL FIX: Update the Objects Found counter when detection completes
                    // IMPORTANT: Preserve the existing count if we had previous detections
                    const counterEl = document.getElementById('detected-objects-count');
                    if (counterEl) {
                      // Get new detection count
                      const newCount = detectionData.trees?.length || detectionData.detections?.length || 0;
                      
                      // Get previous detection count
                      const previousCount = window._previousMLDetectionData?.trees?.length || 
                                           window._previousMLDetectionData?.detections?.length || 0;
                      
                      // Only replace the count if it makes sense - avoid random resets to lower values
                      if (newCount > 0) {
                        // If we're continuing a session, we should keep the max count
                        // This ensures we don't randomly drop to a lower number when doing multiple detections
                        const finalCount = Math.max(newCount, previousCount);
                        counterEl.textContent = finalCount.toString();
                        console.log(`DetectionSidebar: Updated Objects Found counter to ${finalCount} (new: ${newCount}, previous: ${previousCount})`);
                      } else if (previousCount > 0) {
                        // If new count is 0 but we had previous detections, keep previous count
                        counterEl.textContent = previousCount.toString();
                        console.log(`DetectionSidebar: Keeping previous count ${previousCount} (new count was 0)`);
                      }
                    }
                    
                    // Always make the overlay settings visible
                    window.mlOverlaySettings = {
                      ...(window.mlOverlaySettings || {}),
                      showOverlay: true,
                      opacity: 0.3,
                      showSegmentation: true
                    };
                    window.detectionShowOverlay = true;
                    
                    // Also make sure ML overlay is visible with the data
                    if (typeof window.renderMLOverlay === 'function') {
                      const mapInstance = window.map || window.googleMapsInstance || window._googleMap;
                      if (mapInstance) {
                        console.log("DetectionSidebar: Ensuring ML overlay is visible with detection data");
                        
                        // Force ML overlay to be visible with latest data
                        window.renderMLOverlay(
                          mapInstance, 
                          detectionData,
                          {
                            opacity: window.mlOverlaySettings?.opacity || 0.3,
                            showSegmentation: window.mlOverlaySettings?.showSegmentation !== false,
                            forceRenderBoxes: true
                          }
                        );
                        
                        // Ensure the ML overlay is visible (in case it was hidden)
                        const overlay = document.getElementById('ml-detection-overlay');
                        if (overlay) {
                          overlay.style.display = 'block';
                          overlay.style.opacity = '1';
                        }
                        
                        // Also dispatch an event to ensure the overlay is visible
                        window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
                          detail: { 
                            showOverlay: true, 
                            showSegmentation: true, 
                            opacity: window.mlOverlaySettings?.opacity || 0.3,
                            source: 'detection_completion'
                          }
                        }));
                      }
                    }
                    
                    // Show the detection preview
                    if (typeof window.showDetectionPreview === 'function') {
                      console.log("DetectionSidebar: Showing detection preview");
                      window.showDetectionPreview(detectionData);
                    } else {
                      console.warn("DetectionSidebar: showDetectionPreview function not available");
                    }
                  } else {
                    console.warn("DetectionSidebar: No detection data available after detection complete");
                  }
                }, 1000); // Wait a second to ensure data is loaded
                
                // Remove this event listener
                window.removeEventListener('treeDetectionComplete', handleDetectionComplete);
                window.removeEventListener('treeDetectionError', handleDetectionError);
              };
              
              const handleDetectionError = () => {
                console.log("ML detection error");
                updateProgress(0, "Detection failed. Please try again.");
                cleanupDetection();
                
                // Remove this event listener
                window.removeEventListener('treeDetectionComplete', handleDetectionComplete);
                window.removeEventListener('treeDetectionError', handleDetectionError);
              };
              
              window.addEventListener('treeDetectionComplete', handleDetectionComplete);
              window.addEventListener('treeDetectionError', handleDetectionError);
              
              // ENHANCED: Make overlay immediately visible before dispatching detection event
              console.log("DetectionSidebar: Making overlay immediately visible before detection starts");
              
              // Preserve the current detection data and its counter before starting a new detection
              // This ensures we don't lose or reset existing counts during the detection process
              if (window.mlDetectionData) {
                // Save a backup of current detection data
                window._previousMLDetectionData = JSON.parse(JSON.stringify(window.mlDetectionData));
                
                // Update the objects counter with current data to ensure consistency
                const counterEl = document.getElementById('detected-objects-count');
                if (counterEl) {
                  const count = window.mlDetectionData.trees?.length || window.mlDetectionData.detections?.length || 0;
                  console.log(`DetectionSidebar: Preserving current object count: ${count}`);
                  // We'll use the stored data rather than updating DOM directly to maintain React state integrity
                }
                
                // Force overlay to be visible with existing data
                const mapInstance = window.map || window.googleMapsInstance || window._googleMap;
                if (mapInstance && typeof window.renderMLOverlay === 'function') {
                  // Force immediate overlay display with existing data
                  window.renderMLOverlay(
                    mapInstance,
                    window.mlDetectionData,
                    {
                      opacity: window.mlOverlaySettings?.opacity || 0.7,
                      showSegmentation: window.mlOverlaySettings?.showSegmentation !== false,
                      forceRenderBoxes: true,
                      preserveExistingDetections: true // Flag to preserve existing detection data
                    }
                  );
                }
              }
              
              // 2. Manually ensure the overlay is visible if it exists
              if (window._mlDetectionOverlay && window._mlDetectionOverlay.div) {
                window._mlDetectionOverlay.div.style.display = 'block';
              }
              
              // 3. Update any overlay element directly as a fallback
              const overlay = document.getElementById('ml-detection-overlay');
              if (overlay) {
                overlay.style.display = 'block';
              }
              
              // 4. Update global settings to show overlay
              window.mlOverlaySettings = {
                ...(window.mlOverlaySettings || {}),
                showOverlay: true,
                opacity: window.mlOverlaySettings?.opacity || 0.3
              };
              window.detectionShowOverlay = true;
              
              // 5. Now dispatch the detection event
              window.dispatchEvent(new CustomEvent('openTreeDetection', {
                detail: {
                  useSatelliteImagery: true,
                  useRealGemini: geminiParams.includeRiskAnalysis,
                  saveToResponseJson: true,
                  geminiParams: geminiParams,
                  job_id: jobId,
                  source: "detect_button",  // CRITICAL: Required for MapControls to identify this as a detection request
                  forceOverlayVisible: true // Flag to indicate the overlay should be immediately visible
                }
              }));
              
              // Simulate progress updates
              updateProgress(10, "Loading model...");
              setTimeout(() => updateProgress(20, "Processing satellite imagery..."), 500);
              setTimeout(() => updateProgress(30, "Running object detection..."), 1000);
              
              // Set up a safety timeout
              setTimeout(() => {
                // If detection is still running after timeout, force completion
                updateProgress(100, "Detection complete!");
                cleanupDetection();
                window.removeEventListener('treeDetectionComplete', handleDetectionComplete);
                window.removeEventListener('treeDetectionError', handleDetectionError);
              }, 3000); // 3 second safety timeout - reduced from 20 seconds for better UX
            }}
          >
            <BarChart size={14} className="mr-1" />
            Detect
          </Button>
        </div>
        
        {/* Clear All Markers Button */}
        <div className="mt-2">
          <Button
            variant="outline"
            size="sm"
            className="w-full text-red-600 border-red-200 hover:bg-red-50 hover:border-red-300"
            onClick={() => {
              // Confirm before clearing all markers
              if (window.confirm("Are you sure you want to clear all markers?")) {
                // Dispatch event to clear all markers
                window.dispatchEvent(new CustomEvent('clearMLMarkers', {
                  detail: { source: 'clear_button' }
                }));
                
                // Also trigger a reset filters event
                window.dispatchEvent(new CustomEvent('resetFilters'));
                
                // Show confirmation toast
                const toast = document.createElement('div');
                toast.className = 'fixed top-4 right-4 bg-red-50 text-red-800 px-4 py-2 rounded shadow-md z-50 flex items-center';
                toast.innerHTML = `
                  <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                  </svg>
                  All markers cleared
                `;
                document.body.appendChild(toast);
                
                // Remove toast after 2 seconds
                setTimeout(() => {
                  if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                  }
                }, 2000);
              }
            }}
          >
            <Trash size={14} className="mr-1" />
            Clear All Markers
          </Button>
        </div>
      </div>
      
      {/* Resize handle for the sidebar */}
      <div 
        className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize bg-slate-200 hover:bg-blue-400 active:bg-blue-500 transition-colors"
        onMouseDown={(e) => {
          e.preventDefault();
          
          const startX = e.clientX;
          const startWidth = width;
          
          // Function to update badge position more consistently
          const updateBadgePosition = (newWidth) => {
            // Try multiple badge selectors to ensure we find it
            const badge = document.getElementById('detection-debug') || 
                         document.querySelector('.detection-badge') ||
                         document.querySelector('[id*="detection"][id*="badge"]');
                
            if (badge) {
              badge.style.right = `${newWidth}px`;
              badge.style.transition = 'none'; // Disable transitions during resize for smoothness
              console.log(`DetectionSidebar: Updated badge position to right: ${newWidth}px`);
            } else {
              console.log('DetectionSidebar: Badge not found during resize');
            }
          };
          
          const onMouseMove = (moveEvent) => {
            const newWidth = Math.max(300, Math.min(600, startWidth - (moveEvent.clientX - startX)));
            setWidth(newWidth);
            
            // Update map container size immediately for smoother resizing
            const mapContainer = document.querySelector('#map-container');
            if (mapContainer) {
              mapContainer.style.right = `${newWidth}px`;
            }
            
            // Update badge position
            updateBadgePosition(newWidth);
            
            // Force immediate Google Maps resize for smoother resizing
            const mapInstance = window.googleMapsInstance || window._googleMap;
            if (mapInstance && window.google && window.google.maps) {
              window.google.maps.event.trigger(mapInstance, 'resize');
            }
            
            // Dispatch resize event for real-time updates
            const eventDetail = { 
              width: newWidth,
              collapsed: false,
              source: 'detection_sidebar_resize',
              headerCollapsed: headerCollapsed
            };
            
            // Dispatch event for real-time resizing
            window.dispatchEvent(new CustomEvent('detectionSidebarResizing', {
              detail: eventDetail
            }));
          };
          
          const onMouseUp = () => {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            
            // Final badge position update with transition re-enabled
            const badge = document.getElementById('detection-debug');
            if (badge) {
              badge.style.right = `${width}px`;
              badge.style.transition = 'right 0.2s ease'; // Re-enable transitions after resize
            }
            
            // Force map resize
            const mapInstance = window.googleMapsInstance || window._googleMap;
            if (mapInstance && window.google && window.google.maps) {
              window.google.maps.event.trigger(mapInstance, 'resize');
            }
            
            // Force resize to ensure map redraws correctly
            window.dispatchEvent(new Event('resize'));
            
            // Final notification about completed resize
            window.dispatchEvent(new CustomEvent('detectionSidebarToggle', {
              detail: {
                width: width,
                collapsed: false,
                source: 'detection_sidebar_resize_complete',
                headerCollapsed: headerCollapsed
              }
            }));
          };
          
          document.addEventListener('mousemove', onMouseMove);
          document.addEventListener('mouseup', onMouseUp);
        }}
      />
      
      {/* Collapsed sidebar indicator - shown only when sidebar is collapsed */}
      {collapsed && (
        <div 
          className="fixed right-0 top-1/2 transform -translate-y-1/2 bg-white shadow-md rounded-l-md cursor-pointer z-30 transition-all hover:bg-blue-50"
          onClick={() => {
            console.log('DetectionSidebar: Expanding sidebar from collapsed indicator');
            
            // When clicking to expand the sidebar, make sure the map resizes properly
            // Use requestAnimationFrame for smoother visual transitions
            requestAnimationFrame(() => {
              // 1. First update the map container immediately
              const mapContainer = document.querySelector('#map-container');
              if (mapContainer) {
                mapContainer.style.right = `${width}px`;
                console.log(`DetectionSidebar: Set map container right to ${width}px`);
              }
              
              // 2. Get map wrapper and adjust based on header state
              const mapWrapper = document.getElementById('map-wrapper');
              const header = document.querySelector('header');
              if (mapWrapper && header) {
                const headerHeight = header.offsetHeight;
                if (!headerCollapsed) {
                  mapWrapper.style.height = `calc(100vh - ${headerHeight}px)`;
                  mapWrapper.style.top = `${headerHeight}px`;
                  console.log(`DetectionSidebar: Adjusted map wrapper for expanded header: height calc(100vh - ${headerHeight}px), top ${headerHeight}px`);
                }
              }
              
              // 3. Then toggle the collapsed state which will dispatch events
              toggleCollapse();
              
              // 4. Force Google Maps to recalculate its size with a slight delay
              setTimeout(() => {
                const mapInstance = window.googleMapsInstance || window._googleMap;
                if (mapInstance && window.google && window.google.maps) {
                  window.google.maps.event.trigger(mapInstance, 'resize');
                  console.log('DetectionSidebar: Triggered Google Maps resize after expanding');
                }
                
                // Also dispatch a resize event
                window.dispatchEvent(new Event('resize'));
              }, 150);
            });
          }}
          style={{ top: `calc(50% + ${headerCollapsed ? '0px' : '30px'})` }}
        >
          <div className="p-2">
            <ChevronLeft size={16} className="rotate-180" />
          </div>
        </div>
      )}
    </div>
  );
};

export default DetectionSidebar;