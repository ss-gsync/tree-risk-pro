// src/components/visualization/Detection/DetectionSidebarBridge.jsx
//
// This component acts as a bridge between the DOM-based sidebar created
// in Layout/Sidebar/index.jsx and our React-based DetectionSidebar.jsx.
// It coordinates state and events between the two systems.

import React, { useEffect, useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import DetectionSidebar from './DetectionSidebar';
import MLOverlayInitializer from './MLOverlayInitializer';

/**
 * DetectionSidebarBridge - Connects DOM-based sidebar with React components
 * 
 * This component:
 * 1. Detects when the DOM-based sidebar is created
 * 2. Creates a React portal to render our DetectionSidebar inside it
 * 3. Syncs state between DOM elements and React components
 * 4. Ensures MLOverlay is properly initialized
 */
const DetectionSidebarBridge = () => {
  const [sidebarElement, setSidebarElement] = useState(null);
  const [headerCollapsed, setHeaderCollapsed] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.3);
  const [showSegmentation, setShowSegmentation] = useState(true);
  const [trees, setTrees] = useState([]);
  const [selectedTree, setSelectedTree] = useState(null);
  const [currentTreeIndex, setCurrentTreeIndex] = useState(0);
  const [manualPlacement, setManualPlacement] = useState(false);
  const [width, setWidth] = useState(384);
  
  // Track initialization state
  const initialized = useRef(false);
  
  // Function to handle the DOM sidebar initialization
  const initializeDOMSidebar = () => {
    // Find the sidebar or its React container in the DOM
    const domSidebar = document.querySelector('.detection-sidebar');
    const reactContainer = document.querySelector('#detection-sidebar-react-container');
    
    // Use either the sidebar or the container, with preference for the container
    const targetElement = reactContainer || domSidebar;
    
    if (!targetElement || initialized.current) return false;
    
    console.log("DetectionSidebarBridge: Found DOM sidebar, initializing integration with", 
      reactContainer ? "React container" : "sidebar element");
    
    // Store reference to the sidebar
    setSidebarElement(targetElement);
    initialized.current = true;
    
    // Initialize the state from DOM elements
    
    // Check overlay visibility
    const overlay = document.getElementById('ml-detection-overlay');
    if (overlay) {
      setShowOverlay(overlay.style.display !== 'none');
    }
    
    // Get opacity value
    try {
      const savedOpacity = localStorage.getItem('ml-overlay-opacity');
      if (savedOpacity !== null) {
        setOverlayOpacity(parseFloat(savedOpacity));
      }
    } catch (e) {
      console.error("Error reading opacity setting:", e);
    }
    
    // Get segmentation state from global settings
    if (window.mlOverlaySettings) {
      setShowSegmentation(window.mlOverlaySettings.showSegmentation !== false);
    }
    
    // Get header state
    const headerElement = document.querySelector('header');
    const isHeaderCollapsed = headerElement ? 
      headerElement.classList.contains('collapsed') || 
      headerElement.offsetHeight < 50 : true;
    setHeaderCollapsed(isHeaderCollapsed);
    
    return true;
  };
  
  // Proactively make the overlay visible when this component mounts
  useEffect(() => {
    // Force overlay to be visible on first render
    console.log('DetectionSidebarBridge: Forcing overlay to be visible on first render');
    window.mlOverlaySettings = {
      ...(window.mlOverlaySettings || {}),
      showOverlay: true
    };
    window.detectionShowOverlay = true;
    
    // Check if we have an existing overlay, make it visible
    if (window._mlDetectionOverlay && window._mlDetectionOverlay.div) {
      window._mlDetectionOverlay.div.style.display = 'block';
    }
    
    // Dispatch event to notify other components
    window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
      detail: {
        showOverlay: true,
        opacity: window.mlOverlaySettings?.opacity || 0.3,
        source: 'detection_sidebar_bridge_init'
      }
    }));
  }, []);
    
  // Initialize when the component mounts
  useEffect(() => {
    // Remove any existing sidebars first to prevent duplicates
    const cleanupExistingSidebars = () => {
      console.log('DetectionSidebarBridge: Running comprehensive sidebar cleanup');
      
      // Find and remove ALL detection sidebars - even React ones from previous sessions
      const allSidebars = document.querySelectorAll('.detection-sidebar');
      allSidebars.forEach(sidebar => {
        if (sidebar && sidebar.parentNode) {
          console.log('DetectionSidebarBridge: Removing existing sidebar', sidebar.id || sidebar.className);
          sidebar.style.display = 'none';  // First hide it
          sidebar.style.visibility = 'hidden';
          sidebar.parentNode.removeChild(sidebar);
        }
      });
      
      // Find and remove old Object Recognition sidebars by className pattern
      const objectRecogSidebars = document.querySelectorAll('.object-recognition-sidebar, [class*="object-recognition"]');
      objectRecogSidebars.forEach(sidebar => {
        if (sidebar && sidebar.parentNode) {
          console.log('DetectionSidebarBridge: Removing object recognition sidebar', sidebar.id || sidebar.className);
          sidebar.style.display = 'none';
          sidebar.style.visibility = 'hidden';
          sidebar.parentNode.removeChild(sidebar);
        }
      });
      
      // Clean up any old elements that might be left over
      const cleanupElements = (selector, name) => {
        try {
          const elements = document.querySelectorAll(selector);
          elements.forEach(el => {
            if (el && document.body.contains(el) && el.parentNode) {
              console.log(`DetectionSidebarBridge: Removing old ${name}`);
              el.style.display = 'none';
              el.style.visibility = 'hidden';
              el.parentNode.removeChild(el);
            }
          });
        } catch (e) {
          console.error(`Error cleaning up ${name} elements:`, e);
        }
      };
      
      // Clean up various elements
      cleanupElements('#object-recognition-badge', 'badge');
      cleanupElements('.detection-badge', 'badge');
      cleanupElements('#detection-debug', 'detection badge');
      cleanupElements('#ml-detection-overlay', 'ML overlay');
      
      // Delay slightly to ensure DOM operations complete
      setTimeout(() => {
        // Verify cleanup was successful - double check for any lingering elements
        const remainingSidebars = document.querySelectorAll('.detection-sidebar, .object-recognition-sidebar');
        if (remainingSidebars.length > 0) {
          console.log('DetectionSidebarBridge: Found remaining sidebars after cleanup, removing them');
          remainingSidebars.forEach(sidebar => {
            if (sidebar && sidebar.parentNode) {
              sidebar.parentNode.removeChild(sidebar);
            }
          });
        }
      }, 50);
      
      return true;
    };
    
    // Run cleanup first
    cleanupExistingSidebars();
    
    // Then check for existing sidebar
    initializeDOMSidebar();
    
    // Set up a MutationObserver to watch for sidebar creation
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.addedNodes.length) {
          // Check if any of the added nodes or their children are the sidebar
          for (let i = 0; i < mutation.addedNodes.length; i++) {
            const node = mutation.addedNodes[i];
            if (node.nodeType === 1) { // Element node
              if (node.classList?.contains('detection-sidebar')) {
                initializeDOMSidebar();
                return;
              }
              
              // Check children
              const sidebar = node.querySelector?.('.detection-sidebar');
              if (sidebar) {
                initializeDOMSidebar();
                return;
              }
            }
          }
        }
      }
    });
    
    // Start observing document body for sidebar creation
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Listen for detection mode activation event from the sidebar
    const handleMLDetectionActive = (event) => {
      console.log('DetectionSidebarBridge: ML detection active event received', event.detail);
      
      // Immediately mark detection as active for other components
      window.detectionShowOverlay = true;
      window.mlOverlaySettings = {
        ...(window.mlOverlaySettings || {}),
        showOverlay: true,
        pendingButtonTrigger: false
      };
      
      // Ensure MLOverlay is initialized - this is critical for first click functionality
      if (typeof window.ensureMLOverlayInitialized === 'function') {
        console.log('DetectionSidebarBridge: Using global ensureMLOverlayInitialized function');
        const initialized = window.ensureMLOverlayInitialized();
        console.log('DetectionSidebarBridge: Global initialization result:', initialized);
      } else if (typeof MLOverlayModule.ensureInitialized === 'function') {
        console.log('DetectionSidebarBridge: Using module ensureInitialized function');
        const initialized = MLOverlayModule.ensureInitialized();
        console.log('DetectionSidebarBridge: Module initialization result:', initialized);
      }
      
      // CRITICAL FIX: Always make the overlay visible when the Detection sidebar is opened
      console.log('DetectionSidebarBridge: Initializing detection sidebar AND setting overlay to visible');
      
      // Set the overlay to visible by default - this is the key fix
      window.mlOverlaySettings = {
        ...(window.mlOverlaySettings || {}),
        showOverlay: true,  // Make overlay visible by default
        pendingButtonTrigger: false
      };
      window.detectionShowOverlay = true;
      
      // Dispatch event to trigger overlay creation
      window.dispatchEvent(new CustomEvent('openTreeDetection', {
        detail: { 
          buttonTriggered: true,  // Pretend this is a button trigger to force overlay creation
          sidebarInitialization: true,
          initialVisibility: true,  // Flag to indicate this should make overlay visible
          prepareDetection: true
        }
      }));
      
      // Import MLOverlay module directly to ensure it's loaded and initialized
      import('./MLOverlay').then(MLOverlayModule => {
        console.log('DetectionSidebarBridge: MLOverlay module imported directly');
        
        // Try to initialize the module
        if (typeof MLOverlayModule.ensureInitialized === 'function') {
          const initialized = MLOverlayModule.ensureInitialized();
          console.log('DetectionSidebarBridge: MLOverlay initialization result:', initialized);
        }
        
        // Make sure global methods are set
        window.renderMLOverlay = MLOverlayModule.renderMLOverlay;
        window.removeMLOverlay = MLOverlayModule.removeMLOverlay;
        window.updateMLOverlayOpacity = MLOverlayModule.updateMLOverlayOpacity;
        window.updateMLOverlayClasses = MLOverlayModule.updateMLOverlayClasses;
        window.ensureInitialized = MLOverlayModule.ensureInitialized;
      }).catch(err => {
        console.error('DetectionSidebarBridge: Error importing MLOverlay module:', err);
      });
      
      // Check global flag before doing cleanup
      if (!window.isDetectionSidebarActive) {
        // Only do cleanup if sidebar isn't currently being created
        console.log('DetectionSidebarBridge: Running cleanup because sidebar is not marked as active');
        cleanupExistingSidebars();
      } else {
        console.log('DetectionSidebarBridge: Skipping cleanup because sidebar is marked as active');
      }
      
      // Wait a moment to ensure cleanup completes before proceeding
      setTimeout(() => {
        // If sidebar element is passed directly in the event, use that
        if (event.detail && event.detail.sidebarElement) {
          console.log('DetectionSidebarBridge: Using sidebar element from event');
          
          // Make sure the sidebar is properly marked as a React container
          event.detail.sidebarElement.setAttribute('data-react-container', 'true');
          
          // Set the sidebar element in state
          setSidebarElement(event.detail.sidebarElement);
          initialized.current = true;
          
          // Set header collapsed state if provided
          if (typeof event.detail.headerCollapsed !== 'undefined') {
            setHeaderCollapsed(event.detail.headerCollapsed);
          }
          
          // Initialize state from localStorage
          try {
            const savedOpacity = localStorage.getItem('ml-overlay-opacity');
            if (savedOpacity !== null) {
              setOverlayOpacity(parseFloat(savedOpacity));
            }
          } catch (e) {
            console.error("Error reading opacity setting:", e);
          }
          
          // Initialize segmentation state
          if (window.mlOverlaySettings) {
            setShowSegmentation(window.mlOverlaySettings.showSegmentation !== false);
          }
          
          // Make sure content container exists
          let contentContainer = event.detail.sidebarElement.querySelector('#detection-content-container');
          if (!contentContainer) {
            contentContainer = document.createElement('div');
            contentContainer.id = 'detection-content-container';
            contentContainer.style.width = '100%';
            contentContainer.style.height = '100%'; // Full height since React header is inside the content container
            contentContainer.style.overflow = 'auto';
            contentContainer.style.position = 'relative';
            event.detail.sidebarElement.appendChild(contentContainer);
          }
          
          // Update React portal target to this container
          console.log('DetectionSidebarBridge: Portal will render into content container');
          setSidebarElement(event.detail.sidebarElement);
        } else {
          // Find the newly created sidebar in the DOM
          setTimeout(() => {
            const result = initializeDOMSidebar();
            
            if (result) {
              console.log('DetectionSidebarBridge: Successfully initialized after mlDetectionActive event');
              
              // Set header collapsed state if provided
              if (event.detail && typeof event.detail.headerCollapsed !== 'undefined') {
                setHeaderCollapsed(event.detail.headerCollapsed);
              }
            } else {
              console.warn('DetectionSidebarBridge: Failed to initialize after mlDetectionActive event, will retry');
              
              // Try again with a longer delay
              setTimeout(initializeDOMSidebar, 300);
            }
          }, 100);
        }
        
        // Update the detection badge to ensure it's correctly positioned
        setTimeout(() => {
          const badge = document.getElementById('detection-debug');
          if (badge) {
            badge.style.right = `${width}px`; // Ensure badge is positioned relative to sidebar
            
            // Make sure the badge is visible if overlay is visible
            if (showOverlay) {
              badge.style.display = 'block';
              badge.style.opacity = '1';
            }
          }
        }, 200);
      }, 50); // Short delay to ensure cleanup completes
    };
    
    // Listen for header state changes
    const handleHeaderCollapse = (event) => {
      if (event.detail && typeof event.detail.collapsed !== 'undefined') {
        setHeaderCollapsed(event.detail.collapsed);
      }
    };
    
    // Listen for ML overlay settings changes
    const handleOverlaySettingsChanged = (event) => {
      const { showOverlay: newShowOverlay, showSegmentation: newShowSegmentation, opacity: newOpacity, source } = event.detail || {};
      
      // Skip if this event was triggered by this component itself or its children
      if (source === 'detection_sidebar_bridge' || source === 'detection_sidebar') {
        console.log('DetectionSidebarBridge: Skipping self-triggered event to avoid loops');
        return;
      }
      
      // Only update state if values are different to prevent unnecessary renders
      if (typeof newShowOverlay !== 'undefined' && newShowOverlay !== showOverlay) {
        setShowOverlay(newShowOverlay);
      }
      
      if (typeof newShowSegmentation !== 'undefined' && newShowSegmentation !== showSegmentation) {
        setShowSegmentation(newShowSegmentation);
      }
      
      if (typeof newOpacity !== 'undefined' && Math.abs(newOpacity - overlayOpacity) > 0.001) {
        setOverlayOpacity(newOpacity);
      }
    };
    
    // Listen for changes to detected trees
    const handleDetectionUpdate = (event) => {
      if (event.detail && event.detail.trees) {
        const newTrees = event.detail.trees.map((tree) => ({
          ...tree,
          visible: true,
        }));
        
        setTrees(newTrees);
        
        if (newTrees.length > 0 && !selectedTree) {
          setSelectedTree(newTrees[0]);
          setCurrentTreeIndex(0);
        }
      }
    };
    
    // Handle badge creation
    const handleCreateDetectionBadge = (event) => {
      const { width: sidebarWidth, headerCollapsed: isHeaderCollapsed } = event.detail || {};
      const mapContainer = document.getElementById('map-container');
      if (!mapContainer) return;
      
      // Remove any existing badge to avoid duplicates
      const existingBadge = document.getElementById('detection-debug');
      if (existingBadge && existingBadge.parentNode) {
        existingBadge.parentNode.removeChild(existingBadge);
      }
      
      // Create the new badge
      const badge = document.createElement('div');
      badge.id = 'detection-debug';
      badge.textContent = 'DETECTION';
      badge.style.display = showOverlay ? 'block' : 'none';
      badge.style.opacity = showOverlay ? '1' : '0';
      badge.style.position = 'absolute';
      badge.style.top = '0';
      badge.style.right = sidebarWidth || `${width}px`; // Use width from event or component state
      badge.style.background = 'rgba(13, 71, 161, 0.85)';
      badge.style.zIndex = '200';
      badge.style.padding = '5px 12px';
      badge.style.fontSize = '12px';
      badge.style.color = 'white';
      badge.style.fontWeight = '500';
      badge.style.borderBottomLeftRadius = '3px';
      badge.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
      badge.style.letterSpacing = '0.5px';
      badge.style.transition = 'right 0.2s ease'; // Add smooth transition for right property
      
      // Add to map container
      mapContainer.appendChild(badge);
      console.log('DetectionSidebarBridge: Created detection badge');
    };
    
    // Handle sidebar resize (specifically for badge repositioning)
    const handleSidebarResizing = (event) => {
      if (event.detail && typeof event.detail.width === 'number') {
        const newWidth = event.detail.width;
        
        // Update badge position directly
        const badge = document.getElementById('detection-debug');
        if (badge) {
          badge.style.right = `${newWidth}px`;
        }
      }
    };
    
    // Handle sidebar resize completed
    const handleSidebarResized = (event) => {
      if (event.detail && typeof event.detail.width === 'number') {
        const newWidth = event.detail.width;
        
        // Update badge position with animation
        const badge = document.getElementById('detection-debug');
        if (badge) {
          badge.style.right = `${newWidth}px`;
          badge.style.transition = 'right 0.2s ease'; // Ensure smooth transition after resize
          
          console.log(`DetectionSidebarBridge: Updated badge position after resize to ${newWidth}px`);
        }
      }
    };

    // Handle fast inference results from early detection data
    const handleFastInferenceResults = (event) => {
      if (!event.detail) return;
      
      console.log("DetectionSidebarBridge: Received fastInferenceResults event",
                  'job_id:', event.detail.job_id,
                  'trees:', Array.isArray(event.detail.trees) ? event.detail.trees.length : 'none');
      
      // Ensure preliminary detection data is processed
      if (event.detail._preliminary === true) {
        console.log('DetectionSidebarBridge: Processing preliminary detection data');
        
        // Update trees with preliminary data
        if (Array.isArray(event.detail.trees)) {
          const newTrees = event.detail.trees.map((tree) => ({
            ...tree,
            visible: true,
            preliminary: true
          }));
          
          setTrees(newTrees);
          
          if (newTrees.length > 0 && !selectedTree) {
            setSelectedTree(newTrees[0]);
            setCurrentTreeIndex(0);
          }
        }
          
        // Always pass the data to the detection preview for early visualization
        // even if there are no trees yet, as metadata-only is valid for preview
        if (typeof window.showDetectionPreview === 'function') {
          console.log('DetectionSidebarBridge: Showing detection preview with preliminary data');
          
          try {
            // Make direct function call to show preview
            window.showDetectionPreview(event.detail);
            
            // Also try setting global detection data
            if (!window.mlDetectionData) {
              window.mlDetectionData = event.detail;
              
              // Force overlay to be visible
              window.mlOverlaySettings = {
                ...(window.mlOverlaySettings || {}),
                showOverlay: true
              };
              window.detectionShowOverlay = true;
            }
          } catch (error) {
            console.error('Error showing detection preview:', error);
          }
        } else {
          console.warn('DetectionSidebarBridge: window.showDetectionPreview is not available');
        }
      }
    };
    
    // Register event listeners
    window.addEventListener('mlDetectionActive', handleMLDetectionActive);
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    window.addEventListener('mlOverlaySettingsChanged', handleOverlaySettingsChanged);
    window.addEventListener('detectionDataLoaded', handleDetectionUpdate);
    window.addEventListener('createDetectionBadge', handleCreateDetectionBadge);
    window.addEventListener('detectionSidebarResizing', handleSidebarResizing);
    window.addEventListener('detectionSidebarResized', handleSidebarResized);
    document.addEventListener('fastInferenceResults', handleFastInferenceResults);
    
    // Clean up event listeners and observer
    return () => {
      observer.disconnect();
      window.removeEventListener('mlDetectionActive', handleMLDetectionActive);
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
      window.removeEventListener('mlOverlaySettingsChanged', handleOverlaySettingsChanged);
      window.removeEventListener('detectionDataLoaded', handleDetectionUpdate);
      window.removeEventListener('createDetectionBadge', handleCreateDetectionBadge);
      window.removeEventListener('detectionSidebarResizing', handleSidebarResizing);
      window.removeEventListener('detectionSidebarResized', handleSidebarResized);
      document.removeEventListener('fastInferenceResults', handleFastInferenceResults);
    };
  }, []);
  
  // Whenever selectedTree changes, dispatch an event to notify other components
  // Use a ref to track previous values and prevent redundant events
  const prevSelectedRef = useRef({ tree: null, index: -1 });
  
  useEffect(() => {
    if (!selectedTree) return;
    
    // Only dispatch event if tree or index actually changed
    const prevTree = prevSelectedRef.current.tree;
    const prevIndex = prevSelectedRef.current.index;
    
    if (!prevTree || prevTree.id !== selectedTree.id || prevIndex !== currentTreeIndex) {
      console.log('DetectionSidebarBridge: Tree selection changed, dispatching event');
      
      window.dispatchEvent(new CustomEvent('treeSelected', {
        detail: { 
          tree: selectedTree, 
          index: currentTreeIndex,
          source: 'detection_sidebar_bridge'
        }
      }));
      
      // Update the ref
      prevSelectedRef.current = { tree: selectedTree, index: currentTreeIndex };
    }
  }, [selectedTree, currentTreeIndex]);
  
  // Hook up the manual placement toggle to the DOM-based button
  // Use a ref to track previous state
  const prevManualPlacementRef = useRef(manualPlacement);
  
  useEffect(() => {
    // Only dispatch event if state actually changed
    if (prevManualPlacementRef.current !== manualPlacement) {
      console.log(`DetectionSidebarBridge: Manual placement mode ${manualPlacement ? 'enabled' : 'disabled'}`);
      
      if (manualPlacement) {
        window.dispatchEvent(new CustomEvent('enableManualTreePlacement', {
          detail: { source: 'detection_sidebar_bridge' }
        }));
      } else {
        window.dispatchEvent(new CustomEvent('disableManualTreePlacement', {
          detail: { source: 'detection_sidebar_bridge' }
        }));
      }
      
      // Update ref
      prevManualPlacementRef.current = manualPlacement;
    }
  }, [manualPlacement]);
  
  // Simplified functions for tree management
  const removeTree = () => {
    if (!selectedTree) return;
    
    // Mark tree as not visible
    setTrees(prevTrees => 
      prevTrees.map(tree => 
        tree.id === selectedTree.id ? { ...tree, visible: false } : tree
      )
    );
    
    // Find next visible tree
    const visibleTrees = trees.filter(tree => 
      tree.visible && tree.id !== selectedTree.id
    );
    
    if (visibleTrees.length > 0) {
      const nextIndex = Math.min(currentTreeIndex, visibleTrees.length - 1);
      setSelectedTree(visibleTrees[nextIndex]);
      
      // Find actual index in full array
      const newIdx = trees.findIndex(t => t.id === visibleTrees[nextIndex].id);
      setCurrentTreeIndex(newIdx >= 0 ? newIdx : 0);
    } else {
      setSelectedTree(null);
      setCurrentTreeIndex(0);
    }
  };
  
  const validateTree = () => {
    if (!selectedTree) return;
    
    // Update tree with validated status
    setTrees(prevTrees => 
      prevTrees.map(tree => 
        tree.id === selectedTree.id ? { 
          ...tree, 
          validated: true,
          approval_status: 'approved',
          approved_at: new Date().toISOString()
        } : tree
      )
    );
    
    // Update selected tree
    setSelectedTree({
      ...selectedTree,
      validated: true,
      approval_status: 'approved',
      approved_at: new Date().toISOString()
    });
    
    // Go to next tree
    goToNextTree();
  };
  
  const goToNextTree = () => {
    const visibleTrees = trees.filter(tree => tree.visible);
    if (visibleTrees.length === 0) return;
    
    const visibleIndex = selectedTree 
      ? visibleTrees.findIndex(tree => tree.id === selectedTree.id)
      : -1;
    
    const nextVisibleIndex = (visibleIndex + 1) % visibleTrees.length;
    const nextTree = visibleTrees[nextVisibleIndex];
    
    // Find actual index in full array
    const actualIndex = trees.findIndex(tree => tree.id === nextTree.id);
    
    setSelectedTree(nextTree);
    setCurrentTreeIndex(actualIndex >= 0 ? actualIndex : nextVisibleIndex);
  };
  
  const goToPreviousTree = () => {
    const visibleTrees = trees.filter(tree => tree.visible);
    if (visibleTrees.length === 0) return;
    
    const visibleIndex = selectedTree 
      ? visibleTrees.findIndex(tree => tree.id === selectedTree.id)
      : 0;
    
    const prevVisibleIndex = (visibleIndex - 1 + visibleTrees.length) % visibleTrees.length;
    const prevTree = visibleTrees[prevVisibleIndex];
    
    // Find actual index in full array
    const actualIndex = trees.findIndex(tree => tree.id === prevTree.id);
    
    setSelectedTree(prevTree);
    setCurrentTreeIndex(actualIndex >= 0 ? actualIndex : prevVisibleIndex);
  };
  
  // Default empty form data
  const [formData, setFormData] = useState({
    species: '',
    height: 30,
    diameter: 12,
    riskLevel: 'new'
  });
  
  // State for editing
  const [isEditing, setIsEditing] = useState(false);
  
  // Update form data when selected tree changes
  useEffect(() => {
    if (selectedTree) {
      setFormData({
        species: selectedTree.species || 'Unknown Species',
        height: selectedTree.height || 30,
        diameter: selectedTree.diameter || 12,
        riskLevel: selectedTree.risk_level || 'medium'
      });
    }
  }, [selectedTree]);
  
  // Handle form input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };
  
  // Save tree edits
  const saveTreeEdits = () => {
    if (!selectedTree) return;
    
    // Update tree with edited values
    setTrees(prevTrees => 
      prevTrees.map(tree => 
        tree.id === selectedTree.id ? {
          ...tree,
          species: formData.species,
          height: formData.height ? parseFloat(formData.height) : null,
          diameter: formData.diameter ? parseFloat(formData.diameter) : null,
          risk_level: formData.riskLevel
        } : tree
      )
    );
    
    // Update selected tree
    setSelectedTree({
      ...selectedTree,
      species: formData.species,
      height: formData.height ? parseFloat(formData.height) : null,
      diameter: formData.diameter ? parseFloat(formData.diameter) : null,
      risk_level: formData.riskLevel
    });
    
    // Exit edit mode
    setIsEditing(false);
  };
  
  // Simple fake Gemini parameters 
  const [geminiParams, setGeminiParams] = useState({
    detectionThreshold: 0.7,
    maxTrees: 20,
    includeRiskAnalysis: true,
    detailLevel: 'high'
  });
  
  // Update collapsed state for the sidebar
  const [collapsed, setCollapsed] = useState(false);
  
  // Decide where to render our React component
  // Look for a content container first, otherwise use the whole sidebar
  const renderTarget = sidebarElement 
    ? sidebarElement.querySelector('#detection-content-container') || sidebarElement
    : null;
    
  // Helper to get the most appropriate target for the portal
  const getPortalTarget = () => {
    if (!sidebarElement) return null;
    
    // Look for content container first
    const contentContainer = sidebarElement.querySelector('#detection-content-container');
    if (contentContainer) return contentContainer;
    
    // If content container doesn't exist but sidebar does, create one
    const newContainer = document.createElement('div');
    newContainer.id = 'detection-content-container';
    newContainer.style.width = '100%';
    newContainer.style.height = '100%'; // Full height since React header is inside
    newContainer.style.overflow = 'auto';
    newContainer.style.position = 'relative';
    sidebarElement.appendChild(newContainer);
    
    return newContainer;
  };
  
  // If we have a sidebar element, render our DetectionSidebar component inside it
  return (
    <>
      {sidebarElement && createPortal(
        <DetectionSidebar
          trees={trees}
          selectedTree={selectedTree}
          setSelectedTree={setSelectedTree}
          currentTreeIndex={currentTreeIndex}
          setCurrentTreeIndex={setCurrentTreeIndex}
          isEditing={isEditing}
          setIsEditing={setIsEditing}
          formData={formData}
          setFormData={setFormData}
          removeTree={removeTree}
          validateTree={validateTree}
          goToNextTree={goToNextTree}
          goToPreviousTree={goToPreviousTree}
          handleInputChange={handleInputChange}
          saveTreeEdits={saveTreeEdits}
          collapsed={collapsed}
          setCollapsed={setCollapsed}
          width={width}
          setWidth={setWidth}
          manualPlacement={manualPlacement}
          setManualPlacement={setManualPlacement}
          showOverlay={showOverlay}
          setShowOverlay={setShowOverlay}
          overlayOpacity={overlayOpacity}
          setOverlayOpacity={setOverlayOpacity}
          showSegmentation={showSegmentation}
          setShowSegmentation={setShowSegmentation}
          geminiParams={geminiParams}
          setGeminiParams={setGeminiParams}
          headerCollapsed={headerCollapsed}
        />,
        getPortalTarget() || sidebarElement
      )}
      
      {/* Always include MLOverlayInitializer for consistent initialization */}
      <MLOverlayInitializer />
    </>
  );
};

export default DetectionSidebarBridge;