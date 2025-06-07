// src/components/visualization/Detection/DetectionMode.jsx
//
// This is the main controller component for object detection functionality.
// It's been refactored to use DetectionSidebar and DetectionRenderer components
// for better separation of concerns.

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import DetectionSidebar from './DetectionSidebar';
import DetectionRenderer from './DetectionRenderer';
import { useValidation } from '../../../hooks/useReportValidation';

/**
 * Detection Mode Component
 * 
 * Top-level controller for the object detection feature
 * that coordinates between sidebar UI and map rendering.
 */
const DetectionMode = (props) => {
  // Destructure props for safety and easier access
  const { 
    mapRef, 
    validationData, 
    detectedTrees = [], 
    onExitValidation = () => {}, 
    onSaveTrees = () => {},
    headerCollapsed: propHeaderCollapsed 
  } = props;
  
  // Check if this is feature selection mode
  const isFeatureSelectionMode = validationData?.mode === 'feature_selection';
  
  // State for tree data and selection
  const [trees, setTrees] = useState([]);
  const [selectedTree, setSelectedTree] = useState(null);
  const [currentTreeIndex, setCurrentTreeIndex] = useState(0);
  const [isEditing, setIsEditing] = useState(false);
  const [editingBounds, setEditingBounds] = useState(null);
  const [collapsed, setCollapsed] = useState(false); // ALWAYS start uncollapsed
  
  // UI state
  const [activeTab, setActiveTab] = useState('trees'); // 'trees' or 'params'
  const [selectedReports, setSelectedReports] = useState([]);
  const [configVisible, setConfigVisible] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredTrees, setFilteredTrees] = useState([]);
  const [headerCollapsed, setHeaderCollapsed] = useState(true); // Track header collapsed state
  
  // Form data for editing trees
  const [formData, setFormData] = useState({
    species: '',
    height: 30,
    diameter: 12,
    riskLevel: 'new'
  });
  
  // Detection parameters
  const [geminiParams, setGeminiParams] = useState({
    detectionThreshold: 0.7,
    maxTrees: 20,
    includeRiskAnalysis: true,
    detailLevel: 'high'
  });
  
  // Visualization settings
  const [manualPlacement, setManualPlacement] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [width, setWidth] = useState(384); // Width of the sidebar in pixels
  const [showPreview, setShowPreview] = useState(false); // Track if detection preview is showing
  
  // Only use validation hook if we're in the reports tab
  const { validationItems, validateItem, isProcessing, refreshValidationItems } = 
    activeTab === 'reports' ? useValidation() : 
    { validationItems: [], validateItem: () => {}, isProcessing: false, refreshValidationItems: () => {} };
    
  // CRITICAL: Make sure the sidebar stays visible (not collapsed) during detection preview
  useEffect(() => {
    if (showPreview && collapsed) {
      console.log("DetectionMode: Detection preview is showing but sidebar is collapsed - forcing sidebar to show");
      setCollapsed(false);
    }
  }, [showPreview, collapsed, setCollapsed]);
  
  // State for tracking detection status
  const [isDetecting, setIsDetecting] = useState(false);
  
  // State for segmentation masks
  const [showSegmentation, setShowSegmentation] = useState(true);
  
  // Improved header state tracking with props-first approach
  useEffect(() => {
    // First, check if header state was passed as a prop
    if (props.headerCollapsed !== undefined) {
      // Props take precedence over local detection
      setHeaderCollapsed(props.headerCollapsed);
    } else {
      // Fallback to direct DOM detection only if props aren't provided
      try {
        const headerElement = document.querySelector('header');
        const isHeaderCollapsed = headerElement ? 
          headerElement.classList.contains('collapsed') || 
          headerElement.offsetHeight < 50 : true;
        setHeaderCollapsed(isHeaderCollapsed);
      } catch (e) {
        console.log("Error detecting header state:", e);
        // Default to true if detection fails
        setHeaderCollapsed(true);
      }
    }
    
    // Listen for explicit header collapse events
    const handleHeaderCollapse = (event) => {
      if (event.detail && event.detail.collapsed !== undefined) {
        setHeaderCollapsed(event.detail.collapsed);
      }
    };
    
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    return () => {
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
    };
  }, [props.headerCollapsed]);
  
  // Helper function to show detection preview using the global function
  const showDetectionPreview = (data) => {
    if (!data || typeof window.showDetectionPreview !== 'function') return;
    
    console.log("DetectionMode: Showing detection preview via global function");
    try {
      // Ensure no duplicates by removing existing preview
      if (typeof window.destroyDetectionPreview === 'function') {
        window.destroyDetectionPreview();
      }
      
      // Make sidebar visible
      setCollapsed(false);
      
      // Update UI state
      setShowPreview(true);
      
      // Show preview after DOM update
      setTimeout(() => {
        window.showDetectionPreview(data);
      }, 50);
      
      // Add a global listener to prevent any component from collapsing the sidebar
      // while the preview is showing
      const preventCollapseHandler = (event) => {
        if (showPreview) {
          console.log("DetectionMode: Preventing sidebar collapse while preview is showing");
          event.preventDefault();
          event.stopPropagation();
          return false;
        }
      };
      
      // Remove any existing handler before adding a new one
      window.removeEventListener('beforeDetectionSidebarCollapse', preventCollapseHandler, true);
      window.addEventListener('beforeDetectionSidebarCollapse', preventCollapseHandler, true);
    } catch (error) {
      console.error("Error showing detection preview:", error);
    }
  };

  // Centralized progress bar management
  const progressManager = useMemo(() => {
    const createProgressBar = (detectBtn) => {
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
      
      const newProgressBar = document.createElement('div');
      newProgressBar.id = 'detection-progress-bar';
      newProgressBar.style.width = '5%';
      newProgressBar.style.height = '100%';
      newProgressBar.style.backgroundColor = '#3b82f6';
      newProgressBar.style.borderRadius = '3px';
      newProgressBar.style.transition = 'width 0.3s ease';
      
      const newProgressText = document.createElement('div');
      newProgressText.id = 'detection-progress-text';
      newProgressText.style.fontSize = '11px';
      newProgressText.style.color = '#64748b';
      newProgressText.style.marginTop = '4px';
      newProgressText.textContent = 'Loading model...';
      
      progressBarOuter.appendChild(newProgressBar);
      progressContainer.appendChild(progressBarOuter);
      progressContainer.appendChild(newProgressText);
      
      // Add progress bar near the button
      const buttonContainer = detectBtn.parentNode;
      if (buttonContainer) {
        buttonContainer.parentNode.insertBefore(progressContainer, buttonContainer.nextSibling);
      } else {
        // If can't find the button container, add it to the document body
        document.body.appendChild(progressContainer);
      }
      
      return progressContainer;
    };
    
    const update = (percent, message) => {
      const bar = document.getElementById('detection-progress-bar');
      const text = document.getElementById('detection-progress-text');
      
      if (bar) bar.style.width = `${percent}%`;
      if (text) text.textContent = message;
    };
    
    const remove = () => {
      setTimeout(() => {
        const progressContainer = document.getElementById('detection-progress-container');
        if (progressContainer && progressContainer.parentNode) {
          progressContainer.parentNode.removeChild(progressContainer);
        }
      }, 2000);
    };
    
    return {
      createProgressBar,
      update,
      remove
    };
  }, []);
  
  // Centralized detection button management
  const buttonManager = useMemo(() => {
    const setLoading = (btn, isLoading) => {
      if (!btn) return;
      
      if (isLoading) {
        btn.disabled = true;
        btn.style.backgroundColor = '#CBD5E1'; // Light gray
        btn.style.cursor = 'not-allowed';
        btn.textContent = 'Detecting...';
      } else {
        btn.disabled = false;
        btn.style.backgroundColor = '#0d47a1'; // Dark blue
        btn.style.cursor = 'pointer';
        btn.textContent = 'Detect';
      }
    };
    
    return { setLoading };
  }, []);
  
  // Centralized event handling
  const eventManager = useMemo(() => {
    let activeListeners = [];
    
    const registerListeners = (listeners) => {
      // Remove any existing listeners first
      removeAllListeners();
      
      // Add new listeners
      listeners.forEach(({ event, handler }) => {
        window.addEventListener(event, handler);
        activeListeners.push({ event, handler });
      });
    };
    
    const removeAllListeners = () => {
      activeListeners.forEach(({ event, handler }) => {
        window.removeEventListener(event, handler);
      });
      activeListeners = [];
    };
    
    return {
      registerListeners,
      removeAllListeners
    };
  }, []);
  
  // Handle detection button click
  const handleDetectClick = useCallback(() => {
    if (isDetecting) {
      console.log("Already detecting, ignoring click");
      return; // Prevent multiple clicks
    }
    
    console.log("Starting detection process");
    setIsDetecting(true);
    
    // Get and update the Detect button
    const detectBtn = document.getElementById('detect-trees-btn');
    buttonManager.setLoading(detectBtn, true);
    
    // Create progress bar if it doesn't exist
    const progressBar = document.getElementById('detection-progress-bar');
    if (!progressBar && detectBtn) {
      progressManager.createProgressBar(detectBtn);
    }
    
    // Use existing job ID if available, otherwise generate one
    const jobId = window.currentDetectionJobId || `detection_${Date.now()}`;
    // Store it globally so other components can access it
    window.currentDetectionJobId = jobId;
    console.log(`DetectionMode: Using job ID: ${jobId}`);
    
    // Set up event handlers for detection process
    const handleDetectionResults = (event) => {
      if (!event.detail) return;
      
      console.log("DetectionMode: Received detection results event");
      progressManager.update(80, 'Inference complete! Rendering results...');
    };
    
    const handleDetectionComplete = () => {
      setIsDetecting(false);
      progressManager.update(100, 'Detection complete!');
      buttonManager.setLoading(detectBtn, false);
      progressManager.remove();
      eventManager.removeAllListeners();
      
      // Clear safety timeout if it exists
      if (safetyTimeout) clearTimeout(safetyTimeout);
    };
    
    const handleDetectionError = () => {
      setIsDetecting(false);
      progressManager.update(0, 'Detection failed. Please try again.');
      buttonManager.setLoading(detectBtn, false);
      eventManager.removeAllListeners();
      
      // Clear safety timeout if it exists
      if (safetyTimeout) clearTimeout(safetyTimeout);
    };
    
    // Register all event listeners
    eventManager.registerListeners([
      { event: 'detectionDataLoaded', handler: handleDetectionResults },
      { event: 'enterTreeValidationMode', handler: handleDetectionComplete },
      { event: 'treeDetectionError', handler: handleDetectionError },
      { event: 'treeDetectionComplete', handler: handleDetectionComplete }
    ]);
    
    // Safety timeout: ensure detection always completes even if no event is fired
    const safetyTimeout = setTimeout(() => {
      console.log('Detection safety timeout reached - forcing completion');
      handleDetectionComplete();
    }, 30000); // 30 seconds max
    
    // First make sure any cached mapViewInfo is cleared so we get fresh coordinates
    delete window.mapViewInfo;
    
    // Dispatch detection event
    const detectionEvent = new CustomEvent('openTreeDetection', {
      detail: {
        useSatelliteImagery: true,
        useRealGemini: geminiParams.includeRiskAnalysis,
        saveToResponseJson: true,
        geminiParams: geminiParams,
        job_id: jobId,
        buttonTriggered: true,  // Flag to indicate this was triggered by button
        executeMLPipeline: true,  // Specifically request ML pipeline execution
        freshCoordinates: true   // Flag to force using current map coordinates
      },
      buttonTriggered: true  // Flag at event level as well for wider compatibility
    });
    
    console.log("DetectionMode: Dispatching openTreeDetection event with buttonTriggered flag");
    
    // Ensure we set global flags for ML overlay
    window.detectionShowOverlay = true;
    window.mlOverlaySettings = {
      ...(window.mlOverlaySettings || {}),
      showOverlay: true,
      pendingButtonTrigger: false
    };
    
    // Dispatch the event after setting all flags
    window.dispatchEvent(detectionEvent);
    
    // Show initial progress
    progressManager.update(10, 'Loading Grounded SAM model...');
    
    // Simulate progress updates
    setTimeout(() => progressManager.update(20, 'Processing satellite imagery...'), 500);
    setTimeout(() => progressManager.update(30, 'Running object detection...'), 1000);
  }, [
    isDetecting, geminiParams, showPreview, buttonManager, progressManager, eventManager, showDetectionPreview
  ]);
  
  // Add click handler to the detect button - attach via props instead of directly
  // This avoids issues with authentication errors and element event handling
  useEffect(() => {
    // This effect now only handles dynamic detection button events
    // for buttons that are created outside of React lifecycle
    const detectBtn = document.getElementById('detect-trees-btn');
    if (detectBtn && !detectBtn.hasAttribute('data-handler-attached')) {
      detectBtn.addEventListener('click', handleDetectClick);
      detectBtn.setAttribute('data-handler-attached', 'true');
      
      console.log('Attached click handler to existing detection button');
    }
    
    return () => {
      const detectBtn = document.getElementById('detect-trees-btn');
      if (detectBtn && detectBtn.hasAttribute('data-handler-attached')) {
        detectBtn.removeEventListener('click', handleDetectClick);
        detectBtn.removeAttribute('data-handler-attached');
      }
    };
  }, [handleDetectClick]);
  
  // Initialize trees from detected trees prop
  useEffect(() => {
    if (detectedTrees && detectedTrees.length > 0) {
      console.log(`Received ${detectedTrees.length} detected trees to render`);
      
      // Initialize trees from props and set as visible
      const initialTrees = detectedTrees.map(tree => ({
        ...tree,
        visible: true
      }));
      
      setTrees(initialTrees);
      
      // If no selected tree yet, select the first visible tree
      if (!selectedTree && initialTrees.length > 0) {
        setSelectedTree(initialTrees[0]);
        setCurrentTreeIndex(0);
        
        // Pre-populate form data for editing
        setFormData({
          species: initialTrees[0].species || 'Unspecified',
          height: initialTrees[0].height || 'Unknown',
          diameter: initialTrees[0].diameter || 'Unknown',
          riskLevel: initialTrees[0].risk_level || 'Unassigned'
        });
      }
      
      // Ensure detect button is reset when trees are received
      setIsDetecting(false);
      const detectBtn = document.getElementById('detect-trees-btn');
      if (detectBtn) {
        detectBtn.disabled = false;
        detectBtn.style.backgroundColor = '#0d47a1'; // Dark blue
        detectBtn.style.cursor = 'pointer';
        detectBtn.textContent = 'Detect';
      }
    }
  }, [detectedTrees, selectedTree]);
  
  // Handle tree removal
  const removeTree = () => {
    if (!selectedTree) return;
    
    if (!window.confirm("Are you sure you want to remove this object?")) {
      return;
    }
    
    // 1. First, mark tree as not visible instead of removing - this allows undo
    setTrees(prevTrees => 
      prevTrees.map(tree => 
        tree.id === selectedTree.id ? { ...tree, visible: false } : tree
      )
    );
    
    // 2. Get the next visible tree
    const visibleTrees = trees.filter(tree => 
      tree.visible && tree.id !== selectedTree.id
    );
    
    // 3. Select the next tree or null if none left
    if (visibleTrees.length > 0) {
      const nextIndex = Math.min(currentTreeIndex, visibleTrees.length - 1);
      setSelectedTree(visibleTrees[nextIndex]);
      
      // Find the actual index in the original array
      const newIdx = trees.findIndex(t => t.id === visibleTrees[nextIndex].id);
      setCurrentTreeIndex(newIdx >= 0 ? newIdx : 0);
    } else {
      setSelectedTree(null);
      setCurrentTreeIndex(0);
    }
  };
  
  // Approve current tree without editing and persist to ValidationQueue
  const validateTree = () => {
    if (!selectedTree) return;
    
    // Show approval notification
    const notification = document.createElement('div');
    notification.className = 'tree-approved-notification';
    notification.style.position = 'absolute';
    notification.style.top = '70px';
    notification.style.left = '50%';
    notification.style.transform = 'translateX(-50%)';
    notification.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
    notification.style.color = 'white';
    notification.style.padding = '8px 16px';
    notification.style.borderRadius = '4px';
    notification.style.zIndex = '9999';
    notification.style.fontSize = '14px';
    notification.style.fontWeight = '500';
    notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
    notification.textContent = 'Object approved';
    document.body.appendChild(notification);
    
    // Remove notification after 1.5 seconds
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 1500);
    
    // Add validated status to the selected tree
    const updatedTrees = trees.map(tree => {
      if (tree.id === selectedTree.id) {
        return {
          ...tree,
          validated: true,
          approval_status: 'approved', // Add explicit approval status
          approved_at: new Date().toISOString() // Add timestamp
        };
      }
      return tree;
    });
    
    setTrees(updatedTrees);
    setSelectedTree({
      ...selectedTree,
      validated: true,
      approval_status: 'approved',
      approved_at: new Date().toISOString()
    });
    
    // Move to next tree
    goToNextTree();
  };
  
  // Handle next/previous tree
  const goToNextTree = () => {
    // Get only visible trees
    const visibleTrees = trees.filter(tree => tree.visible);
    if (visibleTrees.length === 0) return;
    
    // Find current index among visible trees
    const visibleIndex = selectedTree 
      ? visibleTrees.findIndex(tree => tree.id === selectedTree.id)
      : -1;
    
    // Get next visible index (circular)
    const nextVisibleIndex = (visibleIndex + 1) % visibleTrees.length;
    const nextVisibleTree = visibleTrees[nextVisibleIndex];
    
    // Find actual index in full tree array for carousel navigation tracking
    const actualIndex = trees.findIndex(tree => tree.id === nextVisibleTree.id);
    
    // Update state
    setSelectedTree(nextVisibleTree);
    setCurrentTreeIndex(actualIndex >= 0 ? actualIndex : nextVisibleIndex);
    
    // Update form data for when edit mode is activated
    setFormData({
      species: nextVisibleTree.species || 'Unknown Species',
      height: nextVisibleTree.height || 30,
      diameter: nextVisibleTree.diameter || 12,
      riskLevel: nextVisibleTree.risk_level || 'medium'
    });
  };
  
  const goToPreviousTree = () => {
    // Get only visible trees
    const visibleTrees = trees.filter(tree => tree.visible);
    if (visibleTrees.length === 0) return;
    
    // Find current index among visible trees
    const visibleIndex = selectedTree 
      ? visibleTrees.findIndex(tree => tree.id === selectedTree.id)
      : 0;
    
    // Get previous visible index (circular)
    const prevVisibleIndex = (visibleIndex - 1 + visibleTrees.length) % visibleTrees.length;
    const prevVisibleTree = visibleTrees[prevVisibleIndex];
    
    // Find actual index in full tree array
    const actualIndex = trees.findIndex(tree => tree.id === prevVisibleTree.id);
    
    // Update state
    setSelectedTree(prevVisibleTree);
    setCurrentTreeIndex(actualIndex >= 0 ? actualIndex : prevVisibleIndex);
    
    // Update form data for when edit mode is activated
    setFormData({
      species: prevVisibleTree.species || 'Unknown Species',
      height: prevVisibleTree.height || 30,
      diameter: prevVisibleTree.diameter || 12,
      riskLevel: prevVisibleTree.risk_level || 'medium'
    });
  };
  
  // Handle input changes in form
  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    if (type === 'checkbox') {
      // For checkbox fields
      setFormData(prev => ({ ...prev, [name]: checked }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };
  
  // Save edits to the selected tree 
  const saveTreeEdits = () => {
    if (!selectedTree) return;
    
    // Update the tree with edited values
    const updatedTrees = trees.map(tree => {
      if (tree.id === selectedTree.id) {
        return {
          ...tree,
          species: formData.species,
          height: formData.height ? parseFloat(formData.height) : null,
          diameter: formData.diameter ? parseFloat(formData.diameter) : null,
          risk_level: formData.riskLevel
        };
      }
      return tree;
    });
    
    // Update the trees array
    setTrees(updatedTrees);
    
    // Update the selected tree with edited values
    setSelectedTree({
      ...selectedTree,
      species: formData.species,
      height: formData.height ? parseFloat(formData.height) : null,
      diameter: formData.diameter ? parseFloat(formData.diameter) : null,
      risk_level: formData.riskLevel
    });
    
    // Exit edit mode
    setIsEditing(false);
    
    // Display success message
    const notification = document.createElement('div');
    notification.className = 'edit-success-notification';
    notification.style.position = 'absolute';
    notification.style.top = '70px';
    notification.style.left = '50%';
    notification.style.transform = 'translateX(-50%)';
    notification.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
    notification.style.color = 'white';
    notification.style.padding = '8px 16px';
    notification.style.borderRadius = '4px';
    notification.style.zIndex = '1000';
    notification.style.fontSize = '14px';
    notification.style.fontWeight = '500';
    notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
    notification.textContent = 'Object details updated successfully';
    
    // Add to document safely
    try {
      document.body.appendChild(notification);
      
      // Remove after 1.5 seconds
      setTimeout(() => {
        try {
          if (notification && notification.parentNode && document.body.contains(notification)) {
            notification.parentNode.removeChild(notification);
          }
        } catch (e) {
          console.log("Error removing notification:", e);
        }
      }, 1500);
    } catch (e) {
      console.log("Error showing notification:", e);
    }
  };
  
  // Listen for validation mode cleanup event from MapView but don't force sidebar collapse
  useEffect(() => {
    // Handle validation mode cleanup
    const handleValidationModeCleanup = (event) => {
      console.log("ValidationModeCleanup event received in DetectionMode", event?.detail);
      
      // Immediately start our own cleanup process
      try {
        // Reset state vars to default values
        setTrees([]);
        setSelectedTree(null);
        setCurrentTreeIndex(0);
        setIsEditing(false);
        setEditingBounds(null);
        setSearchQuery('');
        setFilteredTrees([]);
        
        // If preview is showing, destroy it
        if (showPreview) {
          console.log("Cleaning up detection preview");
          if (typeof window.destroyDetectionPreview === 'function') {
            try {
              window.destroyDetectionPreview();
            } catch (cleanupError) {
              console.warn("Error during preview cleanup:", cleanupError);
            }
          }
          setShowPreview(false);
        }
        
        // Notify parent component that we're cleaning up
        if (typeof onExitValidation === 'function') {
          onExitValidation();
        }
      } catch (err) {
        console.error("Error during ValidationMode cleanup:", err);
      }
    };
    
    // Prevent collapsing sidebar when preview is showing
    const preventSidebarCollapse = (event) => {
      if (showPreview) {
        console.log("DetectionMode: Preventing sidebar collapse while preview is showing");
        event.preventDefault();
        event.stopPropagation();
        return false;
      }
    };
    
    // Register listeners for events related to detection
    window.addEventListener('validationModeCleanup', handleValidationModeCleanup);
    window.addEventListener('beforeDetectionSidebarCollapse', preventSidebarCollapse, true);
    
    return () => {
      window.removeEventListener('validationModeCleanup', handleValidationModeCleanup);
      window.removeEventListener('beforeDetectionSidebarCollapse', preventSidebarCollapse, true);
      
      // Clean up detection preview if component unmounts
      if (showPreview) {
        console.log("Component unmounting - cleaning up detection preview");
        if (typeof window.destroyDetectionPreview === 'function') {
          try {
            window.destroyDetectionPreview();
          } catch (cleanupError) {
            console.warn("Error during preview cleanup on unmount:", cleanupError);
          }
        }
      }
    };
  }, [onExitValidation, showPreview]);
  
  // Detect manual placement mode and set up map click handler
  useEffect(() => {
    if (!manualPlacement || !mapRef.current) return;
    
    // Add visual indicator and mouse mode for manual placement
    console.log("Manual placement mode activated");
    
    return () => {
      // Cleanup for manual placement mode
      console.log("Manual placement mode deactivated");
    };
  }, [manualPlacement, mapRef]);
  
  // Dispatch map initialized event when map is available
  useEffect(() => {
    if (mapRef.current && window.google && window.google.maps) {
      console.log("DetectionMode: Map is available, dispatching mapInitialized event");
      // Make map available via window.map
      window.map = window.map || mapRef.current;
      // Dispatch event to notify components that map is available
      window.dispatchEvent(new Event('mapInitialized'));
    }
  }, [mapRef]);
  
  return (
    <>
      {/* Sidebar UI Component - only visible when not collapsed */}
      {!collapsed && (
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
        />
      )}
      
      {/* Map Rendering Component */}
      <DetectionRenderer
        mapRef={mapRef}
        trees={trees}
        selectedTree={selectedTree}
        setSelectedTree={setSelectedTree}
        setCurrentTreeIndex={setCurrentTreeIndex}
        showOverlay={showOverlay}
        overlayOpacity={overlayOpacity}
        jobId={validationData?.jobId}
        setCollapsed={setCollapsed}
        width={width}
        showPreview={showPreview}
        setShowPreview={setShowPreview}
      />
    </>
  );
};

export default DetectionMode;