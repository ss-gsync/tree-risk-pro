import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { X, Check, Edit, Plus, Trash, ChevronLeft, ChevronRight, Save, AlertTriangle, FileText, MapPin, Clock, CheckCircle, XCircle, Search, Database, BarChart, Settings } from 'lucide-react';
import { useValidation } from '../../../hooks/useReportValidation';

const TreeValidationMode = ({ 
  mapRef, 
  validationData, 
  detectedTrees,
  onExitValidation, 
  onSaveTrees 
}) => {
  // Check if this is feature selection mode
  const isFeatureSelectionMode = validationData?.mode === 'feature_selection';
  const [trees, setTrees] = useState([]);
  const [selectedTree, setSelectedTree] = useState(null);
  const [currentTreeIndex, setCurrentTreeIndex] = useState(0);
  const [isEditing, setIsEditing] = useState(false);
  const [editingBounds, setEditingBounds] = useState(null);
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState('trees'); // 'trees' or 'reports'
  const [selectedReports, setSelectedReports] = useState([]);
  const { validationItems, validateItem, isProcessing, refreshValidationItems } = useValidation();
  const [configVisible, setConfigVisible] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredTrees, setFilteredTrees] = useState([]);
  const [geminiParams, setGeminiParams] = useState({
    detectionThreshold: 0.7,
    maxTrees: 20,
    includeRiskAnalysis: true,
    detailLevel: 'high'
  });
  
  /**
   * Legacy collapse toggle function 
   * 
   * Collapse functionality is disabled for this sidebar to maintain consistency,
   * but we still need to adjust the map container to prevent grey areas.
   * This function is kept for API compatibility with other components.
   */
  const toggleCollapse = () => {
    // No-op - we're disabling the collapse functionality for consistency
    // Only adjust the map container to prevent grey area
    const mapContainer = document.querySelector('.flex-1.absolute.inset-0');
    if (mapContainer) {
      mapContainer.style.right = collapsed ? '0px' : '384px';
    }
  };
  
  // Listen for validation queue toggle events and respond politely
  useEffect(() => {
    const handleValidationQueueToggle = (event) => {
      // Only respond to events from ValidationQueue that aren't initialization events
      if (event.detail.source === 'validationQueue' && !event.detail.initializing) {
        // If ValidationQueue is being expanded, collapse this sidebar
        if (!event.detail.collapsed) {
          // Before collapsing, ensure we're not in the middle of our own operation
          setTimeout(() => {
            setCollapsed(true);
            
            // Reset to default tab for politeness
            setActiveTab('trees');
            
            // Also exit validation mode if needed
            if (!isFeatureSelectionMode) {
              // Trigger exit validation mode without dispatching events (to prevent loops)
              if (typeof onExitValidation === 'function') {
                onExitValidation();
              }
            }
            
            // Adjust map container to prevent grey areas
            const mapContainer = document.querySelector('.flex-1.absolute.inset-0');
            if (mapContainer) {
              mapContainer.style.right = '384px'; // Set to Video sidebar width
            }
          }, 50);
        }
      }
    };
    
    // Handle force close requests from other components
    const handleForceCloseTreeDatabase = (event) => {
      if (!collapsed) {
        setCollapsed(true);
        
        // Reset to default tab for politeness
        setActiveTab('trees');
      }
    };
    
    // Handle force close for Object Detection mode
    const handleForceCloseObjectDetection = (event) => {
      if (!collapsed && !isFeatureSelectionMode) {
        // Exit validation mode
        if (typeof onExitValidation === 'function') {
          onExitValidation();
        }
        
        setCollapsed(true);
      }
    };
    
    // Listen for other sidebars toggling
    window.addEventListener('validationQueueToggle', handleValidationQueueToggle);
    window.addEventListener('forceCloseTreeDatabase', handleForceCloseTreeDatabase);
    window.addEventListener('forceCloseObjectDetection', handleForceCloseObjectDetection);
    
    return () => {
      window.removeEventListener('validationQueueToggle', handleValidationQueueToggle);
      window.removeEventListener('forceCloseTreeDatabase', handleForceCloseTreeDatabase);
      window.removeEventListener('forceCloseObjectDetection', handleForceCloseObjectDetection);
    };
  }, [isFeatureSelectionMode, onExitValidation, collapsed]);
  
  // State for controlling Detect button availability
  const [detectButtonDisabled, setDetectButtonDisabled] = useState(false);
  const [detectButtonText, setDetectButtonText] = useState('Detect Trees');
  
  // Listen for tree detection errors to reset the button
  useEffect(() => {
    const handleDetectionError = (event) => {
      console.log("Tree detection error received:", event.detail);
      // Reset button state
      setDetectButtonDisabled(false);
      setDetectButtonText('Detect Trees');
    };
    
    window.addEventListener('treeDetectionError', handleDetectionError);
    
    return () => {
      window.removeEventListener('treeDetectionError', handleDetectionError);
    };
  }, []);
  
  // Dispatch initial state on mount and load validation items
  useEffect(() => {
    // Notify other components of our presence
    window.dispatchEvent(new CustomEvent('validationSidebarToggle', {
      detail: { 
        collapsed,
        source: 'treeValidationMode',
        initializing: true
      }
    }));
    
    // Adjust the map container - important to size correctly
    const mapContainer = document.getElementById('map-container');
    if (mapContainer) {
      mapContainer.style.right = '384px'; // This sidebar is always visible when mounted
    }
    
    // Refresh validation items when the component mounts
    if (isFeatureSelectionMode && refreshValidationItems) {
      refreshValidationItems();
      console.log("Refreshing validation items in Tree Inventory");
    }
    
    // Force map resize to prevent grey areas
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 100);
    
    // Handler for the detection flow completion
    const handleDetectionComplete = (event) => {
      setDetectButtonDisabled(false);
      setDetectButtonText('Detect Trees');
    };
    
    // Listen for detection result events
    window.addEventListener('treeDetectionResult', handleDetectionComplete);
    
    // Clean up when unmounting
    return () => {
      // Before resetting, check if the Video sidebar is visible
      const videoSidebar = document.getElementById('imagery-sidebar');
      const videoSidebarVisible = videoSidebar && videoSidebar.style.width !== '0px';
      
      // Reset map container when sidebar is unmounted
      const mapContainer = document.getElementById('map-container');
      if (mapContainer) {
        // Only reset to 0 if the Video sidebar isn't visible
        if (!videoSidebarVisible) {
          mapContainer.style.right = '0px';
        } else {
          // If Video sidebar is visible, set to its width
          mapContainer.style.right = '384px';
        }
      }
      
      // Notify other components we're leaving
      window.dispatchEvent(new CustomEvent('validationSidebarToggle', {
        detail: { 
          collapsed: true,
          source: 'treeValidationMode',
          unmounting: true
        }
      }));
      
      // Force map resize again after unmounting
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 100);
      
      // Remove event listener
      window.removeEventListener('treeDetectionResult', handleDetectionComplete);
    };
  }, []);
  
  // Fetch validation data from zarr database when in Tree Inventory mode
  useEffect(() => {
    // Only refresh validation items when we're in Tree Inventory (Database) mode
    if (isFeatureSelectionMode && refreshValidationItems) {
      // Check if a specific tab was requested in the validation data
      if (validationData && validationData.tab) {
        console.log(`Setting requested tab: ${validationData.tab}`);
        setActiveTab(validationData.tab);
      } else {
        // Default to Index tab for Database button, Reports tab otherwise
        const defaultTab = (validationData && validationData.jobId === 'tree_inventory') ? 'trees' : 'reports';
        setActiveTab(defaultTab);
      }
      
      // Refresh data from zarr database
      refreshValidationItems();
      console.log("Fetching validation items from zarr database for Database sidebar");
    }
  }, [isFeatureSelectionMode, refreshValidationItems, validationData]);
  const [formData, setFormData] = useState({
    species: '',
    height: 30,
    diameter: 12,
    riskLevel: 'medium'
  });
  const boundingBoxRefs = useRef({});
  
  // Initialize with detected trees
  useEffect(() => {
    if (detectedTrees && detectedTrees.length > 0) {
      const treesWithValidation = detectedTrees.map(tree => ({
        ...tree,
        validated: false, // Track whether this tree has been validated
        edited: false, // Track whether this tree has been edited
        visible: true // Track whether this tree should be displayed
      }));
      
      setTrees(treesWithValidation);
      setSelectedTree(treesWithValidation[0]);
      setCurrentTreeIndex(0);
      
      // Initialize form data from first tree
      const firstTree = treesWithValidation[0];
      if (firstTree) {
        setFormData({
          species: firstTree.species || 'Unknown Species',
          height: firstTree.height || 30,
          diameter: firstTree.diameter || 12,
          riskLevel: firstTree.risk_level || 'medium'
        });
      }
    }
  }, [detectedTrees]);
  
  // Filter trees based on search query
  useEffect(() => {
    if (searchQuery.trim() === '') {
      setFilteredTrees([]);
    } else {
      const query = searchQuery.toLowerCase();
      const filtered = trees.filter(tree => 
        (tree.visible) && (
          (tree.species && tree.species.toLowerCase().includes(query)) ||
          (tree.tree_species && tree.tree_species.toLowerCase().includes(query)) ||
          (tree.id && tree.id.toString().includes(query)) ||
          (tree.location && JSON.stringify(tree.location).toLowerCase().includes(query))
        )
      );
      setFilteredTrees(filtered);
    }
  }, [searchQuery, trees]);
  
  // Select a tree by index
  const selectTreeByIndex = (index) => {
    if (index >= 0 && index < trees.length) {
      setSelectedTree(trees[index]);
      setCurrentTreeIndex(index);
      
      // Update form data
      const selectedTree = trees[index];
      setFormData({
        species: selectedTree.species || 'Unknown Species',
        height: selectedTree.height || 30,
        diameter: selectedTree.diameter || 12,
        riskLevel: selectedTree.risk_level || 'medium'
      });
      
      // Center map on selected tree
      if (mapRef.current && selectedTree.location) {
        const [lng, lat] = selectedTree.location;
        mapRef.current.panTo({ lat, lng });
      }
      
      // Exit editing mode
      setIsEditing(false);
      setEditingBounds(null);
    }
  };
  
  // Handle next/previous tree
  const goToNextTree = () => {
    selectTreeByIndex((currentTreeIndex + 1) % trees.length);
  };
  
  const goToPreviousTree = () => {
    selectTreeByIndex((currentTreeIndex - 1 + trees.length) % trees.length);
  };
  
  // Handle input changes in form
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Save the current tree edits
  const saveTreeEdits = () => {
    if (!selectedTree) return;
    
    const updatedTrees = trees.map(tree => {
      if (tree.id === selectedTree.id) {
        return {
          ...tree,
          species: formData.species,
          height: parseInt(formData.height),
          diameter: parseInt(formData.diameter),
          risk_level: formData.riskLevel,
          validated: true,
          edited: true
        };
      }
      return tree;
    });
    
    setTrees(updatedTrees);
    setSelectedTree({
      ...selectedTree,
      species: formData.species,
      height: parseInt(formData.height),
      diameter: parseInt(formData.diameter),
      risk_level: formData.riskLevel,
      validated: true,
      edited: true
    });
    
    setIsEditing(false);
  };
  
  // Validate current tree without editing and persist to ValidationQueue
  const validateTree = () => {
    if (!selectedTree) return;
    
    // Add validated status to the selected tree
    const updatedTrees = trees.map(tree => {
      if (tree.id === selectedTree.id) {
        return {
          ...tree,
          validated: true
        };
      }
      return tree;
    });
    
    setTrees(updatedTrees);
    setSelectedTree({
      ...selectedTree,
      validated: true
    });
    
    // Prepare tree data with properly formatted fields that Tree Inventory expects
    const validatedTreeData = {
      ...selectedTree,
      validated: true,
      validation_timestamp: Date.now(),
      tree_species: selectedTree.species || selectedTree.tree_species || 'Unknown Species',
      tree_height: selectedTree.height || selectedTree.tree_height || 30,
      tree_diameter: selectedTree.diameter || selectedTree.tree_diameter || 12,
      risk_level: selectedTree.risk_level || 'medium',
      status: 'approved',
      property_type: 'Residential',
      proximity: 'Near structure',
      // Add risk factors for Tree Inventory presentation
      riskFactors: selectedTree.riskFactors || [
        { 
          level: selectedTree.risk_level || 'medium', 
          description: `AI detected risk level: ${selectedTree.risk_level || 'medium'}`
        }
      ],
      // Add location description if missing
      location: selectedTree.location ? {
        ...((typeof selectedTree.location === 'object' && !Array.isArray(selectedTree.location)) 
          ? selectedTree.location 
          : { coordinates: selectedTree.location }),
        description: `Tree group at coordinates [${selectedTree.location[0].toFixed(6)}, ${selectedTree.location[1].toFixed(6)}]`
      } : selectedTree.location,
      // Indicate this is part of a tree group for segmentation compatibility
      tree_group_id: selectedTree.tree_group_id || `group-${Date.now()}-${Math.floor(Math.random() * 1000)}`
    };
    
    // Persist the validated tree to the Tree Inventory through ValidationQueue
    try {
      // Create a custom event to send the validated tree to ValidationQueue
      const validationEvent = new CustomEvent('addValidatedTree', {
        detail: {
          tree: validatedTreeData,
          source: 'treeValidation'
        }
      });
      window.dispatchEvent(validationEvent);
      console.log("Tree validation data sent to Tree Inventory:", validatedTreeData);
    } catch (error) {
      console.error("Error sending validated tree to Tree Inventory:", error);
    }
    
    // Go to next tree
    goToNextTree();
  };
  
  // Remove current tree from validation list
  const removeTree = () => {
    if (!selectedTree) return;
    
    if (window.confirm('Are you sure you want to remove this tree from detection results?')) {
      const updatedTrees = trees.map(tree => {
        if (tree.id === selectedTree.id) {
          return {
            ...tree,
            visible: false  // Mark as not visible instead of removing completely
          };
        }
        return tree;
      });
      
      setTrees(updatedTrees);
      
      // Select next tree
      const visibleTrees = updatedTrees.filter(t => t.visible);
      const nextIndex = Math.min(currentTreeIndex, visibleTrees.length - 1);
      
      if (nextIndex >= 0) {
        selectTreeByIndex(nextIndex);
      } else {
        setSelectedTree(null);
        setCurrentTreeIndex(-1);
      }
    }
  };
  
  // Enter edit mode for bounding box
  const startEditing = () => {
    setIsEditing(true);
    
    if (selectedTree && selectedTree.bbox) {
      setEditingBounds(selectedTree.bbox);
    }
  };
  
  // Function to fetch satellite image for a location
  const fetchStaticMapImage = async (location, zoom = 19) => {
    try {
      // Configure for Google Static Maps API
      const apiKey = 'AIzaSyDEPTUGZu7ZqNcyM4dGaPKrD8yyQgDrk3g'; // This is a placeholder, use your actual API key
      const size = '600x400';
      const mapType = 'satellite';
      const scale = 2; // For higher resolution
      
      // Format location coordinates
      const center = Array.isArray(location) 
        ? `${location[1]},${location[0]}` // [lng, lat] format
        : `${location.lat},${location.lng}`; // {lat, lng} format
      
      // Build Google Static Maps URL
      const staticMapUrl = `https://maps.googleapis.com/maps/api/staticmap?center=${center}&zoom=${zoom}&size=${size}&maptype=${mapType}&scale=${scale}&key=${apiKey}`;
      
      console.log("Fetching satellite image for location:", center);
      
      // Return the URL without making the actual request from frontend
      return staticMapUrl;
    } catch (error) {
      console.error("Error generating static map URL:", error);
      return null;
    }
  };
  
  // Export selected trees to the zarr persistence layer
  const exportSelectedTrees = async () => {
    // If in Reports tab, export the selected reports
    if (isFeatureSelectionMode && activeTab === 'reports') {
      if (selectedReports.length === 0) {
        alert('No reports selected. Please select at least one report to export.');
        return;
      }
      
      // Get the selected validation items
      const reportsToExport = validationItems.filter(item => selectedReports.includes(item.id));
      
      console.log(`Exporting ${reportsToExport.length} selected reports`);
      
      // Enhance reports with satellite imagery where possible
      for (const report of reportsToExport) {
        if (report.location) {
          try {
            // Get satellite image URL for this location
            const imageUrl = await fetchStaticMapImage(report.location);
            if (imageUrl) {
              report.satelliteImageUrl = imageUrl;
              console.log(`Added satellite image URL to report ${report.id}`);
            }
          } catch (error) {
            console.error(`Failed to add satellite image for report ${report.id}:`, error);
          }
        }
      }
      
      // Create an export event that will be handled by the persistence layer
      const exportEvent = new CustomEvent('exportReports', {
        detail: {
          reports: reportsToExport,
          source: 'treeInventory'
        }
      });
      window.dispatchEvent(exportEvent);
      
      // Show success message
      alert(`Successfully exported ${reportsToExport.length} reports with satellite imagery.`);
      
      // Clear selection after export
      setSelectedReports([]);
      
      return;
    }
    
    // If in Object Detection mode (not feature selection mode)
    if (!isFeatureSelectionMode) {
      // Save validated trees to zarr database
      const validatedTrees = trees
        .filter(tree => tree.visible && tree.validated)
        .map(({ validated, edited, visible, ...tree }) => tree);
      
      const unvalidatedCount = trees.filter(tree => tree.visible && !tree.validated).length;
      
      if (unvalidatedCount > 0) {
        if (!window.confirm(`There are ${unvalidatedCount} unvalidated trees. Do you want to continue anyway?`)) {
          return;
        }
      }
      
      if (validatedTrees.length === 0) {
        alert('No trees to save. Please validate at least one tree.');
        return;
      }
      
      // Show a loading indicator
      const loadingMessage = `Enhancing ${validatedTrees.length} tree records with satellite imagery...`;
      console.log(loadingMessage);
      
      try {
        // Enhance trees with satellite imagery
        for (const tree of validatedTrees) {
          if (tree.location) {
            try {
              // Get satellite image URL for this location
              const imageUrl = await fetchStaticMapImage(tree.location);
              if (imageUrl) {
                tree.satelliteImageUrl = imageUrl;
                
                // Also store coordinates info for Gemini API
                tree.coordsInfo = {
                  center: tree.location,
                  zoom: 19,
                  mapType: 'satellite'
                };
                
                console.log(`Added satellite imagery for tree at [${tree.location[0]}, ${tree.location[1]}]`);
              }
            } catch (error) {
              console.error(`Failed to add satellite image for tree:`, error);
            }
          }
        }
        
        // Create an event to save the trees directly to zarr database
        const saveEvent = new CustomEvent('saveTreesToZarr', {
          detail: {
            trees: validatedTrees,
            source: 'objectDetection',
            target: 'zarr',  // Specify the destination as zarr/
            timestamp: Date.now(),
            includeImagery: true  // Flag to indicate satellite imagery is included
          }
        });
        window.dispatchEvent(saveEvent);
        
        // Generate results.json file for Gemini integration
        const results = {
          timestamp: Date.now(),
          job_id: `gemini_${Date.now()}`,
          tree_count: validatedTrees.length,
          trees: validatedTrees.map(tree => ({
            ...tree,
            risk_factors: tree.risk_factors || [],
            detection_method: 'gemini',
            satellite_imagery_url: tree.satelliteImageUrl
          }))
        };
        
        // Save results to backend API
        try {
          const apiEndpoint = '/api/gemini/save-results';
          const response = await fetch(apiEndpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              results: results,
              timestamp: Date.now()
            })
          });
          
          const responseData = await response.json();
          if (responseData.success) {
            console.log(`Successfully saved Gemini results to backend: ${responseData.job_id}`);
          } else {
            console.error(`Error saving Gemini results: ${responseData.message}`);
          }
        } catch (error) {
          console.error("API call to save Gemini results failed:", error);
        }
        
        // Don't automatically open the Video sidebar anymore
        console.log('Tree detection completed successfully, but not automatically opening Video sidebar');
        
        // Show success message
        alert(`Successfully saved ${validatedTrees.length} trees with satellite imagery to the database.`);
        
        // Also call the original save method for backward compatibility
        onSaveTrees(validatedTrees);
      } catch (error) {
        console.error("Error during tree export:", error);
        alert(`Error during export: ${error.message}`);
      }
      
      return;
    }
      
    // If in Trees tab of Database sidebar, export validated trees
    const validatedTrees = trees
      .filter(tree => tree.visible && tree.validated)
      .map(({ validated, edited, visible, ...tree }) => tree);
    
    const unvalidatedCount = trees.filter(tree => tree.visible && !tree.validated).length;
    
    if (unvalidatedCount > 0) {
      if (!window.confirm(`There are ${unvalidatedCount} unselected trees. Do you want to continue anyway?`)) {
        return;
      }
    }
    
    if (validatedTrees.length === 0) {
      alert('No trees to export. Please select at least one tree.');
      return;
    }
    
    // Create an export event that will be handled by the persistence layer
    const exportEvent = new CustomEvent('exportSelectedTrees', {
      detail: {
        trees: validatedTrees,
        source: 'treeInventory',
        format: 'report'
      }
    });
    window.dispatchEvent(exportEvent);
    
    // No need to call onSaveTrees for the Database sidebar reports export
  };
  
  // Calculate progress
  const validatedCount = trees.filter(tree => tree.visible && tree.validated).length;
  const totalVisibleTrees = trees.filter(tree => tree.visible).length;
  const progressPercentage = totalVisibleTrees > 0 
    ? Math.round((validatedCount / totalVisibleTrees) * 100) 
    : 0;
  
  // Add state to track header collapsed state
  const [headerCollapsed, setHeaderCollapsed] = useState(true);
  
  // Listen for header collapse events
  useEffect(() => {
    const handleHeaderCollapse = (event) => {
      setHeaderCollapsed(event.detail.collapsed);
      
      // When header state changes, also update sidebar position immediately
      const detectionSidebar = document.getElementById('tree-detection-sidebar');
      if (detectionSidebar) {
        detectionSidebar.style.top = event.detail.collapsed ? '40px' : '64px';
      }
    };
    
    window.addEventListener('headerCollapse', handleHeaderCollapse);
    
    // Initialize with current header state on mount
    try {
      const header = document.querySelector('header');
      if (header) {
        const isCollapsed = header.classList.contains('h-10');
        setHeaderCollapsed(isCollapsed);
      }
    } catch (e) {
      console.error("Error checking header state:", e);
    }
    
    return () => {
      window.removeEventListener('headerCollapse', handleHeaderCollapse);
    };
  }, []);

  return (
    <div 
      id="tree-detection-sidebar"
      className={`fixed ${headerCollapsed ? 'top-10' : 'top-16'} right-0 bottom-0 w-96 z-50 shadow-xl transition-all duration-300`}
    >
      {/* Removed collapse functionality */}
      
      {/* Main sidebar content */}
      <div className="h-full bg-white bg-opacity-98 shadow-lg flex flex-col transition-all ease-in-out duration-200">
      {/* Header */}
      <div className={`h-10 py-2 px-4 flex justify-between items-center ${isFeatureSelectionMode ? 'bg-emerald-600' : 'bg-blue-600'} text-white transition-all duration-300 border-b border-gray-200`}>
        <div className="flex items-center">
          <div className={`h-6 w-6 rounded-md ${isFeatureSelectionMode ? 'bg-emerald-700' : 'bg-blue-700'} flex items-center justify-center mr-2`}>
            {isFeatureSelectionMode ? <Database size={14} /> : <BarChart size={14} />}
          </div>
          <span className="font-medium text-sm">{isFeatureSelectionMode ? 'Database' : 'Detection'}</span>
        </div>
        <div className="flex items-center space-x-2">
          {!isFeatureSelectionMode && (
            <button 
              onClick={() => setConfigVisible(!configVisible)}
              className={`p-0.5 rounded hover:bg-blue-700 ${configVisible ? 'bg-blue-700' : ''}`}
              title="Gemini Configuration"
            >
              <Settings size={14} className="text-white" />
            </button>
          )}
          {isFeatureSelectionMode && (
            <button 
              onClick={() => {
                // Toggle or create a search panel
                const searchPanel = document.getElementById('database-search-panel');
                if (searchPanel) {
                  searchPanel.classList.toggle('hidden');
                  searchPanel.querySelector('input')?.focus();
                }
              }}
              className={`p-0.5 rounded hover:bg-emerald-700`}
              title="Search Trees"
            >
              <Search size={14} className="text-white" />
            </button>
          )}
          <button 
            onClick={onExitValidation}
            className={`p-0.5 rounded ${isFeatureSelectionMode ? 'hover:bg-emerald-700' : 'hover:bg-blue-700'}`}
            title="Close panel"
          >
            <X size={14} className="text-white" />
          </button>
        </div>
      </div>
      
      {/* Database Search Panel */}
      {isFeatureSelectionMode && (
        <div id="database-search-panel" className="px-4 py-3 border-b bg-emerald-50 hidden">
          <h3 className="text-sm font-medium text-emerald-700 mb-2">Search Tree Database</h3>
          <div className="relative">
            <input
              type="text"
              placeholder="Search by species, location, or ID..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                const query = e.target.value.toLowerCase();
                if (query.trim() === '') {
                  setFilteredTrees([]);
                } else {
                  // Filter trees based on search query
                  const filtered = trees.filter(tree => 
                    (tree.species && tree.species.toLowerCase().includes(query)) ||
                    (tree.tree_species && tree.tree_species.toLowerCase().includes(query)) ||
                    (tree.id && tree.id.toString().includes(query)) ||
                    (tree.location && JSON.stringify(tree.location).toLowerCase().includes(query))
                  );
                  setFilteredTrees(filtered);
                }
              }}
              className="w-full p-2 border border-emerald-300 rounded-md text-sm bg-white text-emerald-800 focus:ring-emerald-500 focus:border-emerald-500 pl-8"
            />
            <Search className="absolute left-2 top-2.5 h-4 w-4 text-emerald-500" />
            {searchQuery && (
              <button 
                onClick={() => {
                  setSearchQuery('');
                  setFilteredTrees([]);
                }}
                className="absolute right-2 top-2.5 text-gray-400 hover:text-gray-600"
              >
                <X size={16} />
              </button>
            )}
          </div>
          <div className="flex justify-between mt-2 text-xs text-emerald-600">
            <span>Found: {searchQuery ? filteredTrees.length : trees.length} trees</span>
            <button 
              onClick={() => {
                const searchPanel = document.getElementById('database-search-panel');
                if (searchPanel) searchPanel.classList.add('hidden');
              }}
              className="hover:underline"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Gemini Configuration Panel */}
      {!isFeatureSelectionMode && configVisible && (
        <div className="px-4 py-3 border-b bg-blue-50">
          <h3 className="text-sm font-medium text-blue-700 mb-2">Gemini Detection Settings</h3>
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-blue-600 mb-1">Detection Threshold</label>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.1"
                value={geminiParams.detectionThreshold}
                onChange={(e) => setGeminiParams({...geminiParams, detectionThreshold: parseFloat(e.target.value)})}
                className="w-full h-2 bg-blue-200 rounded-lg appearance-none cursor-pointer"
              />
              <div className="flex justify-between text-xs text-blue-600">
                <span>0.1</span>
                <span>{geminiParams.detectionThreshold}</span>
                <span>0.9</span>
              </div>
            </div>
            
            <div>
              <label className="block text-xs text-blue-600 mb-1">Max Trees</label>
              <input
                type="number"
                min="5"
                max="50"
                value={geminiParams.maxTrees}
                onChange={(e) => setGeminiParams({...geminiParams, maxTrees: parseInt(e.target.value)})}
                className="w-full p-2 border border-blue-300 rounded-md text-sm bg-white text-blue-800 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            
            <div className="flex items-center">
              <input
                type="checkbox"
                id="includeRiskAnalysis"
                checked={geminiParams.includeRiskAnalysis}
                onChange={(e) => setGeminiParams({...geminiParams, includeRiskAnalysis: e.target.checked})}
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-blue-300 rounded"
              />
              <label htmlFor="includeRiskAnalysis" className="ml-2 block text-sm text-blue-700">
                Include Risk Analysis
              </label>
            </div>
            
            <div>
              <label className="block text-xs text-blue-600 mb-1">Detail Level</label>
              <select
                value={geminiParams.detailLevel}
                onChange={(e) => setGeminiParams({...geminiParams, detailLevel: e.target.value})}
                className="w-full p-2 border border-blue-300 rounded-md text-sm bg-white text-blue-800 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
            </div>
            
            <div className="pt-2">
              <button
                onClick={() => {
                  // Here you would dispatch an event or call a function to apply the Gemini parameters
                  window.dispatchEvent(new CustomEvent('updateGeminiParams', { 
                    detail: geminiParams 
                  }));
                  alert(`Configuration updated:\n${JSON.stringify(geminiParams, null, 2)}`);
                }}
                className="w-full py-2 px-3 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-md shadow-sm transition-colors duration-150"
              >
                Apply Settings
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Detection Button - only show in Object Detection mode */}
      {!isFeatureSelectionMode && (
        <div className="px-4 pt-3 pb-2 border-b">
          <button 
            className={`w-full py-1.5 bg-blue-50 text-blue-600 hover:bg-blue-100 border border-blue-200 text-sm font-medium rounded-md transition-all ${detectButtonDisabled ? 'opacity-70 cursor-not-allowed' : 'hover:shadow-sm'}`}
            disabled={detectButtonDisabled}
            onClick={() => {
              // Disable the button to prevent multiple clicks
              setDetectButtonDisabled(true);
              setDetectButtonText('Detecting Trees...');
              
              // Get Gemini settings from localStorage if available
              let geminiSettings = {
                detectionThreshold: 0.7,
                maxTrees: 20,
                includeRiskAnalysis: true,
                detailLevel: 'high'
              };
              
              try {
                const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
                if (savedSettings) {
                  const settings = JSON.parse(savedSettings);
                  if (settings.geminiSettings) {
                    geminiSettings = { ...geminiSettings, ...settings.geminiSettings };
                  }
                }
              } catch (e) {
                console.error("Error loading Gemini settings:", e);
              }
              
              // Trigger tree detection which will be handled by handleExportData in MapControls
              // Include a flag to use real Gemini results
              window.dispatchEvent(new CustomEvent('openTreeDetection', {
                detail: {
                  useSatelliteImagery: true,
                  geminiParams: geminiSettings,
                  useRealGemini: true, // Flag to use real Gemini results
                  saveToResponseJson: true // Flag to save results to response.json
                }
              }));
              
              // Use a timeout to simulate a delay in case the event doesn't trigger a response
              setTimeout(() => {
                if (detectButtonText === 'Detecting Trees...') {
                  setDetectButtonText('Detect Trees');
                  setDetectButtonDisabled(false);
                }
              }, 30000); // 30 second timeout
            }}
          >
            {detectButtonText}
          </button>
        </div>
      )}
      
      {/* Progress bar */}
      {isFeatureSelectionMode ? (
        // For Tree Inventory mode, show tabs for Trees and Reports
        <div>
          <div className="flex border-b">
            <button
              className={`flex-1 px-4 py-2 text-center font-medium text-sm ${
                activeTab === 'trees' 
                  ? 'border-b-2 border-emerald-600 text-emerald-600' 
                  : 'text-gray-500 hover:text-emerald-600'
              }`}
              onClick={() => setActiveTab('trees')}
            >
              Index
            </button>
            <button
              className={`flex-1 px-4 py-2 text-center font-medium text-sm ${
                activeTab === 'reports' 
                  ? 'border-b-2 border-emerald-600 text-emerald-600' 
                  : 'text-gray-500 hover:text-emerald-600'
              }`}
              onClick={() => setActiveTab('reports')}
            >
              Reports
            </button>
          </div>
          
          {activeTab === 'trees' ? (
            <div className={`px-4 py-2 ${isFeatureSelectionMode ? 'bg-emerald-50' : 'bg-blue-50'}`}>
              <div className="flex justify-between items-center text-sm mb-1">
                <span>{isFeatureSelectionMode ? 'Selected' : 'Progress'}: {validatedCount} of {totalVisibleTrees} trees</span>
                <span className={`font-medium ${isFeatureSelectionMode ? 'text-emerald-600' : 'text-blue-600'}`}>{progressPercentage}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`${isFeatureSelectionMode ? 'bg-emerald-600' : 'bg-blue-600'} rounded-full h-2`}
                  style={{ width: `${progressPercentage}%` }}
                ></div>
              </div>
            </div>
          ) : (
            <div className="px-4 py-2 bg-emerald-50">
              <div className="flex justify-between items-center text-sm mb-1">
                <span>Selected: {selectedReports.length} of {validationItems.length} reports</span>
                <span className="font-medium text-emerald-600">
                  {validationItems.length > 0 
                    ? Math.round((selectedReports.length / validationItems.length) * 100) 
                    : 0}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-emerald-600 rounded-full h-2"
                  style={{ 
                    width: `${validationItems.length > 0 
                      ? Math.round((selectedReports.length / validationItems.length) * 100) 
                      : 0}%` 
                  }}
                ></div>
              </div>
            </div>
          )}
        </div>
      ) : (
        // For Object Detection mode, just show the progress
        <div className={`px-4 py-2 bg-blue-50`}>
          <div className="flex justify-between items-center text-sm mb-1">
            <span>Progress: {validatedCount} of {totalVisibleTrees} areas</span>
            <span className="font-medium text-blue-600">{progressPercentage}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 rounded-full h-2"
              style={{ width: `${progressPercentage}%` }}
            ></div>
          </div>
        </div>
      )}
      
      {/* Tree navigation */}
      <div className="p-4 flex items-center justify-between border-b">
        <button 
          onClick={goToPreviousTree}
          className={`p-1 rounded ${isFeatureSelectionMode ? 'hover:bg-emerald-100 text-emerald-700' : 'hover:bg-blue-100 text-blue-700'}`}
          disabled={trees.length === 0}
        >
          <ChevronLeft size={20} />
        </button>
        
        <div className="text-center">
          <span className={`text-sm ${isFeatureSelectionMode ? 'text-emerald-800' : 'text-blue-800'}`}>
            {isFeatureSelectionMode ? 
              (activeTab === 'reports' ? 'Report' : 'Tree') 
              : 'Area'} {trees.length > 0 ? currentTreeIndex + 1 : 0} of {trees.filter(t => t.visible).length}
          </span>
          <div className="text-xs text-gray-500">
            {selectedTree?.validated 
              ? <span className="text-green-500 flex items-center justify-center">
                  <Check size={12} className="mr-1" />
                  {isFeatureSelectionMode ? 'Selected' : 'Validated'}
                </span>
              : <span className={`${isFeatureSelectionMode ? 'text-emerald-400' : 'text-amber-500'} flex items-center justify-center`}>
                  <AlertTriangle size={12} className="mr-1" />
                  {isFeatureSelectionMode ? 'Requires Selection' : 'Needs Validation'}
                </span>
            }
          </div>
        </div>
        
        <button 
          onClick={goToNextTree}
          className={`p-1 rounded ${isFeatureSelectionMode ? 'hover:bg-emerald-100 text-emerald-700' : 'hover:bg-blue-100 text-blue-700'}`}
          disabled={trees.length === 0}
        >
          <ChevronRight size={20} />
        </button>
      </div>
      
      {/* Thin separator line */}
      <div className={`border-t ${isFeatureSelectionMode ? 'border-emerald-100' : 'border-blue-100'}`}></div>
      
      {/* Main content - either Trees or Reports */}
      {activeTab === 'trees' && (
        <div className="flex-1 overflow-auto">
          {trees.length === 0 ? (
            <div className="text-center pt-8 pb-8 text-gray-500 flex flex-col items-center">
              <span>
                {isFeatureSelectionMode ? 
                  "No index available. Please run Detection." :
                  "No trees available. Please run Detect Trees."}
              </span>
              {/* Thin separator line between messages */}
              <div className="border-t border-gray-200 w-1/2 my-4"></div>
              <span className="text-sm text-gray-400">
                Run Detection to find trees in the current view.
              </span>
            </div>
          ) : searchQuery && filteredTrees.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              No trees match your search query "{searchQuery}".
            </div>
          ) : (
            <div className="space-y-3 p-4">
              {(searchQuery ? filteredTrees : trees.filter(tree => tree.visible)).map((tree, index) => (
                <div 
                  key={tree.id || index}
                  className={`p-3 rounded-lg ${
                    selectedTree && selectedTree.id === tree.id 
                      ? isFeatureSelectionMode ? 'bg-emerald-50 border border-emerald-200' : 'bg-blue-50 border border-blue-200'
                      : 'bg-white border border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-medium">
                        {tree.species || tree.tree_species || "Unknown Species"} - {tree.height || tree.tree_height || 0}ft
                      </h3>
                      <div className="text-xs text-gray-500 mt-1">
                        {tree.location && Array.isArray(tree.location) ? (
                          <div className="flex items-center">
                            <MapPin className="h-3 w-3 mr-1" />
                            {`[${tree.location[0].toFixed(6)}, ${tree.location[1].toFixed(6)}]`}
                          </div>
                        ) : tree.location && tree.location.description ? (
                          <div className="flex items-center">
                            <MapPin className="h-3 w-3 mr-1" />
                            {tree.location.description}
                          </div>
                        ) : null}
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setSelectedTree(tree);
                        setCurrentTreeIndex(index);
                      }}
                      className={`px-2 py-1 text-xs ${
                        tree.validated 
                          ? 'bg-green-100 text-green-700' 
                          : isFeatureSelectionMode ? 'bg-emerald-100 text-emerald-700' : 'bg-blue-100 text-blue-700'
                      } rounded`}
                    >
                      {tree.validated 
                        ? (isFeatureSelectionMode ? 'Selected' : 'Validated') 
                        : (isFeatureSelectionMode ? 'Select' : 'Validate')}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {isFeatureSelectionMode && activeTab === 'reports' ? (
        // REPORTS VIEW
        <div className="flex-1 overflow-auto">
          {validationItems.length === 0 ? (
            <div className="p-4 flex-1 flex items-center justify-center text-gray-500">
              No reports available. Create report from Index.
            </div>
          ) : (
            <div className="flex flex-col">
              <div className="divide-y">
                {validationItems.map((item) => (
                  <div key={item.id} className="p-3 hover:bg-gray-50">
                    <div className="flex justify-between items-start mb-2">
                      <div>
                        <h3 className="font-medium">
                          {item.tree_species} - {item.tree_height}ft 
                          {item.tree_diameter && ` - Diameter: ${item.tree_diameter}in`}
                        </h3>
                        <div className="text-sm text-gray-500 mb-1">
                          <div className="flex items-start">
                            <MapPin className="h-3 w-3 mr-1 mt-0.5" />
                            <span>
                              {item.location?.description || 
                               `Tree at [${item.location?.[0]?.toFixed(6) || '0'}, ${item.location?.[1]?.toFixed(6) || '0'}]`}
                            </span>
                          </div>
                        </div>
                        <div className="flex space-x-2 mt-1">
                          {item.status === 'pending' && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-yellow-100 text-yellow-800">
                              <Clock className="h-3 w-3 mr-1" />
                              Pending
                            </span>
                          )}
                          {item.status === 'approved' && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-100 text-green-800">
                              <CheckCircle className="h-3 w-3 mr-1" />
                              Approved
                            </span>
                          )}
                          {item.status === 'rejected' && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-red-100 text-red-800">
                              <XCircle className="h-3 w-3 mr-1" />
                              Rejected
                            </span>
                          )}
                          
                          {/* Risk level badges */}
                          {(item.risk_level === 'high' || (item.riskFactors && item.riskFactors.some(f => f.level === 'high'))) && (
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-red-100 text-red-800">
                              <AlertTriangle className="h-3 w-3 mr-1" />
                              High Risk
                            </span>
                          )}
                        </div>
                      </div>
                      
                      {/* Report selection checkbox */}
                      <div className="flex items-center">
                        <label className="inline-flex items-center">
                          <input
                            type="checkbox"
                            className="form-checkbox h-4 w-4 text-emerald-600 border-gray-300 rounded"
                            checked={selectedReports.includes(item.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedReports(prev => [...prev, item.id]);
                              } else {
                                setSelectedReports(prev => prev.filter(id => id !== item.id));
                              }
                            }}
                          />
                        </label>
                      </div>
                    </div>
                    
                    {/* Tree attributes (condensed) */}
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs mt-2">
                      <div className="text-gray-700">
                        <span className="font-medium">Size:</span> {
                          item.tree_height <= 30 ? 'Short (<30 ft)' : 
                          item.tree_height <= 50 ? 'Medium (30-50 ft)' : 
                          'Tall (>50 ft)'
                        }
                      </div>
                      <div className="text-gray-700">
                        <span className="font-medium">DBH:</span> {item.tree_diameter || '?'} inches
                      </div>
                      <div className="text-gray-700">
                        <span className="font-medium">Property:</span> {item.property_type || 'Residential'}
                      </div>
                      <div className="text-gray-700">
                        <span className="font-medium">Proximity:</span> {item.proximity || 'Near structure'}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              {/* Thin grey separator at the bottom of reports list */}
              <div className="border-t border-gray-200 mt-4"></div>
            </div>
          )}
        </div>
      ) : (
        // TREES VIEW
        <>
          {/* Thin grey separator between list and details */}
          <div className="border-t border-gray-200"></div>
          
          {/* Selected tree details & form */}
          {selectedTree ? (
            <div className="p-4 flex-1 overflow-y-auto">
              <div className="mb-4">
                <div className="text-sm font-medium mb-1">Tree Details</div>
                
                {isEditing ? (
                  // Edit form
                  <div className="space-y-3">
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Species</label>
                      <select
                        name="species"
                        className="w-full p-2 border rounded focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                        value={formData.species}
                        onChange={handleInputChange}
                      >
                        <option value="Unknown Species">Unknown Species</option>
                        <option value="Live Oak">Live Oak</option>
                        <option value="Post Oak">Post Oak</option>
                        <option value="Bald Cypress">Bald Cypress</option>
                        <option value="Red Oak">Red Oak</option>
                        <option value="Southern Magnolia">Southern Magnolia</option>
                        <option value="Cedar Elm">Cedar Elm</option>
                        <option value="Pecan">Pecan</option>
                        <option value="Shumard Oak">Shumard Oak</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Height (ft)</label>
                      <input
                        type="number"
                        name="height"
                        className="w-full p-2 border rounded focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                        value={formData.height}
                        onChange={handleInputChange}
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Diameter (in)</label>
                      <input
                        type="number"
                        name="diameter"
                        className="w-full p-2 border rounded focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                        value={formData.diameter}
                        onChange={handleInputChange}
                      />
                    </div>
                    
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Risk Level</label>
                      <select
                        name="riskLevel"
                        className="w-full p-2 border rounded focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                        value={formData.riskLevel}
                        onChange={handleInputChange}
                      >
                        <option value="low">Low Risk</option>
                        <option value="medium">Medium Risk</option>
                        <option value="high">High Risk</option>
                      </select>
                    </div>
                    
                    {/* Bounding box editing placeholder - in a real implementation, this would include 
                        interactive controls to adjust the bounding box on the map */}
                    <div className="text-xs text-gray-600 italic">
                      Bounding box editing would be implemented here in a real application,
                      allowing users to drag corners of the box to adjust it
                    </div>
                    
                    <div className="flex space-x-2 mt-4">
                      <button
                        onClick={() => setIsEditing(false)}
                        className="bg-gray-100 hover:bg-gray-200 text-gray-800 px-4 py-2 rounded text-sm"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={saveTreeEdits}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded text-sm"
                      >
                        Save Changes
                      </button>
                    </div>
                  </div>
                ) : (
                  // View mode
                  <div className="bg-gray-50 p-3 rounded border">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <div className="text-xs text-gray-500">Species</div>
                        <div>{selectedTree.species || 'Unknown Species'}</div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500">Height</div>
                        <div>{selectedTree.height || 'Unknown'} ft</div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500">Diameter</div>
                        <div>{selectedTree.diameter || 'Unknown'} in</div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500">Risk Level</div>
                        <div className={selectedTree.risk_level === 'high' 
                          ? 'text-red-600 font-medium' 
                          : selectedTree.risk_level === 'medium' 
                            ? 'text-orange-600 font-medium' 
                            : 'text-green-600 font-medium'
                        }>
                          {selectedTree.risk_level === 'high' 
                            ? 'High Risk' 
                            : selectedTree.risk_level === 'medium' 
                              ? 'Medium Risk' 
                              : 'Low Risk'
                          }
                        </div>
                      </div>
                    </div>
                    
                    <div className="mt-3 pt-3 border-t text-xs text-gray-600">
                      <div>ML Detection Confidence: {selectedTree.confidence 
                        ? `${(selectedTree.confidence * 100).toFixed(1)}%` 
                        : 'Unknown'
                      }</div>
                      {selectedTree.bbox && (
                        <div>Bounding Box: [{selectedTree.bbox.map(v => Math.round(v)).join(', ')}]</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
              
              {/* Action buttons */}
              {!isEditing && (
                <div className="space-y-2">
                  <button
                    onClick={validateTree}
                    className="flex items-center justify-center bg-green-100 hover:bg-green-200 text-green-700 p-2 rounded w-full"
                  >
                    <Check className="h-4 w-4 mr-2" />
                    <span>Validate & Continue</span>
                  </button>
                  
                  <button
                    onClick={startEditing}
                    className="flex items-center justify-center bg-blue-100 hover:bg-blue-200 text-blue-700 p-2 rounded w-full"
                  >
                    <Edit className="h-4 w-4 mr-2" />
                    <span>Edit Tree Data</span>
                  </button>
                  
                  <button
                    onClick={removeTree}
                    className="flex items-center justify-center bg-red-100 hover:bg-red-200 text-red-800 p-2 rounded w-full"
                  >
                    <Trash className="h-4 w-4 mr-2" />
                    <span>Remove Tree</span>
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="p-4 pt-8 pb-4 flex-1 flex flex-col items-center justify-center text-gray-500">
              {isFeatureSelectionMode ? 
                "No trees selected." : 
                "No areas selected."}
              {/* Thin separator line between messages */}
              <div className="border-t border-gray-200 w-1/2 my-4"></div>
              <span className="text-sm text-gray-400">
                {isFeatureSelectionMode ? 
                  "Select a tree from the list above." : 
                  "Select an area from the list above."}
              </span>
            </div>
          )}
        </>
      )}
      
      {/* Thin separator line before footer */}
      <div className={`border-t ${isFeatureSelectionMode ? 'border-emerald-100' : 'border-blue-100'}`}></div>
      
      {/* Footer */}
      <div className="p-4 bg-gray-50 border-t">
        {isFeatureSelectionMode && activeTab === 'reports' ? (
          // Reports export button
          <button
            onClick={exportSelectedTrees}
            className="flex items-center justify-center bg-emerald-600 hover:bg-emerald-700 text-white py-1.5 rounded-md w-full text-sm font-medium"
            disabled={selectedReports.length === 0}
          >
            <FileText className="h-4 w-4 mr-2" />
            <span>Export Reports ({selectedReports.length})</span>
          </button>
        ) : (
          // Trees button
          <button
            onClick={exportSelectedTrees}
            className={`flex items-center justify-center ${
              isFeatureSelectionMode 
              ? 'bg-emerald-600 hover:bg-emerald-700' 
              : 'bg-blue-600 hover:bg-blue-700'
            } text-white py-1.5 rounded-md w-full text-sm font-medium`}
            disabled={validatedCount === 0}
          >
            <Save className="h-4 w-4 mr-2" />
            <span>{isFeatureSelectionMode && activeTab === 'trees' 
              ? 'Create Report'
              : isFeatureSelectionMode && activeTab === 'reports'
              ? `Export Reports (${validatedCount})`
              : `Save for Review (${validatedCount})`}</span>
          </button>
        )}
        
        <div className="mt-2 text-xs text-gray-500 text-center">
          {isFeatureSelectionMode && activeTab === 'reports' ? (
            // Reports footer text
            selectedReports.length === 0 
              ? 'Select at least one report to export'
              : `${selectedReports.length} of ${validationItems.length} reports selected for export`
          ) : (
            // Trees footer text
            validatedCount === 0 
              ? `${isFeatureSelectionMode ? 'Select' : 'Validate'} at least one tree group to ${isFeatureSelectionMode ? 'export' : 'save to database'}`
              : `${validatedCount} of ${totalVisibleTrees} tree groups are ${isFeatureSelectionMode ? 'selected' : 'validated'}`
          )}
        </div>
      </div>
    </div>
    </div>
  );
};

export default TreeValidationMode;