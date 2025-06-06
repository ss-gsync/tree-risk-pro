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
    
    // Adjust the map container to prevent grey area
    const mapContainer = document.querySelector('#map-container');
    if (mapContainer) {
      mapContainer.style.right = newCollapsedState ? '0px' : `${width}px`;
    }
    
    // Dispatch validationSidebarToggle event to notify other components of this change
    window.dispatchEvent(new CustomEvent('validationSidebarToggle', {
      detail: {
        collapsed: newCollapsedState,
        source: 'tree_detection',
        width: newCollapsedState ? 0 : width
      }
    }));
    
    // Force window resize to update map rendering
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 50);
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
    }
    
    // Clear any existing detection data on initialization to prevent auto-rendering
    if (window.mlDetectionData) {
      console.log('DetectionSidebar: Clearing existing detection data to prevent auto-rendering');
      window.mlDetectionData = null;
    }
  }, [visibleTrees]);
  
  // =====================================================================
  // EMERGENCY FIX: COMPLETE REWRITE OF STATE MANAGEMENT TO PREVENT INFINITE UPDATES
  // =====================================================================
  
  // Use a static class variable to track if we've initialized settings
  // This prevents any React state updates from triggering re-renders
  if (!window._mlSettingsInitialized) {
    console.log('DetectionSidebar: ONE-TIME INITIALIZATION of global settings');
    
    // Initialize global settings
    window.mlOverlaySettings = {
      showOverlay: true,
      showSegmentation: true,
      opacity: 0.7
    };
    
    window.detectionShowOverlay = true;
    window._mlSettingsInitialized = true;
  }
  
  // Track last toggle timestamp to prevent duplicate events
  const lastToggleTimeRef = useRef(0);
  
  // Use event handler for updating settings via UI controls
  const handleToggleOverlay = (newValue) => {
    // Prevent duplicate calls within short time window (debounce)
    const now = Date.now();
    if (now - lastToggleTimeRef.current < 100) {
      console.log('DetectionSidebar: Ignoring rapid toggle to prevent infinite loop');
      return;
    }
    lastToggleTimeRef.current = now;
    
    // Only update the global state, NOT the React state
    window.mlOverlaySettings = {
      ...(window.mlOverlaySettings || {}),
      showOverlay: newValue
    };
    window.detectionShowOverlay = newValue;
    
    // Dispatch event with source to prevent infinite loops
    window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
      detail: { 
        showOverlay: newValue,
        showSegmentation: window.mlOverlaySettings.showSegmentation,
        opacity: window.mlOverlaySettings.opacity,
        source: 'detection_sidebar', // Add source to identify event origin
        timestamp: now // Add timestamp to make event unique
      }
    }));
    
    // Update DOM directly
    const overlayEl = document.getElementById('ml-detection-overlay');
    if (overlayEl) {
      overlayEl.style.display = newValue ? 'block' : 'none';
    }
  };
  
  // Track last segmentation toggle timestamp to prevent duplicate events
  const lastSegToggleTimeRef = useRef(0);
  
  // Handle segmentation toggle
  const handleToggleSegmentation = (newValue) => {
    // Prevent duplicate calls within short time window (debounce)
    const now = Date.now();
    if (now - lastSegToggleTimeRef.current < 100) {
      console.log('DetectionSidebar: Ignoring rapid segmentation toggle to prevent infinite loop');
      return;
    }
    lastSegToggleTimeRef.current = now;
    
    // Update global state
    window.mlOverlaySettings = {
      ...(window.mlOverlaySettings || {}),
      showSegmentation: newValue
    };
    
    // Dispatch event with source to prevent infinite loops
    window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
      detail: { 
        showOverlay: window.mlOverlaySettings.showOverlay,
        showSegmentation: newValue,
        opacity: window.mlOverlaySettings.opacity,
        source: 'detection_sidebar', // Add source to identify event origin
        timestamp: now // Add timestamp to make event unique
      }
    }));
    
    // Update DOM directly
    const masks = document.querySelectorAll('.segmentation-mask');
    masks.forEach(mask => {
      mask.style.display = newValue ? 'block' : 'none';
    });
  };
  
  // Track last opacity change timestamp to prevent duplicate events
  const lastOpacityChangeTimeRef = useRef(0);
  
  // Handle opacity change
  const handleOpacityChange = (newValue) => {
    // Prevent duplicate calls within short time window (debounce)
    const now = Date.now();
    if (now - lastOpacityChangeTimeRef.current < 100) {
      console.log('DetectionSidebar: Ignoring rapid opacity change to prevent infinite loop');
      return;
    }
    lastOpacityChangeTimeRef.current = now;
    
    // Update global state
    window.mlOverlaySettings = {
      ...(window.mlOverlaySettings || {}),
      opacity: newValue
    };
    
    // Dispatch event with source to prevent infinite loops
    window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
      detail: { 
        showOverlay: window.mlOverlaySettings.showOverlay,
        showSegmentation: window.mlOverlaySettings.showSegmentation,
        opacity: newValue,
        source: 'detection_sidebar', // Add source to identify event origin
        timestamp: now // Add timestamp to make event unique
      }
    }));
    
    // Update opacity directly
    if (typeof window.updateMLOverlayOpacity === 'function') {
      window.updateMLOverlayOpacity(newValue);
    } else if (typeof window.updateOverlayOpacity === 'function') {
      window.updateOverlayOpacity(newValue);
    }
    
    // Update overlay element directly
    const overlay = document.getElementById('ml-detection-overlay');
    if (overlay) {
      overlay.style.backgroundColor = `rgba(0, 30, 60, ${newValue})`;
    }
  };
  
  // Adjust top position based on header state
  const sidebarTopPosition = headerCollapsed ? '0px' : '60px';
  
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
      {/* Objects count display */}
      <div className="px-3 pt-2 mb-0">
        <div className="flex items-center justify-between">
          <span className="text-xs text-blue-900 font-medium">Objects Found:</span>
          <span className="text-xs bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full font-semibold" id="detected-objects-count">
            {filteredTrees.length}
          </span>
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
          Objects
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
              <h3 className="font-semibold text-sm mb-1">Detection Parameters</h3>
              
              <div className="space-y-2">
                <div>
                  <label className="text-sm text-gray-600 block mb-1">Detection Mode</label>
                  <select
                    className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm"
                    value={geminiParams.detailLevel}
                    onChange={(e) => setGeminiParams(prev => ({
                      ...prev,
                      detailLevel: e.target.value
                    }))}
                  >
                    <option value="high">High Quality (Slower)</option>
                    <option value="medium">Balanced</option>
                    <option value="low">Fast Detection</option>
                  </select>
                </div>
                
                
                {/* Use the React component for Detection Categories */}
                <DetectionCategories />
                
                {/* ML Overlay Controls (direct implementation) */}
                <div className="mt-4">
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
                                const newValue = !showOverlay;
                                console.log(`DetectionSidebar: Toggling overlay visibility to ${newValue ? 'visible' : 'hidden'}`);
                                
                                // Use our global handler instead of React state updates
                                handleToggleOverlay(newValue);
                                
                                // Also update the local state to keep the UI in sync
                                // This won't cause infinite loops since we don't update global state in useEffect
                                setShowOverlay(newValue);
                                
                                // Function to find and update the DETECTION badge
                                const updateBadgeVisibility = () => {
                                  // First try by ID
                                  const badge = document.getElementById('detection-debug');
                                  
                                  if (badge) {
                                    badge.style.display = newValue ? 'block' : 'none';
                                    badge.style.opacity = newValue ? '1' : '0';
                                    console.log(`DetectionSidebar: Badge visibility set to ${newValue ? 'visible' : 'hidden'} via ID`);
                                    return true;
                                  }
                                  
                                  // If not found by ID, try class-based selectors
                                  const badgeByClass = document.querySelector('.detection-badge');
                                  if (badgeByClass) {
                                    badgeByClass.style.display = newValue ? 'block' : 'none';
                                    badgeByClass.style.opacity = newValue ? '1' : '0';
                                    console.log(`DetectionSidebar: Badge visibility set to ${newValue ? 'visible' : 'hidden'} via class`);
                                    return true;
                                  }
                                  
                                  // If still not found, log warning
                                  console.warn('DetectionSidebar: Could not find detection badge to update visibility');
                                  return false;
                                };
                                
                                // Update badge visibility
                                updateBadgeVisibility();
                                
                                // Create badge if needed and overlay is visible
                                if (!updateBadgeVisibility() && newValue) {
                                  console.log('DetectionSidebar: Badge not found, creating new one');
                                  // Try to create the badge via event
                                  window.dispatchEvent(new CustomEvent('createDetectionBadge', {
                                    detail: { width: width, headerCollapsed: headerCollapsed }
                                  }));
                                }
                                
                                // Dispatch event for other components
                                window.dispatchEvent(new CustomEvent('mlOverlaySettingsChanged', {
                                  detail: { 
                                    showOverlay: newValue, 
                                    showSegmentation, 
                                    opacity: overlayOpacity
                                  }
                                }));
                                
                                console.log(`DetectionSidebar: ML Overlay and badge visibility update completed`);
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
                                const newValue = !showSegmentation;
                                
                                // Use global handler instead of React state updates
                                handleToggleSegmentation(newValue);
                                
                                // Also update local state to keep UI in sync
                                setShowSegmentation(newValue);
                                
                                console.log(`DetectionSidebar: Segmentation mask visibility set to ${newValue ? 'visible' : 'hidden'}`);
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
                          Overlay Opacity ({Math.round(overlayOpacity * 100)}%)
                        </label>
                        <input
                          id="opacity-slider"
                          type="range"
                          min="0"
                          max="1"
                          step="0.05"
                          value={window.mlOverlaySettings?.opacity || overlayOpacity}
                          onChange={(e) => {
                            const newValue = parseFloat(e.target.value);
                            
                            // Use global handler instead of React state updates
                            handleOpacityChange(newValue);
                            
                            // Also update local state to keep UI in sync
                            setOverlayOpacity(newValue);
                            
                            console.log(`DetectionSidebar: Updating overlay opacity to ${newValue}`);
                          }}
                          onMouseUp={() => {
                            // Use our handler for the final update when the slider is released
                            handleOpacityChange(overlayOpacity);
                          }}
                          className="w-full"
                        />
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Detection preview is now a modal popup, not rendered in sidebar */}
                
                <div>
                  <label className="text-sm text-gray-600 block mb-1">Analytics Categories</label>
                  <div className="grid grid-cols-2 gap-1">
                    <div className="flex items-center text-xs bg-gray-50 rounded p-1">
                      <input 
                        type="checkbox" 
                        id="risk-analysis" 
                        checked={geminiParams.includeRiskAnalysis} 
                        onChange={(e) => setGeminiParams(prev => ({
                          ...prev,
                          includeRiskAnalysis: e.target.checked
                        }))}
                        className="mr-1 h-3 w-3" 
                      />
                      <label htmlFor="risk-analysis" className="text-xs">Risk Assessment</label>
                    </div>
                    <div className="flex items-center text-xs bg-gray-50 rounded p-1">
                      <input 
                        type="checkbox" 
                        id="species-detection" 
                        defaultChecked={true} 
                        className="mr-1 h-3 w-3" 
                      />
                      <label htmlFor="species-detection" className="text-xs">Species Detection</label>
                    </div>
                    <div className="flex items-center text-xs bg-gray-50 rounded p-1">
                      <input 
                        type="checkbox" 
                        id="health-assessment" 
                        defaultChecked={true} 
                        className="mr-1 h-3 w-3" 
                      />
                      <label htmlFor="health-assessment" className="text-xs">Health Assessment</label>
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
                  className="w-full border border-gray-300 rounded text-xs py-1"
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
            className="flex-none px-4"
            id="detect-trees-btn"
            onClick={e => {
              console.log("DetectionSidebar: Detect button clicked - starting ML pipeline");
              
              // Show a loading state for the button
              const button = e.currentTarget;
              const originalText = button.innerHTML;
              button.innerHTML = '<svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Detecting...';
              button.disabled = true;
              button.style.backgroundColor = '#CBD5E1'; // Light gray
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
              console.log(`DetectionSidebar: Using job ID: ${jobId}`);
              
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
              
              // Dispatch event to trigger detection
              window.dispatchEvent(new CustomEvent('openTreeDetection', {
                detail: {
                  useSatelliteImagery: true,
                  useRealGemini: geminiParams.includeRiskAnalysis,
                  saveToResponseJson: true,
                  geminiParams: geminiParams,
                  job_id: jobId,
                  buttonTriggered: true  // Flag to indicate this was triggered by button click
                },
                buttonTriggered: true  // Add flag at event level too for backwards compatibility
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
              }, 20000); // 20 second safety timeout
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
            
            // Update map container size
            const mapContainer = document.querySelector('#map-container');
            if (mapContainer) {
              mapContainer.style.right = `${newWidth}px`;
            }
            
            // Update badge position - call our enhanced function
            updateBadgePosition(newWidth);
            
            // Also dispatch event during resize for other components that might need it
            window.dispatchEvent(new CustomEvent('detectionSidebarResizing', {
              detail: { width: newWidth }
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
            
            // Force resize to ensure map redraws correctly
            window.dispatchEvent(new Event('resize'));
            
            // Dispatch event to notify other components of the sidebar width change
            window.dispatchEvent(new CustomEvent('detectionSidebarResized', {
              detail: { width: width }
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
          onClick={toggleCollapse}
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