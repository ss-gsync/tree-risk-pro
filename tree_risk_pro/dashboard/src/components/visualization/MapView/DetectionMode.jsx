import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { X, Check, Edit, Plus, Trash, ChevronLeft, ChevronRight, Save, AlertTriangle, FileText, MapPin, Clock, CheckCircle, XCircle, Search, Database, BarChart, Settings, Image, Eye, EyeOff, Sliders } from 'lucide-react';
import MLOverlay from './MLOverlay';
import { useValidation } from '../../../hooks/useReportValidation';

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
  const [trees, setTrees] = useState([]);
  const [selectedTree, setSelectedTree] = useState(null);
  const [currentTreeIndex, setCurrentTreeIndex] = useState(0);
  const [isEditing, setIsEditing] = useState(false);
  const [editingBounds, setEditingBounds] = useState(null);
  const [collapsed, setCollapsed] = useState(false); // ALWAYS start uncollapsed
  
  // Removed component mounting log to prevent console flood
  const [activeTab, setActiveTab] = useState('trees'); // 'trees' or 'reports'
  const [selectedReports, setSelectedReports] = useState([]);
  // Only use validation hook if we're in the reports tab
  const { validationItems, validateItem, isProcessing, refreshValidationItems } = 
    activeTab === 'reports' ? useValidation() : 
    { validationItems: [], validateItem: () => {}, isProcessing: false, refreshValidationItems: () => {} };
  const [configVisible, setConfigVisible] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredTrees, setFilteredTrees] = useState([]);
  const [headerCollapsed, setHeaderCollapsed] = useState(true); // Track header collapsed state
  
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
  const [formData, setFormData] = useState({
    species: '',
    height: 30,
    diameter: 12,
    riskLevel: 'new'
  });
  const [geminiParams, setGeminiParams] = useState({
    detectionThreshold: 0.7,
    maxTrees: 20,
    includeRiskAnalysis: true,
    detailLevel: 'high'
  });
  
  // State for controlling manual tree placement
  const [manualPlacement, setManualPlacement] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.3);
  const [width, setWidth] = useState(384); // Width of the sidebar in pixels
  
  // Define startEditing function
  const startEditing = () => setIsEditing(true);

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
  
  // Manual tree placement functionality with geocoding
  const handleMapClick = useCallback((event) => {
    if (!manualPlacement || !mapRef.current) return;
    
    // Get click coordinates
    const lat = event.latLng.lat();
    const lng = event.latLng.lng();
    
    console.log(`Manual tree placement at [${lng}, ${lat}]`);
    
    // Create a new tree at the clicked location with default values
    const newTree = {
      id: `manual_tree_${Date.now()}`,
      location: [lng, lat],
      height: null,
      species: "",
      diameter: null,
      risk_level: "new",
      confidence: 1.0,
      manually_placed: true,
      visible: true,
      source: "manual_placement",
      address: "Retrieving address..."
    };
    
    // Use Google Maps Geocoding API to get the address
    if (window.google && window.google.maps && window.google.maps.Geocoder) {
      const geocoder = new window.google.maps.Geocoder();
      geocoder.geocode({ location: { lat, lng } }, (results, status) => {
        if (status === "OK" && results[0]) {
          const address = results[0].formatted_address;
          console.log(`Geocoded address: ${address}`);
          
          // Update the tree with the address
          newTree.address = address;
          
          // Update the tree in the state if it was already added
          setTrees(prev => prev.map(t => 
            t.id === newTree.id ? { ...t, address } : t
          ));
          
          // Update selected tree if this is the selected one
          if (selectedTree && selectedTree.id === newTree.id) {
            setSelectedTree(prev => ({ ...prev, address }));
          }
        } else {
          console.warn("Geocoder failed due to: " + status);
          newTree.address = "Address not available";
        }
      });
    } else {
      console.warn("Google Maps Geocoder not available");
      newTree.address = "Address not available";
    }
    
    // Add tree to the list
    setTrees(prev => [...prev, newTree]);
    
    // Select the new tree
    setSelectedTree(newTree);
    setCurrentTreeIndex(trees.length);
    
    // Update form data with empty values for manual placement
    setFormData({
      species: "",
      height: "",
      diameter: "",
      riskLevel: "new"
    });
    
    // Create a simple green dot at the click location on the ML overlay
    try {
      // Add a Google Maps Circle for the green dot - much simpler than custom overlays
      if (window.google && window.google.maps) {
        const mapInstance = mapRef.current.getMap?.() || mapRef.current;
        
        // Create green dot circle - smaller, thinner and much more transparent
        const greenDot = new window.google.maps.Circle({
          center: { lat, lng },
          radius: 1.5, // Smaller radius in meters
          fillColor: '#2ECC71', // No risk green
          fillOpacity: 0.4, // More transparent
          strokeColor: '#FFFFFF',
          strokeWeight: 1, // Thinner stroke
          strokeOpacity: 0.6, // More transparent stroke
          map: mapInstance,
          zIndex: 1001 // Above other elements
        });
        
        // Store circle reference in markers array
        if (mapRef.current.getMarkers) {
          const markers = mapRef.current.getMarkers();
          markers.push(greenDot);
        }
        
        // Animate circle size
        let currentRadius = 1.5;
        const targetRadius = 3;
        const step = 0.2;
        
        const animateGrowth = () => {
          if (currentRadius < targetRadius) {
            currentRadius += step;
            greenDot.setRadius(currentRadius);
            window.requestAnimationFrame(animateGrowth);
          }
        };
        
        animateGrowth();
      }
    } catch (error) {
      console.error("Error adding green dot to ML overlay:", error);
    }
    
    // Add marker to map using a completely different approach
    if (mapRef.current) {
      try {
        if (mapRef.current.getMap && window.google && window.google.maps) {
          console.log("Creating draggable marker using DOM approach");
          const mapInstance = mapRef.current.getMap();
          
          // CREATE MARKER USING HTML DOM ELEMENT OVERLAY
          // Create a custom overlay class for our marker if it doesn't exist
          if (!window.CustomMarkerOverlay) {
            window.CustomMarkerOverlay = class extends window.google.maps.OverlayView {
              constructor(position, map, content, options = {}) {
                super();
                this.position = position;
                this.content = content;
                this.options = options;
                this.map = map;
                this.div = null;
                this.isDragging = false;
                this.dragStartPosition = null;
                this.draggedPosition = null;
                this.setMap(map);
              }
              
              onAdd() {
                // Create the outer div that will contain the marker
                const div = document.createElement('div');
                div.style.position = 'absolute';
                div.style.cursor = this.options.draggable ? 'move' : 'pointer';
                div.style.zIndex = this.options.zIndex || 100;
                div.style.userSelect = 'none';
                div.classList.add('custom-marker');
                
                // Add the inner content - safely
                try {
                  // Check if content already has a parent
                  if (this.content.parentNode) {
                    try {
                      this.content.parentNode.removeChild(this.content);
                    } catch (e) {
                      console.error("Error removing content from parent:", e);
                    }
                  }
                  
                  // Now try to append
                  div.appendChild(this.content);
                } catch (e) {
                  console.error("Error appending content to div:", e);
                  // Create a backup content if original fails
                  try {
                    const backupContent = this.content.cloneNode(true);
                    div.appendChild(backupContent);
                  } catch (e2) {
                    console.error("Even backup content failed:", e2);
                  }
                }
                
                // Store for use later
                this.div = div;
                
                // Add click event listener
                div.addEventListener('click', (e) => {
                  e.stopPropagation();
                  if (this.options.onClick) {
                    this.options.onClick();
                  }
                });
                
                // Make draggable if needed
                if (this.options.draggable) {
                  div.addEventListener('mousedown', (e) => {
                    e.stopPropagation();
                    this.isDragging = true;
                    this.dragStartPosition = { x: e.clientX, y: e.clientY };
                    this.div.style.zIndex = (this.options.zIndex || 100) + 50;
                    
                    // Add temporary global event listeners
                    document.addEventListener('mousemove', this.handleMouseMove);
                    document.addEventListener('mouseup', this.handleMouseUp);
                  });
                }
                
                // Add to the overlay's panes - with extra safety
                try {
                  const panes = this.getPanes();
                  
                  // Make sure panes exist and overlayMouseTarget is valid
                  if (panes && panes.overlayMouseTarget) {
                    // First check if div is already in DOM
                    if (div.parentNode) {
                      try {
                        div.parentNode.removeChild(div);
                      } catch (e) {
                        console.error("Error removing div from current parent:", e);
                      }
                    }
                    
                    // Try to append the div to the target pane
                    panes.overlayMouseTarget.appendChild(div);
                  } else {
                    console.error("Invalid panes structure for overlay:", panes);
                    
                    // Fallback to map container if panes not available
                    const mapContainer = document.getElementById('map-container');
                    if (mapContainer) {
                      mapContainer.appendChild(div);
                      console.log("Used fallback container for marker");
                    }
                  }
                } catch (e) {
                  console.error("Error adding div to overlay panes:", e);
                  
                  // Last resort fallback - try to add to document body
                  try {
                    document.body.appendChild(div);
                    console.log("Used body fallback for marker");
                  } catch (e2) {
                    console.error("All fallbacks failed for marker div:", e2);
                  }
                }
              }
              
              handleMouseMove = (e) => {
                if (!this.isDragging) return;
                
                // Calculate new position
                const proj = this.getProjection();
                const position = proj.fromLatLngToDivPixel(this.position);
                const offsetX = e.clientX - this.dragStartPosition.x;
                const offsetY = e.clientY - this.dragStartPosition.y;
                
                // Apply new position to div
                this.div.style.left = `${position.x + offsetX}px`;
                this.div.style.top = `${position.y + offsetY}px`;
                
                // Update the dragged position
                const newPixelPos = new window.google.maps.Point(
                  position.x + offsetX,
                  position.y + offsetY
                );
                this.draggedPosition = proj.fromDivPixelToLatLng(newPixelPos);
              }
              
              handleMouseUp = (e) => {
                if (!this.isDragging) return;
                this.isDragging = false;
                
                // Clean up event listeners
                document.removeEventListener('mousemove', this.handleMouseMove);
                document.removeEventListener('mouseup', this.handleMouseUp);
                
                // Reset z-index
                this.div.style.zIndex = this.options.zIndex || 100;
                
                // If we have a dragged position, update and notify
                if (this.draggedPosition) {
                  this.position = this.draggedPosition;
                  this.draw(); // Redraw at the new position
                  
                  if (this.options.onDragEnd) {
                    this.options.onDragEnd(this.draggedPosition);
                  }
                  
                  this.draggedPosition = null;
                }
              }
              
              draw() {
                if (!this.div) return;
                
                // Calculate position on the map
                const proj = this.getProjection();
                const position = proj.fromLatLngToDivPixel(this.position);
                
                // Place the div at the right position
                this.div.style.left = `${position.x - (this.options.anchorX || 10)}px`;
                this.div.style.top = `${position.y - (this.options.anchorY || 10)}px`;
              }
              
              onRemove() {
                // Clean up event listeners
                document.removeEventListener('mousemove', this.handleMouseMove);
                document.removeEventListener('mouseup', this.handleMouseUp);
                
                // Remove the element from the DOM - safely
                if (this.div) {
                  try {
                    // First check if div is still in DOM and has a parent
                    if (this.div.parentNode) {
                      // Check if the div is actually in the document
                      if (document.body.contains(this.div)) {
                        try {
                          this.div.parentNode.removeChild(this.div);
                          console.log("Marker div removed successfully");
                        } catch (err) {
                          console.log("Safe error removing marker:", err);
                        }
                      } else {
                        console.log("Marker div is not in document - skipping removal");
                        // Don't attempt to remove if not in document to avoid errors
                      }
                    } else {
                      console.log("Marker div has no parent - skipping removal");
                    }
                  } catch (e) {
                    console.error("Error removing marker div:", e);
                    
                    // Hide it if removal fails
                    try {
                      this.div.style.display = 'none';
                      this.div.style.visibility = 'hidden';
                      this.div.style.opacity = '0';
                    } catch (e2) {} 
                  } finally {
                    // Always clear the reference
                    this.div = null;
                  }
                }
              }
              
              setPosition(latLng) {
                this.position = latLng;
                this.draw();
              }
              
              getPosition() {
                return this.position;
              }
            };
          }
          
          // Create the visual marker element
          const markerElement = document.createElement('div');
          markerElement.style.width = '20px';
          markerElement.style.height = '20px';
          markerElement.style.borderRadius = '50%';
          markerElement.style.backgroundColor = '#2ECC71';
          markerElement.style.border = '2px solid #1E8449';
          markerElement.style.boxShadow = '0 0 8px rgba(0,0,0,0.3)';
          
          // Create a marker using our custom overlay
          const customMarker = new window.CustomMarkerOverlay(
            { lat, lng },
            mapInstance,
            markerElement,
            {
              draggable: true,
              zIndex: 1000,
              anchorX: 10,
              anchorY: 10,
              onClick: () => {
                console.log("Custom marker clicked:", newTree);
                
                // Select this tree
                setSelectedTree(newTree);
                
                // Find index of this tree for the current index
                const treeIndex = trees.findIndex(t => t.id === newTree.id);
                if (treeIndex !== -1) {
                  setCurrentTreeIndex(treeIndex);
                }
                
                // Show notification
                const clickNotification = document.createElement('div');
                clickNotification.className = 'tree-selected-notification';
                clickNotification.style.position = 'absolute';
                clickNotification.style.top = '70px';
                clickNotification.style.left = '50%';
                clickNotification.style.transform = 'translateX(-50%)';
                clickNotification.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
                clickNotification.style.color = 'white';
                clickNotification.style.padding = '8px 16px';
                clickNotification.style.borderRadius = '4px';
                clickNotification.style.zIndex = '9999';
                clickNotification.style.fontSize = '14px';
                clickNotification.style.fontWeight = '500';
                clickNotification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
                clickNotification.textContent = 'Object selected';
                document.body.appendChild(clickNotification);
                
                // Remove notification after 1.5 seconds - safely
                setTimeout(() => {
                  try {
                    // Check if notification still exists and is in the document
                    if (clickNotification.parentNode && document.body.contains(clickNotification)) {
                      clickNotification.parentNode.removeChild(clickNotification);
                    } else {
                      console.log("Notification already removed or not in document");
                    }
                  } catch (e) {
                    console.error("Error removing click notification:", e);
                    // Try to hide it if removal fails
                    try {
                      clickNotification.style.display = 'none';
                      clickNotification.style.visibility = 'hidden';
                    } catch (e2) {}
                  }
                }, 1500);
              },
              onDragEnd: (newPos) => {
                const newLat = newPos.lat();
                const newLng = newPos.lng();
                
                // Update the tree location in state
                setTrees(prev => prev.map(t => 
                  t.id === newTree.id 
                    ? { ...t, location: [newLng, newLat] } 
                    : t
                ));
                
                // Update selected tree if this is the selected one
                if (selectedTree && selectedTree.id === newTree.id) {
                  setSelectedTree(prev => ({ ...prev, location: [newLng, newLat] }));
                }
                
                // Show notification
                const notification = document.createElement('div');
                notification.className = 'tree-repositioned-notification';
                notification.style.position = 'absolute';
                notification.style.top = '70px';
                notification.style.left = '50%';
                notification.style.transform = 'translateX(-50%)';
                notification.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
                notification.style.color = 'white';
                notification.style.padding = '8px 16px';
                notification.style.borderRadius = '4px';
                notification.style.zIndex = '9999';
                notification.style.fontSize = '14px';
                notification.style.fontWeight = '500';
                notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
                notification.textContent = 'Object position updated';
                document.body.appendChild(notification);
                
                // Remove notification after 1.5 seconds - safely
                setTimeout(() => {
                  try {
                    // Check if notification still exists and is in the document
                    if (notification.parentNode && document.body.contains(notification)) {
                      notification.parentNode.removeChild(notification);
                    } else {
                      console.log("Tree reposition notification already removed or not in document");
                    }
                  } catch (e) {
                    console.error("Error removing reposition notification:", e);
                    // Try to hide it if removal fails
                    try {
                      notification.style.display = 'none';
                      notification.style.visibility = 'hidden';
                    } catch (e2) {}
                  }
                }, 1500);
              }
            }
          );
          
          // Store reference to the custom marker
          newTree.customMarker = customMarker;
          
          // Create and show successful placement notification
          const placementNotification = document.createElement('div');
          placementNotification.className = 'tree-placed-notification';
          placementNotification.style.position = 'absolute';
          placementNotification.style.top = '70px';
          placementNotification.style.left = '50%';
          placementNotification.style.transform = 'translateX(-50%)';
          placementNotification.style.backgroundColor = 'rgba(34, 197, 94, 0.9)';
          placementNotification.style.color = 'white';
          placementNotification.style.padding = '8px 16px';
          placementNotification.style.borderRadius = '4px';
          placementNotification.style.zIndex = '9999';
          placementNotification.style.fontSize = '14px';
          placementNotification.style.fontWeight = '500';
          placementNotification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
          placementNotification.textContent = 'Custom marker placed successfully';
          
          // Add a small note about what this tree will feed into
          const noteSpan = document.createElement('span');
          noteSpan.style.display = 'block';
          noteSpan.style.fontSize = '11px';
          noteSpan.style.opacity = '0.9';
          noteSpan.style.marginTop = '4px';
          noteSpan.textContent = 'Custom marker ready for interaction';
          placementNotification.appendChild(noteSpan);
          
          // Add to document
          document.body.appendChild(placementNotification);
          
          // Remove after 1.5 seconds - safely
          setTimeout(() => {
            try {
              // Check if notification still exists and is in the document
              if (placementNotification.parentNode && document.body.contains(placementNotification)) {
                placementNotification.parentNode.removeChild(placementNotification);
              } else {
                console.log("Placement notification already removed or not in document");
              }
            } catch (e) {
              console.error("Error removing placement notification:", e);
              // Try to hide it if removal fails
              try {
                placementNotification.style.display = 'none';
                placementNotification.style.visibility = 'hidden';
              } catch (e2) {}
            }
          }, 1500);
          
          // Store the marker reference in the map's markers array if needed
          if (mapRef.current.getMarkers) {
            const markers = mapRef.current.getMarkers();
            markers.push(customMarker);
          }
        }
        // Fallback to renderDetectedTrees if direct creation is not possible
        else if (mapRef.current.renderDetectedTrees) {
          console.log("Fallback: Using renderDetectedTrees to add manual tree marker (not draggable)");
          mapRef.current.renderDetectedTrees([newTree], 'manual');
        }
        
        // Show notification for successful placement
        const notification = document.createElement('div');
        notification.className = 'tree-placed-notification';
        notification.style.position = 'absolute';
        notification.style.top = '70px';
        notification.style.left = '50%';
        notification.style.transform = 'translateX(-50%)';
        notification.style.backgroundColor = 'rgba(34, 197, 94, 0.9)'; // green-600 with opacity
        notification.style.color = 'white';
        notification.style.padding = '8px 16px';
        notification.style.borderRadius = '4px';
        notification.style.zIndex = '1000';
        notification.style.fontSize = '14px';
        notification.style.fontWeight = '500';
        notification.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
        notification.textContent = 'Tree placed successfully - added to validation queue';
        
        // Add a small note about what this tree will feed into
        const noteSpan = document.createElement('span');
        noteSpan.style.display = 'block';
        noteSpan.style.fontSize = '11px';
        noteSpan.style.opacity = '0.9';
        noteSpan.style.marginTop = '4px';
        noteSpan.textContent = 'Tree ready for review in Detection sidebar';
        notification.appendChild(noteSpan);
        
        // Add to document
        document.body.appendChild(notification);
        
        // Remove after 1.5 seconds - safely
        setTimeout(() => {
          try {
            // Check if notification still exists and is in the document
            if (notification.parentNode && document.body.contains(notification)) {
              notification.parentNode.removeChild(notification);
            } else {
              console.log("Tree placement notification already removed or not in document");
            }
          } catch (e) {
            console.error("Error removing tree notification:", e);
            // Try to hide it if removal fails
            try {
              notification.style.display = 'none';
              notification.style.visibility = 'hidden';
            } catch (e2) {}
          }
        }, 1500);
      } catch (error) {
        console.error("Error adding manual tree marker:", error);
      }
    }
    
    console.log("Added manually placed tree:", newTree);
    
    // We don't automatically exit manual placement mode anymore - it stays active until user right-clicks
    // Don't remove the cursor indicator as we're keeping the placement mode active
    const indicator = document.getElementById('placement-cursor-indicator');
    if (indicator) {
      // Just update style to show success briefly
      indicator.style.backgroundColor = 'rgba(25, 220, 25, 0.6)';
      setTimeout(() => {
        try {
          if (indicator && document.body.contains(indicator) && indicator.parentNode) {
            indicator.style.backgroundColor = 'rgba(46, 204, 113, 0.5)';
          }
        } catch (e) {
          console.error("Error resetting indicator style:", e);
        }
      }, 200);
    }
    
    // Don't reset cursor styles as we're keeping the placement mode active
    // We still want the crosshair cursor for continuing to place trees
    
    // Don't remove global style or deactivate the placement button
    // as we want manual placement mode to continue until right-click
    
  }, [manualPlacement, mapRef, trees.length, setFormData]);
  
  // Set up click listener for manual placement
  useEffect(() => {
    if (!mapRef.current || !manualPlacement) return;
    
    // Add click and right-click listeners to map
    console.log("Setting up manual tree placement mode");
    const mapInstance = mapRef.current.getMap?.() || mapRef.current;
    
    if (!mapInstance) {
      console.warn("No map instance available for click listener");
      return;
    }
    
    // Important: Configure the overlay to allow marker interaction
    // The ML overlay now handles marker interaction dynamically
    const overlayElements = document.getElementsByClassName('ml-detection-overlay');
    if (overlayElements.length > 0) {
      for (let i = 0; i < overlayElements.length; i++) {
        // Let the overlay handle events with its smart detection logic
        overlayElements[i].setAttribute('data-mode', 'placement');
        console.log("Set ML overlay to placement mode for marker interaction");
      }
    }
    
    // We want to directly apply styles to the map container
    const mapContainer = document.getElementById('map-container');
    const mapDiv = mapInstance.getDiv ? mapInstance.getDiv() : null;
    
    // Create left-click listener for tree placement
    const clickListener = mapInstance.addListener('click', (event) => {
      // Capture the click event and handle it
      console.log("Map click detected in manual placement mode");
      handleMapClick(event);
    });
    
    // Create right-click listener to exit manual placement mode
    const rightClickListener = mapInstance.addListener('rightclick', (event) => {
      console.log("Right-click detected, exiting manual placement mode");
      
      // Exit manual placement mode
      setManualPlacement(false);
      
      // Clean up cursor indicator
      const indicator = document.getElementById('placement-cursor-indicator');
      if (indicator && indicator.parentNode) {
        indicator.remove();
      }
      
      // Reset cursor styles
      if (mapContainer) {
        mapContainer.style.cursor = '';
        
        // Reset child elements
        const childElements = mapContainer.querySelectorAll('*');
        childElements.forEach(element => {
          element.style.cursor = '';
        });
      }
      
      // Remove global style
      const styleElement = document.getElementById('manual-placement-cursor-style');
      if (styleElement) {
        styleElement.remove();
      }
      
      // Update button UI to reflect manual placement mode deactivation
      const placementButton = document.querySelector('[title*="Placement mode active"]');
      if (placementButton) {
        placementButton.classList.remove('bg-green-600', 'ring-2', 'ring-green-300', 'ring-opacity-50');
        placementButton.style.transform = 'scale(1)';
        placementButton.style.boxShadow = 'none';
      }
      
      // Show confirmation toast
      const toast = document.createElement('div');
      toast.className = 'placement-exit-toast';
      toast.style.position = 'absolute';
      toast.style.top = '70px';
      toast.style.left = '50%';
      toast.style.transform = 'translateX(-50%)';
      toast.style.backgroundColor = 'rgba(59, 130, 246, 0.9)';
      toast.style.color = 'white';
      toast.style.padding = '8px 16px';
      toast.style.borderRadius = '4px';
      toast.style.zIndex = '1000';
      toast.style.fontSize = '14px';
      toast.style.fontWeight = '500';
      toast.style.boxShadow = '0 2px 4px rgba(0,0,0,0.2)';
      toast.textContent = 'Manual placement mode exited';
      document.body.appendChild(toast);
      
      // Remove after 1.5 seconds
      setTimeout(() => {
        if (toast.parentNode) {
          toast.parentNode.removeChild(toast);
        }
      }, 1500);
    });
    
    // Change cursor to crosshair for better placement feedback - apply to both elements
    if (mapDiv) {
      mapDiv.style.cursor = 'crosshair !important';
      console.log("Set cursor to crosshair on map div");
    }
    
    if (mapContainer) {
      mapContainer.style.cursor = 'crosshair !important';
      console.log("Set cursor to crosshair on map container");
      
      // Apply cursor to all child elements as well to ensure it shows everywhere
      const childElements = mapContainer.querySelectorAll('*');
      childElements.forEach(element => {
        element.style.cursor = 'crosshair';
      });
    }
    
    // Add global styles to ensure cursor is crosshair over the map
    const style = document.createElement('style');
    style.id = 'manual-placement-cursor-style';
    style.innerHTML = `
      #map-container, #map-container * {
        cursor: crosshair !important;
      }
      .ml-detection-overlay {
        pointer-events: none !important;
      }
    `;
    document.head.appendChild(style);
    
    // Add visual indicator element that follows cursor
    const indicator = document.createElement('div');
    indicator.id = 'placement-cursor-indicator';
    indicator.style.position = 'absolute';
    indicator.style.width = '20px';
    indicator.style.height = '20px';
    indicator.style.borderRadius = '50%';
    indicator.style.border = '2px solid white';
    indicator.style.backgroundColor = 'rgba(46, 204, 113, 0.5)'; // No risk green
    indicator.style.transform = 'translate(-50%, -50%)';
    indicator.style.pointerEvents = 'none';
    indicator.style.zIndex = '9999';
    indicator.style.boxShadow = '0 0 4px rgba(0,0,0,0.5)';
    document.body.appendChild(indicator);
    
    // Track mouse position
    const trackMouse = (e) => {
      indicator.style.left = `${e.clientX}px`;
      indicator.style.top = `${e.clientY}px`;
    };
    
    document.addEventListener('mousemove', trackMouse);
    
    return () => {
      // Clean up listeners when unmounting or disabling manual placement
      if (window.google?.maps?.event) {
        window.google.maps.event.removeListener(clickListener);
        window.google.maps.event.removeListener(rightClickListener);
      }
      
      // Reset cursor on both elements
      if (mapDiv) {
        mapDiv.style.cursor = '';
      }
      
      if (mapContainer) {
        mapContainer.style.cursor = '';
        
        // Reset child elements
        const childElements = mapContainer.querySelectorAll('*');
        childElements.forEach(element => {
          element.style.cursor = '';
        });
      }
      
      // Remove global style
      const styleElement = document.getElementById('manual-placement-cursor-style');
      if (styleElement) {
        styleElement.remove();
      }
      
      // Remove cursor indicator
      const indicator = document.getElementById('placement-cursor-indicator');
      if (indicator) {
        indicator.remove();
      }
      
      // Remove mousemove listener
      document.removeEventListener('mousemove', trackMouse);
      
      console.log("Removed click listener and reset cursor for manual tree placement");
    };
  }, [mapRef, manualPlacement, handleMapClick]);

  // Effect to handle sidebar resizing
  useEffect(() => {
    // Update map container when width changes
    if (!collapsed) {
      const mapContainer = document.querySelector('#map-container');
      if (mapContainer) {
        mapContainer.style.right = `${width}px`;
      }
      
      // Notify other components about the resize
      const resizeEvent = new CustomEvent('sidebarResized', {
        detail: { width, source: 'detectionMode' }
      });
      window.dispatchEvent(resizeEvent);
      
      // Trigger window resize to update map
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 50);
    }
  }, [width, collapsed]);

  // Initialize trees from detected trees prop
  useEffect(() => {
    if (detectedTrees && detectedTrees.length > 0) {
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
      
      // Add markers to the map for visualization
      if (mapRef.current && mapRef.current.renderDetectedTrees && window.google) {
        console.log(`Rendering ${initialTrees.length} detected trees from ML pipeline`);
        
        // Use slight delay to ensure map is fully loaded
        setTimeout(() => {
          if (mapRef.current) {
            mapRef.current.renderDetectedTrees(initialTrees, 'ai');
          }
        }, 100);
      }
    }
  }, [detectedTrees, mapRef, selectedTree]);

  // Listen for tree selection and repositioning events from map markers
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
        
        console.log("Found and selected tree in list:", treeInList);
      } else {
        console.warn("Could not find selected tree in current trees list");
      }
    };
    
    // Handle marker repositioning events
    const handleTreeRepositioned = (event) => {
      if (!event.detail) return;
      
      const { tree, newLocation } = event.detail;
      
      // Update the tree location in our trees state
      setTrees(prev => prev.map(t => 
        t.id === tree.id 
          ? { ...t, location: newLocation } 
          : t
      ));
      
      // Also update selected tree if it's the one that was moved
      if (selectedTree && selectedTree.id === tree.id) {
        setSelectedTree(prev => ({ ...prev, location: newLocation }));
      }
    };
    
    window.addEventListener('treeSelected', handleTreeSelected);
    window.addEventListener('treeRepositioned', handleTreeRepositioned);
    
    return () => {
      window.removeEventListener('treeSelected', handleTreeSelected);
      window.removeEventListener('treeRepositioned', handleTreeRepositioned);
    };
  }, [selectedTree, trees]);

  // Listen for validation mode cleanup event from MapView
  useEffect(() => {
    const handleValidationModeCleanup = (event) => {
      console.log("ValidationModeCleanup event received in DetectionMode", event?.detail);
      
      // Immediately start our own cleanup process
      try {
        // 1. Set collapsed state to hide sidebar UI
        setCollapsed(true);
        
        // 2. Find and hide our component's DOM elements
        const selfElement = document.querySelector('.detection-sidebar');
        if (selfElement) {
          // Add cleanup classes 
          selfElement.classList.add('detection-sidebar-removed');
          selfElement.classList.add('cleanup-in-progress');
          
          // Apply hiding styles
          try {
            selfElement.style.display = 'none';
            selfElement.style.visibility = 'hidden';
            selfElement.style.opacity = '0';
            selfElement.style.transform = 'translateX(100%)';
            selfElement.style.width = '0px';
            selfElement.style.pointerEvents = 'none';
            selfElement.style.zIndex = '-1';
          } catch (err) {
            console.log("Error applying hiding styles during cleanup:", err);
          }
        }
        
        // 3. Reset state vars to default values
        setTrees([]);
        setSelectedTree(null);
        setCurrentTreeIndex(0);
        setIsEditing(false);
        setEditingBounds(null);
        setSearchQuery('');
        setFilteredTrees([]);
        
        // 4. Notify parent component that we're cleaning up
        if (typeof onExitValidation === 'function') {
          onExitValidation();
        }
      } catch (err) {
        console.error("Error during ValidationMode cleanup:", err);
      }
    };
    
    // Register listener for the cleanup event
    window.addEventListener('validationModeCleanup', handleValidationModeCleanup);
    
    return () => {
      window.removeEventListener('validationModeCleanup', handleValidationModeCleanup);
    };
  }, [onExitValidation]);

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
                
                // Reset map container width to fill the screen
                setTimeout(() => {
                  const mapContainer = document.querySelector('#map-container');
                  if (mapContainer) {
                    mapContainer.style.right = '0px';
                    
                    // Trigger window resize to force map to redraw
                    window.dispatchEvent(new Event('resize'));
                  }
                }, 100);
              }
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
    
    // Handler for openFeatureSelection events (Database button)
    const handleOpenFeatureSelection = (event) => {
      // This event is triggered by the Database button - we need to close the Detection sidebar
      console.log("openFeatureSelection event received in DetectionMode - FORCING CLEANUP");
      
      // CRITICAL: Start extensive cleanup process to ensure this component is fully removed

      // 1. Set collapsed state to true to trigger CSS transitions
      setCollapsed(true);
      
      // 2. Call exit validation function to clean up parent state
      if (typeof onExitValidation === 'function') {
        onExitValidation();
      }
      
      // 3. Reset map container width immediately
      const mapContainer = document.querySelector('#map-container');
      if (mapContainer) {
        mapContainer.style.right = '0px';
      }
      
      // 4. Mark element for complete removal and apply safe DOM manipulations
      // Find the component root element
      const sidebar = document.querySelector('.detection-sidebar');
      if (sidebar) {
        // Add class to mark for complete removal in render function
        sidebar.classList.add('detection-sidebar-removed');
        
        // Safely apply style changes in try/catch to prevent DOM errors
        try {
          sidebar.style.display = 'none';
          sidebar.style.visibility = 'hidden';
          sidebar.style.opacity = '0';
          sidebar.style.transform = 'translateX(100%)';
          sidebar.style.width = '0px';
          sidebar.style.pointerEvents = 'none';
          sidebar.style.zIndex = '-1'; // Move behind everything
        } catch (e) {
          console.log("Error manipulating sidebar DOM", e);
        }
        
        // We no longer try to actually remove the DOM element as it causes errors
      }
      
      // 5. Notify all listeners that this sidebar is closing
      window.dispatchEvent(new CustomEvent('validationSidebarToggle', {
        detail: {
          collapsed: true,
          source: 'detection_closing_for_database',
          width: 0,
          forceRemove: true,
          forceClose: true
        }
      }));
      
      // 6. Force resize after closure
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 50);
      
      // 7. Run another cleanup pass after a delay
      setTimeout(() => {
        // Find any lingering detection sidebars and mark them for removal
        const sidebars = document.querySelectorAll('.detection-sidebar');
        sidebars.forEach(sidebar => {
          if (sidebar) {
            // Mark for complete removal
            sidebar.classList.add('detection-sidebar-removed');
            
            // Apply safe style changes
            try {
              sidebar.style.display = 'none';
              sidebar.style.visibility = 'hidden';
              sidebar.style.opacity = '0';
              sidebar.style.width = '0px';
              sidebar.style.pointerEvents = 'none';
              sidebar.style.zIndex = '-1';
            } catch (err) {
              console.log("Error in sidebar cleanup:", err);
            }
          }
        });
        
        // Force another resize
        window.dispatchEvent(new Event('resize'));
      }, 200);
    };
    
    // Handle force close for Object Detection mode
    const handleForceCloseObjectDetection = (event) => {
      console.log("forceCloseObjectDetection event received in DetectionMode");
      
      // CRITICAL: Always do a full cleanup regardless of current state
      
      // 1. Set collapsed state to true to trigger CSS transitions
      setCollapsed(true);
      
      // 2. Call exit validation function to clean up parent state
      if (typeof onExitValidation === 'function') {
        onExitValidation();
      }
      
      // 3. Reset map container width immediately
      const mapContainer = document.querySelector('#map-container');
      if (mapContainer) {
        mapContainer.style.right = '0px';
      }
      
      // 4. Mark element for complete removal and apply safe DOM manipulations
      // Find the component root element (direct self-reference)
      const selfElement = document.querySelector('.detection-sidebar');
      if (selfElement) {
        // Add class to mark for complete removal in render function
        selfElement.classList.add('detection-sidebar-removed');
        
        // Safely apply style changes in try/catch to prevent DOM errors
        try {
          selfElement.style.display = 'none';
          selfElement.style.visibility = 'hidden';
          selfElement.style.opacity = '0';
          selfElement.style.transform = 'translateX(100%)';
          selfElement.style.width = '0px';
          selfElement.style.pointerEvents = 'none';
          selfElement.style.zIndex = '-1'; // Move behind everything
        } catch (e) {
          console.log("Error manipulating sidebar DOM", e);
        }
      }
      
      // 5. Notify all listeners that this sidebar is closing
      window.dispatchEvent(new CustomEvent('validationSidebarToggle', {
        detail: {
          collapsed: true,
          source: 'tree_detection',
          width: 0,
          forceRemove: true,
          forceClose: true
        }
      }));
      
      // 6. Force resize to ensure proper rendering
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
        
        // Do another resize after a bit longer delay to ensure everything has settled
        setTimeout(() => {
          window.dispatchEvent(new Event('resize'));
        }, 300);
      }, 50);
    };
    
    // Listen for other sidebars toggling
    window.addEventListener('validationQueueToggle', handleValidationQueueToggle);
    window.addEventListener('forceCloseTreeDatabase', handleForceCloseTreeDatabase);
    window.addEventListener('forceCloseObjectDetection', handleForceCloseObjectDetection);
    window.addEventListener('openFeatureSelection', handleOpenFeatureSelection);
    
    return () => {
      window.removeEventListener('validationQueueToggle', handleValidationQueueToggle);
      window.removeEventListener('forceCloseTreeDatabase', handleForceCloseTreeDatabase);
      window.removeEventListener('forceCloseObjectDetection', handleForceCloseObjectDetection);
      window.removeEventListener('openFeatureSelection', handleOpenFeatureSelection);
    };
  }, [isFeatureSelectionMode, onExitValidation, collapsed]);

  // Handle tree removal
  const removeTree = () => {
    if (!selectedTree) return;
    
    if (!window.confirm("Are you sure you want to remove this tree?")) {
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
    setCurrentTreeIndex(nextVisibleIndex); // Use index in visible trees array
    
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
    
    // Update state
    setSelectedTree(prevVisibleTree);
    setCurrentTreeIndex(prevVisibleIndex); // Use index in visible trees array
    
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
      // For checkbox fields, explicitly use boolean
      const newValue = Boolean(checked);
      console.log(`Setting ${name} to ${newValue}`);
      
      // Special handling for Gemini setting
      if (name === 'useGemini') {
        console.log(`= Changing Gemini setting to: ${newValue}`);
      }
      
      setFormData(prev => ({ ...prev, [name]: newValue }));
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
    notification.style.backgroundColor = 'rgba(34, 197, 94, 0.9)'; // green-600 with opacity
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

  // TODO: Add your save trees handling logic here
  const saveTreesToValidationQueue = () => {
    const validatedTrees = trees.filter(tree => tree.visible && tree.validated);
    if (validatedTrees.length === 0) {
      window.alert('No approved trees to save. Please approve at least one tree.');
      return;
    }

    // Save trees to validation queue
    onSaveTrees(validatedTrees);
    
    // Show success message
    window.alert(`${validatedTrees.length} approved trees have been saved to the validation queue.`);
  };

  // Calculate progress statistics
  const totalVisibleTrees = trees.filter(tree => tree.visible).length;
  const validatedCount = trees.filter(tree => tree.visible && tree.validated).length;
  const progressPercentage = totalVisibleTrees === 0 ? 0 : Math.round((validatedCount / totalVisibleTrees) * 100);

  // Listen for direct commands to collapse/close the sidebar from other components
  useEffect(() => {
    const handleSetDetectionModeState = (event) => {
      if (event.detail) {
        if (event.detail.active === false || event.detail.forceCollapse === true) {
          console.log("DetectionMode: Received force collapse command");
          
          // IMPORTANT: Only update internal state, do NOT call onExitValidation
          // to prevent infinite recursion
          setCollapsed(true);
          
          // Reset map container size directly
          const mapContainer = document.querySelector('#map-container');
          if (mapContainer) {
            mapContainer.style.right = '0px';
          }
        } else if (event.detail.active === true) {
          console.log("DetectionMode: Received activation command with force:", event.detail.force);
          
          // Force is prioritized for stubborn UIs
          if (event.detail.force) {
            // Ensure critical DOM changes
            const sidebar = document.querySelector('.detection-sidebar');
            if (sidebar) {
              sidebar.style.display = 'block';
              sidebar.style.visibility = 'visible';
              sidebar.style.opacity = '1';
              sidebar.style.transform = 'translateX(0)';
              sidebar.style.width = `${width}px`;
              sidebar.style.right = '0px';
              console.log("Applied direct DOM styles to detection sidebar", sidebar);
            }
          }
          
          // Set internal state
          setCollapsed(false);
          
          // Update map container width
          const mapContainer = document.querySelector('#map-container');
          if (mapContainer) {
            mapContainer.style.right = `${width}px`;
          }
          
          // Make detection debug visible
          const debugDiv = document.getElementById('detection-debug');
          if (debugDiv) {
            debugDiv.style.display = 'block';
          }
        }
      }
    };
    
    window.addEventListener('setDetectionModeState', handleSetDetectionModeState);
    
    return () => {
      window.removeEventListener('setDetectionModeState', handleSetDetectionModeState);
    };
  }, [width]);

  // Get header height only once using a ref to prevent infinite renders
  const [topPosition, setTopPosition] = useState('40px');
  
  // Use effect to measure header height only on mount
  useEffect(() => {
    const measureHeaderHeight = () => {
      const headerElement = document.querySelector('header');
      if (!headerElement) return '40px'; // Default fallback height
      
      // Measure actual rendered height
      const actualHeight = headerElement.getBoundingClientRect().height;
      
      // Add 2px buffer to prevent any possible overlap
      return `${Math.ceil(actualHeight) + 2}px`;
    };
    
    // Only update this once on mount
    setTopPosition(measureHeaderHeight());
  }, []);
  
  // Important cleanup for when component is about to unmount
  useEffect(() => {
    // Return cleanup function
    return () => {
      // On unmount, clean up any orphaned sidebar elements that might be left in the DOM
      try {
        const existingSidebars = document.querySelectorAll('.detection-sidebar');
        existingSidebars.forEach(sidebar => {
          if (sidebar) {
            sidebar.classList.add('detection-sidebar-removed');
            sidebar.style.display = 'none';
            sidebar.style.visibility = 'hidden';
            sidebar.style.opacity = '0';
            sidebar.style.pointerEvents = 'none';
            sidebar.style.zIndex = '-1';
          }
        });
      } catch (e) {
        // Silent catch - we're in cleanup, no need to log errors
      }
    };
  }, []);
  
  // ALWAYS render the component - never bail out
  // Removed the check that skipped rendering
  
  return (
    <div 
      className={`detection-sidebar fixed right-0 bottom-0 bg-white z-30 border-l border-gray-200 shadow-lg transition-all duration-300 ease-in-out ${collapsed ? 'translate-x-full detection-sidebar-collapsed' : 'translate-x-0'}`}
      style={{ 
        width: collapsed ? '0px' : `${width}px`,
        top: propHeaderCollapsed !== undefined ? (propHeaderCollapsed ? '40px' : '64px') : (headerCollapsed ? '40px' : '64px'), // Use props first, fallback to state
        zIndex: 1000, // Higher z-index to ensure it's above other elements
        visibility: collapsed ? 'hidden' : 'visible', // Ensure completely hidden when collapsed
        display: collapsed ? 'none' : 'block', // Further ensure it's completely removed when collapsed
        opacity: collapsed ? 0 : 1, // Add opacity transition
        pointerEvents: collapsed ? 'none' : 'auto' // Prevent interactions when collapsed
      }}
    >
      {/* Left resize handle with thicker light blue bar on hover */}
      <div 
        className="absolute left-0 top-0 bottom-0 w-3 cursor-ew-resize z-10 hover:before:opacity-100 before:opacity-0 before:content-[''] before:absolute before:left-0 before:top-0 before:bottom-0 before:w-1 before:bg-blue-300/40 before:transition-opacity before:duration-150" 
        style={{ left: '6px' }}
        title="Drag to resize sidebar"
        onMouseDown={(e) => {
          e.preventDefault();
          const startX = e.clientX;
          const startWidth = width;
          const windowWidth = window.innerWidth;
          
          const handleMouseMove = (moveEvent) => {
            const deltaX = startX - moveEvent.clientX;
            const newWidth = Math.min(Math.max(startWidth + deltaX, 300), 600); // Constrain between 300px and 600px
            setWidth(newWidth);
            
            // Adjust map container if needed
            const mapContainer = document.querySelector('#map-container');
            if (mapContainer) {
              mapContainer.style.right = `${newWidth}px`;
            }
          };
          
          const handleMouseUp = () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            
            // Trigger resize to ensure map renders correctly after resize
            window.dispatchEvent(new Event('resize'));
          };
          
          document.addEventListener('mousemove', handleMouseMove);
          document.addEventListener('mouseup', handleMouseUp);
        }}
      />
      {/* Compact header matching the slate color theme */}
      <div className="flex items-center justify-between py-1 px-2 bg-slate-100 border-b border-slate-200">
        <div className="flex items-center">
          <h3 className="text-slate-700 font-medium text-sm flex items-center">
            <>
              <svg className="h-3.5 w-3.5 mr-1.5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              Object Recognition
            </>
          </h3>
        </div>
        <div className="flex items-center space-x-1">
          {/* Pin button for manual placement in header */}
          {!isFeatureSelectionMode && (
            <button
              onClick={() => setManualPlacement(!manualPlacement)}
              title={manualPlacement ? "Placement mode active - right-click to disable" : "Enable manual placement"}
              className={`p-1 rounded-md ${manualPlacement ? 'bg-green-100 text-green-700' : 'text-slate-600 hover:bg-slate-200'}`}
            >
              <MapPin size={16} />
            </button>
          )}
          <button
            onClick={() => {
              onExitValidation();
              
              // Reset map container width to fill the screen
              const mapContainer = document.querySelector('#map-container');
              if (mapContainer) {
                mapContainer.style.right = '0px';
                
                // Trigger window resize to force map to redraw
                setTimeout(() => {
                  window.dispatchEvent(new Event('resize'));
                }, 50);
              }
            }}
            className="p-1 text-slate-600 hover:bg-slate-200 rounded-md"
            title="Close sidebar"
          >
            <X size={16} />
          </button>
        </div>
      </div>
      
      {/* Tree detection section */}
      {!collapsed && !isFeatureSelectionMode && (
        <div className="p-4 border-b">
          {/* Objects progress indicator */}
          <div className="mb-3 bg-slate-50 p-2 rounded border border-slate-200">
            <div className="flex justify-between items-center text-xs text-slate-600 mb-1.5">
              <span>Objects: {validatedCount} of {totalVisibleTrees} validated</span>
              <span className="font-medium text-slate-700">{progressPercentage}%</span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-1.5">
              <div 
                className="bg-slate-700 rounded-full h-1.5"
                style={{ width: `${progressPercentage}%` }}
              ></div>
            </div>
          </div>
          
          {/* Detection buttons row - disabled with coming soon message */}
          <div className="grid grid-cols-1 gap-2">
            {/* Combined Detection & Segmentation button - disabled with coming soon */}
            <Button
              disabled={true}
              className="bg-slate-500 text-white flex items-center justify-center py-3 rounded-md shadow-sm cursor-not-allowed"
            >
              <svg className="h-4 w-4 mr-1.5" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <span className="text-sm">ML Detection & Segmentation (Coming Soon)</span>
            </Button>
            
            {/* Additional note about manual detection */}
            <div className="bg-blue-50 text-blue-700 p-2 text-xs rounded-md border border-blue-200">
              <p className="flex items-center">
                <MapPin className="h-3 w-3 mr-1 flex-shrink-0" />
                <span>Use manual placement to add objects to the map</span>
              </p>
            </div>
          </div>
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
            <div className={`px-4 py-2 ${isFeatureSelectionMode ? 'bg-emerald-50' : 'bg-blue-50'} border-t border-gray-200`}>
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
                  style={{ width: `${validationItems.length > 0 ? (selectedReports.length / validationItems.length) * 100 : 0}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>
      ) : (
        // Empty div to maintain structure
        <div></div>
      )}

      {/* Thin separator line */}
      <div className={`border-t ${isFeatureSelectionMode ? 'border-emerald-100' : 'border-blue-100'}`}></div>
      
      {/* Main content - either Trees or Reports */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'trees' ? (
          <div className="flex flex-col h-full">
            {/* Counter and navigation - sleeker height */}
            <div className="flex-shrink-0 px-4 py-1 flex items-center justify-center">
              {/* Object navigation controls */}
              <div className="flex justify-between items-center w-full mb-1 bg-gray-50 rounded-md py-1 px-2 shadow-sm">
                <button 
                  onClick={goToPreviousTree}
                  className={`p-0.5 rounded ${isFeatureSelectionMode ? 'hover:bg-emerald-100 text-emerald-700' : 'hover:bg-blue-100 text-blue-700'}`}
                  disabled={trees.length === 0 || trees.filter(t => t.visible).length <= 1}
                >
                  <ChevronLeft size={16} />
                </button>
                
                <div className="text-center">
                  <span className="text-xs font-medium">
                    Object {trees.filter(t => t.visible).length > 0 ? currentTreeIndex + 1 : 0}/{trees.filter(t => t.visible).length}
                  </span>
                  <div className="text-xs text-gray-500">
                    {selectedTree?.validated 
                      ? <span className="text-green-500 flex items-center justify-center text-[10px]">
                          <Check size={9} className="mr-0.5" />
                          Approved
                        </span>
                      : <span className="text-amber-500 flex items-center justify-center text-[10px]">
                          <AlertTriangle size={9} className="mr-0.5" />
                          Pending
                        </span>
                    }
                  </div>
                </div>
                
                <button 
                  onClick={goToNextTree}
                  className={`p-0.5 rounded ${isFeatureSelectionMode ? 'hover:bg-emerald-100 text-emerald-700' : 'hover:bg-blue-100 text-blue-700'}`}
                  disabled={trees.length === 0 || trees.filter(t => t.visible).length <= 1}
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
            
            {/* Information card - with reduced spacing below */}
            <div className="px-4 pb-1">
              {trees.length === 0 ? (
                <div className="text-center flex flex-col items-center py-3 bg-gray-50 rounded-md">
                  <svg className="w-8 h-8 mb-2 text-gray-300" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span className="font-medium text-gray-600 text-xs">
                    {isFeatureSelectionMode ? 
                      "No objects in index. Please run detection first." :
                      "No objects available. Please run detection."}
                  </span>
                </div>
              ) : searchQuery && filteredTrees.length === 0 ? (
                <div className="text-center text-gray-500 py-3 bg-gray-50 rounded-md">
                  <span className="font-medium text-xs">No objects match your search query "{searchQuery}"</span>
                </div>
              ) : (
                /* Information card */
                (() => {
                  const visibleTrees = searchQuery ? filteredTrees : trees.filter(tree => tree.visible);
                  const tree = visibleTrees[currentTreeIndex] || visibleTrees[0];
                  
                  // If we have a valid tree to display
                  if (tree) {
                    return (
                      <div 
                        key={tree.id}
                        className={`p-2 rounded-lg w-full ${
                          selectedTree && selectedTree.id === tree.id 
                            ? isFeatureSelectionMode ? 'bg-emerald-50 border border-emerald-200' : 'bg-blue-50 border border-blue-200'
                            : 'bg-white border border-gray-200'
                        }`}
                      >
                        <div className="space-y-1.5">
                          {/* Object name */}
                          <h3 className="font-medium text-gray-800">
                            {tree.species || tree.tree_species ? 
                              (tree.species || tree.tree_species) : 
                              <span className="text-gray-400 italic">Unspecified</span>}
                          </h3>
                          
                          {/* Address - displayed more prominently but more compact */}
                          {tree.address && (
                            <div className="text-xs bg-white p-1 rounded border border-gray-200">
                              <div className="font-medium text-gray-700 leading-tight">Property Address</div>
                              <div className="text-gray-600 leading-tight">{tree.address}</div>
                            </div>
                          )}
                          
                          {/* Location coordinates */}
                          <div className="grid grid-cols-2 gap-x-2 text-xs">
                            <div>
                              <span className="text-gray-500">Lat:</span> {" "}
                              <span className="font-medium">
                                {tree.location ? tree.location[1].toFixed(6) : 'Unknown'}
                              </span>
                            </div>
                            <div>
                              <span className="text-gray-500">Lon:</span> {" "}
                              <span className="font-medium">
                                {tree.location ? tree.location[0].toFixed(6) : 'Unknown'}
                              </span>
                            </div>
                          </div>
                          
                          {/* Validation status indicator */}
                          {tree.validated && (
                            <div className="flex items-center justify-center mt-1 pt-1 border-t border-gray-100">
                              <span className="inline-flex items-center py-0.5 px-2 text-xs bg-green-50 text-green-700 rounded-full">
                                <Check className="h-3 w-3 mr-1" />
                                Approved
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  }
                  return null;
                })()
              )}
            </div>
            
            {/* Object details separator line - reduced spacing */}
            <div className="border-t border-gray-200 mx-4 my-1"></div>
            
            {/* Object details section */}
            {selectedTree && (
              <div className="px-4 pt-1 pb-2">
                {isEditing ? (
                  /* Edit Mode - Tree Properties Form */
                  <div className="bg-white rounded-md border shadow-sm overflow-hidden">
                    <div className="bg-slate-50 p-3 border-b">
                      <h3 className="text-base font-medium text-slate-700">Edit Object Information</h3>
                      <p className="text-xs text-slate-500 mt-1">Update detection data for this object</p>
                    </div>
                    
                    <div className="p-4 space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Species</label>
                        <input
                          type="text"
                          name="species"
                          value={formData.species}
                          onChange={handleInputChange}
                          className="w-full rounded-md border-gray-300 shadow-sm focus:border-slate-500 focus:ring-1 focus:ring-slate-500 p-2 text-sm"
                          placeholder="Enter tree species"
                        />
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Height (ft)</label>
                          <input
                            type="number"
                            name="height"
                            value={formData.height}
                            onChange={handleInputChange}
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-slate-500 focus:ring-1 focus:ring-slate-500 p-2 text-sm"
                            placeholder="Height in feet"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Diameter (in)</label>
                          <input
                            type="number"
                            name="diameter"
                            value={formData.diameter}
                            onChange={handleInputChange}
                            className="w-full rounded-md border-gray-300 shadow-sm focus:border-slate-500 focus:ring-1 focus:ring-slate-500 p-2 text-sm"
                            placeholder="Diameter in inches"
                          />
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Risk Level</label>
                        <div className="grid grid-cols-3 gap-2">
                          <label className={`flex-1 relative flex items-center border px-3 py-2.5 rounded-md cursor-pointer ${
                            formData.riskLevel === "new" 
                              ? "bg-slate-100 border-slate-300 ring-1 ring-slate-500" 
                              : "bg-white border-gray-300 hover:bg-gray-50"
                          }`}>
                            <input
                              type="radio"
                              name="riskLevel"
                              value="new"
                              checked={formData.riskLevel === "new"}
                              onChange={handleInputChange}
                              className="sr-only"
                            />
                            <span className="text-sm font-medium">New</span>
                          </label>
                          
                          <label className={`flex-1 relative flex items-center border px-3 py-2.5 rounded-md cursor-pointer ${
                            formData.riskLevel === "medium" 
                              ? "bg-orange-50 border-orange-300 ring-1 ring-orange-500" 
                              : "bg-white border-gray-300 hover:bg-gray-50"
                          }`}>
                            <input
                              type="radio"
                              name="riskLevel"
                              value="medium"
                              checked={formData.riskLevel === "medium"}
                              onChange={handleInputChange}
                              className="sr-only"
                            />
                            <span className="text-sm font-medium">Medium</span>
                          </label>
                          
                          <label className={`flex-1 relative flex items-center border px-3 py-2.5 rounded-md cursor-pointer ${
                            formData.riskLevel === "high" 
                              ? "bg-red-50 border-red-300 ring-1 ring-red-500" 
                              : "bg-white border-gray-300 hover:bg-gray-50"
                          }`}>
                            <input
                              type="radio"
                              name="riskLevel"
                              value="high"
                              checked={formData.riskLevel === "high"}
                              onChange={handleInputChange}
                              className="sr-only"
                            />
                            <span className="text-sm font-medium">Risk</span>
                          </label>
                        </div>
                      </div>
                      
                      {/* Source information */}
                      <div className="bg-slate-50 p-3 rounded-md border text-xs text-slate-500 mt-2">
                        <div className="flex items-center">
                          <span className="font-medium mr-1">Source:</span>
                          <span className="ml-auto bg-slate-100 text-slate-700 px-2 py-0.5 rounded">
                            {selectedTree.manually_placed ? 'Manual Placement' : 'ML Detection'}
                          </span>
                        </div>
                        
                        {!selectedTree.manually_placed && selectedTree.bbox && (
                          <div className="mt-2">
                            <span className="font-medium">Bounding Box:</span> [{selectedTree.bbox.map(v => Math.round(v)).join(', ')}]
                          </div>
                        )}
                      </div>
                      
                      {/* Action buttons */}
                      <div className="flex space-x-3 mt-6 pt-2 border-t">
                        <button
                          onClick={() => setIsEditing(false)}
                          className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2.5 rounded-md text-sm font-medium border border-gray-300 shadow-sm"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={saveTreeEdits}
                          className="flex-1 bg-slate-700 hover:bg-slate-800 text-white px-4 py-2.5 rounded-md text-sm font-medium shadow-sm"
                        >
                          Save Changes
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  /* View mode - compact layout with proper object details */
                  <div className="bg-white p-3 rounded-md border shadow-sm">
                    <div className="flex justify-between items-center pb-1.5 mb-2 border-b">
                      <h3 className="text-base font-medium text-gray-700">Details</h3>
                      <span className="text-xs font-medium px-2 py-1 bg-slate-100 rounded-md text-slate-700">
                        Object {currentTreeIndex + 1}/{trees.filter(t => t.visible).length}
                      </span>
                    </div>
                    
                    {/* Species name with more prominence */}
                    <div className="mb-3">
                      <div className="text-xs text-gray-500 mb-0.5 font-medium">Name / Species</div>
                      <div className="bg-gray-50 p-2 rounded border border-gray-200 font-medium">
                        {selectedTree.species ? selectedTree.species : 
                          <span className="text-gray-400 italic">Unspecified</span>}
                      </div>
                    </div>
                    
                    {/* Smaller details in grid */}
                    <div className="grid grid-cols-2 gap-x-2 gap-y-2 text-sm mb-3">
                      <div>
                        <div className="text-xs text-gray-500 mb-0.5 font-medium">Height (ft)</div>
                        <div className="bg-gray-50 p-1.5 rounded border border-gray-200 text-xs">
                          {selectedTree.height ? selectedTree.height : 
                            <span className="text-gray-400 italic">Unknown</span>}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500 mb-0.5 font-medium">DBH (in)</div>
                        <div className="bg-gray-50 p-1.5 rounded border border-gray-200 text-xs">
                          {selectedTree.diameter ? selectedTree.diameter : 
                            <span className="text-gray-400 italic">Unknown</span>}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500 mb-0.5 font-medium">Source</div>
                        <div className="bg-slate-50 text-slate-600 p-1.5 rounded border border-slate-200 text-xs">
                          {selectedTree.manually_placed ? 'Manual Placement' : 'ML Detection'}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-gray-500 mb-0.5 font-medium">Status</div>
                        <div className={`p-1.5 rounded border text-xs ${
                          selectedTree.risk_level === 'new' ? 'bg-slate-100 text-slate-700 border-slate-200' :
                          selectedTree.risk_level === 'medium' ? 'bg-orange-50 text-orange-600 border-orange-200' :
                          selectedTree.risk_level === 'high' ? 'bg-red-50 text-red-600 border-red-200' :
                          'bg-green-50 text-green-600 border-green-200'
                        }`}>
                          {selectedTree.risk_level === 'new' ? 'New' :
                           selectedTree.risk_level === 'medium' ? 'Medium' :
                           selectedTree.risk_level === 'high' ? 'Risk' :
                           'No Risk'}
                        </div>
                      </div>
                      
                    </div>
                    
                    {/* Action buttons at bottom */}
                    <div className="mt-3 pt-2 border-t">
                      <div className="flex space-x-2">
                        <button
                          onClick={validateTree}
                          className="flex-1 flex items-center justify-center bg-green-50 hover:bg-green-100 text-green-700 py-2 px-3 rounded-md border border-green-200 shadow-sm transition-all duration-150 hover:shadow"
                        >
                          <Check className="h-4 w-4 mr-1" />
                          <span className="text-xs font-medium">Approve</span>
                        </button>
                        
                        <button
                          onClick={startEditing}
                          className="flex-1 flex items-center justify-center bg-blue-50 hover:bg-blue-100 text-blue-700 py-2 px-3 rounded-md border border-blue-200 shadow-sm transition-all duration-150 hover:shadow"
                        >
                          <svg className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                          </svg>
                          <span className="text-xs font-medium">Edit</span>
                        </button>
                        
                        <button
                          onClick={removeTree}
                          className="flex-1 flex items-center justify-center bg-red-50 hover:bg-red-100 text-red-700 py-2 px-3 rounded-md border border-red-200 shadow-sm transition-all duration-150 hover:shadow"
                        >
                          <Trash className="h-4 w-4 mr-1" />
                          <span className="text-xs font-medium">Delete</span>
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
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
                          <span className="font-medium">Species:</span> {item.tree_species || 'Unknown'}
                        </div>
                        {item.tree_diameter && (
                          <div className="text-gray-700">
                            <span className="font-medium">Diameter:</span> {item.tree_diameter} inches
                          </div>
                        )}
                        <div className="text-gray-700">
                          <span className="font-medium">Risk:</span> {
                            item.risk_level === 'high' ? 'High' : 
                            item.risk_level === 'medium' ? 'Medium' : 
                            'Low'
                          }
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Thin separator line before footer */}
      <div className={`border-t ${isFeatureSelectionMode ? 'border-emerald-100' : 'border-blue-100'}`}></div>
      
      {/* Footer */}
      <div className="p-4 bg-gray-50 border-t shadow-inner">
        {isFeatureSelectionMode && activeTab === 'reports' ? (
          // Reports export button
          <div className="space-y-3">
            <button
              onClick={saveTreesToValidationQueue}
              className="flex items-center justify-center bg-emerald-600 hover:bg-emerald-700 text-white py-2.5 rounded-md w-full text-sm font-medium shadow-sm transition-all duration-150 hover:shadow hover:translate-y-[-1px]"
              disabled={selectedReports.length === 0}
            >
              <FileText className="h-4 w-4 mr-2" />
              <span className="font-medium">Export Reports ({selectedReports.length})</span>
            </button>
            
            <div className="mt-2 text-xs text-gray-600 bg-white p-2.5 rounded-md border border-gray-200 flex items-center justify-center">
              {selectedReports.length === 0 
                ? 'Select at least one report to export'
                : `${selectedReports.length} of ${validationItems.length} reports selected for export`}
            </div>
          </div>
        ) : (
          // Trees button
          <div>
            <button
              onClick={saveTreesToValidationQueue}
              className={`flex items-center justify-center ${
                isFeatureSelectionMode 
                ? 'bg-emerald-600 hover:bg-emerald-700' 
                : 'bg-slate-700 hover:bg-slate-800'
              } text-white py-1.5 rounded-md w-full text-sm font-medium shadow-sm transition-all duration-150 hover:shadow hover:translate-y-[-1px]`}
              disabled={validatedCount === 0}
            >
              <Save className="h-4 w-4 mr-2" />
              <span className="font-medium">{isFeatureSelectionMode && activeTab === 'trees' 
                ? 'Create Report'
                : isFeatureSelectionMode && activeTab === 'reports'
                ? `Export Reports (${validatedCount})`
                : `Save for Review (${validatedCount})`}</span>
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DetectionMode;