// src/components/visualization/MapView/TreeTaggingMode.jsx

import React, { useState, useEffect } from 'react';
import { Plus, X, Check, TreePine } from 'lucide-react';
import TreeTagForm from './TreeTagForm';

/**
 * A component for enabling and controlling the tree tagging mode
 * Allows users to tag trees in 3D view with enhanced visualization
 */
const TreeTaggingMode = ({ 
  mapRef, 
  isTaggingMode, 
  setIsTaggingMode, 
  is3DMode, 
  onTagComplete 
}) => {
  const [currentTag, setCurrentTag] = useState(null);
  const [tagHistory, setTagHistory] = useState([]);
  const [tempMarker, setTempMarker] = useState(null);
  const [showTagForm, setShowTagForm] = useState(false);
  
  // When tagging mode changes, set up listeners
  useEffect(() => {
    if (!mapRef?.current) return;
    
    let mapClickListener = null;
    
    // Set up map click listener when tagging mode is active
    if (isTaggingMode) {
      console.log('Enabling tree tagging mode');
      
      // Add instruction tooltip
      const map = mapRef.current.getMap();
      
      // Change cursor to crosshair
      if (map) {
        map.setOptions({ draggableCursor: 'crosshair' });
      }
      
      // Setup click listener for map
      mapClickListener = map?.addListener('click', handleMapClick);
    } else {
      // Reset cursor
      const map = mapRef.current.getMap();
      if (map) {
        map.setOptions({ draggableCursor: null });
      }
      
      // Remove temporary marker if present
      if (tempMarker) {
        tempMarker.setMap(null);
        setTempMarker(null);
      }
    }
    
    // Cleanup listener on mode change or unmount
    return () => {
      if (mapClickListener) {
        window.google?.maps?.event.removeListener(mapClickListener);
      }
      
      // Reset cursor
      const map = mapRef.current?.getMap();
      if (map) {
        map.setOptions({ draggableCursor: null });
      }
    };
  }, [isTaggingMode, mapRef]);
  
  // Handle map clicks when in tagging mode
  const handleMapClick = (event) => {
    if (!mapRef?.current || !isTaggingMode) return;
    
    const { latLng } = event;
    const lat = latLng.lat();
    const lng = latLng.lng();
    
    console.log(`Tree tagged at: ${lat}, ${lng}`);
    
    // Remove previous temporary marker
    if (tempMarker) {
      tempMarker.setMap(null);
    }
    
    // Create a new marker at the clicked location
    const map = mapRef.current.getMap();
    
    // Create a tree marker with custom icon
    const newMarker = new window.google.maps.Marker({
      position: { lat, lng },
      map: map,
      icon: {
        path: 'M12 22c4.2 0 8-3.22 8-8.2c0-3.18-2.45-6.92-7.34-11.23c-.38-.33-.95-.33-1.33 0C6.45 6.88 4 10.62 4 13.8C4 18.78 7.8 22 12 22z',
        fillColor: '#10b981', // Emerald green
        fillOpacity: 0.9,
        strokeWeight: 1.5,
        strokeColor: 'white',
        strokeOpacity: 0.9,
        scale: 1.8,
        anchor: new window.google.maps.Point(12, 22),
      },
      title: 'New Tree Tag',
      animation: window.google.maps.Animation.DROP,
      draggable: true // Allow the marker to be adjusted
    });
    
    // Set this as the current temporary marker
    setTempMarker(newMarker);
    
    // Create tag data object
    const newTag = {
      id: `tree-tag-${Date.now()}`,
      position: { lat, lng },
      timestamp: new Date().toISOString(),
      is3D: is3DMode
    };
    
    // Set as current tag
    setCurrentTag(newTag);
    
    // Add drag end listener to update position
    newMarker.addListener('dragend', () => {
      const newPosition = newMarker.getPosition();
      setCurrentTag(prev => ({
        ...prev,
        position: { 
          lat: newPosition.lat(), 
          lng: newPosition.lng() 
        }
      }));
    });
  };
  
  // Function to open the form for more details
  const openTagForm = () => {
    setShowTagForm(true);
  };
  
  // Function to handle form submission
  const handleFormSubmit = (formData) => {
    if (!currentTag) return;
    
    // Add form data to the tag
    const completeTag = {
      ...currentTag,
      details: formData,
      timestamp: new Date().toISOString()
    };
    
    // Add to tag history
    setTagHistory(prev => [...prev, completeTag]);
    
    // Callback with tag data
    if (onTagComplete) {
      onTagComplete(completeTag);
    }
    
    // Convert temp marker to permanent marker
    if (tempMarker) {
      // Change icon to be permanent (slightly different design)
      tempMarker.setIcon({
        path: 'M12 22c4.2 0 8-3.22 8-8.2c0-3.18-2.45-6.92-7.34-11.23c-.38-.33-.95-.33-1.33 0C6.45 6.88 4 10.62 4 13.8C4 18.78 7.8 22 12 22z',
        fillColor: '#059669', // Darker green
        fillOpacity: 0.9,
        strokeWeight: 1.5,
        strokeColor: 'white',
        strokeOpacity: 0.9,
        scale: 1.8,
        anchor: new window.google.maps.Point(12, 22),
      });
      
      // Make it non-draggable now that it's saved
      tempMarker.setDraggable(false);
      
      // Add to the map reference marker array for proper tracking
      if (mapRef.current) {
        mapRef.current.getMarkers().push(tempMarker);
      }
      
      // Clear temp marker reference
      setTempMarker(null);
    }
    
    // Reset current tag and close form
    setCurrentTag(null);
    setShowTagForm(false);
  };
  
  // Function to cancel the current tag
  const cancelCurrentTag = () => {
    // Remove the temporary marker
    if (tempMarker) {
      tempMarker.setMap(null);
      setTempMarker(null);
    }
    
    // Reset current tag and close form
    setCurrentTag(null);
    setShowTagForm(false);
  };
  
  return (
    <div className="flex flex-col items-center">
      {/* Button to enable/disable tagging mode */}
      <button
        onClick={() => setIsTaggingMode(!isTaggingMode)}
        className={`
          flex items-center justify-center p-2 rounded-md w-full mb-2
          ${isTaggingMode 
            ? 'bg-green-500 text-white hover:bg-green-600' 
            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }
        `}
        title={isTaggingMode ? "Exit tagging mode" : "Enter tree tagging mode"}
      >
        <TreePine className="h-4 w-4 mr-1.5" />
        <span className="text-xs font-medium">
          {isTaggingMode ? "Exit Tagging Mode" : "Tag Trees"}
        </span>
      </button>
      
      {/* Action buttons for current tag */}
      {isTaggingMode && currentTag && !showTagForm && (
        <div className="flex space-x-2 mt-1 w-full">
          <button
            onClick={openTagForm}
            className="flex items-center justify-center p-1.5 bg-green-500 text-white rounded-md flex-1 hover:bg-green-600"
            title="Add tree details"
          >
            <Check className="h-3.5 w-3.5 mr-1" />
            <span className="text-xs">Add Details</span>
          </button>
          
          <button
            onClick={cancelCurrentTag}
            className="flex items-center justify-center p-1.5 bg-red-100 text-red-700 rounded-md flex-1 hover:bg-red-200"
            title="Cancel this tree tag"
          >
            <X className="h-3.5 w-3.5 mr-1" />
            <span className="text-xs">Cancel</span>
          </button>
        </div>
      )}
      
      {/* Form overlay for adding tree details */}
      {showTagForm && currentTag && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="max-w-md w-full mx-auto">
            <TreeTagForm 
              onSubmit={handleFormSubmit}
              onCancel={cancelCurrentTag}
            />
          </div>
        </div>
      )}
      
      {/* Instructions when in tagging mode */}
      {isTaggingMode && !currentTag && !showTagForm && (
        <p className="text-xs text-gray-500 italic mt-1">
          {is3DMode 
            ? "Click on the 3D map to tag a tree location" 
            : "Enable 3D mode for best tagging accuracy"
          }
        </p>
      )}
      
      {/* Tags count */}
      {tagHistory.length > 0 && (
        <div className="text-xs text-gray-600 mt-2">
          <span className="font-medium">{tagHistory.length}</span> trees tagged
        </div>
      )}
    </div>
  );
};

export default TreeTaggingMode;