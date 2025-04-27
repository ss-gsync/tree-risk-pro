// src/components/visualization/MapView/Map3DToggle.jsx

import React from 'react';
import { Box, Grid } from 'lucide-react';

/**
 * A toggle button component for switching between 2D and 3D map views
 * 
 * @param {boolean} is3DMode - Current state of 3D mode
 * @param {function} onToggle - Function to call when toggling 3D mode
 * @param {boolean} disabled - Whether the toggle is disabled
 */
const Map3DToggle = ({ is3DMode, onToggle, disabled = false }) => {
  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      className={`
        map-3d-toggle
        flex items-center justify-center px-3 py-1
        rounded-md text-xs border 
        transition-all duration-100 ease-in-out
        ${disabled 
          ? 'opacity-60 cursor-not-allowed text-gray-400 bg-gray-50 border-gray-200' 
          : is3DMode 
            ? 'text-blue-600 bg-blue-50 hover:bg-blue-100 border-blue-200' 
            : 'text-gray-700 bg-white hover:bg-gray-50 border-gray-200'
        }
      `}
      title={is3DMode ? "Switch to 2D view" : "Switch to 3D view"}
    >
      {is3DMode ? "2D View" : "3D View"}
    </button>
  );
};

export default Map3DToggle;