// src/components/visualization/Detection/DetectionCategories.jsx
//
// A dedicated component for the Detection Categories section
// Extracted from Sidebar/index.jsx to improve code organization

import React, { useState, useEffect, useRef } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

/**
 * DetectionCategories Component
 * 
 * Handles all the functionality from the original Sidebar/index.jsx
 * implementation but in a cleaner React component.
 */
const DetectionCategories = () => {
  // State for controlling dropdown visibility (collapsed by default)
  const [isExpanded, setIsExpanded] = useState(false);
  const [detectedCount, setDetectedCount] = useState(0);
  const counterRef = useRef(null);
  
  // Updated categories to match Grounded SAM backend categories for tree risk assessment
  const classes = [
    { id: 'healthy_tree', label: 'Healthy Trees', color: '#16a34a', checked: true },
    { id: 'hazardous_tree', label: 'Hazardous Trees', color: '#8b5cf6', checked: true },
    { id: 'dead_tree', label: 'Dead Trees', color: '#6b7280', checked: true },
    { id: 'low_canopy_tree', label: 'Low Canopy', color: '#0ea5e9', checked: true },
    { id: 'pest_disease_tree', label: 'Pest/Disease', color: '#84cc16', checked: true },
    { id: 'flood_prone_tree', label: 'Flood-Prone', color: '#0891b2', checked: true },
    { id: 'utility_conflict_tree', label: 'Utility Conflict', color: '#3b82f6', checked: true },
    { id: 'structural_hazard_tree', label: 'Structural Hazard', color: '#0d9488', checked: true },
    { id: 'fire_risk_tree', label: 'Fire Risk', color: '#4f46e5', checked: true }
  ];
  
  // Initialize global state and listen for counter updates
  useEffect(() => {
    // Initialize visibility state to match updated categories
    window.visibleDetectionCategories = window.visibleDetectionCategories || {
      healthy_tree: true,
      hazardous_tree: true,
      dead_tree: true,
      low_canopy_tree: true,
      pest_disease_tree: true,
      flood_prone_tree: true,
      utility_conflict_tree: true,
      structural_hazard_tree: true,
      fire_risk_tree: true
    };
    
    // Initialize detection parameters - exactly as in the original code
    window.detectionParams = window.detectionParams || {
      confidenceThreshold: 0.3,
      model: 'grounded_sam',
      withSegmentation: true
    };
    
    // Listen for detection events to update the counter
    const handleDetectionDisplayed = (e) => {
      if (e.detail && e.detail.treeCount) {
        updateCounter(e.detail.treeCount);
      }
    };
    
    window.addEventListener('detectionDisplayed', handleDetectionDisplayed);
    
    // Also export the counter element to the window for other components
    if (counterRef.current) {
      window.detectedComponentsCounter = counterRef.current;
    }
    
    return () => {
      window.removeEventListener('detectionDisplayed', handleDetectionDisplayed);
    };
  }, []);
  
  // Function to update counter with animation - just like the original
  const updateCounter = (count) => {
    setDetectedCount(count);
    
    if (counterRef.current) {
      // Apply animation effect to counter
      counterRef.current.style.transition = 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
      counterRef.current.style.transform = 'scale(1.2)';
      counterRef.current.style.backgroundColor = 'rgba(13, 71, 161, 0.15)';
      
      // Reset after animation
      setTimeout(() => {
        if (counterRef.current) {
          counterRef.current.style.transform = 'scale(1)';
          counterRef.current.style.backgroundColor = 'rgba(241, 245, 249, 0.6)';
        }
      }, 300);
    }
  };
  
  // Update counter on component mount to show checked categories count
  useEffect(() => {
    // Count how many categories are checked
    const checkedCount = classes.filter(cls => cls.checked).length;
    setDetectedCount(checkedCount);
  }, []);
  
  // Handle category toggle - same functionality as the original
  const handleCategoryToggle = (cls, isChecked) => {
    // Update global state
    window.visibleDetectionCategories[cls.id] = isChecked;
    
    // Find all markers/boxes with this category
    const elements = document.querySelectorAll(`.detection-box.${cls.id}-box`);
    console.log(`Toggling visibility of ${elements.length} ${cls.id} elements to ${isChecked}`);
    
    // Update visibility
    elements.forEach(el => {
      el.style.display = isChecked ? 'block' : 'none';
      el.style.opacity = isChecked ? '1' : '0';
      el.style.visibility = isChecked ? 'visible' : 'hidden';
    });
    
    // Update the counter to show the number of checked categories
    const checkedCount = document.querySelectorAll('#detection-categories-section input[type="checkbox"]:checked').length;
    setDetectedCount(checkedCount);
    
    // Dispatch event for React components - exactly as in the original
    window.dispatchEvent(new CustomEvent('detectionCategoryToggle', {
      detail: { 
        category: cls.id, 
        visible: isChecked 
      }
    }));
  };
  
  return (
    <div id="detection-categories-section" className="px-3 py-2">
      {/* Clickable header with counter - serves as dropdown toggle */}
      <div 
        className="flex justify-between items-center px-2 py-1.5 bg-white border border-slate-200 rounded cursor-pointer hover:bg-slate-50"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center">
          {/* Neural network/segmentation inspired icon */}
          <svg 
            width="14" 
            height="14" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2" 
            strokeLinecap="round" 
            strokeLinejoin="round"
            className="mr-1.5 text-slate-500"
          >
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
            <line x1="12" y1="22.08" x2="12" y2="12"></line>
          </svg>
          <span className="text-sm font-medium text-slate-600">Detection Categories</span>
        </div>
        
        <div className="flex items-center">
          {/* Counter badge */}
          <span 
            ref={counterRef}
            id="detected-components-count"
            className="text-xs font-semibold text-slate-600 bg-slate-100 px-2 py-0.5 rounded border border-slate-200 mr-2"
          >
            {detectedCount}
          </span>
          
          {/* Dropdown indicator */}
          {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>
      
      {/* Categories grid with smoother animation */}
      <div 
        className="overflow-hidden transition-all duration-300 mt-2"
        style={{ 
          maxHeight: isExpanded ? '200px' : '0px',
          opacity: isExpanded ? 1 : 0
        }}
      >
        <div className="grid grid-cols-2 gap-1.5 p-2 border border-slate-200 rounded bg-white">
          {classes.map(cls => (
            <div key={cls.id} className="flex items-center mb-0.5">
              {/* Color indicator - exact same style as the original */}
              <div 
                className="w-2.5 h-2.5 rounded-full mr-2 flex-shrink-0"
                style={{ 
                  backgroundColor: cls.color,
                  boxShadow: 'inset 0 0 0 1px rgba(0,0,0,0.1)' 
                }}
              />
              
              {/* Checkbox with same ID as the original for compatibility */}
              <input
                type="checkbox"
                id={cls.id}
                defaultChecked={cls.checked}
                className="mr-1.5 h-3 w-3 cursor-pointer"
                onChange={(e) => handleCategoryToggle(cls, e.target.checked)}
              />
              
              {/* Label with same properties as the original */}
              <label 
                htmlFor={cls.id}
                className="text-xs cursor-pointer truncate"
                style={{
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}
              >
                {cls.label}
              </label>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DetectionCategories;