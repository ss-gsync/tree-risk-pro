// src/components/visualization/Detection/index.js
// Unified detection module that integrates all functionality

// Export all detection-related components for easier importing
export * as MLOverlay from './MLOverlay';
export { default as ObjectDetection } from './ObjectDetection';
export { default as DetectionPreview } from './DetectionPreview';
export { default as DetectionSidebar } from './DetectionSidebar';
export { default as DetectionSidebarBridge } from './DetectionSidebarBridge';
export { default as DetectionRenderer } from './DetectionRenderer';
export { default as DetectionCategories } from './DetectionCategories';
export { default as DetectionMode } from './DetectionMode';
export { default as MLOverlayInitializer } from './MLOverlayInitializer';
export { default as TreeInfo } from './TreeInfo';
export { default as TreeTagForm } from './TreeTagForm';

// Import the MLOverlay module to ensure it's loaded
import * as MLOverlayModule from './MLOverlay';

// Export service functions
export { 
  loadDetectionData, 
  loadDetectionMetadata, 
  applyDetectionOverlay,
  initializeMap
} from './detectionService';

// Re-export MLOverlay module for easy access
export { MLOverlayModule };

// Import the detection preview functions
import { showDetectionPreview, destroyDetectionPreview } from './DetectionPreview';
// Import the detection service
import * as detectionService from './detectionService';

// Define the tree categories for consistency
const treeCategories = [
  'healthy_tree',
  'hazardous_tree',
  'dead_tree',
  'low_canopy_tree',
  'pest_disease_tree',
  'flood_prone_tree',
  'utility_conflict_tree',
  'structural_hazard_tree',
  'fire_risk_tree'
];

// Function to show detection preview with improved data handling
function enhancedShowDetectionPreview(data) {
  if (!data) return;
  
  // Show the preview directly
  console.log("Showing detection preview immediately");
  showDetectionPreview(data);
}

// Function to apply detection overlay using the new unified service
async function enhancedApplyDetection(detectionId, appendMode = false) {
  console.log(`Enhanced apply detection: ${detectionId}, append: ${appendMode}`);
  
  try {
    // Use the integrated detection service
    const result = await detectionService.applyDetectionOverlay(detectionId, appendMode);
    
    if (result) {
      console.log(`Applied detection ${detectionId} successfully with ${result.trees?.length || 0} trees`);
      
      // Also show in preview if requested
      if (window.detectionPreviewEnabled) {
        enhancedShowDetectionPreview(result);
      }
      
      return result;
    } else {
      console.error(`Failed to apply detection ${detectionId}`);
      return null;
    }
  } catch (error) {
    console.error(`Error in enhancedApplyDetection: ${error.message}`);
    return null;
  }
}

// Export functions globally for easy access from anywhere
console.log('DETECTION INDEX: Setting up global functions');
window.showDetectionPreview = enhancedShowDetectionPreview;
window.destroyDetectionPreview = destroyDetectionPreview;

// Export the new unified detection application function
window.applyDetection = enhancedApplyDetection;

// For direct access to the detection service
window.detectionService = detectionService;

console.log('DETECTION INDEX: Global functions set', 
           'showDetectionPreview:', typeof window.showDetectionPreview === 'function',
           'destroyDetectionPreview:', typeof window.destroyDetectionPreview === 'function',
           'applyDetection:', typeof window.applyDetection === 'function');

// Export for module imports
export { 
  enhancedShowDetectionPreview, 
  destroyDetectionPreview,
  enhancedApplyDetection as applyDetection,
  detectionService,
  treeCategories
};

// Add a function to remove the detection overlay
export function removeDetectionOverlay() {
  if (typeof window.removeDetectionOverlay === 'function') {
    return window.removeDetectionOverlay();
  } else {
    // Manual fallback
    const overlays = document.querySelectorAll('#ml-detection-overlay, #detection-boxes-container');
    overlays.forEach(el => el.parentNode && el.parentNode.removeChild(el));
    return true;
  }
}

// Make remove function available globally
window.removeDetectionOverlay = removeDetectionOverlay;

// Export a complete treeDetection object for compatibility with standalone scripts
// Get functions from the MLOverlay module - handle the case where they're not defined yet
const renderMLOverlay = MLOverlayModule.renderMLOverlay || 
                      ((...args) => {
                         console.log('Deferring renderMLOverlay call until Google Maps is loaded');
                         return setTimeout(() => {
                           if (window.renderMLOverlay) window.renderMLOverlay(...args);
                         }, 1000);
                       });
const mlRemoveOverlay = MLOverlayModule.removeMLOverlay;
const updateOverlayOpacity = MLOverlayModule.updateMLOverlayOpacity;
const updateDetectionClasses = MLOverlayModule.updateMLOverlayClasses;

// Make these available globally with safety checks
window.renderDetectionOverlay = renderMLOverlay; // For backward compatibility
window.mlUpdateOverlayOpacity = updateOverlayOpacity;
window.mlUpdateDetectionClasses = updateDetectionClasses;
window.ensureMLOverlayInitialized = MLOverlayModule.ensureInitialized;

window.treeDetection = {
  ...detectionService,
  applyDetection: enhancedApplyDetection,
  removeOverlay: removeDetectionOverlay,
  renderDetectionOverlay: renderMLOverlay, // For backward compatibility
  renderMLOverlay: renderMLOverlay,
  updateOverlayOpacity,
  updateDetectionClasses,
  ensureInitialized: MLOverlayModule.ensureInitialized
};