// Detection Preview Component - Popup for ML Results
import React, { useState, useEffect } from 'react';
import * as ReactDOM from 'react-dom';
import { 
  BarChart, X, Check, Info, AlertTriangle, Eye, Layers, ChevronDown, 
  Maximize2, Camera, FileBarChart2, CircleOff, CircleCheck,
  Building, Zap, TreePine, Filter, Tag, MapPin
} from 'lucide-react';

/**
 * DetectionPreview Component as Popup
 */
const DetectionPopup = ({ data, onClose }) => {
  console.log('DETECTION PREVIEW: Data received:', data);
  
  // Always prioritize showing results if we have any trees data
  const hasData = Array.isArray(data?.trees) && data.trees.length > 0;
  
  // Only show preliminary state briefly, especially if we have actual data
  // Force isPreliminary to false if we have tree data
  const isPreliminary = hasData ? false : (data?.status === 'processing' || data?._preliminary === true);
  
  // IMPORTANT: Always use the job_id from the data object returned by the server
  // This is the SINGLE SOURCE OF TRUTH for job IDs
  const jobId = data?.job_id;
  console.log('DETECTION PREVIEW: Using server-returned job ID:', jobId);
  
  // Also update the global currentDetectionJobId to maintain consistency
  if (jobId) {
    window.currentDetectionJobId = jobId;
    console.log('DETECTION PREVIEW: Updated global job ID to server value:', jobId);
  }
  
  // Image paths
  // Keep the full path with /ttt prefix since that's how the Express server is configured
  // Check if jobId already starts with "detection_" to avoid duplication
  const jobPrefix = jobId && !jobId.startsWith('detection_') ? 'detection_' : '';
  const formattedJobId = jobId ? `${jobPrefix}${jobId}` : null;
  
  // Use formatted job ID to avoid duplication in paths
  const visualizationImage = formattedJobId ? 
    `/ttt/data/ml/detection_${formattedJobId.replace('detection_', '')}/ml_response/combined_visualization.jpg` : null;
  const satelliteImage = formattedJobId ? 
    `/ttt/data/ml/detection_${formattedJobId.replace('detection_', '')}/satellite_${formattedJobId.replace('detection_', '')}.jpg` : null;
    
  // State for fullscreen view
  const [fullscreenImage, setFullscreenImage] = useState(null);
  
  console.log('DETECTION PREVIEW: Using sanitized job ID:', formattedJobId ? formattedJobId.replace('detection_', '') : 'none');
    
  console.log('DETECTION PREVIEW: Image paths:', { visualizationImage, satelliteImage });
  
  // Debug image loading by creating test image elements
  useEffect(() => {
    if (visualizationImage) {
      const img = new Image();
      img.onload = () => console.log(`Successfully loaded visualization image: ${visualizationImage}`);
      img.onerror = (err) => console.error(`Failed to load visualization image: ${visualizationImage}`, err);
      img.src = visualizationImage;
    }
    
    if (satelliteImage) {
      const img = new Image();
      img.onload = () => console.log(`Successfully loaded satellite image: ${satelliteImage}`);
      img.onerror = (err) => console.error(`Failed to load satellite image: ${satelliteImage}`, err);
      img.src = satelliteImage;
    }
  }, [visualizationImage, satelliteImage]);
  
  // Tree counts
  const treeCount = Array.isArray(data?.trees) ? data.trees.length : 0;
  
  // Get counts by category
  const treeCategoryCounts = {
    healthy_tree: 0,
    hazardous_tree: 0,
    dead_tree: 0,
    low_canopy_tree: 0,
    pest_disease_tree: 0,
    flood_prone_tree: 0,
    utility_conflict_tree: 0,
    structural_hazard_tree: 0,
    fire_risk_tree: 0
  };
  
  // Count trees by category
  if (Array.isArray(data?.trees)) {
    data.trees.forEach(tree => {
      const category = tree.category || tree.class || 'healthy_tree';
      if (treeCategoryCounts.hasOwnProperty(category)) {
        treeCategoryCounts[category]++;
      } else {
        // Default to healthy if unknown category
        treeCategoryCounts.healthy_tree++;
      }
    });
  }
  
  // Category display names mapping
  const categoryLabels = {
    healthy_tree: 'Healthy Trees',
    hazardous_tree: 'Hazardous Trees',
    dead_tree: 'Dead Trees',
    low_canopy_tree: 'Low Canopy',
    pest_disease_tree: 'Pest/Disease',
    flood_prone_tree: 'Flood-Prone',
    utility_conflict_tree: 'Utility Conflict',
    structural_hazard_tree: 'Structural Hazard',
    fire_risk_tree: 'Fire Risk'
  };
  
  // Category colors
  const categoryColors = {
    healthy_tree: '#16a34a',
    hazardous_tree: '#8b5cf6',
    dead_tree: '#6b7280',
    low_canopy_tree: '#0ea5e9',
    pest_disease_tree: '#84cc16',
    flood_prone_tree: '#0891b2',
    utility_conflict_tree: '#3b82f6',
    structural_hazard_tree: '#0d9488',
    fire_risk_tree: '#4f46e5'
  };
  
  // Stats
  const detectionTime = data?.timestamp;
  const modelType = data?.model_type || data?.metadata?.model || 'Standard Detection';
  
  // Tab state
  const [activeTab, setActiveTab] = useState('results');
  
  // Handle view on map click
  const handleViewOnMap = () => {
    // Create detection data
    const detectionData = { ...data, job_id: jobId };
    
    // Set global flags
    window.detectionShowOverlay = true;
    window.mlDetectionData = detectionData;
    
    // Dispatch event
    window.dispatchEvent(new CustomEvent('enterDetectionMode', {
      detail: {
        jobId: jobId,
        data: detectionData,
        showOverlay: true,
        forceVisible: true,
        forceRenderBoxes: true
      }
    }));
    
    // Close the popup
    setTimeout(() => onClose(), 100);
  };
  
  // Escape key handler
  useEffect(() => {
    const handleEscKey = (event) => {
      if (event.key === 'Escape') onClose();
    };
    
    document.addEventListener('keydown', handleEscKey);
    return () => document.removeEventListener('keydown', handleEscKey);
  }, [onClose]);
  
  if (!data) return null;
  
  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-[9999] overflow-y-auto backdrop-blur-sm">
      <div 
        className="bg-white rounded-md shadow-xl w-full mx-4 md:mx-auto overflow-hidden flex flex-col"
        style={{ maxWidth: '800px', maxHeight: '90vh' }}
      >
        {/* Header */}
        <div className="flex justify-between items-center p-3 bg-slate-100 border-b border-slate-200">
          <div>
            <h2 className="text-md font-bold flex items-center">
              <BarChart size={16} className="mr-1.5 text-blue-600" />
              Detection Results
            </h2>
            {/* Status messages removed as requested */}
          </div>
          
          <div className="flex items-center space-x-2">
            {/* Tabs */}
            <div className="hidden md:flex bg-slate-200 rounded-sm p-0.5 mr-2">
              <button
                onClick={() => setActiveTab('results')}
                className={`px-3 py-0.5 text-xs font-medium rounded-sm ${
                  activeTab === 'results' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-700'
                }`}
              >
                Results
              </button>
              <button
                onClick={() => setActiveTab('stats')}
                className={`px-3 py-0.5 text-xs font-medium rounded-sm ${
                  activeTab === 'stats' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-700'
                }`}
              >
                Stats
              </button>
              <button
                onClick={() => setActiveTab('details')}
                className={`px-3 py-0.5 text-xs font-medium rounded-sm ${
                  activeTab === 'details' ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-700'
                }`}
              >
                Details
              </button>
            </div>
            
            <button 
              onClick={onClose}
              className="p-1 rounded-sm hover:bg-slate-200"
            >
              <X size={16} />
            </button>
          </div>
        </div>
        
        {/* Content */}
        <div className="p-4 overflow-y-auto">
          {/* Results Tab */}
          {activeTab === 'results' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Image column */}
              <div>
                <h3 className="text-sm font-medium text-slate-700 mb-2">Detection Visualization</h3>
                <div className="border border-slate-200 rounded-md bg-white overflow-hidden shadow-sm" style={{ height: '240px' }}>
                  {visualizationImage ? (
                    <div className="relative w-full h-full">
                      <img 
                        src={visualizationImage} 
                        alt="ML detection visualization" 
                        className="w-full h-full object-cover cursor-pointer"
                        onClick={() => setFullscreenImage(visualizationImage)}
                        onLoad={() => console.log('Visualization image loaded successfully in UI')}
                        onError={(e) => {
                          console.log('Error loading visualization image, trying satellite image');
                          // Try satellite image as fallback
                          if (satelliteImage) {
                            console.log('Trying satellite image instead:', satelliteImage);
                            e.target.src = satelliteImage;
                            e.target.alt = "Satellite image";
                            
                            // Add onerror handler for satellite image too
                            e.target.onerror = () => {
                              console.log('Satellite image also failed to load');
                              // If both fail, show a fallback message
                              const parent = e.target.parentNode;
                              if (parent) {
                                const fallback = document.createElement('div');
                                fallback.className = "absolute inset-0 flex flex-col items-center justify-center bg-slate-50";
                                fallback.innerHTML = `
                                  <svg class="w-8 h-8 text-slate-300 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                  </svg>
                                  <p class="text-xs text-slate-500">Failed to load images</p>
                                  <p class="text-xs text-slate-400">Job ID: ${jobId || 'Unknown'}</p>
                                `;
                                parent.appendChild(fallback);
                              }
                            };
                          }
                        }}
                      />
                      <div className="absolute bottom-2 right-2 bg-white bg-opacity-75 text-xs p-1 rounded">
                        Job ID: {jobId}
                      </div>
                      <div className="absolute top-2 right-2 bg-white bg-opacity-75 rounded p-1">
                        <Maximize2 
                          size={16} 
                          className="text-slate-600 cursor-pointer"
                          onClick={() => setFullscreenImage(visualizationImage)} 
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center bg-slate-50">
                      <Camera size={36} className="text-slate-300 mb-2" />
                      <p className="text-xs text-slate-500">Visualization not available</p>
                      <p className="text-xs text-slate-400">{jobId || 'Unknown job ID'}</p>
                    </div>
                  )}
                  
                  {/* Fullscreen Image Modal */}
                  {fullscreenImage && (
                    <div 
                      className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-[10000]"
                      onClick={() => setFullscreenImage(null)}
                    >
                      <div className="relative max-w-[90vw] max-h-[90vh]">
                        <img 
                          src={fullscreenImage} 
                          alt="Detection visualization fullscreen" 
                          className="max-w-full max-h-[90vh] object-contain"
                        />
                        <button 
                          className="absolute top-4 right-4 bg-white bg-opacity-75 rounded-full p-2"
                          onClick={() => setFullscreenImage(null)}
                        >
                          <X size={20} className="text-black" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Results column */}
              <div>
                <h3 className="text-sm font-medium text-slate-700 mb-2">Detection Summary</h3>
                <div className="border border-slate-200 rounded-md bg-white p-3 shadow-sm h-full">
                  {treeCount > 0 ? (
                    <div className="space-y-3 h-full flex flex-col">
                      <div className="flex items-center justify-between p-2 bg-green-50 rounded border border-green-100">
                        <div className="flex items-center">
                          <TreePine size={16} className="text-green-600 mr-2" />
                          <span className="text-sm text-slate-700">Trees Detected</span>
                        </div>
                        <span className="text-sm font-medium text-green-700">{treeCount}</span>
                      </div>
                      
                      <div className="flex-grow overflow-y-auto">
                        <div className="text-xs text-slate-600 mb-2 font-semibold">Analysis Results:</div>
                        <div className="text-sm text-slate-700 leading-relaxed">
                          <p>
                            Satellite imagery analysis completed successfully using {modelType} machine learning model. 
                            {treeCount > 0 ? ` A total of ${treeCount} trees were detected in the target area.` : ' No trees were detected in the target area.'}
                          </p>
                          
                          {/* Only show category breakdown if we have trees */}
                          {treeCount > 0 && (
                            <div className="mt-2">
                              <div className="text-xs font-medium mb-1.5">Categories breakdown:</div>
                              <div className="grid grid-cols-2 gap-x-2 gap-y-1">
                                {Object.entries(treeCategoryCounts).map(([category, count]) => 
                                  count > 0 ? (
                                    <div key={category} className="flex items-center text-xs">
                                      <div className="w-2 h-2 rounded-full mr-1" style={{backgroundColor: categoryColors[category]}}></div>
                                      <span className="truncate" title={categoryLabels[category]}>{categoryLabels[category]}:</span>
                                      <span className="ml-auto font-medium">{count}</span>
                                    </div>
                                  ) : null
                                )}
                              </div>
                            </div>
                          )}
                          
                          <p className="mt-2 text-xs text-slate-500">
                            To view the detected trees on the map, click the "View on Map" button below.
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-4 h-full flex flex-col items-center justify-center">
                      <div className="flex flex-col items-center justify-center text-slate-500">
                        <AlertTriangle size={24} className="mb-2 text-amber-500" />
                        <span className="text-sm font-medium">No trees detected</span>
                        <p className="text-xs text-slate-400 mt-1">
                          Try a different location or adjust the view
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          
          {/* Stats Tab */}
          {activeTab === 'stats' && (
            <div className="bg-white rounded-md border border-slate-200 shadow-sm p-4">
              <h3 className="text-sm font-semibold text-slate-800 mb-3">Tree Detection Analytics</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="bg-green-50 rounded-md p-3 border border-green-100">
                  <div className="text-xs text-green-700 mb-1">Trees Detected</div>
                  <div className="text-xl font-bold text-green-800">{treeCount}</div>
                  <div className="text-xs text-green-600 mt-1">Total count from aerial imagery</div>
                </div>
                
                <div className="bg-blue-50 rounded-md p-3 border border-blue-100">
                  <div className="text-xs text-blue-700 mb-1">Detection Quality</div>
                  <div className="text-xl font-bold text-blue-800">
                    {treeCount > 10 ? 'High' : treeCount > 0 ? 'Medium' : 'N/A'}
                  </div>
                  <div className="text-xs text-blue-600 mt-1">Based on detection confidence</div>
                </div>
                
                <div className="bg-indigo-50 rounded-md p-3 border border-indigo-100">
                  <div className="text-xs text-indigo-700 mb-1">Model</div>
                  <div className="text-lg font-bold text-indigo-800 truncate" title={modelType}>
                    {modelType}
                  </div>
                  <div className="text-xs text-indigo-600 mt-1">Machine learning algorithm</div>
                </div>
              </div>
              
              <div className="border-t border-slate-100 pt-4 pb-2">
                <h4 className="text-sm font-medium text-slate-700 mb-3">Detection Metrics</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-2">
                  <div className="flex justify-between py-1 border-b border-slate-100">
                    <span className="text-slate-600 text-xs font-medium">Detection Time</span>
                    <span className="text-slate-800 text-xs">
                      {new Date(detectionTime).toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <div className="flex justify-between py-1 border-b border-slate-100">
                    <span className="text-slate-600 text-xs font-medium">Detection Date</span>
                    <span className="text-slate-800 text-xs">
                      {new Date(detectionTime).toLocaleDateString()}
                    </span>
                  </div>
                  
                  <div className="flex justify-between py-1 border-b border-slate-100">
                    <span className="text-slate-600 text-xs font-medium">Average Confidence</span>
                    <span className="text-slate-800 text-xs">
                      {Array.isArray(data?.trees) && data.trees.length > 0
                        ? `${Math.round(data.trees.reduce((sum, tree) => sum + (tree.confidence || 0), 0) / data.trees.length * 100)}%`
                        : 'N/A'}
                    </span>
                  </div>
                  
                  <div className="flex justify-between py-1 border-b border-slate-100">
                    <span className="text-slate-600 text-xs font-medium">Processing Status</span>
                    <span className="text-xs">
                      <span className="text-green-600 flex items-center">
                        <Check size={12} className="mr-1" />
                        Complete
                      </span>
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Category distribution section */}
              {treeCount > 0 && (
                <div className="border-t border-slate-100 pt-4">
                  <h4 className="text-sm font-medium text-slate-700 mb-3">Category Distribution</h4>
                  <div className="grid grid-cols-1 gap-y-2">
                    {Object.entries(treeCategoryCounts).map(([category, count]) => {
                        const percentage = count > 0 ? Math.round((count / treeCount) * 100) : 0;
                        return (
                          <div key={category} className="flex items-center">
                            <div className="w-3 h-3 rounded-full mr-2" style={{backgroundColor: categoryColors[category]}}></div>
                            <span className="text-xs text-slate-600">{categoryLabels[category]}</span>
                            <div className="ml-auto flex items-center">
                              <span className="text-xs font-medium text-slate-700 mr-2">{count}</span>
                              <div className="w-20 bg-slate-100 h-1.5 rounded overflow-hidden">
                                <div 
                                  className="h-full rounded" 
                                  style={{
                                    width: `${percentage}%`, 
                                    backgroundColor: categoryColors[category]
                                  }}
                                ></div>
                              </div>
                              <span className="text-xs text-slate-500 ml-1">{percentage}%</span>
                            </div>
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Details Tab */}
          {activeTab === 'details' && (
            <div className="bg-white rounded-md border border-slate-200 shadow-sm p-4">
              <h3 className="text-sm font-semibold text-slate-800 mb-3">Technical Details</h3>
              
              <div className="space-y-4">
                <div className="border rounded-md p-3 bg-slate-50">
                  <h4 className="text-xs font-semibold text-slate-700 mb-2">Detection Parameters</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-600 text-xs">Job ID:</span>
                      <span className="font-medium text-slate-800 text-xs">{jobId || 'N/A'}</span>
                    </div>
                    
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-600 text-xs">Model:</span>
                      <span className="font-medium text-slate-800 text-xs">{modelType}</span>
                    </div>
                    
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-600 text-xs">Timestamp:</span>
                      <span className="font-medium text-slate-800 text-xs">{detectionTime}</span>
                    </div>
                    
                    <div className="flex justify-between py-1">
                      <span className="text-slate-600 text-xs">Imagery Source:</span>
                      <span className="font-medium text-slate-800 text-xs">Satellite</span>
                    </div>
                  </div>
                </div>
                
                <div className="border rounded-md p-3 bg-slate-50">
                  <h4 className="text-xs font-semibold text-slate-700 mb-2">Processing Information</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-600 text-xs">Status:</span>
                      <span className="font-medium text-xs flex items-center">
                        <span className="text-green-600 flex items-center">
                          <Check size={12} className="mr-1" />
                          Complete
                        </span>
                      </span>
                    </div>
                    
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-600 text-xs">Tree Count:</span>
                      <span className="font-medium text-slate-800 text-xs">{treeCount}</span>
                    </div>
                    
                    <div className="flex justify-between py-1 border-b border-slate-100">
                      <span className="text-slate-600 text-xs">Detection Path:</span>
                      <span className="font-medium text-slate-800 text-xs truncate" style={{maxWidth: '200px'}} title={data?.ml_response_dir || 'N/A'}>
                        {data?.ml_response_dir || 'N/A'}
                      </span>
                    </div>
                    
                    <div className="flex justify-between py-1">
                      <span className="text-slate-600 text-xs">API Version:</span>
                      <span className="font-medium text-slate-800 text-xs">v0.2.3</span>
                    </div>
                  </div>
                </div>
                
                {/* Visualization info */}
                <div className="border rounded-md p-3 bg-slate-50">
                  <h4 className="text-xs font-semibold text-slate-700 mb-2">Visualization Assets</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center py-1 border-b border-slate-100">
                      <div className="w-6 text-center">
                        {visualizationImage ? 
                          <Check size={12} className="text-green-600 inline" /> : 
                          <CircleOff size={12} className="text-slate-400 inline" />
                        }
                      </div>
                      <span className="text-slate-600">Detection Visualization</span>
                    </div>
                    
                    <div className="flex items-center py-1">
                      <div className="w-6 text-center">
                        {satelliteImage ? 
                          <Check size={12} className="text-green-600 inline" /> : 
                          <CircleOff size={12} className="text-slate-400 inline" />
                        }
                      </div>
                      <span className="text-slate-600">Satellite Image</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Footer */}
        <div className="border-t border-slate-200 p-3 bg-white flex items-center justify-between">
          <div className="text-xs text-slate-500">
            <MapPin size={12} className="inline mr-1" />
            Ready for map display
          </div>
          
          <div className="flex space-x-2">
            <button
              onClick={onClose}
              className="px-3 py-1 border border-slate-300 bg-white text-slate-700 rounded-sm text-xs"
            >
              <X size={12} className="inline mr-1" />
              Close
            </button>
            <button
              onClick={handleViewOnMap}
              className="px-3 py-1 bg-blue-600 text-white rounded-sm text-xs"
              style={{ backgroundColor: '#0d47a1' }}
            >
              <Eye size={12} className="inline mr-1" />
              View on Map
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Render the popup into a container
 */
function renderPopup(data, container) {
  try {
    console.log('DETECTION PREVIEW: Rendering popup');
    
    // Create container for React
    const freshContainer = document.createElement('div');
    freshContainer.id = 'detection-popup-inner-container';
    freshContainer.style.width = '100%';
    freshContainer.style.height = '100%';
    
    // Clear existing content
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    container.appendChild(freshContainer);
    
    // Use React 18 createRoot if available
    if (ReactDOM.createRoot) {
      const root = ReactDOM.createRoot(freshContainer);
      container._reactRoot = root;
      root.render(<DetectionPopup data={data} onClose={() => destroyDetectionPreview()} />);
    } else {
      ReactDOM.render(
        <DetectionPopup data={data} onClose={() => destroyDetectionPreview()} />,
        freshContainer
      );
    }
  } catch (error) {
    console.error('DETECTION PREVIEW: Error rendering popup:', error);
  }
}

/**
 * Show the detection preview
 */
function showDetectionPreview(data) {
  console.log('DETECTION PREVIEW: showDetectionPreview called with data:', data);
  
  if (!data) {
    console.error('DETECTION PREVIEW: No data provided');
    return;
  }
  
  // Ensure trees array exists
  if (!Array.isArray(data.trees)) {
    data.trees = [];
  }
  
  // Clean up any existing preview
  try {
    const existingContainer = document.getElementById('detection-popup-container');
    if (existingContainer && existingContainer.parentNode) {
      existingContainer.parentNode.removeChild(existingContainer);
    }
  } catch (error) {
    console.error('DETECTION PREVIEW: Error removing existing preview:', error);
  }
  
  // Create container
  const popupContainer = document.createElement('div');
  popupContainer.id = 'detection-popup-container';
  popupContainer.setAttribute('data-persistent', 'true');
  popupContainer.style.zIndex = '9999';
  popupContainer.style.position = 'fixed';
  popupContainer.style.top = '0';
  popupContainer.style.left = '0';
  popupContainer.style.width = '100%';
  popupContainer.style.height = '100%';
  
  // Add to document
  document.body.appendChild(popupContainer);
  window._detectionPopupContainer = popupContainer;
  
  // Render popup
  renderPopup(data, popupContainer);
  
  // Prevent scrolling
  document.body.style.overflow = 'hidden';
  
  // Store data for reuse
  window._lastPreviewData = data;
}

/**
 * Remove the detection preview
 */
function destroyDetectionPreview() {
  console.log('DETECTION PREVIEW: destroyDetectionPreview called');
  
  const popupContainer = window._detectionPopupContainer || 
                        document.getElementById('detection-popup-container');
  
  if (popupContainer) {
    try {
      // Unmount React component
      if (popupContainer._reactRoot) {
        popupContainer._reactRoot.unmount();
        popupContainer._reactRoot = null;
      } else if (ReactDOM.unmountComponentAtNode) {
        const innerContainer = document.getElementById('detection-popup-inner-container');
        if (innerContainer) {
          ReactDOM.unmountComponentAtNode(innerContainer);
        }
        ReactDOM.unmountComponentAtNode(popupContainer);
      }
      
      // Remove from DOM
      if (popupContainer.parentNode) {
        popupContainer.parentNode.removeChild(popupContainer);
      }
      
      // Restore scrolling
      document.body.style.overflow = '';
      
      // Clear reference
      window._detectionPopupContainer = null;
    } catch (error) {
      console.error('DETECTION PREVIEW: Error destroying preview:', error);
    }
  }
}

// Set up global functions
window.showDetectionPreview = showDetectionPreview;
window.destroyDetectionPreview = destroyDetectionPreview;

// Set up event listeners
document.addEventListener('DOMContentLoaded', () => {
  window.showDetectionPreview = showDetectionPreview;
  window.destroyDetectionPreview = destroyDetectionPreview;
  
  // Listen for fastInferenceResults event
  document.addEventListener('fastInferenceResults', event => {
    if (event.detail) {
      console.log('DETECTION PREVIEW: Received fastInferenceResults event');
      showDetectionPreview(event.detail);
    }
  });
});

// Export functions
export { showDetectionPreview, destroyDetectionPreview };
export default { showDetectionPreview, destroyDetectionPreview };