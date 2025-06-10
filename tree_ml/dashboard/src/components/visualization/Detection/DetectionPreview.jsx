// Detection Preview Component - Popup for ML Results
import React, { useState, useEffect } from 'react';
import * as ReactDOM from 'react-dom';
import { 
  BarChart, X, Check, Info, AlertTriangle, Eye, Layers, ChevronDown, 
  Maximize2, Camera, FileBarChart2, CircleOff, CircleCheck,
  Building, Zap, TreePine, Filter, Tag, MapPin
} from 'lucide-react';

/**
 * DetectionPopup Component - Displays detection results in a modal
 */
const DetectionPopup = ({ data, onClose }) => {
  console.log('DetectionPopup: Rendering with data:', {
    hasData: !!data,
    hasDetections: !!data?.detections,
    hasTrees: !!data?.trees,
    treeCount: Array.isArray(data?.trees) ? data.trees.length : 0,
    detectionCount: Array.isArray(data?.detections) ? data.detections.length : 0,
    jobId: data?.job_id
  });
  
  // Get all tree data from any available source
  const treeData = useTreeData(data);
  
  // Image paths
  const jobId = data?.job_id;
  const imagePath = data?.paths?.visualizationImage || 
    (jobId ? `/api/ml/detection/${jobId}/visualization` : null);
  const satellitePath = data?.paths?.satelliteImage || 
    (jobId ? `/api/ml/detection/${jobId}/satellite` : null);
    
  // Debug logs
  console.log(`DetectionPreview - Image paths:`, {
    jobId,
    visualizationPath: imagePath,
    satellitePath,
    dataKeys: data ? Object.keys(data) : [],
    hasPaths: !!data?.paths,
    pathsKeys: data?.paths ? Object.keys(data.paths) : []
  });
  
  // UI state
  const [fullscreenImage, setFullscreenImage] = useState(null);
  const [activeTab, setActiveTab] = useState('results');
  
  // Load tree categories from available data
  const { categories, treeCount } = useCategoryCounts(treeData);
  
  // Basic metadata
  const detectionTime = data?.timestamp || new Date().toISOString();
  const modelType = data?.model_type || data?.metadata?.model || 'Standard Detection';
  
  // Setup escape key handler
  useEffect(() => {
    const handleEscKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleEscKey);
    return () => document.removeEventListener('keydown', handleEscKey);
  }, [onClose]);
  
  // View on map handler
  const handleViewOnMap = () => {
    // Update global flags
    window.detectionShowOverlay = true;
    window.mlDetectionData = data;
    
    // Dispatch event to trigger map view
    window.dispatchEvent(new CustomEvent('enterDetectionMode', {
      detail: {
        jobId,
        data,
        showOverlay: true,
        forceVisible: true,
        forceRenderBoxes: true
      }
    }));
    
    // Close the popup
    setTimeout(onClose, 100);
  };
  
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
                  {imagePath ? (
                    <div className="relative w-full h-full">
                      <img 
                        src={imagePath} 
                        alt="ML detection visualization" 
                        className="w-full h-full object-cover cursor-pointer"
                        onClick={() => setFullscreenImage(imagePath)}
                        onError={(e) => {
                          console.error('Error loading visualization image');
                          e.target.style.display = 'none';
                          
                          const parent = e.target.parentNode;
                          if (parent) {
                            const fallback = document.createElement('div');
                            fallback.className = "absolute inset-0 flex flex-col items-center justify-center bg-slate-50";
                            fallback.innerHTML = `
                              <svg class="w-8 h-8 text-slate-300 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                              </svg>
                              <p class="text-xs text-slate-500">Failed to load detection image</p>
                              <p class="text-xs text-slate-400">Job ID: ${jobId || 'Unknown'}</p>
                            `;
                            parent.appendChild(fallback);
                          }
                        }}
                      />
                      <div className="absolute bottom-2 right-2 bg-white bg-opacity-75 text-xs p-1 rounded">
                        Job ID: {jobId || 'Unknown'}
                      </div>
                      <div className="absolute top-2 right-2 bg-white bg-opacity-75 rounded p-1">
                        <Maximize2 
                          size={16} 
                          className="text-slate-600 cursor-pointer"
                          onClick={() => setFullscreenImage(imagePath)} 
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
                          onError={() => setFullscreenImage(null)}
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
                            Satellite imagery analysis completed using ML detection model. 
                            {treeCount > 0 ? ` Found ${treeCount} trees in the target area.` : ' No trees detected in the target area.'}
                          </p>
                          
                          {/* Category breakdown */}
                          {treeCount > 0 && (
                            <div className="mt-2">
                              <div className="text-xs font-medium mb-1.5">Categories breakdown:</div>
                              <div className="grid grid-cols-2 gap-x-2 gap-y-1">
                                {Object.entries(categories).map(([category, count]) => 
                                  count > 0 ? (
                                    <div key={category} className="flex items-center text-xs">
                                      <div className="w-2 h-2 rounded-full mr-1" style={{backgroundColor: getCategoryColor(category)}}></div>
                                      <span className="truncate" title={getCategoryLabel(category)}>{getCategoryLabel(category)}:</span>
                                      <span className="ml-auto font-medium">{count}</span>
                                    </div>
                                  ) : null
                                )}
                              </div>
                            </div>
                          )}
                          
                          <p className="mt-2 text-xs text-slate-500">
                            Click "View on Map" to see trees on the map.
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
                          Try a different location or adjust parameters
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
                  <div className="text-base font-bold text-green-800">{treeCount}</div>
                  <div className="text-xs text-green-600 mt-1">Total tree count</div>
                </div>
                
                <div className="bg-blue-50 rounded-md p-3 border border-blue-100">
                  <div className="text-xs text-blue-700 mb-1">Detection Quality</div>
                  <div className="text-base font-bold text-blue-800">
                    Standard
                  </div>
                  <div className="text-xs text-blue-600 mt-1">Baseline detection</div>
                </div>
                
                <div className="bg-indigo-50 rounded-md p-3 border border-indigo-100">
                  <div className="text-xs text-indigo-700 mb-1">Model</div>
                  <div className="text-base font-bold text-indigo-800 truncate" title="Grounded SAM">
                    Grounded SAM
                  </div>
                  <div className="text-xs text-indigo-600 mt-1">Detection + Segmentation</div>
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
                    <span className="text-slate-600 text-xs font-medium">Status</span>
                    <span className="text-green-600 text-xs flex items-center">
                      <Check size={12} className="mr-1" />
                      Complete
                    </span>
                  </div>
                </div>
              </div>
              
              {/* Categories distribution */}
              {treeCount > 0 && (
                <div className="border-t border-slate-100 pt-4">
                  <h4 className="text-sm font-medium text-slate-700 mb-3">Category Distribution</h4>
                  <div className="grid grid-cols-1 gap-y-2">
                    {Object.entries(categories).map(([category, count]) => {
                      if (count > 0) {
                        const percentage = Math.round((count / treeCount) * 100);
                        return (
                          <div key={category} className="flex items-center">
                            <div className="w-3 h-3 rounded-full mr-2" style={{backgroundColor: getCategoryColor(category)}}></div>
                            <span className="text-xs text-slate-600">{getCategoryLabel(category)}</span>
                            <div className="ml-auto flex items-center">
                              <span className="text-xs font-medium text-slate-700 mr-2">{count}</span>
                              <div className="w-20 bg-slate-100 h-1.5 rounded overflow-hidden">
                                <div 
                                  className="h-full rounded" 
                                  style={{
                                    width: `${percentage}%`, 
                                    backgroundColor: getCategoryColor(category)
                                  }}
                                ></div>
                              </div>
                              <span className="text-xs text-slate-500 ml-1">{percentage}%</span>
                            </div>
                          </div>
                        );
                      }
                      return null;
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
                      <span className="text-slate-600 text-xs">Detected Trees:</span>
                      <span className="font-medium text-slate-800 text-xs">{treeCount}</span>
                    </div>
                  </div>
                </div>
                
                {/* Data sources */}
                <div className="border rounded-md p-3 bg-slate-50">
                  <h4 className="text-xs font-semibold text-slate-700 mb-2">Data Sources</h4>
                  <div className="space-y-2">
                    <div className="flex items-center py-1 border-b border-slate-100">
                      <div className="w-6 text-center">
                        {imagePath ? 
                          <Check size={12} className="text-green-600 inline" /> : 
                          <CircleOff size={12} className="text-slate-400 inline" />
                        }
                      </div>
                      <span className="text-slate-600 text-xs">ML Visualization</span>
                    </div>
                    
                    <div className="flex items-center py-1 border-b border-slate-100">
                      <div className="w-6 text-center">
                        {satellitePath ? 
                          <Check size={12} className="text-green-600 inline" /> : 
                          <CircleOff size={12} className="text-slate-400 inline" />
                        }
                      </div>
                      <span className="text-slate-600 text-xs">Satellite Image</span>
                    </div>
                    
                    <div className="flex items-center py-1">
                      <div className="w-6 text-center">
                        {treeCount > 0 ? 
                          <Check size={12} className="text-green-600 inline" /> : 
                          <CircleOff size={12} className="text-slate-400 inline" />
                        }
                      </div>
                      <span className="text-slate-600 text-xs">Tree Detection Data</span>
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
 * Hook to extract and normalize tree data from various formats
 */
function useTreeData(data) {
  const [trees, setTrees] = useState([]);
  
  useEffect(() => {
    if (!data) return;
    
    // Extract trees from any available data source
    let extractedTrees = [];
    
    // Try data.trees array first
    if (Array.isArray(data.trees) && data.trees.length > 0) {
      console.log(`Using ${data.trees.length} trees from data.trees`);
      extractedTrees = data.trees;
    } 
    // Try data.detections array next
    else if (Array.isArray(data.detections) && data.detections.length > 0) {
      console.log(`Using ${data.detections.length} trees from data.detections`);
      extractedTrees = data.detections;
    }
    // Try raw trees.json format (success/detections)
    else if (data.success === true && Array.isArray(data.detections)) {
      console.log(`Using ${data.detections.length} trees from raw trees.json format`);
      extractedTrees = data.detections;
    }
    // Try embedded raw_trees_data
    else if (data.raw_trees_data && data.raw_trees_data.success === true && 
             Array.isArray(data.raw_trees_data.detections)) {
      console.log(`Using ${data.raw_trees_data.detections.length} trees from raw_trees_data`);
      extractedTrees = data.raw_trees_data.detections;
    }
    
    // If we found trees, normalize them
    if (extractedTrees.length > 0) {
      // Normalize tree objects
      const normalizedTrees = extractedTrees.map(tree => {
        // Get original class/category (spaces included)
        const originalClass = tree.class || tree.category || 'healthy tree';
        
        // Convert to normalized form
        const normalizedClass = typeof originalClass === 'string' 
          ? originalClass.toLowerCase().replace(/\s+/g, '_') 
          : 'healthy_tree';
        
        return {
          ...tree,
          class: normalizedClass,
          category: normalizedClass
        };
      });
      
      setTrees(normalizedTrees);
      console.log(`Normalized ${normalizedTrees.length} trees for display`);
    } else {
      console.warn('No tree data found in any format');
      
      // CRITICAL FIX: Check for other detection data formats as a last resort
      if (data && data.job_id) {
        console.log(`Attempting to load detection data from direct API for job ${data.job_id}`);
        // Try loading trees.json directly from the API as a fallback
        fetch(`/api/ml/detection/${data.job_id}/trees`)
          .then(response => {
            if (response.ok) return response.json();
            throw new Error(`Failed to load trees data: ${response.status}`);
          })
          .then(treesData => {
            console.log(`Successfully loaded trees data from API with ${treesData.detections?.length || 0} detections`);
            
            // If we have detections, update our local state
            if (treesData.detections && treesData.detections.length > 0) {
              const normalizedTrees = treesData.detections.map(detection => {
                // Convert to normalized form
                const originalClass = detection.class || detection.category || 'healthy tree';
                const normalizedClass = typeof originalClass === 'string' 
                  ? originalClass.toLowerCase().replace(/\s+/g, '_') 
                  : 'healthy_tree';
                
                return {
                  ...detection,
                  class: normalizedClass,
                  category: normalizedClass
                };
              });
              setTrees(normalizedTrees);
              console.log(`Loaded ${normalizedTrees.length} trees from direct API call`);
            } else {
              setTrees([]);
            }
          })
          .catch(err => {
            console.error('Error loading trees data from API:', err);
            setTrees([]);
          });
      } else {
        setTrees([]);
      }
    }
  }, [data]);
  
  return trees;
}

/**
 * Hook to count trees by category
 */
function useCategoryCounts(trees) {
  const [categoryData, setCategoryData] = useState({ 
    categories: {}, 
    treeCount: 0 
  });
  
  useEffect(() => {
    // Initialize category counters
    const categoryCounts = {
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
    if (Array.isArray(trees) && trees.length > 0) {
      trees.forEach(tree => {
        // Use category or class, normalize to underscore format
        let category = tree.category || tree.class || 'healthy_tree';
        
        // Normalize if it's a string
        if (typeof category === 'string') {
          category = category.toLowerCase().replace(/\s+/g, '_');
        }
        
        // Count in appropriate category
        if (categoryCounts.hasOwnProperty(category)) {
          categoryCounts[category]++;
        } else {
          // Default to healthy if unknown category
          categoryCounts.healthy_tree++;
        }
      });
    }
    
    // Update state
    setCategoryData({
      categories: categoryCounts,
      treeCount: trees.length
    });
  }, [trees]);
  
  return categoryData;
}

/**
 * Get display name for a category
 */
function getCategoryLabel(category) {
  const labels = {
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
  
  return labels[category] || category.replace('_', ' ');
}

/**
 * Get color for a category
 */
function getCategoryColor(category) {
  const colors = {
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
  
  return colors[category] || '#16a34a';
}

/**
 * Create a direct fallback loader to load trees.json if needed
 */
async function loadTreesJsonDirectly(jobId) {
  if (!jobId) return null;
  
  try {
    console.log(`Direct loader: Attempting to load trees.json for job ${jobId}`);
    const response = await fetch(`/api/ml/detection/${jobId}/trees`);
    
    if (!response.ok) {
      throw new Error(`Failed to load trees.json: ${response.status}`);
    }
    
    const data = await response.json();
    console.log(`Direct loader: Successfully loaded trees.json with ${data.detections?.length || 0} detections`);
    return data;
  } catch (error) {
    console.error(`Direct loader error:`, error);
    return null;
  }
}

/**
 * Render the popup into a container
 */
function renderPopup(data, container) {
  try {
    // Create React container
    const reactContainer = document.createElement('div');
    reactContainer.id = 'detection-popup-inner-container';
    
    // Clear any existing content
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    
    // Add new container
    container.appendChild(reactContainer);
    
    // Use appropriate React rendering method
    if (ReactDOM.createRoot) {
      const root = ReactDOM.createRoot(reactContainer);
      container._reactRoot = root;
      root.render(<DetectionPopup data={data} onClose={() => destroyDetectionPreview()} />);
    } else {
      ReactDOM.render(
        <DetectionPopup data={data} onClose={() => destroyDetectionPreview()} />, 
        reactContainer
      );
    }
  } catch (error) {
    console.error('Error rendering detection popup:', error);
  }
}

/**
 * Show the detection preview
 */
async function showDetectionPreview(data) {
  console.log('showDetectionPreview called with data:', {
    hasData: !!data,
    hasDetections: !!data?.detections,
    hasTrees: !!data?.trees,
    hasJobId: !!data?.job_id,
    jobId: data?.job_id
  });
  
  if (!data) {
    console.error('No data provided to showDetectionPreview');
    return;
  }
  
  // Remove any existing preview
  destroyDetectionPreview();
  
  // Extract job ID
  const jobId = data.job_id;
  
  // If there's no tree data in the provided object but we have a job ID,
  // try to load the trees.json file directly as a fallback
  let enrichedData = { ...data };
  
  if ((!data.trees || data.trees.length === 0) && 
      (!data.detections || data.detections.length === 0) && 
      jobId) {
    console.log(`No tree data found, attempting direct trees.json load for job ${jobId}`);
    const treesJson = await loadTreesJsonDirectly(jobId);
    
    if (treesJson && treesJson.success && Array.isArray(treesJson.detections)) {
      console.log(`Direct load successful, found ${treesJson.detections.length} trees`);
      
      // Add the tree data to our enriched data object
      enrichedData = {
        ...data,
        // Add raw data
        raw_trees_data: treesJson,
        // Also add direct detections array
        detections: treesJson.detections,
        // And populate trees array
        trees: treesJson.detections.map(detection => {
          const className = detection.class ? detection.class.replace(/\s+/g, '_') : 'healthy_tree';
          return {
            ...detection,
            class: className,
            category: className
          };
        })
      };
    }
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
  
  // Render popup with enriched data
  renderPopup(enrichedData, popupContainer);
  
  // Prevent scrolling
  document.body.style.overflow = 'hidden';
  
  // Store data for reuse
  window._lastPreviewData = enrichedData;
}

/**
 * Remove the detection preview
 */
function destroyDetectionPreview() {
  const container = document.getElementById('detection-popup-container');
  
  if (container) {
    try {
      // Unmount React component
      if (container._reactRoot) {
        container._reactRoot.unmount();
      } else if (ReactDOM.unmountComponentAtNode) {
        const innerContainer = document.getElementById('detection-popup-inner-container');
        if (innerContainer) {
          ReactDOM.unmountComponentAtNode(innerContainer);
        }
        ReactDOM.unmountComponentAtNode(container);
      }
      
      // Remove from DOM
      if (container.parentNode) {
        container.parentNode.removeChild(container);
      }
      
      // Restore scrolling
      document.body.style.overflow = '';
      
      // Clear reference
      window._detectionPopupContainer = null;
    } catch (error) {
      console.error('Error cleaning up detection preview:', error);
    }
  }
}

// Set up global functions
window.showDetectionPreview = showDetectionPreview;
window.destroyDetectionPreview = destroyDetectionPreview;

// Set up event listeners
document.addEventListener('DOMContentLoaded', () => {
  // Ensure global functions are available
  window.showDetectionPreview = showDetectionPreview;
  window.destroyDetectionPreview = destroyDetectionPreview;
  
  // Listen for events that should trigger the preview
  document.addEventListener('fastInferenceResults', event => {
    if (event.detail) {
      console.log('Received fastInferenceResults event');
      showDetectionPreview(event.detail);
    }
  });
  
  document.addEventListener('detectionDataLoaded', event => {
    if (event.detail) {
      console.log('Received detectionDataLoaded event');
      showDetectionPreview(event.detail);
    }
  });
  
  // Add helper function to check for existing detection data
  window.checkAndShowDetectionPreview = () => {
    if (window._lastPreviewData) {
      console.log('Found cached detection data');
      showDetectionPreview(window._lastPreviewData);
      return true;
    } else if (window.mlDetectionData) {
      console.log('Found global mlDetectionData');
      showDetectionPreview(window.mlDetectionData);
      return true;
    }
    return false;
  };
});

// Export functions
export { showDetectionPreview, destroyDetectionPreview };
export default { showDetectionPreview, destroyDetectionPreview };