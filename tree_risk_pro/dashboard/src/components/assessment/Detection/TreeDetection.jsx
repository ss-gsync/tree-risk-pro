// src/components/assessment/Detection/TreeDetection.jsx

import React, { useState, useEffect } from 'react';
import { Scan, AlertTriangle, Leaf, RefreshCw, CheckCircle, X, Sparkles } from 'lucide-react';
import { DetectionService } from '../../../services/api/apiService';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '../../ui/card';
import { Button } from '../../ui/button';
import { Label } from '../../ui/label';
import { Input } from '../../ui/input';
import { Textarea } from '../../ui/textarea';
import { useSelector } from 'react-redux';

const TreeDetection = ({ onDetectionComplete, onClose }) => {
  const [isDetecting, setIsDetecting] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState(null);
  const [showModal, setShowModal] = useState(true);
  // Define initial form data with the Gemini flag from local storage
  const initialGeminiSetting = (() => {
    try {
      const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        return Boolean(parsedSettings.useGeminiForTreeDetection);
      }
    } catch (e) {
      console.error('Error reading initial Gemini setting:', e);
    }
    return true; // Default to using Gemini AI if settings not found
  })();
  
  console.log("Initial Gemini setting:", initialGeminiSetting);
  
  // Force localStorage to use ML pipeline (Gemini is disabled)
  try {
    const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
    if (settings.useGeminiForTreeDetection !== false) {
      settings.useGeminiForTreeDetection = false;
      localStorage.setItem('treeRiskDashboardSettings', JSON.stringify(settings));
      console.log("Updated localStorage to use ML pipeline instead of Gemini");
    }
  } catch (e) {
    console.error("Error updating localStorage settings:", e);
    // Create default settings if needed
    localStorage.setItem('treeRiskDashboardSettings', JSON.stringify({
      map3DApi: 'cesium',
      defaultView: '2d',
      useGeminiForTreeDetection: false,
      theme: 'light'
    }));
  }
  
  const [formData, setFormData] = useState({
    imageUrl: '',
    coordinates: '',
    areaId: `area_${new Date().getTime()}`,
    useCurrentView: true,
    useGemini: initialGeminiSetting // Default to using Gemini from settings
  });
  const [error, setError] = useState(null);
  const [treeCount, setTreeCount] = useState(0);
  
  // Get current map view from Redux store
  const mapState = useSelector(state => state.map);
  
  
  // Initialize from settings and current map coordinates
  useEffect(() => {
    // Load settings from localStorage
    try {
      const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        // Set useGemini based on settings (default to true if not found)
        setFormData(prev => ({
          ...prev,
          useGemini: parsedSettings.useGeminiForTreeDetection !== false // Default to true
        }));
      }
    } catch (e) {
      console.error('Error loading settings:', e);
    }
    
    // Prefill coordinates from current map view
    if (mapState.center && mapState.zoom) {
      setFormData(prev => ({
        ...prev,
        coordinates: JSON.stringify({
          center: mapState.center,
          zoom: mapState.zoom,
          bounds: [
            [mapState.center[0] - 0.01, mapState.center[1] - 0.01],
            [mapState.center[0] + 0.01, mapState.center[1] + 0.01]
          ]
        })
      }));
    }
  }, [mapState.center, mapState.zoom]);
  
  // This effect has been removed since we now use the backend for all Gemini processing
  
  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setIsDetecting(true);
    
    // Get the latest setting from localStorage
    try {
      const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        // Update form data with settings
        setFormData(prev => ({
          ...prev,
          useGemini: parsedSettings.useGeminiForTreeDetection
        }));
      }
    } catch (e) {
      console.error('Error loading Gemini setting from localStorage:', e);
    }
    
    try {
      // Prepare request data
      // Get the current settings directly from localStorage to ensure accuracy
      const currentSettings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
      const useGemini = currentSettings.useGeminiForTreeDetection === true;
      
      const requestData = {
        area_id: formData.areaId,
        debug: true,
        include_bounding_boxes: true,
        assistance_mode: true,
        use_gemini_for_tree_detection: useGemini,
        dashboard_request: true
      };
      
      console.log(`Using Gemini for detection: ${useGemini}`);
      
      console.log("Detection request data:", {
        area_id: formData.areaId,
        useGemini: useGemini  // Use the value we just determined directly from localStorage
      });
      
      // Capture the current map view - only allow detection in 3D mode
      if (formData.useCurrentView && mapState.center) {
        // Check if 3D mode is active based on the button state
        const mapButton = document.querySelector('.map-3d-toggle');
        const is3DMode = mapButton && mapButton.textContent.includes('2D');
        
        // Only allow detection in 3D mode
        if (!is3DMode) {
          setError('Tree detection is only available in 3D view mode. Please switch to 3D view and try again.');
          setIsDetecting(false);
          return;
        }
        
        // Create a precise bounding box for better detection context
        const boundsOffset = 0.003; // Approximately 300 meters at the equator
        
        // Set up coordinates for the request
        requestData.coordinates = {
          center: mapState.center,
          zoom: mapState.zoom,
          bounds: [
            [mapState.center[0] - boundsOffset, mapState.center[1] - boundsOffset],
            [mapState.center[0] + boundsOffset, mapState.center[1] + boundsOffset]
          ],
          is3D: true // Just one simple flag
        };
        
        // Add map view capture event to get 3D data
        try {
          // Attempt to dispatch an event to capture current map view
          const captureEvent = new CustomEvent('captureMapViewForDetection', {
            detail: { requestData, useGemini: formData.useGemini }
          });
          window.dispatchEvent(captureEvent);
          console.log("Dispatched map view capture event");
          
          // Set the Gemini and dashboard flags (these should be already set in the requestData object)
        } catch (captureError) {
          console.warn("Error dispatching map capture event:", captureError);
        }
      } else if (formData.imageUrl) {
        requestData.image_url = formData.imageUrl;
      } else if (formData.coordinates) {
        try {
          requestData.coordinates = JSON.parse(formData.coordinates);
        } catch (error) {
          setError('Invalid coordinates format');
          setIsDetecting(false);
          return;
        }
      } else {
        setError('Either 3D view or coordinates must be provided');
        setIsDetecting(false);
        return;
      }
      
      // If using Gemini, ensure we have the map view data needed
      if (formData.useGemini) {
        try {
          const mapRef = window.mapRef;
          if (!mapRef || !mapRef.current) {
            setError('Map reference not available');
            setIsDetecting(false);
            return;
          }
          
          // Get current map view if not already captured
          if (!requestData.map_view_info) {
            try {
              // Use captureCurrentView to get map data including image URL if possible
              const mapViewInfo = await mapRef.current.captureCurrentView();
              console.log("Captured map view for Gemini:", mapViewInfo);
              
              // Check if we have an image URL - required for Gemini tree detection
              if (mapViewInfo.viewData && mapViewInfo.viewData.imageUrl) {
                const imageUrl = mapViewInfo.viewData.imageUrl;
                const length = imageUrl ? imageUrl.length : 0;
                
                console.log("Image URL successfully captured, length:", length);
                
                if (length < 1000) {
                  console.error("Image URL too short to be valid:", length);
                  setError("Unable to capture a valid map image. Tree detection with Gemini requires a map image.");
                  setIsDetecting(false);
                  return;
                }
                
                console.log("Image URL begins with:", imageUrl.substring(0, 30) + "...");
              } else {
                console.log("No image URL captured - Gemini tree detection requires an image");
                setError("Unable to capture map image. Tree detection with Gemini requires a map image.");
                setIsDetecting(false);
                return;
              }
              
              // Add the view info to the request
              requestData.map_view_info = mapViewInfo;
              console.log("Using captureCurrentView with direct canvas capture");
            } catch (urlError) {
              console.error("Failed to capture map view URL:", urlError);
              setError(`Failed to capture map view: ${urlError.message}`);
              setIsDetecting(false);
              return;
            }
          }
          
          // Explicitly set the Gemini flag again to ensure it's included
          requestData.use_gemini_for_tree_detection = true;
          requestData.dashboard_request = true;
          console.log("Set use_gemini_for_tree_detection=true for map view data");
        } catch (error) {
          console.error('Error capturing map view for Gemini:', error);
          setIsDetecting(false);
          setError(`Failed to capture map view: ${error.message}`);
          return;
        }
      }
      
      // Start detection job (synchronous process)
      try {
        // Ensure we're using the correct setting from localStorage
        const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
        const useGemini = settings.useGeminiForTreeDetection === true;
        console.log(`Using Gemini setting from localStorage: ${useGemini}`);
        requestData.use_gemini_for_tree_detection = useGemini;
        requestData.dashboard_request = true;
        
        // Log the exact request being sent
        console.log("Final detection request:", JSON.stringify({
          ...requestData,
          use_gemini_for_tree_detection: useGemini
        }, null, 2));
        
        // The backend will now process the entire request synchronously
        // and return the complete results when done
        const detectionResults = await DetectionService.detectTrees(requestData);
        
        // Log the raw results for debugging
        console.log("Detection results:", JSON.stringify(detectionResults, null, 2));
        
        // Update state with results
        setIsDetecting(false);
        // Make sure we have valid job_id to avoid "undefined" in API calls
        const resultJobId = detectionResults.job_id || detectionResults.area_id || formData.areaId;
        setJobId(resultJobId);
        setJobStatus(detectionResults);
        setTreeCount(detectionResults.tree_count || 0);
        
        // Check for errors
        if (detectionResults.status === 'error') {
          console.log("Error in tree detection:", detectionResults.error || "Unknown error");
          setError(detectionResults.error || "An error occurred during tree detection. Please check the log files andtry again.");
          return;
        }
        
        // Only check tree count if no error was already set
        // This prevents overriding error messages with a generic "No trees" message
        if (detectionResults.status !== 'error' && 
            (detectionResults.tree_count === 0 || !detectionResults.trees || detectionResults.trees.length === 0)) {
          console.log("No trees detected in this area");
          setError('No trees detected in this area.');
          return;
        }
        
        // Trees were detected successfully
        console.log(`Found ${detectionResults.trees.length} trees in detection results`);
        setShowModal(false);
        
        // Create and dispatch event to enter tree validation mode with the detection results
        const validationEvent = new CustomEvent('enterTreeValidationMode', {
          detail: {
            detectionJobId: resultJobId, // Use the validated job ID
            areaId: formData.areaId,
            treeCount: detectionResults.tree_count || detectionResults.trees.length,
            trees: detectionResults.trees || [] // Include detected trees directly
          }
        });
        
        console.log("Dispatching validation event:", validationEvent);
        window.dispatchEvent(validationEvent);
        
        // Pass the detection results to parent component
        if (onDetectionComplete) {
          onDetectionComplete(detectionResults);
        }
        
      } catch (error) {
        console.error('Error during tree detection:', error);
        setIsDetecting(false);
        setError(`Tree detection failed: ${error.message}`);
      }
    } catch (error) {
      console.error('Error starting detection:', error);
      setError(`Failed to start tree detection: ${error.message}`);
      setIsDetecting(false);
    }
  };
  
  // Load Gemini setting from localStorage on component mount and check API configuration
  React.useEffect(() => {
    try {
      const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        // Update the form with the user's Gemini preference from settings
        const isGeminiEnabled = Boolean(parsedSettings.useGeminiForTreeDetection);
        console.log("Loaded Gemini setting from localStorage:", isGeminiEnabled);
        
        // Ensure the type is boolean and update state
        setFormData(prev => ({
          ...prev,
          useGemini: isGeminiEnabled
        }));
      }
      
      // Check API configuration to ensure Gemini is available
      fetch('/api/config')
        .then(response => response.json())
        .then(config => {
          console.log("API configuration:", config);
          if (config.geminiEnabled && !config.geminiStatus) {
            console.warn("Gemini is enabled but not initialized properly on the backend");
          }
        })
        .catch(err => {
          console.error("Failed to check API configuration:", err);
        });
    } catch (e) {
      console.error('Error loading Gemini setting from localStorage:', e);
    }
  }, []);
  
  // Handle modal close
  const handleCloseModal = () => {
    if (!isDetecting) {
      setShowModal(false);
      setError(null);
      
      if (onClose) {
        onClose();
      }
    }
  };
  
  // Handle form field changes
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    if (type === 'checkbox') {
      // For checkbox fields, explicitly use boolean
      const newValue = Boolean(checked);
      console.log(`Setting ${name} to ${newValue}`);
      
      // Special handling for Gemini setting
      if (name === 'useGemini') {
        console.log(`🔍 Changing Gemini setting to: ${newValue}`);
      }
      
      setFormData(prev => ({ ...prev, [name]: newValue }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };
  
  return (
    <>
      {/* Status display */}
      {jobId && !showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md bg-white">
            <CardHeader className="pb-2">
              <CardTitle className="text-md">
                <div className="flex items-center">
                  {isDetecting ? (
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <CheckCircle className="h-4 w-4 mr-2 text-green-500" />
                  )}
                  Tree Detection
                </div>
              </CardTitle>
              <CardDescription className="text-xs">
                Job ID: {jobId}
              </CardDescription>
            </CardHeader>
            <CardContent className="pb-2">
              <div className="space-y-2">
                <div className="text-sm">
                  Status: <span className={`font-medium ${jobStatus?.status === 'complete' ? 'text-green-600' : 'text-amber-600'}`}>
                    {jobStatus?.status === 'complete' ? 'Complete' : 'Processing'}
                  </span>
                </div>
                
                {jobStatus?.status === 'complete' && (
                  <div className="text-sm">
                    Trees detected: <span className="font-medium text-green-600">{treeCount}</span>
                  </div>
                )}
                
                {jobStatus?.message && (
                  <div className="text-sm text-gray-500">
                    {jobStatus.message}
                  </div>
                )}
              </div>
            </CardContent>
            <CardFooter className="pt-2">
              <Button 
                variant="outline" 
                size="sm" 
                className="mt-2 text-xs px-2 py-1"
                onClick={() => {
                  setJobId(null);
                  setJobStatus(null);
                  setIsDetecting(false);
                  
                  if (onClose) {
                    onClose();
                  }
                }}
              >
                Close
              </Button>
            </CardFooter>
          </Card>
        </div>
      )}
      
      {/* Detection Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-md">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h3 className="text-lg font-medium">Detect Trees</h3>
                <p className="text-sm text-blue-600 font-medium">Only available in 3D view mode</p>
              </div>
              <button 
                onClick={handleCloseModal}
                className="text-gray-400 hover:text-gray-500"
                disabled={isDetecting}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            {error && (
              <div className="mb-4 p-2 bg-red-100 text-red-700 rounded-md flex items-center">
                <AlertTriangle className="h-4 w-4 mr-2" />
                {error}
              </div>
            )}
            
            <form onSubmit={handleSubmit}>
              <div className="space-y-4">
                <div>
                  <Label htmlFor="areaId" className="text-sm font-medium text-gray-700">
                    Area ID
                  </Label>
                  <Input
                    id="areaId"
                    name="areaId"
                    value={formData.areaId}
                    onChange={handleChange}
                    className="w-full mt-1"
                    placeholder="Enter area identifier"
                  />
                </div>
                
                <div className="space-y-2">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="useCurrentView"
                      name="useCurrentView"
                      checked={formData.useCurrentView}
                      onChange={handleChange}
                      className="h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300 rounded"
                    />
                    <label htmlFor="useCurrentView" className="ml-2 block text-sm text-gray-700">
                      Use current map view
                    </label>
                  </div>
                  
                  {/* Gemini AI option removed - using ML pipeline only */}
                </div>
                
                {!formData.useCurrentView && (
                  <>
                    <div>
                      <Label htmlFor="imageUrl" className="text-sm font-medium text-gray-700">
                        Image URL (Optional)
                      </Label>
                      <Input
                        id="imageUrl"
                        name="imageUrl"
                        value={formData.imageUrl}
                        onChange={handleChange}
                        className="w-full mt-1"
                        placeholder="Enter image URL"
                        disabled={isDetecting}
                      />
                    </div>
                    
                    <div>
                      <Label htmlFor="coordinates" className="text-sm font-medium text-gray-700">
                        Coordinates (Optional)
                      </Label>
                      <Textarea
                        id="coordinates"
                        name="coordinates"
                        value={formData.coordinates}
                        onChange={handleChange}
                        className="w-full mt-1"
                        rows={4}
                        placeholder="Enter coordinates JSON"
                        disabled={isDetecting}
                      />
                    </div>
                  </>
                )}
              </div>
              
              <div className="mt-5 sm:mt-6">
                <Button
                  type="submit"
                  className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 text-white font-medium rounded-md"
                  disabled={isDetecting}
                >
                  {isDetecting ? (
                    <>
                      <RefreshCw className="animate-spin h-4 w-4 mr-2" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Leaf className="h-4 w-4 mr-2" />
                      Start Tree Detection
                    </>
                  )}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

export default TreeDetection;