import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Search, Home, Compass, Filter, Download, Box, X } from 'lucide-react';
import { DetectionService } from '../../../services/api/apiService';

/**
 * CesiumControls - Controls for the Cesium 3D viewer
 * Provides controls for tree detection, camera movement, and filtering
 */
const CesiumControls = ({ cesiumRef }) => {
  const dispatch = useDispatch();
  const [isExporting, setIsExporting] = useState(false);
  const [exportStatus, setExportStatus] = useState('');
  const [controlsExpanded, setControlsExpanded] = useState(true);
  const [cameraInfo, setCameraInfo] = useState({
    longitude: 0,
    latitude: 0,
    height: 0,
    heading: 0,
    pitch: 0
  });
  
  // Get the current camera position when the component mounts and when it changes
  useEffect(() => {
    if (!cesiumRef || !cesiumRef.current) return;
    
    const viewer = cesiumRef.current.getCesiumViewer();
    if (!viewer) return;
    
    // Function to update camera information
    const updateCameraInfo = () => {
      const camera = viewer.camera;
      const position = camera.positionCartographic;
      
      if (position) {
        setCameraInfo({
          longitude: window.Cesium.Math.toDegrees(position.longitude).toFixed(6),
          latitude: window.Cesium.Math.toDegrees(position.latitude).toFixed(6),
          height: (position.height / 1000).toFixed(2), // Convert to km for display
          heading: window.Cesium.Math.toDegrees(camera.heading).toFixed(1),
          pitch: window.Cesium.Math.toDegrees(camera.pitch).toFixed(1)
        });
      }
    };
    
    // Update initially
    updateCameraInfo();
    
    // Set up event listener for camera changes
    const cameraListener = viewer.camera.changed.addEventListener(updateCameraInfo);
    
    // Clean up
    return () => {
      try {
        cameraListener();
      } catch (e) {
        console.warn("Error removing camera listener:", e);
      }
    };
  }, [cesiumRef]);
  
  // Handle tree detection with ML pipeline
  const handleDetectTrees = async () => {
    if (!cesiumRef || !cesiumRef.current) {
      alert("3D viewer not ready. Please try again in a moment.");
      return;
    }
    
    // Show confirmatation dialog
    const detectionChoice = window.confirm(
      'Detect trees in this 3D view? This will process imagery through the ML pipeline to identify trees. You will be able to review and edit the results before adding them to the database.'
    );
    
    if (!detectionChoice) {
      return;
    }
    
    try {
      setIsExporting(true);
      setExportStatus("Initializing tree detection...");
      
      // Area ID based on current time
      const areaId = `area_${new Date().getTime()}`;
      
      // Capture the current 3D view
      setExportStatus("Capturing 3D view...");
      
      const captureData = await cesiumRef.current.captureCurrentView();
      if (!captureData) {
        throw new Error("Failed to capture 3D view data");
      }
      
      // Prepare request data for ML pipeline
      const requestData = {
        area_id: areaId,
        debug: true,
        include_bounding_boxes: true,
        assistance_mode: true,
        capture_method: captureData.captureMethod || "cesium_3dtiles",
        is_3d_capture: true
      };
      
      // Add image data if available
      if (captureData.imageData) {
        requestData.map_image = captureData.imageData;
        setExportStatus("3D image captured successfully!");
      }
      
      // Add view data
      requestData.map_view_info = {
        viewData: captureData.viewData,
        is3D: true,
        mapType: "cesium_3dtiles"
      };
      
      // Run the ML pipeline
      setExportStatus("Processing with ML pipeline...");
      console.log("Sending 3D detection request:", requestData);
      
      const result = await DetectionService.detectTrees(requestData);
      const jobId = result.job_id;
      
      setExportStatus("Waiting for detection results...");
      
      // Poll for results
      let statusResult = await DetectionService.getDetectionStatus(jobId);
      let attempts = 0;
      
      while (statusResult.status !== 'complete' && attempts < 30) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        statusResult = await DetectionService.getDetectionStatus(jobId);
        setExportStatus(`Processing job: ${statusResult.message || statusResult.status}`);
        attempts++;
      }
      
      // Check final status
      if (statusResult.status !== 'complete') {
        throw new Error("Detection timed out");
      }
      
      // Get tree count
      const treeCount = statusResult.tree_count || 0;
      
      if (treeCount === 0) {
        setExportStatus("No trees detected in this area");
        setTimeout(() => setExportStatus(""), 3000);
        setIsExporting(false);
        return;
      }
      
      setExportStatus(`${treeCount} trees detected!`);
      
      // Add detected trees to the 3D view
      if (statusResult.trees && statusResult.trees.length > 0) {
        const markers = cesiumRef.current.addTreeMarkers(statusResult.trees);
        console.log(`Added ${markers.length} tree markers to 3D view`);
      }
      
      // Done
      setTimeout(() => setExportStatus(""), 3000);
      setIsExporting(false);
    } catch (error) {
      console.error("Error detecting trees:", error);
      setExportStatus(`Error: ${error.message}`);
      setTimeout(() => setExportStatus(""), 5000);
      setIsExporting(false);
    }
  };
  
  // Handle resetting the camera to default position
  const handleResetCamera = () => {
    if (!cesiumRef || !cesiumRef.current) return;
    
    const viewer = cesiumRef.current.getCesiumViewer();
    if (!viewer) return;
    
    // Reset to a default position
    viewer.camera.setView({
      destination: window.Cesium.Cartesian3.fromDegrees(-96.7969, 32.7767, 5000), // Dallas area
      orientation: {
        heading: window.Cesium.Math.toRadians(0),
        pitch: window.Cesium.Math.toRadians(-45),
        roll: 0
      }
    });
  };
  
  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-medium">Photorealistic 3D View</h3>
        <button 
          onClick={() => setControlsExpanded(!controlsExpanded)}
          className="text-gray-500 hover:text-gray-700"
        >
          {controlsExpanded ? <X size={16} /> : <Compass size={16} />}
        </button>
      </div>
      
      {controlsExpanded && (
        <>
          {/* Camera information */}
          <div className="text-xs text-gray-500 mb-3 bg-gray-50 p-2 rounded">
            <div className="flex justify-between">
              <span>Position:</span>
              <span className="font-mono">
                {cameraInfo.latitude}, {cameraInfo.longitude}
              </span>
            </div>
            <div className="flex justify-between">
              <span>Height:</span>
              <span className="font-mono">{cameraInfo.height} km</span>
            </div>
            <div className="flex justify-between">
              <span>Heading:</span>
              <span className="font-mono">{cameraInfo.heading}°</span>
            </div>
            <div className="flex justify-between">
              <span>Pitch:</span>
              <span className="font-mono">{cameraInfo.pitch}°</span>
            </div>
          </div>
          
          {/* Quick actions */}
          <div className="flex space-x-2 mb-3">
            <button
              onClick={handleResetCamera}
              className="flex items-center justify-center p-2 bg-gray-100 rounded-md hover:bg-gray-200 flex-1"
              title="Reset camera"
            >
              <Home className="h-4 w-4 mr-1" />
              <span className="text-xs">Reset</span>
            </button>
          </div>
          
          {/* Tree detection button */}
          <button
            onClick={handleDetectTrees}
            disabled={isExporting}
            className={`flex items-center justify-center p-2 ${
              isExporting 
                ? 'bg-blue-100 text-blue-700' 
                : 'bg-blue-50 text-blue-600 hover:bg-blue-100'
            } rounded-md w-full cursor-pointer mb-3`}
          >
            {isExporting ? (
              <>
                <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full mr-2"></div>
                <span className="text-xs">{exportStatus || "Processing..."}</span>
              </>
            ) : (
              <>
                <Box className="h-4 w-4 mr-1" />
                <span className="text-xs">Detect Trees in View</span>
              </>
            )}
          </button>
          
          {/* Status message */}
          {exportStatus && !isExporting && (
            <div className="text-xs text-center text-gray-700 mb-3">
              {exportStatus}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default CesiumControls;