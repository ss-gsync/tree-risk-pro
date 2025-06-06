// src/components/visualization/Response/ResponseViewer.jsx
//
// Component for displaying detection response data from the ML pipeline
// This handles both the visualization and data loading from detection_*

import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../../../ui/card';
import { MapPin, FileText, FileJson, Tree, Map, Eye } from 'lucide-react';
import { Button } from '../../../ui/button';

/**
 * ResponseViewer Component
 * 
 * Displays detection response data from the ML pipeline
 * Handles loading detection results and presenting visualization
 */
const ResponseViewer = ({ 
  detectionId, 
  onViewOnMap, 
  onClose
}) => {
  const [loading, setLoading] = useState(true);
  const [metadata, setMetadata] = useState(null);
  const [treeCount, setTreeCount] = useState(0);
  const [error, setError] = useState(null);

  // Load metadata and tree count from the ML response
  useEffect(() => {
    const loadResponseData = async () => {
      if (!detectionId) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        
        // Attempt to load metadata.json
        const metadataPath = `/data/ml/${detectionId}/ml_response/metadata.json`;
        const metadataResponse = await fetch(metadataPath);
        
        if (!metadataResponse.ok) {
          throw new Error(`Failed to load metadata from ${metadataPath}`);
        }
        
        const metadataJson = await metadataResponse.json();
        setMetadata(metadataJson);
        
        // Get tree count from metadata or try to load trees.json
        if (metadataJson.detection_count !== undefined) {
          setTreeCount(metadataJson.detection_count);
        } else {
          // Try to fetch trees.json to count trees
          try {
            const treesPath = `/data/ml/${detectionId}/ml_response/trees.json`;
            const treesResponse = await fetch(treesPath);
            
            if (treesResponse.ok) {
              const treesData = await treesResponse.json();
              if (treesData.trees) {
                setTreeCount(treesData.trees.length);
              }
            }
          } catch (treeError) {
            console.error("Failed to load trees data:", treeError);
          }
        }
        
        setLoading(false);
      } catch (err) {
        console.error("Error loading detection response:", err);
        setError(err.message);
        setLoading(false);
      }
    };
    
    loadResponseData();
  }, [detectionId]);

  // Handle view on map button click
  const handleViewOnMap = () => {
    if (onViewOnMap && metadata) {
      onViewOnMap(detectionId, metadata);
    }
  };

  // Render error state
  if (error) {
    return (
      <Card className="w-full shadow-md">
        <CardHeader className="pb-2">
          <CardTitle className="text-red-500 flex items-center text-sm">
            <FileText className="h-4 w-4 mr-2" />
            Error Loading Detection Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-red-500">{error}</p>
          <Button 
            variant="outline" 
            size="sm" 
            className="mt-4" 
            onClick={onClose}
          >
            Close
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Render loading state
  if (loading) {
    return (
      <Card className="w-full shadow-md">
        <CardHeader className="pb-2">
          <CardTitle className="text-md flex items-center">
            <FileText className="h-4 w-4 mr-2" />
            Loading Detection Results...
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Render no data state
  if (!metadata) {
    return (
      <Card className="w-full shadow-md">
        <CardHeader className="pb-2">
          <CardTitle className="text-md flex items-center">
            <FileText className="h-4 w-4 mr-2" />
            No Detection Results
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm">No detection results available for this ID.</p>
          <Button 
            variant="outline" 
            size="sm" 
            className="mt-4" 
            onClick={onClose}
          >
            Close
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Render detection results
  return (
    <Card className="w-full shadow-md">
      <CardHeader className="pb-2">
        <CardTitle className="text-md flex items-center">
          <FileText className="h-4 w-4 mr-2" />
          Detection Results
          <span className="text-xs ml-2 text-gray-500">
            ({metadata.job_id || detectionId})
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {/* Detection summary */}
        <div className="mb-4 p-2 bg-gray-50 rounded-md">
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center">
              <Tree className="h-4 w-4 mr-2 text-green-600" />
              <span className="text-sm font-medium">Trees Detected:</span>
              <span className="ml-1 text-sm">{treeCount}</span>
            </div>
            <div className="flex items-center">
              <MapPin className="h-4 w-4 mr-2 text-blue-600" />
              <span className="text-sm font-medium">Source:</span>
              <span className="ml-1 text-sm">{metadata.source || "Unknown"}</span>
            </div>
            <div className="flex items-center">
              <FileJson className="h-4 w-4 mr-2 text-purple-600" />
              <span className="text-sm font-medium">Model:</span>
              <span className="ml-1 text-sm">{metadata.model_type || "Unknown"}</span>
            </div>
            <div className="flex items-center">
              <Map className="h-4 w-4 mr-2 text-orange-600" />
              <span className="text-sm font-medium">Coordinates:</span>
              <span className="ml-1 text-sm">{metadata.coordinate_system || "Unknown"}</span>
            </div>
          </div>
        </div>

        {/* Detection Image */}
        <div className="mb-4">
          <div className="relative w-full aspect-video bg-gray-100 rounded-md overflow-hidden">
            <img 
              src={`/data/ml/${detectionId}/ml_response/combined_visualization.jpg`}
              alt="Detection visualization"
              className="w-full h-full object-contain"
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23ccc' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='2' ry='2'%3E%3C/rect%3E%3Ccircle cx='8.5' cy='8.5' r='1.5'%3E%3C/circle%3E%3Cpolyline points='21 15 16 10 5 21'%3E%3C/polyline%3E%3C/svg%3E";
              }}
            />
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex justify-between mt-4">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={onClose}
          >
            Close
          </Button>
          <Button 
            variant="default" 
            size="sm"
            onClick={handleViewOnMap}
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            <Eye className="h-4 w-4 mr-2" />
            View on Map
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

export default ResponseViewer;