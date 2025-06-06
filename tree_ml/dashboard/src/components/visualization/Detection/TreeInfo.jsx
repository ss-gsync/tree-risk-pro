// src/components/visualization/MapView/TreeInfo.jsx

import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { X, Trees, Ruler, TriangleAlert, Info, Calendar } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { clearSelectedFeature } from '../MapView/mapSlice';
import { TreeService } from '../../../services/api/apiService';

const TreeInfo = () => {
  const dispatch = useDispatch();
  const { selectedFeature } = useSelector((state) => state.map);
  const [treeData, setTreeData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    if (selectedFeature && selectedFeature.type === 'tree') {
      fetchTreeDetails(selectedFeature.data.id);
    } else {
      setTreeData(null);
    }
  }, [selectedFeature]);
  
  const fetchTreeDetails = async (treeId) => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await TreeService.getTree(treeId);
      setTreeData(data);
    } catch (err) {
      console.error('Error fetching tree data:', err);
      setError('Failed to load tree details');
    } finally {
      setIsLoading(false);
    }
  };
  
  if (!selectedFeature || selectedFeature.type !== 'tree') {
    return null;
  }
  
  // Loading state
  if (isLoading) {
    return (
      <Card className="absolute right-4 top-4 z-20 w-80 shadow-lg bg-white">
        <CardHeader className="py-3 border-b flex flex-row justify-between items-center">
          <CardTitle className="text-sm flex items-center">
            <Trees className="h-4 w-4 mr-2" />
            Loading Tree Information
          </CardTitle>
          <button 
            onClick={() => dispatch(clearSelectedFeature())}
            className="h-6 w-6 rounded-full flex items-center justify-center hover:bg-gray-100"
          >
            <X className="h-4 w-4" />
          </button>
        </CardHeader>
        <CardContent className="p-6 flex justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </CardContent>
      </Card>
    );
  }
  
  // Error state
  if (error) {
    return (
      <Card className="absolute right-4 top-4 z-20 w-80 shadow-lg bg-white">
        <CardHeader className="py-3 border-b flex flex-row justify-between items-center">
          <CardTitle className="text-sm flex items-center">
            <TriangleAlert className="h-4 w-4 mr-2 text-red-500" />
            Error
          </CardTitle>
          <button 
            onClick={() => dispatch(clearSelectedFeature())}
            className="h-6 w-6 rounded-full flex items-center justify-center hover:bg-gray-100"
          >
            <X className="h-4 w-4" />
          </button>
        </CardHeader>
        <CardContent className="p-4">
          <p className="text-red-500">{error}</p>
          <button 
            onClick={() => fetchTreeDetails(selectedFeature.data.id)}
            className="mt-2 w-full p-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </CardContent>
      </Card>
    );
  }
  
  // Use loaded data if available, fall back to basic data from selection
  const tree = treeData || selectedFeature.data;
  
  // Define risk level colors
  const riskColors = {
    high: 'text-red-600 bg-red-50',
    medium: 'text-orange-600 bg-orange-50',
    low: 'text-green-600 bg-green-50',
    unknown: 'text-gray-600 bg-gray-50'
  };
  
  const riskColor = riskColors[tree.risk_level] || riskColors.unknown;
  
  return (
    <Card className="absolute right-4 top-4 z-20 w-80 shadow-lg bg-white">
      <CardHeader className="py-3 border-b flex flex-row justify-between items-center">
        <CardTitle className="text-sm flex items-center">
          <Trees className="h-4 w-4 mr-2" />
          Tree Information
        </CardTitle>
        <button 
          onClick={() => dispatch(clearSelectedFeature())}
          className="h-6 w-6 rounded-full flex items-center justify-center hover:bg-gray-100"
        >
          <X className="h-4 w-4" />
        </button>
      </CardHeader>
      
      <CardContent className="p-4">
        <div className="mb-4">
          <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${riskColor}`}>
            <TriangleAlert className="h-3 w-3 mr-1" />
            {tree.risk_level?.toUpperCase() || 'UNKNOWN'} RISK
          </div>
        </div>
      
        <div className="space-y-3">
          <div className="flex items-start">
            <div className="w-24 text-xs text-gray-500">Species</div>
            <div className="flex-1 font-medium">{tree.species || 'Unknown'}</div>
          </div>
          
          <div className="flex items-start">
            <div className="w-24 text-xs text-gray-500">Height</div>
            <div className="flex-1 font-medium">{tree.height} meters</div>
          </div>
          
          <div className="flex items-start">
            <div className="w-24 text-xs text-gray-500">Canopy Width</div>
            <div className="flex-1 font-medium">{tree.canopy_width || 'N/A'} meters</div>
          </div>
          
          <div className="flex items-start">
            <div className="w-24 text-xs text-gray-500">ID</div>
            <div className="flex-1 font-mono text-sm">{tree.id}</div>
          </div>
          
          <div className="flex items-start">
            <div className="w-24 text-xs text-gray-500">Location</div>
            <div className="flex-1 font-mono text-xs">
              {tree.latitude.toFixed(6)}, {tree.longitude.toFixed(6)}
            </div>
          </div>
          
          {tree.last_assessment && (
            <div className="flex items-start">
              <div className="w-24 text-xs text-gray-500">Last Assessment</div>
              <div className="flex-1 font-medium">
                <div className="flex items-center">
                  <Calendar className="h-3 w-3 mr-1" />
                  {tree.last_assessment}
                </div>
              </div>
            </div>
          )}
          
          {tree.notes && (
            <div className="mt-3 p-2 bg-gray-50 rounded text-sm">
              <div className="flex items-center text-xs text-gray-500 mb-1">
                <Info className="h-3 w-3 mr-1" />
                Notes
              </div>
              {tree.notes}
            </div>
          )}
        </div>
      </CardContent>
      
      <CardFooter className="flex justify-between p-3 border-t">
        <button className="text-xs px-3 py-1 bg-blue-50 text-blue-600 rounded hover:bg-blue-100">
          View Full Details
        </button>
        <button className="text-xs px-3 py-1 bg-green-50 text-green-600 rounded hover:bg-green-100">
          Validation Queue
        </button>
      </CardFooter>
    </Card>
  );
};

export default TreeInfo;