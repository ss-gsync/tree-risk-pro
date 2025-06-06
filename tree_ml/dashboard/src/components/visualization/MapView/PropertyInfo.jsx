// src/components/visualization/MapView/PropertyInfo.jsx

import React, { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { X, Home, MapPin, Trees, FileText, AlertTriangle, ArrowRight } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { clearSelectedFeature } from './mapSlice';
import { PropertyService, TreeService } from '../../../services/api/apiService';

const PropertyInfo = () => {
  const dispatch = useDispatch();
  const { selectedFeature } = useSelector((state) => state.map);
  const [isLoading, setIsLoading] = useState(false);
  const [propertyDetails, setPropertyDetails] = useState(null);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    if (selectedFeature && selectedFeature.type === 'property') {
      fetchPropertyDetails(selectedFeature.data.id);
    } else {
      setPropertyDetails(null);
    }
  }, [selectedFeature]);
  
  const fetchPropertyDetails = async (propertyId) => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Fetch detailed property info
      const property = await PropertyService.getProperty(propertyId);
      const trees = await PropertyService.getPropertyTrees(propertyId);
      
      setPropertyDetails({
        ...property,
        trees,
        tree_count: trees.length,
        high_risk_count: trees.filter(tree => tree.risk_level === 'high').length
      });
    } catch (err) {
      console.error('Error fetching property details:', err);
      setError('Failed to load property details');
    } finally {
      setIsLoading(false);
    }
  };
  
  if (!selectedFeature || selectedFeature.type !== 'property') {
    return null;
  }
  
  const property = selectedFeature.data;
  
  // Define risk level colors
  const riskColors = {
    high: 'text-red-600 bg-red-50',
    medium: 'text-orange-600 bg-orange-50',
    low: 'text-green-600 bg-green-50',
    unknown: 'text-gray-600 bg-gray-50'
  };
  
  // Loading state
  if (isLoading) {
    return (
      <Card className="absolute right-4 top-4 z-20 w-80 shadow-lg bg-white">
        <CardHeader className="py-3 border-b flex flex-row justify-between items-center">
          <CardTitle className="text-sm flex items-center">
            <Home className="h-4 w-4 mr-2" />
            Loading Property...
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
            <AlertTriangle className="h-4 w-4 mr-2 text-red-500" />
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
            onClick={() => fetchPropertyDetails(property.id)}
            className="mt-2 w-full p-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Retry
          </button>
        </CardContent>
      </Card>
    );
  }
  
  // Use propertyDetails if available, otherwise fall back to the basic data
  const displayProperty = propertyDetails || property;
  const riskColor = riskColors[displayProperty.risk_level] || riskColors.unknown;
  
  return (
    <Card className="absolute right-4 top-4 z-20 w-80 shadow-lg bg-white">
      <CardHeader className="py-3 border-b flex flex-row justify-between items-center">
        <CardTitle className="text-sm flex items-center">
          <Home className="h-4 w-4 mr-2" />
          Property Information
        </CardTitle>
        <button 
          onClick={() => dispatch(clearSelectedFeature())}
          className="h-6 w-6 rounded-full flex items-center justify-center hover:bg-gray-100"
        >
          <X className="h-4 w-4" />
        </button>
      </CardHeader>
      
      <CardContent className="p-4">
        <div className="mb-3">
          <h3 className="font-medium">{property.address}</h3>
          <div className="text-sm text-gray-500 flex items-center">
            <MapPin className="h-3 w-3 mr-1" />
            {property.city || 'Austin'}, TX
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="p-2 bg-blue-50 rounded">
            <div className="text-xs text-gray-500 mb-1">Property ID</div>
            <div className="font-medium text-sm">{property.id}</div>
          </div>
          
          <div className="p-2 bg-green-50 rounded">
            <div className="text-xs text-gray-500 mb-1">Trees</div>
            <div className="font-medium text-sm flex items-center">
              <Trees className="h-3 w-3 mr-1" />
              {property.tree_count || '12'} trees
            </div>
          </div>
        </div>
        
        <div className="mb-4">
          <div className="text-xs text-gray-500 mb-2">Risk Assessment</div>
          <div className="p-2 bg-gray-50 rounded flex items-center justify-between">
            <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${riskColor}`}>
              <AlertTriangle className="h-3 w-3 mr-1" />
              {property.risk_level?.toUpperCase() || 'LOW'} RISK
            </div>
            <div className="text-sm">
              {property.high_risk_count || '1'} high risk trees
            </div>
          </div>
        </div>
        
        <div className="space-y-2">
          <div className="p-2 bg-gray-50 rounded flex items-center justify-between hover:bg-gray-100 cursor-pointer">
            <div className="flex items-center">
              <FileText className="h-4 w-4 mr-2 text-blue-500" />
              <span className="text-sm">Last Assessment Report</span>
            </div>
            <ArrowRight className="h-4 w-4 text-gray-400" />
          </div>
          
          <div className="p-2 bg-gray-50 rounded flex items-center justify-between hover:bg-gray-100 cursor-pointer">
            <div className="flex items-center">
              <Trees className="h-4 w-4 mr-2 text-green-500" />
              <span className="text-sm">View All Trees</span>
            </div>
            <ArrowRight className="h-4 w-4 text-gray-400" />
          </div>
        </div>
      </CardContent>
      
      <CardFooter className="flex justify-between p-3 border-t">
        <button className="text-xs px-3 py-1 bg-blue-50 text-blue-600 rounded hover:bg-blue-100">
          Full Details
        </button>
        <button className="text-xs px-3 py-1 bg-green-50 text-green-600 rounded hover:bg-green-100">
          Generate Report
        </button>
      </CardFooter>
    </Card>
  );
};

export default PropertyInfo;