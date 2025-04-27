// src/components/assessment/AssessmentPanel.jsx

import React, { useState, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { 
  Activity, 
  AlertTriangle, 
  ChevronDown, 
  ChevronRight, 
  FileText, 
  Filter,
  MoreHorizontal, 
  RefreshCw, 
  Search,
  Tree
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { PropertyService, TreeService } from '../services/api/apiService';

const AssessmentPanel = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedProperty, setExpandedProperty] = useState(null);
  const [properties, setProperties] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const { selectedFeature } = useSelector((state) => state.map);

  const fetchData = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Fetch properties using the API service
      const propertiesData = await PropertyService.getProperties();
      
      // Fetch trees for each property
      const propertiesWithTrees = await Promise.all(
        propertiesData.map(async (property) => {
          // Use the API service to get trees for the property
          const propertyTrees = await PropertyService.getPropertyTrees(property.id);
          
          return {
            ...property,
            trees: propertyTrees
          };
        })
      );
      
      setProperties(propertiesWithTrees);
    } catch (error) {
      console.error('Error fetching properties:', error);
      setError('Failed to load properties. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch properties data
  useEffect(() => {
    fetchData();
  }, []);

  const filteredProperties = properties.filter(property => 
    property.address.toLowerCase().includes(searchQuery.toLowerCase()) ||
    property.trees.some(tree => 
      tree.species && tree.species.toLowerCase().includes(searchQuery.toLowerCase())
    )
  );

  const togglePropertyExpand = (propertyId) => {
    if (expandedProperty === propertyId) {
      setExpandedProperty(null);
    } else {
      setExpandedProperty(propertyId);
    }
  };
  
  const getRiskColor = (risk_level) => {
    switch (risk_level) {
      case 'high': return 'text-red-600';
      case 'medium': return 'text-orange-600';
      case 'low': return 'text-green-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="h-full flex flex-col">
      <CardHeader className="py-3 border-b">
        <div className="flex justify-between items-center">
          <CardTitle className="text-sm flex items-center">
            <Tree className="h-4 w-4 mr-2" />
            Assessment Panel
          </CardTitle>
          <div className="flex">
            <button className="p-1 rounded-full hover:bg-gray-100" title="Refresh">
              <RefreshCw className="h-4 w-4 text-gray-500" />
            </button>
            <button className="p-1 rounded-full hover:bg-gray-100" title="More options">
              <MoreHorizontal className="h-4 w-4 text-gray-500" />
            </button>
          </div>
        </div>
      </CardHeader>
      
      <div className="p-3 border-b">
        <div className="relative">
          <input
            type="text"
            placeholder="Search properties or trees..."
            className="w-full pl-8 pr-3 py-2 border rounded-md text-sm"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-400" />
        </div>
      </div>
      
      <CardContent className="p-0 flex-1 overflow-auto">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-full p-4 text-center text-red-500">
            <div>
              <AlertTriangle className="h-10 w-10 mx-auto mb-2" />
              <p>{error}</p>
              <button 
                onClick={() => fetchData()} 
                className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
              >
                Retry
              </button>
            </div>
          </div>
        ) : filteredProperties.length === 0 ? (
          <div className="flex items-center justify-center h-full p-4 text-center text-gray-500">
            <div>
              <Filter className="h-10 w-10 mx-auto mb-2" />
              <p>No properties match your search</p>
            </div>
          </div>
        ) : (
          <div className="divide-y">
            {filteredProperties.map((property) => (
              <div key={property.id} className="py-2">
                <div 
                  className="flex items-center justify-between px-4 py-2 cursor-pointer hover:bg-gray-50"
                  onClick={() => togglePropertyExpand(property.id)}
                >
                  <div>
                    <h3 className="font-medium text-sm">{property.address}</h3>
                    <div className="text-xs text-gray-500 flex items-center mt-1">
                      <Tree className="h-3 w-3 mr-1" />
                      {property.trees.length} trees
                      
                      {/* Show high risk indicator if any high risk trees */}
                      {property.trees.some(tree => tree.risk_level === 'high') && (
                        <span className="ml-2 flex items-center text-red-600">
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          High risk
                        </span>
                      )}
                    </div>
                  </div>
                  <div>
                    {expandedProperty === property.id ? (
                      <ChevronDown className="h-4 w-4 text-gray-500" />
                    ) : (
                      <ChevronRight className="h-4 w-4 text-gray-500" />
                    )}
                  </div>
                </div>
                
                {/* Expanded tree list */}
                {expandedProperty === property.id && (
                  <div className="px-4 pt-1 pb-2">
                    {property.trees.length === 0 ? (
                      <div className="text-sm text-gray-500 italic py-2">
                        No trees recorded for this property
                      </div>
                    ) : (
                      <div className="space-y-2 mt-1">
                        {property.trees.map((tree) => (
                          <div 
                            key={tree.id}
                            className="flex items-center justify-between p-2 rounded-md bg-gray-50 hover:bg-gray-100 cursor-pointer"
                          >
                            <div>
                              <div className="flex items-center">
                                <span className={`font-medium ${getRiskColor(tree.risk_level)}`}>
                                  {tree.species}
                                </span>
                                <span className="mx-1">•</span>
                                <span className="text-xs">{tree.height}ft</span>
                              </div>
                              
                              {/* Risk factors */}
                              {tree.risk_factors && tree.risk_factors.length > 0 && (
                                <div className="mt-1">
                                  {tree.risk_factors.map((factor, idx) => (
                                    <div 
                                      key={idx}
                                      className={`text-xs flex items-center ${getRiskColor(factor.level)}`}
                                    >
                                      <AlertTriangle className="h-3 w-3 mr-1" />
                                      {factor.description}
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                            
                            <div className="flex">
                              <button 
                                className="p-1 rounded hover:bg-gray-200" 
                                title="View details"
                              >
                                <FileText className="h-4 w-4 text-gray-600" />
                              </button>
                              <button 
                                className="p-1 rounded hover:bg-gray-200" 
                                title="Risk assessment"
                              >
                                <Activity className="h-4 w-4 text-gray-600" />
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    
                    <button className="mt-3 px-3 py-1 text-xs text-blue-600 bg-blue-50 rounded hover:bg-blue-100 w-full">
                      Generate Property Report
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </div>
  );
};

export default AssessmentPanel;