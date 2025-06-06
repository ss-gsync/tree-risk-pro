// src/components/visualization/MapView/LayerManager.jsx

import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Layers, Eye, EyeOff, Map, Trees, Activity, Grid3X3 } from 'lucide-react';
import { toggleLayer, setActiveBasemap } from './mapSlice';

const LayerManager = () => {
  const dispatch = useDispatch();
  const { visibleLayers, activeBasemap } = useSelector((state) => state.map);

  const availableLayers = [
    { id: 'properties', name: 'Properties', icon: <Map className="h-4 w-4 mr-2" /> },
    { id: 'trees', name: 'Trees', icon: <Trees className="h-4 w-4 mr-2" /> },
    { id: 'risk', name: 'Risk Zones', icon: <Activity className="h-4 w-4 mr-2" /> },
    { id: 'lidar', name: 'LiDAR Data', icon: <Grid3X3 className="h-4 w-4 mr-2" /> },
  ];

  const basemapOptions = [
    { id: 'roadmap', name: 'Road Map' },
    { id: 'satellite', name: 'Satellite' },
    { id: 'hybrid', name: 'Hybrid' },
    { id: 'terrain', name: 'Terrain' }
  ];

  return (
    <Card className="absolute bottom-4 right-4 z-10 w-64 bg-white bg-opacity-90 shadow-lg">
      <CardHeader className="py-3">
        <CardTitle className="text-sm flex items-center">
          <Layers className="h-4 w-4 mr-2" />
          Map Layers
        </CardTitle>
      </CardHeader>
      <CardContent className="py-2">
        <div className="mb-4">
          <h3 className="text-xs font-semibold mb-2 text-gray-500">BASEMAP</h3>
          <div className="grid grid-cols-2 gap-1">
            {basemapOptions.map((basemap) => (
              <button
                key={basemap.id}
                className={`text-xs p-1 rounded ${
                  activeBasemap === basemap.id 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => dispatch(setActiveBasemap(basemap.id))}
              >
                {basemap.name}
              </button>
            ))}
          </div>
        </div>
        
        <h3 className="text-xs font-semibold mb-2 text-gray-500">DATA LAYERS</h3>
        <div className="space-y-1">
          {availableLayers.map((layer) => (
            <div 
              key={layer.id}
              className="flex items-center justify-between p-2 rounded-md hover:bg-gray-100"
            >
              <div className="flex items-center">
                {layer.icon}
                <span className="text-sm">{layer.name}</span>
              </div>
              <button
                onClick={() => dispatch(toggleLayer(layer.id))}
                className={`p-1 rounded-full ${
                  visibleLayers.includes(layer.id) 
                    ? 'bg-blue-100' 
                    : 'bg-gray-100'
                }`}
                aria-label={`Toggle ${layer.name} layer`}
              >
                {visibleLayers.includes(layer.id) ? (
                  <Eye className="h-4 w-4 text-blue-600" />
                ) : (
                  <EyeOff className="h-4 w-4 text-gray-400" />
                )}
              </button>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default LayerManager;