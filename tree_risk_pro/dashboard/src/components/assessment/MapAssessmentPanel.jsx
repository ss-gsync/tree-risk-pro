import React, { useState, useRef, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Card, CardContent } from '../ui/card';
import MapView from '../visualization/MapView/MapView';
import MapControls from '../visualization/MapView/MapControls';
import CesiumViewer from '../visualization/MapView/CesiumViewer';
import CesiumControls from '../visualization/MapView/CesiumControls';
import TreeDetection from './Detection/TreeDetection';
import ValidationSystem from './Validation/ValidationSystem';

/**
 * MapAssessmentPanel - Main panel for map visualization and tree assessment
 * Contains both Google Maps and Cesium 3D Tiles viewers
 */
const MapAssessmentPanel = () => {
  const [viewMode, setViewMode] = useState('cesium'); // 'google' or 'cesium' - default to 3D
  const mapRef = useRef(null);
  const cesiumRef = useRef(null);
  const [mapData, setMapData] = useState({});
  const [apiKey, setApiKey] = useState('');
  
  // Get Google Maps API key from environment
  useEffect(() => {
    const key = import.meta.env.VITE_GOOGLE_MAPS_API_KEY || '';
    setApiKey(key);
    
    // Listen for map mode toggle events from MapControls
    const handleMapModeToggle = (event) => {
      const { mode } = event.detail;
      console.log(`Switching map view mode to: ${mode}`);
      setViewMode(mode);
    };
    
    window.addEventListener('toggleMapMode', handleMapModeToggle);
    
    // Clean up event listener
    return () => {
      window.removeEventListener('toggleMapMode', handleMapModeToggle);
    };
  }, []);

  const handleMapDataLoaded = (data) => {
    setMapData(data);
  };

  return (
    <Card className="flex flex-col h-full">
      <CardContent className="p-0 flex flex-col h-full">
        <Tabs defaultValue="map" className="flex flex-col h-full">
          <TabsList className="bg-transparent border-b px-4 py-1">
            <TabsTrigger value="map">Map</TabsTrigger>
            <TabsTrigger value="detection">Detection Results</TabsTrigger>
            <TabsTrigger value="validation">Validation</TabsTrigger>
          </TabsList>
          
          <TabsContent value="map" className="flex flex-1 h-full p-0 m-0">
            <div className="flex h-full w-full">
              {/* Main map view */}
              <div className="flex-1 h-full relative">
                {/* Removed toggle buttons - using Map3DToggle in MapControls instead */}
                
                {viewMode === 'google' ? (
                  <MapView ref={mapRef} onDataLoaded={handleMapDataLoaded} />
                ) : (
                  <CesiumViewer ref={cesiumRef} onDataLoaded={handleMapDataLoaded} apiKey={apiKey} />
                )}
              </div>
              
              {/* Controls sidebar */}
              <div className="w-64 border-l p-4 overflow-y-auto">
                {viewMode === 'google' ? (
                  <MapControls mapRef={mapRef} mapDataRef={mapData} />
                ) : (
                  <CesiumControls cesiumRef={cesiumRef} />
                )}
              </div>
            </div>
          </TabsContent>
          
          <TabsContent value="detection" className="p-4 flex-1 overflow-auto">
            <TreeDetection />
          </TabsContent>
          
          <TabsContent value="validation" className="p-0 flex-1 overflow-auto">
            <ValidationSystem />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default MapAssessmentPanel;