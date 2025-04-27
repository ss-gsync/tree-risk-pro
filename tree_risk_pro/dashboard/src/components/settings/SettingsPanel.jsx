// src/components/settings/SettingsPanel.jsx

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '../ui/card';
import { Label } from '../ui/label';
import { Button } from '../ui/button';
import { save as saveToast, info as infoToast } from '../../utils/toast';

const SettingsPanel = () => {
  // Initialize settings from localStorage or use defaults
  const [settings, setSettings] = useState(() => {
    try {
      const savedSettings = localStorage.getItem('treeRiskDashboardSettings');
      if (savedSettings) {
        return JSON.parse(savedSettings);
      }
    } catch (e) {
      console.error('Error loading settings:', e);
    }
    
    // Default settings
    return {
      map3DApi: 'cesium',         // 'cesium' or 'javascript'
      defaultView: '2d',          // '2d' or '3d'
      theme: 'light',             // 'light' or 'dark'
      defaultMapType: 'roadmap',  // 'roadmap', 'satellite', 'hybrid', 'terrain'
      showHighRiskByDefault: false, // Whether to filter to high risk trees by default
      mapSettings: {
        showTerrain: false,       // Show terrain in 3D view
        showLabels: true,         // Show labels in satellite view
        zoomLevel: 10,            // Default zoom level
        center: [-96.7800, 32.8600] // Default center coordinates [lng, lat]
      },
      geminiSettings: {
        detectionThreshold: 0.7,  // 0.1 to 0.9
        maxTrees: 20,             // 5 to 50
        includeRiskAnalysis: true,
        detailLevel: 'high'       // 'low', 'medium', 'high'
      },
      autoSaveDetectionResults: true
    };
  });
  
  // Track whether settings have been changed
  const [isChanged, setIsChanged] = useState(false);
  
  // Track the current 3D mode to apply changes without reload
  const [currentIs3DMode, setCurrentIs3DMode] = useState(false);
  
  // Listen for 3D mode changes from other components
  useEffect(() => {
    const handleMapModeChange = (event) => {
      const { mode } = event.detail;
      setCurrentIs3DMode(mode === '3D');
    };
    
    window.addEventListener('mapModeChanged', handleMapModeChange);
    
    // Check initial 3D state from the global variable
    if (typeof window.is3DModeActive !== 'undefined') {
      setCurrentIs3DMode(window.is3DModeActive);
    }
    
    return () => {
      window.removeEventListener('mapModeChanged', handleMapModeChange);
    };
  }, []);
  
  /**
   * Handles form field changes in the settings form
   * 
   * This function supports both simple top-level settings and nested settings
   * (like mapSettings.showLabels). It preserves existing values while updating
   * only the changed field.
   * 
   * @param {Object} e - The event object from the form field change
   */
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;
    
    // Handle nested settings (e.g., mapSettings.showLabels)
    if (name.includes('.')) {
      const [parent, child] = name.split('.');
      setSettings(prevSettings => ({
        ...prevSettings,
        [parent]: {
          ...prevSettings[parent],
          [child]: newValue
        }
      }));
    } else {
      // Handle top-level settings
      setSettings(prevSettings => ({
        ...prevSettings,
        [name]: newValue
      }));
    }
    
    // Mark as changed so the Save button becomes enabled
    setIsChanged(true);
  };
  
  /**
   * Applies all settings without requiring a page reload
   * 
   * This function handles saving settings to localStorage, then dispatches
   * events to notify other components about the changes. It includes special
   * handling for 3D API changes, which require exiting 3D mode first.
   */
  const handleSave = () => {
    // Store current values for comparison
    const previousSettings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
    const previousApi = previousSettings.map3DApi || 'cesium';
    
    // Save settings to localStorage
    localStorage.setItem('treeRiskDashboardSettings', JSON.stringify(settings));
    
    // Show success toast
    saveToast('Settings saved successfully');
    
    // Check if 3D API setting changed, which requires special handling
    if (settings.map3DApi !== previousApi) {
      // Create a custom event to notify the MapView component
      const apiChangeEvent = new CustomEvent('map3DApiChanged', {
        detail: {
          newApi: settings.map3DApi,
          previousApi: previousApi
        }
      });
      
      // If currently in 3D mode, we need to exit it first
      if (currentIs3DMode) {
        // Notify user about exiting 3D mode
        infoToast('Exiting 3D mode to apply API change...');
        
        // Create a toggle event to exit 3D mode
        const toggleEvent = new CustomEvent('requestToggle3D', {
          detail: { force2D: true }
        });
        
        // Dispatch both events with a slight delay between them
        window.dispatchEvent(toggleEvent);
        
        // Send the API change event after exiting 3D mode
        setTimeout(() => {
          window.dispatchEvent(apiChangeEvent);
          
          // Tell user they can re-enter 3D mode with the new API
          infoToast('API changed to ' + (settings.map3DApi === 'cesium' ? 'Google Map Tiles API (Cesium)' : 'Google Maps JavaScript API') + '. You can now re-enter 3D mode.');
        }, 300);
      } else {
        // If not in 3D mode, just dispatch the API change event
        window.dispatchEvent(apiChangeEvent);
      }
    }
    
    // Trigger a storage event to notify other components of all setting changes
    window.dispatchEvent(new Event('storage'));
    
    setIsChanged(false);
  };
  
  /**
   * Resets all settings to their default values
   * 
   * This doesn't save the reset immediately - the user still needs to click Save
   * to apply the changes. This allows previewing the defaults before committing.
   */
  const handleReset = () => {
    // Reset to defaults
    setSettings({
      map3DApi: 'cesium',
      defaultView: '2d',
      theme: 'light',
      defaultMapType: 'roadmap',
      showHighRiskByDefault: false,
      mapSettings: {
        showTerrain: false,
        showLabels: true,
        zoomLevel: 10,
        center: [-96.7800, 32.8600]
      },
      geminiSettings: {
        detectionThreshold: 0.7,
        maxTrees: 20,
        includeRiskAnalysis: true,
        detailLevel: 'high'
      },
      autoSaveDetectionResults: true
    });
    
    // Mark as changed so the Save button becomes enabled
    setIsChanged(true);
  };
  
  return (
    <div className="settings-panel p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Settings</h1>
        {/* 
          Return to Map button - preserves 3D state when returning to map view
          1. Navigates back to the Map view first
          2. If user was in 3D mode, waits for map to load then triggers 3D view
          3. Ensures map is properly resized after navigation
        */}
        <button
          onClick={() => {
            // First check if we're currently in 3D mode
            const is3DMode = typeof window.is3DModeActive !== 'undefined' ? window.is3DModeActive : false;
            
            // Navigate back to Map view first
            window.dispatchEvent(new CustomEvent('navigateTo', { detail: { view: 'Map' } }));
            
            // If we need to go back to 3D mode, trigger it directly
            if (is3DMode) {
              console.log("Settings: Returning to 3D view");
              // Give some time for the map to load
              setTimeout(() => {
                // Trigger toggle to 3D mode
                window.dispatchEvent(new CustomEvent('requestToggle3DViewType', {
                  detail: { show3D: true }
                }));
                
                // Log that we've sent the event
                console.log("Settings: Sent 3D toggle event");
              }, 300);
            }
            
            // Ensure the map state is rendered correctly
            setTimeout(() => {
              window.dispatchEvent(new Event('resize'));
            }, 500);
          }}
          className="px-4 py-2 bg-gray-100 text-gray-800 border border-gray-200 rounded-md hover:bg-gray-200 transition-colors"
        >
          Back to Map
        </button>
      </div>
      
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Map Settings</CardTitle>
          <CardDescription>
            Configure map display and 3D view settings
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="defaultView">Default Map View</Label>
            <select
              id="defaultView"
              name="defaultView"
              value={settings.defaultView}
              onChange={handleChange}
              className="w-full p-2 bg-white text-gray-800 border border-gray-200 rounded-md focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
            >
              <option value="2d">2D View</option>
              <option value="3d">3D View</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="defaultMapType">Default Map Type</Label>
            <select
              id="defaultMapType"
              name="defaultMapType"
              value={settings.defaultMapType}
              onChange={handleChange}
              className="w-full p-2 bg-white text-gray-800 border border-gray-200 rounded-md focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
            >
              <option value="roadmap">Road Map</option>
              <option value="satellite">Satellite</option>
              <option value="hybrid">Hybrid</option>
              <option value="terrain">Terrain</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2 pt-2">
            <input
              type="checkbox"
              id="showHighRiskByDefault"
              name="showHighRiskByDefault"
              checked={settings.showHighRiskByDefault}
              onChange={handleChange}
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <Label htmlFor="showHighRiskByDefault" className="cursor-pointer">Show only high-risk trees by default</Label>
          </div>
          
          <div className="space-y-2 mt-4 pt-4 border-t border-gray-200">
            <Label htmlFor="map3DApi">3D Map API</Label>
            <select
              id="map3DApi"
              name="map3DApi"
              value={settings.map3DApi}
              onChange={handleChange}
              className="w-full p-2 bg-white text-gray-800 border border-gray-200 rounded-md focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
            >
              <option value="cesium">Google Map Tiles API (Cesium)</option>
              <option value="javascript">Google Maps JavaScript API</option>
            </select>
            <p className="text-sm text-gray-500">
              The API used for 3D rendering when switching to 3D view:
              <ul className="ml-5 mt-1 list-disc">
                <li>Google Map Tiles API (Cesium) - Higher quality 3D with photorealistic tiles</li>
                <li>Google Maps JavaScript API - Standard Google Maps 3D buildings</li>
              </ul>
            </p>
          </div>
          
          <div className="bg-blue-50 p-3 rounded-md mt-4 text-sm">
            <h4 className="font-medium text-blue-700 mb-2">3D Navigation Instructions</h4>
            
            {settings.map3DApi === 'cesium' ? (
              <div className="text-gray-700">
                <p className="mb-2"><strong>Google Map Tiles API (Cesium):</strong></p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>Hold <strong>Ctrl</strong> while dragging the mouse to tilt and rotate</li>
                  <li>Use scroll wheel to zoom in/out</li>
                  <li>Hold <strong>Shift</strong> to accelerate movement</li>
                  <li>Right-click and drag to rotate around a point</li>
                </ul>
              </div>
            ) : (
              <div className="text-gray-700">
                <p className="mb-2"><strong>Google Maps JavaScript API:</strong></p>
                <ul className="list-disc pl-5 space-y-1">
                  <li>Left-click and drag to pan</li>
                  <li>Use scroll wheel to zoom in/out</li>
                  <li>Hold <strong>Ctrl + drag</strong> or right-click and drag to rotate</li>
                  <li>Hold <strong>Shift + drag</strong> to tilt the view</li>
                </ul>
              </div>
            )}
          </div>
          
          <div className="border-t border-gray-200 pt-4 mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">Map Display Settings</h4>
            
            <div className="flex items-center space-x-2 mt-3">
              <input
                type="checkbox"
                id="showLabels"
                name="mapSettings.showLabels"
                checked={settings.mapSettings?.showLabels !== false}
                onChange={(e) => {
                  // Update the setting
                  setSettings(prev => ({
                    ...prev,
                    mapSettings: {
                      ...prev.mapSettings,
                      showLabels: e.target.checked
                    }
                  }));
                  
                  // Flag that changes have been made
                  setIsChanged(true);
                }}
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <Label htmlFor="showLabels" className="cursor-pointer">Show labels in satellite view</Label>
            </div>
            
            <div className="flex items-center space-x-2 mt-2">
              <input
                type="checkbox"
                id="showTerrain"
                name="mapSettings.showTerrain"
                checked={settings.mapSettings?.showTerrain === true}
                onChange={(e) => setSettings(prev => ({
                  ...prev,
                  mapSettings: {
                    ...prev.mapSettings,
                    showTerrain: e.target.checked
                  }
                }))}
                className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <Label htmlFor="showTerrain" className="cursor-pointer">Show terrain in 3D view</Label>
            </div>
          </div>
        </CardContent>
      </Card>
      
      
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Interface Settings</CardTitle>
          <CardDescription>
            Customize the dashboard appearance
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="theme">Theme</Label>
            <select
              id="theme"
              name="theme"
              value={settings.theme}
              onChange={handleChange}
              className="w-full p-2 bg-white text-gray-800 border border-gray-200 rounded-md focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
            >
              <option value="light">Light</option>
              <option value="dark">Dark (Coming soon)</option>
            </select>
          </div>
        </CardContent>
      </Card>
      
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Gemini AI Settings</CardTitle>
          <CardDescription>
            Configure tree detection parameters
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="geminiSettings.detectionThreshold">Detection Threshold ({settings.geminiSettings?.detectionThreshold || 0.7})</Label>
            <input
              type="range"
              id="geminiSettings.detectionThreshold"
              name="geminiSettings.detectionThreshold"
              min="0.1"
              max="0.9"
              step="0.1"
              value={settings.geminiSettings?.detectionThreshold || 0.7}
              onChange={(e) => {
                const value = parseFloat(e.target.value);
                setSettings(prev => ({
                  ...prev,
                  geminiSettings: {
                    ...prev.geminiSettings,
                    detectionThreshold: value
                  }
                }));
                setIsChanged(true);
              }}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>0.1 (More trees, less confidence)</span>
              <span>0.9 (Fewer trees, more confidence)</span>
            </div>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="geminiSettings.maxTrees">Maximum Tree Count</Label>
            <input
              type="number"
              id="geminiSettings.maxTrees"
              name="geminiSettings.maxTrees"
              min="5"
              max="50"
              value={settings.geminiSettings?.maxTrees || 20}
              onChange={(e) => {
                const value = parseInt(e.target.value);
                setSettings(prev => ({
                  ...prev,
                  geminiSettings: {
                    ...prev.geminiSettings,
                    maxTrees: value
                  }
                }));
                setIsChanged(true);
              }}
              className="w-full p-2 bg-white text-gray-800 border border-gray-200 rounded-md focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
            />
            <p className="text-sm text-gray-500">
              Maximum number of trees to detect in a single analysis (5-50)
            </p>
          </div>
          
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="geminiSettings.includeRiskAnalysis"
              name="geminiSettings.includeRiskAnalysis"
              checked={settings.geminiSettings?.includeRiskAnalysis !== false}
              onChange={(e) => {
                const checked = e.target.checked;
                setSettings(prev => ({
                  ...prev,
                  geminiSettings: {
                    ...prev.geminiSettings,
                    includeRiskAnalysis: checked
                  }
                }));
                setIsChanged(true);
              }}
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <Label htmlFor="geminiSettings.includeRiskAnalysis" className="cursor-pointer">Include risk analysis in detection</Label>
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="geminiSettings.detailLevel">Detail Level</Label>
            <select
              id="geminiSettings.detailLevel"
              name="geminiSettings.detailLevel"
              value={settings.geminiSettings?.detailLevel || 'high'}
              onChange={(e) => {
                const value = e.target.value;
                setSettings(prev => ({
                  ...prev,
                  geminiSettings: {
                    ...prev.geminiSettings,
                    detailLevel: value
                  }
                }));
                setIsChanged(true);
              }}
              className="w-full p-2 bg-white text-gray-800 border border-gray-200 rounded-md focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
            >
              <option value="low">Low (Faster, less detailed)</option>
              <option value="medium">Medium (Balanced)</option>
              <option value="high">High (Slower, more detailed)</option>
            </select>
          </div>
          
          <div className="flex items-center space-x-2 pt-2">
            <input
              type="checkbox"
              id="autoSaveDetectionResults"
              name="autoSaveDetectionResults"
              checked={settings.autoSaveDetectionResults !== false}
              onChange={handleChange}
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <Label htmlFor="autoSaveDetectionResults" className="cursor-pointer">Automatically save detection results</Label>
          </div>
        </CardContent>
      </Card>
      
      <div className="flex justify-end space-x-4">
        <Button
          variant="outline"
          onClick={handleReset}
        >
          Reset to Defaults
        </Button>
        <Button
          onClick={handleSave}
          disabled={!isChanged}
        >
          Save Settings
        </Button>
      </div>
    </div>
  );
};

export default SettingsPanel;