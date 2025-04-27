// src/App.jsx
import React, { Suspense, useRef, useState } from 'react';
import { Provider } from 'react-redux';
import { store } from './store';
import Layout from './components/common/Layout';
import LoadingSpinner from './components/common/Loading';
import MapView from './components/visualization/MapView/MapView.jsx';
import ValidationQueue from './components/assessment/Validation/ValidationQueue';
import MapAssessmentPanel from './components/assessment/MapAssessmentPanel';
import Login from './components/auth/Login';
import { AuthProvider, useAuth } from './components/auth/AuthContext';
import { Layers, Globe } from 'lucide-react';

// Import the settings panel
import SettingsPanel from './components/settings/SettingsPanel';

const DashboardContent = () => {
  const mapRef = useRef(null);
  const cesiumRef = useRef(null);
  const mapDataRef = useRef({
    trees: [],
    properties: []
  });
  const [showNewInterface, setShowNewInterface] = useState(false); // Default to classic view
  const [currentView, setCurrentView] = useState('Map'); // Default to Map view
  
  // Add effect to handle both map filtering events and 3D view toggle
  React.useEffect(() => {
    // Handler for 3D view toggle request from MapControls
    const handleToggle3DViewRequest = (event) => {
      const { show3D } = event.detail;
      setShowNewInterface(false); // Always stay in Classic View
      
      // Set global flag for 3D mode
      window.is3DModeActive = show3D;
    };
    
    // Handler for navigation events from components
    const handleNavigationEvent = (event) => {
      const { view } = event.detail;
      if (view) {
        console.log("Navigation event received:", view);
        handleNavigation(view);
      }
    };
    
    // Add event listeners
    window.addEventListener('requestToggle3DViewType', handleToggle3DViewRequest);
    window.addEventListener('navigateTo', handleNavigationEvent);
    
    // Handler for the new applyMapFilters event that supports all risk levels
    const handleMapFilters = (event) => {
      console.log("Applying map filters:", event.detail.filters);
      const { riskLevel } = event.detail.filters;
      
      // Set global variables
      window.currentRiskFilter = riskLevel;
      window.highRiskFilterActive = riskLevel === 'high';
      window.showOnlyHighRiskTrees = riskLevel !== 'all';
      
      console.log("Filter state set: currentRiskFilter =", window.currentRiskFilter,
                  "highRiskFilterActive =", window.highRiskFilterActive,
                  "showOnlyHighRiskTrees =", window.showOnlyHighRiskTrees);
      
      if (mapRef.current && mapDataRef.current) {
        const googleMap = mapRef.current.getMap();
        if (!googleMap) return;
        
        // Clear existing markers
        const markers = mapRef.current.getMarkers();
        if (markers) {
          markers.forEach(marker => {
            if (marker.setMap) {
              marker.setMap(null);
            } else if (marker.map) {
              marker.map = null;
            }
          });
          
          // Clear the markers array
          mapRef.current.getMarkers().length = 0;
        }
        
        // Refresh markers with the selected filter
        const refreshEvent = new CustomEvent('refreshMapMarkers', { 
          detail: { 
            riskFilter: riskLevel
          } 
        });
        window.dispatchEvent(refreshEvent);
        
        // Update the validation queue filter
        if (window.setValidationRiskFilter) {
          window.setValidationRiskFilter(riskLevel);
        }
      }
    };
    
    // Legacy handler for high risk trees only (for backward compatibility)
    const showHighRiskTrees = () => {
      console.log("Showing high risk trees");
      // Set global variables to indicate high risk filter is active
      window.currentRiskFilter = 'high';
      window.highRiskFilterActive = true;
      window.showOnlyHighRiskTrees = true;
      
      if (mapRef.current && mapDataRef.current) {
        const googleMap = mapRef.current.getMap();
        if (!googleMap) return;
        
        // Dispatch events to update filters
        const filterEvent = new CustomEvent('filterHighRiskOnly', { 
          detail: { active: true } 
        });
        window.dispatchEvent(filterEvent);
        
        // Refresh map markers to show only high risk trees
        const refreshEvent = new CustomEvent('refreshMapMarkers', { 
          detail: { 
            riskFilter: 'high'
          } 
        });
        window.dispatchEvent(refreshEvent);
        
        // Update the validation queue filter
        if (window.setValidationRiskFilter) {
          window.setValidationRiskFilter('high');
        }
      }
    };
    
    // Function to reset filters
    const resetFilters = () => {
      console.log("Resetting filters");
      // Set global flags
      window.currentRiskFilter = 'all';
      window.highRiskFilterActive = false;
      window.showOnlyHighRiskTrees = false;
      
      // Dispatch events to reset filters
      const filterEvent = new CustomEvent('filterHighRiskOnly', { 
        detail: { active: false } 
      });
      window.dispatchEvent(filterEvent);
      
      // Refresh map markers to show all trees
      const refreshEvent = new CustomEvent('refreshMapMarkers', { 
        detail: { 
          riskFilter: 'all'
        } 
      });
      window.dispatchEvent(refreshEvent);
      
      // Update the validation queue filter
      if (window.setValidationRiskFilter) {
        window.setValidationRiskFilter('all');
      }
    };
    
    // Add event listeners
    window.addEventListener('applyMapFilters', handleMapFilters);
    window.addEventListener('showHighRiskTrees', showHighRiskTrees);
    window.addEventListener('resetFilters', resetFilters);
    
    // Cleanup
    return () => {
      window.removeEventListener('applyMapFilters', handleMapFilters);
      window.removeEventListener('showHighRiskTrees', showHighRiskTrees);
      window.removeEventListener('resetFilters', resetFilters);
      window.removeEventListener('requestToggle3DViewType', handleToggle3DViewRequest);
      window.removeEventListener('navigateTo', handleNavigationEvent);
    };
  }, []);
  
  // Handler for navigation from sidebar
  const handleNavigation = (view) => {
    console.log("Navigating to view:", view);
    setCurrentView(view);
    
    // When navigating back to Map view, ensure the UI is reset properly
    if (view === "Map") {
      setShowNewInterface(false);
      
      // Preserve the 3D mode state if it's set globally
      if (typeof window.is3DModeActive !== 'undefined') {
        console.log("Preserving 3D mode state:", window.is3DModeActive);
        // Dispatch an event to ensure 3D mode is maintained if needed
        if (window.is3DModeActive) {
          // Wait a small delay to ensure the map is ready before applying 3D mode
          setTimeout(() => {
            window.dispatchEvent(new CustomEvent('mapModeChanged', { 
              detail: { mode: '3D', tilt: 45 } 
            }));
          }, 300);
        }
      }
    }
  };

  return (
    <Layout mapRef={mapRef} mapDataRef={mapDataRef} onNavigate={handleNavigation}>
      <Suspense fallback={<LoadingSpinner />}>
        {currentView === 'Settings' ? (
          <SettingsPanel />
        ) : (
          // Main dashboard content (Map + Validation Queue)
          showNewInterface ? (
            <div className="h-full">
              <MapAssessmentPanel />
            </div>
          ) : (
            <div className="flex h-full relative">
              <div className="flex-1 absolute inset-0 transition-all duration-300" id="map-container" style={{ 
                right: "0px" // Will dynamically adjust when sidebars open
              }}>
                <MapView 
                  ref={mapRef} 
                  onDataLoaded={(data) => {
                    mapDataRef.current = data;
                  }}
                />
                {/* MapControls moved to sidebar */}
              </div>
              <div id="imagery-sidebar" className="absolute right-0 top-0 bottom-0 transition-all duration-300 border-l border-gray-200 overflow-hidden w-0 z-10">
                <ValidationQueue />
              </div>
            </div>
          )
        )}
      </Suspense>
    </Layout>
  );
};

/**
 * Main application content with authentication flow
 * Renders login screen, loading state, or dashboard based on auth status
 */
const AppContent = () => {
  const { isAuthenticated, isValidating, login } = useAuth();

  // Display loading spinner during authentication check
  if (isValidating) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-200">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-800 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // Security: Verify authentication status before showing content
  // This prevents content from being shown briefly before auth check completes
  if (!isAuthenticated) {
    // Add a unique key to force complete remount after logout
    return <Login onLogin={login} key={`login-${Date.now()}`} />;
  }

  // Render dashboard for authenticated users
  return <DashboardContent />;
};

/**
 * Root application component
 * Sets up Redux store and authentication provider
 */
const App = () => {
  return (
    <Provider store={store}>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </Provider>
  );
};

export default App;