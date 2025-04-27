// src/hooks/useGoogleMapsApi.js

import { useState, useEffect } from 'react';

// Keep track of loading state globally
let isLoading = false;
let googleMapsPromise = null;

// Create a global method to check if Google Maps is fully loaded
window.isGoogleMapsFullyLoaded = function() {
  return !!(window.google && 
            window.google.maps && 
            window.google.maps.Map && 
            window.google.maps.OverlayView);
};

/**
 * Custom hook to load the Google Maps API
 * @param {string} apiKey - Your Google Maps API key
 * @param {Array} libraries - Array of libraries to load (e.g., ['places', 'geometry'])
 * @param {boolean} enableWebGL - Whether to enable WebGL for 3D maps
 * @returns {Object} - { isLoaded, loadError }
 */
export const useGoogleMapsApi = (apiKey, libraries = ['marker'], enableWebGL = true) => {
  const [isLoaded, setIsLoaded] = useState(window.isGoogleMapsFullyLoaded());
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    // If Google Maps is already available, set as loaded
    if (window.isGoogleMapsFullyLoaded()) {
      setIsLoaded(true);
      console.log('Google Maps API already loaded');
      return;
    }

    // If already loading, use existing promise
    if (isLoading) {
      googleMapsPromise
        .then(() => {
          setIsLoaded(true);
          console.log('Google Maps API loaded through existing promise');
        })
        .catch((error) => setLoadError(error));
      return;
    }

    const loadGoogleMapsApi = () => {
      isLoading = true;

      // Log API key details (without exposing the full key)
      if (apiKey) {
        const keyStart = apiKey.substring(0, 6);
        const keyEnd = apiKey.substring(apiKey.length - 4);
        const keyLength = apiKey.length;
        console.log(`Using Maps API key: ${keyStart}...${keyEnd} (${keyLength} chars)`);
        
        // Validate key format
        if (!apiKey.startsWith('AIza')) {
          console.error('Google Maps API key appears to be invalid - should start with "AIza"');
        }
      } else {
        console.warn('No Google Maps API key provided!');
      }

      // Create only one global callback function
      window.googleMapsInitialized = () => {
        console.log('Google Maps API loaded successfully');
        isLoading = false;
        
        // Wait a small bit to ensure all components are fully loaded
        setTimeout(() => {
          setIsLoaded(true);
        }, 100);
      };
      
      // Create error handler function
      window.googleMapsError = (error) => {
        console.error('Google Maps API failed to load:', error);
        if (error.includes('InvalidKeyMapError') || error.includes('MissingKeyMapError')) {
          console.error(`
            API Key Error: Your Google Maps API key is invalid or restricted.
            1. Check if the Maps JavaScript API is enabled in your Google Cloud Console
            2. Verify the key has no domain/IP restrictions or add this domain
            3. Check the API key billing account is active
          `);
        }
        isLoading = false;
        setLoadError(error);
      };

      // Create the script element
      const script = document.createElement('script');
      
      // Create a unique set of libraries to prevent duplicates
      // Note: 'webgl' is not a valid Google Maps library, but we'll handle 3D features with WebGL support
      let uniqueLibraries = [...new Set(libraries.filter(lib => lib !== 'webgl'))];
      // Valid Google Maps libraries include: places, drawing, geometry, visualization, marker
      
      // Check if we're using a demo/placeholder key or no key
      if (!apiKey || apiKey === 'your_maps_api_key' || apiKey === 'AIzaSyBnqECCMkYdFkO84ZpZv5ZNG2hh8Pci6C0') {
        console.warn('Using demo/placeholder Google Maps API key. For production, use a proper API key.');
        // For development/demo purposes, try to load map with restrictions
        script.src = `https://maps.googleapis.com/maps/api/js?v=beta&callback=googleMapsInitialized&error=googleMapsError${
          uniqueLibraries.length > 0 ? `&libraries=${uniqueLibraries.join(',')}` : ''
        }`;
      } else {
        // Normal loading with provided API key
        script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&v=beta&callback=googleMapsInitialized&error=googleMapsError${
          uniqueLibraries.length > 0 ? `&libraries=${uniqueLibraries.join(',')}` : ''
        }`;
      }

      // Add WebGL specific parameters if enabled
      if (enableWebGL) {
        // Using the beta channel for 3D features
        console.log('Enabling WebGL for 3D maps');
      }
      
      script.async = true;
      script.defer = true;

      // Create a promise to track loading
      googleMapsPromise = new Promise((resolve, reject) => {
        window.googleMapsInitialized = () => {
          console.log('Google Maps API loaded successfully through promise');
          isLoading = false;
          
          // Short delay to ensure all APIs are ready
          setTimeout(() => {
            setIsLoaded(true);
            resolve();
          }, 100);
        };
        
        script.onerror = (error) => {
          isLoading = false;
          setLoadError(error);
          reject(error);
        };
      });

      // Add the script to the DOM
      document.head.appendChild(script);
    };

    loadGoogleMapsApi();

    // Cleanup
    return () => {
      // Only clean up callback if this component is unmounting
      // We don't want to remove the script or callback if other components still need Google Maps
    };
  }, [apiKey, libraries.join(','), enableWebGL]);

  return { isLoaded, loadError };
};

export default useGoogleMapsApi;