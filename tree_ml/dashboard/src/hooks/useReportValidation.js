// src/hooks/useReportValidation.js

import { useState, useEffect, useCallback } from 'react';
import { ValidationService } from '../services/api/apiService';

export const useValidation = () => {
  const [validationItems, setValidationItems] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const refreshValidationItems = useCallback(() => {
    // Increment the refresh trigger to force a refresh
    setRefreshTrigger(prev => prev + 1);
  }, []);

  // Only fetch once on mount, or when explicitly triggered
  useEffect(() => {
    // Track if component is mounted to prevent state updates after unmount
    let isMounted = true;
    
    const fetchValidationItems = async () => {
      try {
        // Fetch validation items from backend API
        const data = await ValidationService.getValidationQueue();
        // Only update state if component is still mounted
        if (isMounted) {
          setValidationItems(data);
        }
      } catch (err) {
        console.error('Error fetching validation queue:', err);
        if (isMounted) {
          setError(err);
        }
      }
    };

    fetchValidationItems();
    
    // Cleanup function
    return () => {
      isMounted = false;
    };
  }, [refreshTrigger]);

  const validateItem = async (itemId, status) => {
    setIsProcessing(true);
    try {
      // Call the backend API to update validation status
      const updatedItem = await ValidationService.updateValidationStatus(itemId, status);
      
      // Update local state to reflect the change
      setValidationItems(items =>
        items.map(item =>
          item.id === itemId ? updatedItem : item
        )
      );
    } catch (err) {
      console.error('Error updating validation status:', err);
      setError(err);
    } finally {
      setIsProcessing(false);
    }
  };

  return { validationItems, validateItem, isProcessing, error, refreshValidationItems };
};