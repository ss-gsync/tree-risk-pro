// src/hooks/useValidation.js

import { useState, useEffect } from 'react';
import { ValidationService } from '../services/api/apiService';

export const useValidation = (initialFilters = {}) => {
  const [validationItems, setValidationItems] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState(initialFilters);
  const [isProcessing, setIsProcessing] = useState(false);

  // Fetch validation items when filters change
  useEffect(() => {
    const fetchValidationItems = async () => {
      try {
        setIsLoading(true);
        setError(null);
        
        const data = await ValidationService.getValidationQueue(filters);
        setValidationItems(data);
      } catch (err) {
        console.error('Error fetching validation items:', err);
        setError(err.message || 'Failed to load validation items');
      } finally {
        setIsLoading(false);
      }
    };

    fetchValidationItems();
  }, [filters]);

  // Function to update validation status
  const validateItem = async (itemId, status, notes = {}) => {
    setIsProcessing(true);
    try {
      const updatedItem = await ValidationService.updateValidationStatus(itemId, status, notes);
      
      // Update local state to reflect the change
      setValidationItems(items =>
        items.map(item =>
          item.id === itemId ? updatedItem : item
        )
      );
      
      return updatedItem;
    } catch (err) {
      console.error('Error updating validation status:', err);
      setError(err.message || 'Failed to update validation status');
      throw err;
    } finally {
      setIsProcessing(false);
    }
  };

  // Function to update filters
  const updateFilters = (newFilters) => {
    setFilters(prev => ({
      ...prev,
      ...newFilters
    }));
  };

  return { 
    validationItems, 
    validateItem, 
    isLoading, 
    isProcessing, 
    error,
    filters,
    updateFilters
  };
};