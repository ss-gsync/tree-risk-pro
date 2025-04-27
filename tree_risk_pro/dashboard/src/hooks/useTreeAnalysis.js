// src/hooks/useTreeAnalysis.js

import { useState, useEffect } from 'react';
import { TreeService } from '../services/api/apiService';

export const useTreeAnalysis = (treeId) => {
  const [treeData, setTreeData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isUpdating, setIsUpdating] = useState(false);

  useEffect(() => {
    const fetchTreeData = async () => {
      if (!treeId) {
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        const data = await TreeService.getTree(treeId);
        
        // Format data if needed
        const formattedData = {
          ...data,
          // Generate growth data if not present in the API response
          growthData: data.growthData || generateMockGrowthData()
        };
        
        setTreeData(formattedData);
      } catch (err) {
        console.error('Error fetching tree data:', err);
        setError(err.message || 'Failed to load tree data');
      } finally {
        setIsLoading(false);
      }
    };

    fetchTreeData();
  }, [treeId]);

  // Function to update tree assessment
  const updateTreeAssessment = async (assessmentData) => {
    if (!treeId) return;

    try {
      setIsUpdating(true);
      const updatedTree = await TreeService.updateTreeAssessment(treeId, assessmentData);
      
      // Update local state
      setTreeData(prev => ({
        ...prev,
        ...updatedTree
      }));
      
      return updatedTree;
    } catch (err) {
      console.error('Error updating tree assessment:', err);
      setError(err.message || 'Failed to update tree assessment');
      throw err;
    } finally {
      setIsUpdating(false);
    }
  };

  // Generate mock growth data for development purposes
  const generateMockGrowthData = () => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return months.map((month, index) => ({
      month,
      growth: ((index + 1) * 0.1 + Math.random() * 0.05).toFixed(2) * 1
    }));
  };

  return { 
    treeData, 
    isLoading, 
    error,
    updateTreeAssessment,
    isUpdating
  };
};