// src/hooks/useGeminiTreeDetection.js

import { useState, useCallback, useEffect } from 'react';
import { analyzeTreeWithGemini } from '../services/api/geminiService';
import * as toast from '../utils/toast';

/**
 * Custom hook for analyzing trees using Gemini AI
 * 
 * This hook provides functionality to analyze trees using the Gemini AI service,
 * with state management for loading, results, and error handling.
 * 
 * @returns {Object} - Tree analysis state and methods
 */
export const useGeminiTreeDetection = () => {
  // State for analysis process
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isGeminiEnabled, setIsGeminiEnabled] = useState(false);

  // Check if Gemini is enabled from settings
  useEffect(() => {
    try {
      const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
      const useGemini = settings?.geminiSettings?.useGeminiForAnalytics === true || 
                      settings?.gemini?.useGeminiForAnalytics === true;
      setIsGeminiEnabled(useGemini);
    } catch (e) {
      console.error('Error loading Gemini settings:', e);
      setIsGeminiEnabled(false);
    }
  }, []);

  // Listen for settings changes
  useEffect(() => {
    const handleStorageChange = () => {
      try {
        const settings = JSON.parse(localStorage.getItem('treeRiskDashboardSettings') || '{}');
        const useGemini = settings?.geminiSettings?.useGeminiForAnalytics === true || 
                          settings?.gemini?.useGeminiForAnalytics === true;
        setIsGeminiEnabled(useGemini);
      } catch (e) {
        console.error('Error updating Gemini settings:', e);
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, []);

  /**
   * Analyze a tree using Gemini AI
   * 
   * @param {Object} treeData - Tree data to analyze
   * @returns {Promise<Object>} - Analysis results
   */
  const analyzeTree = useCallback(async (treeData) => {
    if (!isGeminiEnabled) {
      setAnalysisError('Gemini AI analytics is disabled. Enable it in Settings.');
      return null;
    }

    try {
      setIsAnalyzing(true);
      setAnalysisError(null);
      
      if (!treeData || !treeData.id) {
        throw new Error('Valid tree data is required for analysis');
      }
      
      // Call the Gemini analysis API
      const result = await analyzeTreeWithGemini(treeData);
      
      if (result.success) {
        setAnalysisResult(result);
        toast.success('Tree analysis complete');
        return result;
      } else {
        throw new Error(result.message || 'Tree analysis failed');
      }
    } catch (error) {
      console.error('Error analyzing tree with Gemini:', error);
      setAnalysisError(error.message || 'Tree analysis failed');
      toast.error('Analysis failed: ' + (error.message || 'Unknown error'));
      return null;
    } finally {
      setIsAnalyzing(false);
    }
  }, [isGeminiEnabled]);

  /**
   * Clear the current analysis results
   */
  const clearAnalysis = useCallback(() => {
    setAnalysisResult(null);
    setAnalysisError(null);
  }, []);

  return {
    analyzeTree,
    clearAnalysis,
    isAnalyzing,
    analysisError,
    analysisResult,
    isGeminiEnabled
  };
};

export default useGeminiTreeDetection;